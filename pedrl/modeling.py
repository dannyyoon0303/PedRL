"""Model/tokenizer loading and batched generation helpers."""

import random
from typing import List, Optional

import numpy as np
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, adapter_dir: Optional[str] = None, trainable_adapter: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=pick_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable_adapter)
    return model


def make_lora_config(cfg) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )


@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
) -> List[List[dict]]:
    """Generate completions for chat-templated prompt strings.

    Returns, per prompt, a list of {"text": str, "ids": list[int]} of length
    num_return_sequences.
    """
    model.eval()
    device = next(model.parameters()).device
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    do_sample = temperature > 0.0
    outs: List[List[dict]] = []
    prompts_per_call = max(1, batch_size // num_return_sequences)
    for start in range(0, len(prompts), prompts_per_call):
        chunk = prompts[start : start + prompts_per_call]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        comp = gen[:, prompt_len:]
        for i in range(len(chunk)):
            group = []
            for j in range(num_return_sequences):
                ids = comp[i * num_return_sequences + j]
                ids = ids[ids != tokenizer.pad_token_id].tolist()
                # strip trailing EOS for clean re-scoring
                while ids and ids[-1] == tokenizer.eos_token_id:
                    ids = ids[:-1]
                group.append({"text": tokenizer.decode(ids, skip_special_tokens=True), "ids": ids})
            outs.append(group)

    tokenizer.padding_side = old_side
    return outs
