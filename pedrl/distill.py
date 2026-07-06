"""Stage 2 — build the distillation corpus and assimilate it into the student.

2a) Sample the trained teacher (base + teacher LoRA, privileged prompt) K times
    per problem, keep correct completions, and for each problem keep the one
    with the highest G_spike under the frozen student ("most learnable" demo).

2b) Surprisal-gated knowledge assimilation: train a FRESH student LoRA on
    (student_prompt, teacher_completion) with per-token weights

        w_t = sigmoid( kappa * ( log pi_S(tau_t | x, tau_<t) - gamma ) )

    computed from the CURRENT student's own (detached) log-probs, so tokens the
    student finds implausible are down-weighted instead of dominating the update.

    L_assim = E[ (1/sum_t w_t) * sum_t w_t * CE_t ]

With cfg.gating=False this reduces to plain rejection-sampling SFT — the
natural ablation/baseline.
"""

import json
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from .config import PedRLConfig
from .data import answers_match, extract_prediction, load_gsm8k
from .modeling import (
    batch_generate,
    load_model,
    load_tokenizer,
    make_lora_config,
    pick_dtype,
    set_seed,
)
from .surprisal import score_completions


# --------------------------------------------------------------------------
# 2a: corpus generation
# --------------------------------------------------------------------------

def build_corpus(cfg: PedRLConfig) -> str:
    import contextlib

    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)
    ds = load_gsm8k(cfg, tokenizer, split="train", n=cfg.n_distill,
                    filter_path=cfg.train_filter)

    # teacher_adapter_dir == "none" means the untrained (base) model acts as the
    # privileged teacher — the rejection-sampling baseline / step-0 curve point
    adapter = cfg.teacher_adapter_dir
    if adapter.lower() == "none":
        adapter = None
    model = load_model(cfg.model_name, adapter_dir=adapter)
    model.eval()

    def student_ctx():
        return model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()

    print(f"[distill] sampling teacher: {len(ds)} problems x {cfg.distill_samples_per_problem}")
    groups = batch_generate(
        model,
        tokenizer,
        list(ds["prompt"]),
        max_new_tokens=cfg.max_completion_length,
        batch_size=cfg.gen_batch_size,
        temperature=cfg.distill_temperature,
        top_p=cfg.distill_top_p,
        num_return_sequences=cfg.distill_samples_per_problem,
    )

    kept, n_any_correct = [], 0
    for ex, group in zip(ds, groups):
        correct = [g for g in group if answers_match(extract_prediction(g["text"]), ex["answer"])]
        if not correct:
            continue
        n_any_correct += 1
        # pick the most learnable correct demo under the frozen student
        with student_ctx():
            scores = score_completions(
                model, tokenizer,
                [ex["student_prompt"]] * len(correct),
                [g["ids"] for g in correct],
                beta=cfg.spike_beta, lam=cfg.spike_lambda,
            )
        best = max(range(len(correct)), key=lambda i: scores[i].g)
        kept.append({
            "student_prompt": ex["student_prompt"],
            "completion_text": correct[best]["text"],
            "completion_ids": correct[best]["ids"],
            "answer": ex["answer"],
            "g_spike": scores[best].g,
            "mean_logp": scores[best].mean_logp,
        })

    os.makedirs(os.path.dirname(cfg.distill_corpus_path) or ".", exist_ok=True)
    with open(cfg.distill_corpus_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    mean_g = sum(r["g_spike"] for r in kept) / max(1, len(kept))
    print(f"[distill] kept {len(kept)}/{len(ds)} problems "
          f"(teacher pass@{cfg.distill_samples_per_problem}={n_any_correct/len(ds):.2f}, "
          f"mean G_spike of kept demos={mean_g:.3f})")
    return cfg.distill_corpus_path


# --------------------------------------------------------------------------
# 2b: surprisal-gated assimilation
# --------------------------------------------------------------------------

class CorpusDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_len: int):
        self.examples = []
        for r in rows:
            p_ids = tokenizer(r["student_prompt"], add_special_tokens=False)["input_ids"]
            c_ids = list(r["completion_ids"]) + [tokenizer.eos_token_id]
            ids = (p_ids + c_ids)[:max_len]
            labels = ([-100] * len(p_ids) + c_ids)[:max_len]
            if not any(l != -100 for l in labels):
                continue
            self.examples.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


class GatedAssimilationTrainer(Trainer):
    def __init__(self, *args, gate_kappa: float = 2.0, gate_gamma: float = -3.5,
                 gating: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_kappa = gate_kappa
        self.gate_gamma = gate_gamma
        self.gating = gating

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = out.logits[:, :-1].float()
        targets = labels[:, 1:]
        mask = targets != -100

        logp = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        ce = -token_logp

        if self.gating:
            w = torch.sigmoid(self.gate_kappa * (token_logp.detach() - self.gate_gamma))
        else:
            w = torch.ones_like(ce)
        w = w * mask

        per_seq = (w * ce).sum(dim=-1) / w.sum(dim=-1).clamp(min=1e-6)
        loss = per_seq.mean()
        return (loss, out) if return_outputs else loss


def assimilate(cfg: PedRLConfig) -> str:
    from peft import get_peft_model

    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)
    with open(cfg.distill_corpus_path) as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise RuntimeError("distillation corpus is empty — teacher produced no correct samples")

    dataset = CorpusDataset(rows, tokenizer, cfg.max_prompt_length + cfg.max_completion_length + 8)
    base = load_model(cfg.model_name)
    model = get_peft_model(base, make_lora_config(cfg))
    model.print_trainable_parameters()

    dtype = pick_dtype()
    args = TrainingArguments(
        output_dir=os.path.join(cfg.output_dir, "assimilation"),
        num_train_epochs=cfg.assim_epochs,
        learning_rate=cfg.assim_lr,
        per_device_train_batch_size=cfg.assim_batch_size,
        gradient_accumulation_steps=cfg.assim_grad_accum,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
    )
    trainer = GatedAssimilationTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
        gate_kappa=cfg.gate_kappa,
        gate_gamma=cfg.gate_gamma,
        gating=cfg.gating,
    )
    trainer.train()

    os.makedirs(cfg.student_adapter_dir, exist_ok=True)
    model.save_pretrained(cfg.student_adapter_dir)
    tokenizer.save_pretrained(cfg.student_adapter_dir)
    print(f"[assimilate] saved student adapter to {cfg.student_adapter_dir}")
    return cfg.student_adapter_dir
