"""Greedy pass@1 evaluation on the GSM8K test split."""

import json
import os
from typing import Optional

from .config import PedRLConfig
from .data import answers_match, extract_prediction, load_gsm8k
from .modeling import batch_generate, load_model, load_tokenizer, set_seed


def evaluate(cfg: PedRLConfig, adapter_dir: Optional[str] = None, tag: str = "base",
             filter_path: Optional[str] = None, n: Optional[int] = None,
             model=None, tokenizer=None) -> float:
    set_seed(cfg.seed)
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg.model_name)
    ds = load_gsm8k(cfg, tokenizer, split="test", n=n if n is not None else cfg.n_eval,
                    filter_path=filter_path)

    if model is None:
        model = load_model(cfg.model_name, adapter_dir=adapter_dir)
    model.eval()

    print(f"[eval:{tag}] {len(ds)} problems, greedy decoding")
    groups = batch_generate(
        model,
        tokenizer,
        list(ds["student_prompt"]),
        max_new_tokens=cfg.eval_max_new_tokens,
        batch_size=cfg.eval_batch_size,
        temperature=0.0,
    )

    records, n_correct = [], 0
    for ex, group in zip(ds, groups):
        text = group[0]["text"]
        pred = extract_prediction(text)
        ok = answers_match(pred, ex["answer"])
        n_correct += int(ok)
        records.append({
            "question": ex["question"],
            "gold": ex["answer"],
            "pred": pred,
            "correct": ok,
            "completion": text,
        })

    acc = n_correct / len(ds)
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"eval_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"tag": tag, "n": len(ds), "accuracy": acc, "records": records}, f, indent=2)
    print(f"[eval:{tag}] pass@1 = {acc:.3f}  ({n_correct}/{len(ds)})  -> {out_path}")
    return acc


def evaluate_all(cfg: PedRLConfig, adapter_dir: Optional[str], tag: str) -> float:
    """Standard eval slice, plus the held-out hard slice when use_hard_set.
    Loads the model once for both."""
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_model(cfg.model_name, adapter_dir=adapter_dir)
    acc = evaluate(cfg, tag=tag, model=model, tokenizer=tokenizer)
    if cfg.use_hard_set:
        evaluate(cfg, tag=f"{tag}_hard", filter_path=cfg.hard_test_path,
                 n=cfg.n_eval_hard, model=model, tokenizer=tokenizer)
    return acc
