"""Greedy pass@1 evaluation on the held-out slice of verified problems.

A problem counts as solved when the extracted fix passes ALL of its example
tests. Per-category and per-level breakdowns are recorded alongside the
overall accuracy.
"""

import json
import os
from typing import Optional

from pedrl.modeling import batch_generate, load_model, load_tokenizer, set_seed

from .config import DebugPedRLConfig
from .data import extract_code, is_echo, load_debugbench, row_problem
from .verifier import grade_many


def evaluate(cfg: DebugPedRLConfig, adapter_dir: Optional[str] = None, tag: str = "base",
             filter_path: Optional[str] = None, n: Optional[int] = None,
             model=None, tokenizer=None) -> float:
    set_seed(cfg.seed)
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg.model_name)
    ds = load_debugbench(cfg, tokenizer, split="test",
                         n=n if n is not None else cfg.n_eval,
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

    codes = [extract_code(g[0]["text"]) for g in groups]
    problems = [row_problem(row) for row in ds]
    fracs = grade_many(codes, problems, cfg.time_limit, cfg.verify_workers)

    records, n_correct = [], 0
    by_cat, by_level = {}, {}
    for row, group, code, frac in zip(ds, groups, codes, fracs):
        ok = frac >= 1.0
        n_correct += int(ok)
        for key, table in [(row["category"], by_cat), (row["level"], by_level)]:
            c, t = table.get(key, (0, 0))
            table[key] = (c + int(ok), t + 1)
        records.append({
            "slug": row["slug"],
            "category": row["category"],
            "level": row["level"],
            "pass_frac": frac,
            "correct": ok,
            "echo": is_echo(code, row["buggy_code"]),
            "completion": group[0]["text"],
        })

    acc = n_correct / len(ds)
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"eval_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "tag": tag,
            "n": len(ds),
            "accuracy": acc,
            "accuracy_by_category": {c: p / t for c, (p, t) in sorted(by_cat.items())},
            "accuracy_by_level": {l: p / t for l, (p, t) in sorted(by_level.items())},
            "records": records,
        }, f, indent=2)
    print(f"[eval:{tag}] pass@1 = {acc:.3f}  ({n_correct}/{len(ds)})  -> {out_path}")
    return acc


def evaluate_all(cfg: DebugPedRLConfig, adapter_dir: Optional[str], tag: str) -> float:
    """Standard eval slice, plus the held-out hard slice when use_hard_set.
    Loads the model once for both."""
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_model(cfg.model_name, adapter_dir=adapter_dir)
    acc = evaluate(cfg, tag=tag, model=model, tokenizer=tokenizer)
    if cfg.use_hard_set:
        evaluate(cfg, tag=f"{tag}_hard", filter_path=cfg.hard_test_path,
                 n=cfg.n_eval_hard, model=model, tokenizer=tokenizer)
    return acc
