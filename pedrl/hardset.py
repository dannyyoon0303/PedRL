"""filter-hard stage — construct the sparse-reward regime the method targets.

Screens a pool of problems with the BASE student (k samples each, no hint) and
keeps only problems it fails every time. On this subset, on-policy RL reward is
~zero by construction (vanilla GRPO stalls) and every correct teacher demo
carries information the student does not already have — the blog's ~8%-pass@1
setting, reconstructed from GSM8K's hard tail.

Writes, for both splits:
  hard_train.json — teacher training + distillation problems
  hard_test.json  — held-out hard eval slice (selection uses the base student
                    only, so later student improvements on it are legitimate)

Files hold indices into the seed-shuffled split; load_gsm8k(filter_path=...)
consumes them. Idempotent: existing files are kept (delete to re-screen).
"""

import json
import os

from .config import PedRLConfig
from .data import answers_match, extract_prediction, load_gsm8k
from .modeling import batch_generate, load_model, load_tokenizer, set_seed


def build_hard_set(cfg: PedRLConfig) -> None:
    set_seed(cfg.seed)
    todo = [
        ("train", cfg.hard_pool, cfg.hard_train_path),
        ("test", cfg.hard_test_pool, cfg.hard_test_path),
    ]
    todo = [(s, p, path) for s, p, path in todo if not os.path.exists(path)]
    if not todo:
        print("[filter-hard] hard-set files already exist — skipping (delete to re-screen)")
        return

    tokenizer = load_tokenizer(cfg.model_name)
    model = load_model(cfg.model_name)  # base student, no adapter, no hint
    model.eval()

    for split, pool, out_path in todo:
        ds = load_gsm8k(cfg, tokenizer, split=split, n=pool)
        print(f"[filter-hard] screening {len(ds)} {split} problems x {cfg.hard_k} samples")
        groups = batch_generate(
            model,
            tokenizer,
            list(ds["student_prompt"]),
            max_new_tokens=cfg.max_completion_length,
            batch_size=cfg.gen_batch_size,
            temperature=cfg.distill_temperature,
            top_p=cfg.distill_top_p,
            num_return_sequences=cfg.hard_k,
        )
        hard = [
            i for i, (ex, group) in enumerate(zip(ds, groups))
            if not any(answers_match(extract_prediction(g["text"]), ex["answer"]) for g in group)
        ]
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "split": split,
                "pool": pool,
                "k": cfg.hard_k,
                "model": cfg.model_name,
                "indices": hard,
            }, f)
        print(f"[filter-hard] {split}: {len(hard)}/{len(ds)} hard "
              f"(student pass@{cfg.hard_k} = {1 - len(hard)/len(ds):.2f}) -> {out_path}")
