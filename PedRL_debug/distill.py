"""Stage 2 — distillation corpus for debugging + gated assimilation.

2a) build_corpus: sample the trained teacher K times per problem, keep
    completions whose extracted fix passes ALL tests, and per problem keep
    the one with the highest G_spike under the frozen student.

2b) assimilate: unchanged from pedrl — the surprisal-gated trainer is
    task-agnostic, so we reuse pedrl.distill.assimilate directly (it reads
    the corpus jsonl, which uses the same schema).
"""

import json
import os

from pedrl.distill import assimilate  # re-exported for run.py  # noqa: F401
from pedrl.modeling import batch_generate, load_model, load_tokenizer, set_seed
from pedrl.surprisal import score_completions

from .config import DebugPedRLConfig
from .data import extract_code, load_debugbench, row_problem
from .verifier import grade_many


def build_corpus(cfg: DebugPedRLConfig) -> str:
    import contextlib

    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)
    ds = load_debugbench(cfg, tokenizer, split="train", n=cfg.n_distill,
                         filter_path=cfg.train_filter)

    # teacher_adapter_dir == "none": the untrained base model as privileged
    # teacher — the rejection-sampling baseline / step-0 curve point
    adapter = cfg.teacher_adapter_dir
    if adapter.lower() == "none":
        adapter = None
    model = load_model(cfg.model_name, adapter_dir=adapter)
    model.eval()

    def student_ctx():
        return model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()

    k = cfg.distill_samples_per_problem
    print(f"[distill] sampling teacher: {len(ds)} problems x {k}")
    groups = batch_generate(
        model,
        tokenizer,
        list(ds["prompt"]),
        max_new_tokens=cfg.max_completion_length,
        batch_size=cfg.gen_batch_size,
        temperature=cfg.distill_temperature,
        top_p=cfg.distill_top_p,
        num_return_sequences=k,
    )

    # grade every sample in one parallel pass
    codes = [extract_code(g["text"]) for group in groups for g in group]
    problems = [row_problem(row) for row in ds for _ in range(k)]
    fracs = grade_many(codes, problems, cfg.time_limit, cfg.verify_workers)

    kept, n_any_correct = [], 0
    for gi, (row, group) in enumerate(zip(ds, groups)):
        correct = [g for j, g in enumerate(group) if fracs[gi * k + j] >= 1.0]
        if not correct:
            continue
        n_any_correct += 1
        with student_ctx():
            scores = score_completions(
                model, tokenizer,
                [row["student_prompt"]] * len(correct),
                [g["ids"] for g in correct],
                beta=cfg.spike_beta, lam=cfg.spike_lambda,
            )
        best = max(range(len(correct)), key=lambda i: scores[i].g)
        kept.append({
            "student_prompt": row["student_prompt"],
            "completion_text": correct[best]["text"],
            "completion_ids": correct[best]["ids"],
            "slug": row["slug"],
            "g_spike": scores[best].g,
            "mean_logp": scores[best].mean_logp,
        })

    os.makedirs(os.path.dirname(cfg.distill_corpus_path) or ".", exist_ok=True)
    with open(cfg.distill_corpus_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    mean_g = sum(r["g_spike"] for r in kept) / max(1, len(kept))
    print(f"[distill] kept {len(kept)}/{len(ds)} problems "
          f"(teacher pass@{k}={n_any_correct/len(ds):.2f}, "
          f"mean G_spike of kept demos={mean_g:.3f})")
    return cfg.distill_corpus_path
