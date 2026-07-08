"""probe + filter-hard — the hypothesis check and the sparse-regime builder.

probe (run BEFORE any training) is the core experiment of this variant:
sample the SAME base model k times blind (student prompt) and k times with
the privileged witness (bug_explanation in the system prompt). The domain
hypothesis is that debugging, unlike math, is reversal-friendly: the witness
converts problems the model is unconditionally incapable of fixing into
fixable ones. It holds if the privileged pass rate is far above the blind
pass rate — that gap is exactly the signal pedagogical distillation can
transfer. Also reported per bug category (syntax / logic / reference /
multiple) and per difficulty level.

filter-hard mirrors pedrl/hardset.py: keep only problems the blind base
student fails at all of hard_k samples, for both the train pool and the
held-out eval slice, so on-policy RL reward is sparse by construction.
"""

import json
import os

from pedrl.modeling import batch_generate, load_model, load_tokenizer, set_seed

from .config import DebugPedRLConfig
from .data import extract_code, load_debugbench, row_problem
from .verifier import grade_many


def _grade_groups(cfg, ds, groups):
    """Per problem: list of pass fractions, one per sampled completion."""
    codes, problems = [], []
    for row, group in zip(ds, groups):
        for g in group:
            codes.append(extract_code(g["text"]))
            problems.append(row_problem(row))
    fracs = grade_many(codes, problems, cfg.time_limit, cfg.verify_workers)
    k = len(groups[0])
    return [fracs[i * k : (i + 1) * k] for i in range(len(groups))]


def build_hard_set(cfg: DebugPedRLConfig) -> None:
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
        ds = load_debugbench(cfg, tokenizer, split=split, n=pool)
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
        frac_groups = _grade_groups(cfg, ds, groups)
        hard = [
            row["id"] for row, fracs in zip(ds, frac_groups)
            if not any(f >= 1.0 for f in fracs)
        ]
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "split": split,
                "pool": pool,
                "k": cfg.hard_k,
                "model": cfg.model_name,
                "ids": hard,
            }, f)
        print(f"[filter-hard] {split}: {len(hard)}/{len(ds)} hard "
              f"(student pass@{cfg.hard_k} = {1 - len(hard)/len(ds):.2f}) -> {out_path}")


def probe(cfg: DebugPedRLConfig) -> dict:
    """Blind vs privileged pass rate of the SAME untrained model.

    Reports, per condition:
      pass@1       — per-sample all-tests-pass rate (reward density)
      any_correct  — fraction of k-sample groups with >=1 passing rollout
      mixed_groups — fraction with 0 < n_pass < k (the groups GRPO can learn from)
    plus per-category and per-level breakdowns of pass@1.

    Uses the hard train subset when use_hard_set (run filter-hard first);
    otherwise the plain train pool — the general witness-gap measurement.
    """
    set_seed(cfg.seed)
    k = cfg.probe_k or cfg.num_generations
    tokenizer = load_tokenizer(cfg.model_name)
    ds = load_debugbench(cfg, tokenizer, split="train", n=cfg.probe_n,
                         filter_path=cfg.train_filter)
    model = load_model(cfg.model_name)  # base model in BOTH conditions
    model.eval()

    results = {"n": len(ds), "k": k, "model": cfg.model_name,
               "privileged": cfg.privileged, "hard_set": bool(cfg.use_hard_set),
               "conditions": {}}
    for cond, column in [("blind", "student_prompt"), ("privileged", "prompt")]:
        print(f"[probe] sampling {cond}: {len(ds)} problems x {k} "
              f"(temp={cfg.grpo_temperature}, mirroring GRPO)")
        groups = batch_generate(
            model,
            tokenizer,
            list(ds[column]),
            max_new_tokens=cfg.max_completion_length,
            batch_size=cfg.gen_batch_size,
            temperature=cfg.grpo_temperature,
            top_p=cfg.distill_top_p,
            num_return_sequences=k,
        )
        frac_groups = _grade_groups(cfg, ds, groups)

        n_pass_total, n_any, n_mixed = 0, 0, 0
        by_cat, by_level = {}, {}
        for row, fracs in zip(ds, frac_groups):
            nc = sum(f >= 1.0 for f in fracs)
            n_pass_total += nc
            n_any += nc > 0
            n_mixed += 0 < nc < k
            for key, table in [(row["category"], by_cat), (row["level"], by_level)]:
                c, t = table.get(key, (0, 0))
                table[key] = (c + nc, t + k)
        results["conditions"][cond] = {
            "pass1": n_pass_total / (len(ds) * k),
            "any_correct": n_any / len(ds),
            "mixed_groups": n_mixed / len(ds),
            "pass1_by_category": {c: p / t for c, (p, t) in sorted(by_cat.items())},
            "pass1_by_level": {l: p / t for l, (p, t) in sorted(by_level.items())},
        }

    out_path = os.path.join(cfg.output_dir, "probe.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    blind = results["conditions"]["blind"]
    priv = results["conditions"]["privileged"]
    print(f"\n[probe] witness gap on {len(ds)} problems, {k} samples each "
          f"(privileged = {cfg.privileged}):")
    for cond, r in results["conditions"].items():
        print(f"  {cond:>11}: pass@1={r['pass1']:.3f}  "
              f">=1 pass in group={r['any_correct']:.2f}  "
              f"mixed (learnable) groups={r['mixed_groups']:.2f}")
    print("  pass@1 by category (blind -> privileged):")
    for cat in blind["pass1_by_category"]:
        b = blind["pass1_by_category"][cat]
        p = priv["pass1_by_category"].get(cat, 0.0)
        print(f"    {cat:>16}: {b:.3f} -> {p:.3f}")
    gap = priv["pass1"] - blind["pass1"]
    if gap > 0.15:
        print(f"[probe] HYPOTHESIS HOLDS: witness lifts pass@1 by {gap:+.3f} — "
              f"debugging benefits strongly from the hint")
    elif priv["pass1"] > 1.5 * max(blind["pass1"], 1e-9):
        print(f"[probe] hypothesis PARTIAL: relative lift "
              f"{priv['pass1']/max(blind['pass1'],1e-9):.1f}x but absolute gap {gap:+.3f} "
              f"is small — consider --set privileged=solution or the hard preset")
    else:
        print(f"[probe] hypothesis WEAK on this slice (gap {gap:+.3f}) — try "
              f"--set privileged=solution, the hard preset, or a smaller model")
    print(f"-> {out_path}")
    return results
