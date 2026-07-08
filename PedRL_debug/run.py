#!/usr/bin/env python
"""PedRL-on-DebugBench pipeline runner. Run from the repo root:

  python PedRL_debug/run.py build-verified [--preset ...]  # reconstruct executable tests (once)
  python PedRL_debug/run.py probe          [--preset ...]  # HYPOTHESIS CHECK: blind vs privileged
                                                           # pass rate of the same base model
  python PedRL_debug/run.py eval-base      [--preset ...]  # baseline student pass@1
  python PedRL_debug/run.py teacher        [--preset ...]  # stage 1: GRPO pedagogical teacher
  python PedRL_debug/run.py corpus         [--preset ...]  # stage 2a: sample teacher, keep passing fixes
  python PedRL_debug/run.py assimilate     [--preset ...]  # stage 2b: surprisal-gated distillation
  python PedRL_debug/run.py eval-student   [--preset ...]  # student pass@1 after assimilation
  python PedRL_debug/run.py all            [--preset ...]  # the pipeline above, in order

Presets: smoke (T4 pipeline check, Coder-0.5B) | poc (dense, Coder-1.5B) |
         hard (sparse hard subset, Llama-3.2-3B, A100 — runs filter-hard first,
               evals also report the held-out hard slice as eval_*_hard.json)

Analysis / baselines:

  python PedRL_debug/run.py filter-hard    [--preset ...]  # keep problems the blind student fails 0/k
  python PedRL_debug/run.py baseline-rl    [--preset ...]  # vanilla GRPO, same rollout budget
  python PedRL_debug/run.py curve-pedrl    [--preset ...]  # pass@1 vs teacher rollouts (per checkpoint)
  python PedRL_debug/run.py curve-baseline [--preset ...]  # pass@1 vs rollouts for vanilla GRPO
  python PedRL_debug/run.py student-rl     [--preset ...]  # optional stage 3
  python PedRL_debug/run.py eval-adapter --set eval_adapter_dir=... --set eval_tag=...
  python PedRL_debug/run.py eval-leetcode  --set eval_tag=student   # re-grade an existing
      # eval_<tag>.json on LeetCode's full hidden suites (official judge; needs
      # LEETCODE_SESSION in the env + leetcode_env installed; ~15 s/submission)

Ablations: --no-gating (plain rejection-sampling SFT); --set privileged=solution
(hand the teacher the full patch instead of the bug explanation).
Any DebugPedRLConfig field can be overridden with --set key=value (repeatable).
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PedRL_debug.config import DebugPedRLConfig, apply_preset

STAGES = [
    "build-verified", "probe", "eval-base", "teacher", "corpus", "assimilate",
    "eval-student", "all", "filter-hard", "baseline-rl", "student-rl",
    "eval-adapter", "eval-leetcode", "curve-pedrl", "curve-baseline",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--preset", choices=["smoke", "poc", "hard"], default="poc")
    p.add_argument("--no-gating", action="store_true", help="disable the surprisal gate (SFT ablation)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any DebugPedRLConfig field, e.g. --set teacher_steps=200")
    return p.parse_args()


def build_config(args) -> DebugPedRLConfig:
    import dataclasses

    cfg = apply_preset(DebugPedRLConfig(), args.preset)
    if args.no_gating:
        cfg.gating = False
    field_names = {f.name for f in dataclasses.fields(DebugPedRLConfig)}
    for kv in args.set:
        key, _, value = kv.partition("=")
        if key not in field_names:
            raise SystemExit(f"unknown config field: {key}")
        current = getattr(cfg, key)
        if isinstance(current, bool):
            setattr(cfg, key, value.lower() in ("1", "true", "yes"))
        elif isinstance(current, int):
            setattr(cfg, key, int(value))
        elif isinstance(current, float):
            setattr(cfg, key, float(value))
        else:
            setattr(cfg, key, value)
    return cfg.finalize()


# ---------------------------------------------------------------------------
# orchestration helpers (parent process; heavy work runs in subprocesses)
# ---------------------------------------------------------------------------

def _sub(args, stage: str, extra_sets=()):
    cmd = [sys.executable, __file__, stage, "--preset", args.preset]
    if args.no_gating:
        cmd.append("--no-gating")
    for kv in list(args.set) + list(extra_sets):
        cmd += ["--set", kv]
    print(f"\n{'=' * 60}\n>>> {' '.join(cmd[1:])}\n{'=' * 60}")
    subprocess.run(cmd, check=True)


def list_checkpoints(root: str):
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        m = re.fullmatch(r"step_(\d+)", name)
        if m:
            out.append((int(m.group(1)), os.path.join(root, name)))
    return sorted(out)


def _read_acc(cfg, tag: str):
    with open(os.path.join(cfg.output_dir, f"eval_{tag}.json")) as f:
        return json.load(f)["accuracy"]


def _write_curve(cfg, name: str, points):
    path = os.path.join(cfg.output_dir, f"curve_{name}.json")
    with open(path, "w") as f:
        json.dump({"rollouts_per_step": cfg.rollouts_per_step, "points": points}, f, indent=2)
    print(f"\n[curve:{name}]")
    for pt in points:
        print(f"  step {pt['step']:>4}  rollouts {pt['rollouts']:>5}  pass@1 = {pt['accuracy']:.3f}")
    print(f"-> {path}")


def _curve_eval_sets(cfg):
    sets = [f"n_eval={cfg.curve_n_eval}"]
    if cfg.use_hard_set:
        sets.append(f"eval_filter_path={cfg.hard_test_path}")
    return sets


def curve_baseline(args, cfg):
    ckpts = list_checkpoints(os.path.join(cfg.output_dir, "baseline_adapter_checkpoints"))
    if not ckpts:
        raise SystemExit("no baseline checkpoints found — run `baseline-rl` first")
    points = []
    base_tag = "base_hard" if cfg.use_hard_set else "base"
    base_eval = os.path.join(cfg.output_dir, f"eval_{base_tag}.json")
    if os.path.exists(base_eval):
        with open(base_eval) as f:
            points.append({"step": 0, "rollouts": 0, "accuracy": json.load(f)["accuracy"]})
    for step, path in ckpts:
        tag = f"baseline_step{step}"
        _sub(args, "eval-adapter",
             [f"eval_adapter_dir={path}", f"eval_tag={tag}"] + _curve_eval_sets(cfg))
        points.append({"step": step, "rollouts": step * cfg.rollouts_per_step,
                       "accuracy": _read_acc(cfg, tag)})
    _write_curve(cfg, "baseline", points)


def curve_pedrl(args, cfg):
    ckpts = list_checkpoints(cfg.teacher_adapter_dir + "_checkpoints")
    if not ckpts:
        raise SystemExit("no teacher checkpoints found — run `teacher` first")
    curve_dir = os.path.join(cfg.output_dir, "curve")
    points = []
    for step, adapter in [(0, "none")] + ckpts:
        tag = f"pedrl_step{step}"
        corpus = os.path.join(curve_dir, f"corpus_step{step}.jsonl")
        student = os.path.join(curve_dir, f"student_step{step}")
        _sub(args, "corpus", [
            f"teacher_adapter_dir={adapter}", f"distill_corpus_path={corpus}",
            f"n_distill={cfg.curve_n_distill}",
        ])
        _sub(args, "assimilate", [
            f"distill_corpus_path={corpus}", f"student_adapter_dir={student}",
        ])
        _sub(args, "eval-adapter",
             [f"eval_adapter_dir={student}", f"eval_tag={tag}"] + _curve_eval_sets(cfg))
        points.append({"step": step, "rollouts": step * cfg.rollouts_per_step,
                       "accuracy": _read_acc(cfg, tag)})
    _write_curve(cfg, "pedrl", points)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def run_stage(stage: str, cfg: DebugPedRLConfig):
    if stage == "build-verified":
        from PedRL_debug.verifier import build_verified
        build_verified(cfg)
    elif stage == "probe":
        from PedRL_debug.probe import probe
        probe(cfg)
    elif stage == "filter-hard":
        from PedRL_debug.probe import build_hard_set
        build_hard_set(cfg)
    elif stage == "eval-base":
        from PedRL_debug.evaluate import evaluate_all
        evaluate_all(cfg, adapter_dir=None, tag="base")
    elif stage == "teacher":
        from PedRL_debug.teacher import train_teacher
        train_teacher(cfg, pedagogical=True)
    elif stage == "corpus":
        from PedRL_debug.distill import build_corpus
        build_corpus(cfg)
    elif stage == "assimilate":
        from PedRL_debug.distill import assimilate
        assimilate(cfg)
    elif stage == "eval-student":
        from PedRL_debug.evaluate import evaluate_all
        tag = "student" if cfg.gating else "student_sft_ablation"
        evaluate_all(cfg, adapter_dir=cfg.student_adapter_dir, tag=tag)
    elif stage == "eval-adapter":
        from PedRL_debug.evaluate import evaluate
        adapter = cfg.eval_adapter_dir or None
        evaluate(cfg, adapter_dir=adapter, tag=cfg.eval_tag or "adapter",
                 filter_path=cfg.eval_filter_path or None)
    elif stage == "eval-leetcode":
        # official-judge re-grade of an existing eval file; no GPU, network only
        from PedRL_debug.leetcode_eval import evaluate_leetcode
        evaluate_leetcode(cfg, tag=cfg.eval_tag or "student")
    elif stage == "baseline-rl":
        from PedRL_debug.teacher import train_teacher
        from PedRL_debug.evaluate import evaluate_all
        out = os.path.join(cfg.output_dir, "baseline_adapter")
        train_teacher(cfg, pedagogical=False, save_dir=out,
                      max_steps=cfg.teacher_steps, log_name="baseline")
        evaluate_all(cfg, adapter_dir=out, tag="baseline_rl")
    elif stage == "student-rl":
        from PedRL_debug.teacher import train_teacher
        from PedRL_debug.evaluate import evaluate_all
        out = os.path.join(cfg.output_dir, "student_rl_adapter")
        train_teacher(cfg, pedagogical=False, init_adapter=cfg.student_adapter_dir, save_dir=out)
        evaluate_all(cfg, adapter_dir=out, tag="student_rl")
    else:
        raise ValueError(stage)


def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.output_dir, "config.json"))

    if args.stage == "all":
        stages = ["build-verified", "eval-base", "teacher", "corpus",
                  "assimilate", "eval-student"]
        if cfg.use_hard_set:
            stages.insert(1, "filter-hard")
        for stage in stages:
            _sub(args, stage)
    elif args.stage == "curve-baseline":
        curve_baseline(args, cfg)
    elif args.stage == "curve-pedrl":
        curve_pedrl(args, cfg)
    else:
        run_stage(args.stage, cfg)


if __name__ == "__main__":
    main()
