#!/usr/bin/env python
"""PedRL pipeline runner.

Stages (run each in its own process so GPU memory is fully released between them):

  python run.py eval-base   [--preset smoke|poc]   # baseline student pass@1
  python run.py teacher     [--preset ...]         # stage 1: GRPO pedagogical teacher
  python run.py corpus      [--preset ...]         # stage 2a: sample teacher, filter, rank by G_spike
  python run.py assimilate  [--preset ...]         # stage 2b: surprisal-gated distillation
  python run.py eval-student[--preset ...]         # student pass@1 after assimilation
  python run.py student-rl  [--preset ...]         # optional stage 3: plain GRPO on the student
  python run.py all         [--preset ...]         # eval-base -> teacher -> corpus -> assimilate -> eval-student

Ablation: add --no-gating to `assimilate` for plain rejection-sampling SFT.
Any PedRLConfig field can be overridden with --set key=value (repeatable).
"""

import argparse
import dataclasses
import os
import subprocess
import sys

from pedrl.config import PedRLConfig, apply_preset

STAGES = ["eval-base", "teacher", "corpus", "assimilate", "eval-student", "student-rl", "all"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--preset", choices=["smoke", "poc"], default="poc")
    p.add_argument("--no-gating", action="store_true", help="disable the surprisal gate (SFT ablation)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any PedRLConfig field, e.g. --set teacher_steps=200")
    return p.parse_args()


def build_config(args) -> PedRLConfig:
    cfg = apply_preset(PedRLConfig(), args.preset)
    if args.no_gating:
        cfg.gating = False
    field_types = {f.name: f.type for f in dataclasses.fields(PedRLConfig)}
    for kv in args.set:
        key, _, value = kv.partition("=")
        if key not in field_types:
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
    return cfg


def run_stage(stage: str, cfg: PedRLConfig):
    if stage == "eval-base":
        from pedrl.evaluate import evaluate
        evaluate(cfg, adapter_dir=None, tag="base")
    elif stage == "teacher":
        from pedrl.teacher import train_teacher
        train_teacher(cfg, pedagogical=True)
    elif stage == "corpus":
        from pedrl.distill import build_corpus
        build_corpus(cfg)
    elif stage == "assimilate":
        from pedrl.distill import assimilate
        assimilate(cfg)
    elif stage == "eval-student":
        from pedrl.evaluate import evaluate
        tag = "student" if cfg.gating else "student_sft_ablation"
        evaluate(cfg, adapter_dir=cfg.student_adapter_dir, tag=tag)
    elif stage == "student-rl":
        from pedrl.teacher import train_teacher
        from pedrl.evaluate import evaluate
        out = os.path.join(cfg.output_dir, "student_rl_adapter")
        train_teacher(cfg, pedagogical=False, init_adapter=cfg.student_adapter_dir, save_dir=out)
        evaluate(cfg, adapter_dir=out, tag="student_rl")
    else:
        raise ValueError(stage)


def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.output_dir, "config.json"))

    if args.stage == "all":
        # run each stage as a subprocess so GPU memory is released between stages
        passthrough = ["--preset", args.preset] + sum([["--set", kv] for kv in args.set], [])
        if args.no_gating:
            passthrough.append("--no-gating")
        for stage in ["eval-base", "teacher", "corpus", "assimilate", "eval-student"]:
            print(f"\n{'=' * 60}\n>>> stage: {stage}\n{'=' * 60}")
            subprocess.run([sys.executable, __file__, stage] + passthrough, check=True)
    else:
        run_stage(args.stage, cfg)


if __name__ == "__main__":
    main()
