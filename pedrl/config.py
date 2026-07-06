"""Configuration for Pedagogical RL (PedRL).

One dataclass holds every knob for the three stages:
  1. teacher  — GRPO on the privileged teacher with reward R * G_spike
  2. distill  — sample the teacher, then surprisal-gated assimilation into the student
  3. eval     — greedy pass@1 on GSM8K test
"""

from dataclasses import dataclass, field, fields
from typing import Optional
import json
import os


@dataclass
class PedRLConfig:
    # ---- model ----
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs"
    seed: int = 42

    # LoRA (used for both the teacher adapter and the student adapter)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # ---- data ----
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    n_train: int = 256            # problems used for teacher GRPO
    n_distill: int = 256          # problems used to build the assimilation corpus
    n_eval: int = 200             # test problems for evaluation
    privileged: str = "answer"    # "answer" | "solution" — what the teacher gets to see

    # ---- shared generation ----
    max_prompt_length: int = 512
    max_completion_length: int = 384

    # ---- spike-aware learnability score G_spike ----
    # G = exp( -(lam/beta) * log( (1/T) * sum_t exp(beta * d_t) ) )
    # d_t = log pi_S(argmax token) - log pi_S(actual token)  >= 0
    spike_beta: float = 5.0       # beta -> 0: average surprise; beta -> inf: max surprise
    spike_lambda: float = 0.5     # overall strength of the learnability penalty

    # ---- stage 1: teacher GRPO ----
    teacher_steps: int = 120
    teacher_lr: float = 1e-5
    num_generations: int = 8      # group size for GRPO
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    grpo_temperature: float = 0.9
    grpo_kl_beta: float = 0.0     # 0 => no reference model kept in memory
    logging_steps: int = 5

    # ---- stage 2: teacher sampling + assimilation ----
    distill_samples_per_problem: int = 4
    distill_temperature: float = 0.8
    distill_top_p: float = 0.95
    gen_batch_size: int = 16      # sequences per generate() call
    assim_epochs: int = 2
    assim_lr: float = 1e-4
    assim_batch_size: int = 4
    assim_grad_accum: int = 4
    # token gate w_t = sigmoid(kappa * (log pi_S(token) - gamma))
    gate_kappa: float = 2.0
    gate_gamma: float = -3.5
    gating: bool = True           # False => plain rejection-sampling SFT baseline

    # ---- stage 3 (optional): plain GRPO on the student, initialized from assimilation ----
    student_rl_steps: int = 60

    # ---- eval ----
    eval_max_new_tokens: int = 512
    eval_batch_size: int = 16

    # ---- analysis: adapter checkpoints + learning curves ----
    checkpoint_every: int = 30    # save the GRPO adapter every N steps (0 = off)
    curve_n_distill: int = 128    # problems per corpus when building the PedRL curve
    curve_n_eval: int = 100       # eval problems per curve point

    # ---- paths (empty = auto-derived from output_dir in finalize()) ----
    teacher_adapter_dir: str = ""   # "none" = use the base model as the (untrained) teacher
    student_adapter_dir: str = ""
    distill_corpus_path: str = ""
    eval_adapter_dir: str = ""      # for the eval-adapter stage
    eval_tag: str = ""              # output name for the eval-adapter stage

    def finalize(self) -> "PedRLConfig":
        """Fill in any path left empty. Call after all overrides are applied."""
        if not self.teacher_adapter_dir:
            self.teacher_adapter_dir = os.path.join(self.output_dir, "teacher_adapter")
        if not self.student_adapter_dir:
            # the SFT ablation gets its own directory so it never clobbers the gated student
            name = "student_adapter" if self.gating else "student_adapter_sft"
            self.student_adapter_dir = os.path.join(self.output_dir, name)
        if not self.distill_corpus_path:
            self.distill_corpus_path = os.path.join(self.output_dir, "distill_corpus.jsonl")
        return self

    @property
    def rollouts_per_step(self) -> int:
        """Completions consumed per GRPO optimizer step (single GPU)."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({f_.name: getattr(self, f_.name) for f_ in fields(self)}, f, indent=2)


def filter_kwargs_for_dataclass(cls, kwargs: dict, label: str = "") -> dict:
    """Keep only kwargs that `cls` (a dataclass, e.g. trl.GRPOConfig) accepts.
    TRL renames/removes config fields across versions; dropping unknown ones
    keeps us compatible with whatever version Colab installs."""
    import dataclasses

    valid = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(k for k in kwargs if k not in valid)
    if dropped:
        print(f"[{label or cls.__name__}] this version does not support "
              f"{dropped} — dropping")
    return {k: v for k, v in kwargs.items() if k in valid}


def apply_preset(cfg: PedRLConfig, preset: Optional[str]) -> PedRLConfig:
    """Presets: 'smoke' verifies the full pipeline in minutes; 'poc' is the real run."""
    if preset in (None, "poc"):
        return cfg
    if preset == "smoke":
        cfg.n_train = 16
        cfg.n_distill = 16
        cfg.n_eval = 20
        cfg.teacher_steps = 6
        cfg.num_generations = 4
        cfg.per_device_train_batch_size = 4
        cfg.gradient_accumulation_steps = 1
        cfg.max_completion_length = 192
        cfg.eval_max_new_tokens = 256
        cfg.distill_samples_per_problem = 2
        cfg.assim_epochs = 1
        cfg.student_rl_steps = 4
        cfg.logging_steps = 1
        cfg.checkpoint_every = 3
        cfg.curve_n_distill = 8
        cfg.curve_n_eval = 10
        return cfg
    raise ValueError(f"unknown preset: {preset}")
