"""Configuration for PedRL on DebugBench (bug fixing with a privileged witness).

Same three-stage recipe as pedrl/ (teacher GRPO -> corpus -> gated
assimilation), but the task is: given buggy LeetCode code, produce fixed code
that passes executable tests reconstructed from the problem's worked examples.

Privileged information c given to the teacher:
  - "explanation": the dataset's bug_explanation — a diagnosis of the inserted
    bug(s). This is the interesting condition: a *witness* for debugging.
  - "solution": the full corrected code (the patch) — the upper bound.

The domain hypothesis: unlike math (where knowing the final answer often does
not help construct the derivation), debugging is a reversal-friendly task —
given the witness, the fix is far easier than the blind search. If true, the
privileged/blind pass-rate gap (the `probe` stage) is large, and pedagogical
distillation of that gap should beat vanilla GRPO on sample efficiency.
"""

from dataclasses import dataclass, fields
from typing import Optional
import json
import os


@dataclass
class DebugPedRLConfig:
    # ---- model ----
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    output_dir: str = "outputs_debug"
    seed: int = 42

    # LoRA (used for both the teacher adapter and the student adapter)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # ---- data ----
    dataset_name: str = "Rtian/DebugBench"
    language: str = "python3"          # only python3 is locally executable
    # Output of the build-verified stage: problems whose example tests the
    # reference solution passes AND the buggy code fails. Shared across
    # presets/output dirs — built once.
    verified_path: str = "PedRL_debug/verified_debugbench.json"
    n_test_holdout: int = 150          # verified problems held out for eval
    n_train: int = 256                 # problems used for teacher GRPO
    n_distill: int = 256               # problems used to build the assimilation corpus
    n_eval: int = 150                  # held-out problems for evaluation
    privileged: str = "explanation"    # "explanation" (witness) | "solution" (patch)
    teacher_system: str = ""           # custom template ({explanation} / {solution})

    # ---- verifier ----
    time_limit: float = 4.0            # seconds per test case
    verify_workers: int = 8            # parallel grading subprocesses
    partial_credit: bool = False       # reward = frac tests passed (else all-or-nothing)

    # ---- official LeetCode judge (eval-leetcode stage; NEVER used in training) ----
    # Requires LEETCODE_SESSION in the environment and the leetcode_env package
    # (pip install git+https://github.com/GammaTauAI/leetcode-hard-gym.git).
    leetcode_cooldown: float = 15.0    # seconds between submissions — keep it civil
    leetcode_submit_all: bool = False  # also submit local-fail completions (spends quota)

    # ---- hard-subset mode (sparse-reward regime) ----
    use_hard_set: bool = False
    hard_pool: int = 512               # train problems screened by filter-hard
    hard_test_pool: int = 150          # held-out problems screened for the hard eval slice
    hard_k: int = 4                    # hard = base student fails all k samples
    n_eval_hard: int = 100
    # probe: blind vs privileged pass rate of the SAME base model — the
    # witness-gap hypothesis check (run before any training)
    probe_n: int = 96
    probe_k: int = 0                   # 0 = num_generations

    # ---- shared generation ----
    max_prompt_length: int = 1280      # problems w/ longer prompts are dropped at load
    max_completion_length: int = 768   # diagnosis + full corrected code

    # ---- spike-aware learnability score G_spike ----
    spike_beta: float = 5.0
    spike_lambda: float = 0.5

    # ---- stage 1: teacher GRPO ----
    teacher_steps: int = 100
    teacher_lr: float = 1e-5
    num_generations: int = 8
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    grpo_temperature: float = 0.9
    grpo_kl_beta: float = 0.0
    logging_steps: int = 5

    # ---- stage 2: teacher sampling + assimilation ----
    distill_samples_per_problem: int = 4
    distill_temperature: float = 0.8
    distill_top_p: float = 0.95
    gen_batch_size: int = 16
    assim_epochs: int = 2
    assim_lr: float = 1e-4
    assim_batch_size: int = 4
    assim_grad_accum: int = 4
    gate_kappa: float = 2.0
    gate_gamma: float = -3.5
    gating: bool = True

    # ---- stage 3 (optional): plain GRPO on the student ----
    student_rl_steps: int = 60

    # ---- eval ----
    eval_max_new_tokens: int = 896
    eval_batch_size: int = 16

    # ---- analysis: adapter checkpoints + learning curves ----
    checkpoint_every: int = 25
    curve_n_distill: int = 96
    curve_n_eval: int = 100

    # ---- paths (empty = auto-derived from output_dir in finalize()) ----
    teacher_adapter_dir: str = ""
    student_adapter_dir: str = ""
    distill_corpus_path: str = ""
    eval_adapter_dir: str = ""
    eval_tag: str = ""
    eval_filter_path: str = ""
    hard_train_path: str = ""
    hard_test_path: str = ""

    def finalize(self) -> "DebugPedRLConfig":
        if not self.teacher_adapter_dir:
            self.teacher_adapter_dir = os.path.join(self.output_dir, "teacher_adapter")
        if not self.student_adapter_dir:
            name = "student_adapter" if self.gating else "student_adapter_sft"
            self.student_adapter_dir = os.path.join(self.output_dir, name)
        if not self.distill_corpus_path:
            self.distill_corpus_path = os.path.join(self.output_dir, "distill_corpus.jsonl")
        if not self.hard_train_path:
            self.hard_train_path = os.path.join(self.output_dir, "hard_train.json")
        if not self.hard_test_path:
            self.hard_test_path = os.path.join(self.output_dir, "hard_test.json")
        return self

    @property
    def train_filter(self) -> Optional[str]:
        return self.hard_train_path if self.use_hard_set else None

    @property
    def rollouts_per_step(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({f_.name: getattr(self, f_.name) for f_ in fields(self)}, f, indent=2)


def apply_preset(cfg: DebugPedRLConfig, preset: Optional[str]) -> DebugPedRLConfig:
    """Presets:
    - smoke: full-pipeline check in minutes (T4)
    - poc:   dense-reward run, Qwen2.5-Coder-1.5B (T4/A100)
    - hard:  sparse-reward hard subset, Llama-3.2-3B (A100) — a NON-coder model
             on problems it can't fix blind, witness-only privilege
    """
    if preset in (None, "poc"):
        return cfg
    if preset == "hard":
        # Mirror pedrl's hard preset philosophy: a model not specialized for the
        # domain (Llama, not a Coder model), restricted to problems the blind
        # student fails 0/k, so on-policy reward is sparse by construction.
        cfg.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        cfg.output_dir = "outputs_debug_hard"
        cfg.use_hard_set = True
        cfg.n_train = 96
        cfg.n_distill = 96
        cfg.teacher_steps = 80
        cfg.checkpoint_every = 20
        cfg.distill_samples_per_problem = 6
        cfg.curve_n_distill = 64
        cfg.gen_batch_size = 24
        cfg.eval_batch_size = 24
        return cfg
    if preset == "smoke":
        cfg.model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        cfg.n_train = 12
        cfg.n_distill = 12
        cfg.n_eval = 16
        cfg.teacher_steps = 6
        cfg.num_generations = 4
        cfg.per_device_train_batch_size = 4
        cfg.gradient_accumulation_steps = 1
        cfg.max_completion_length = 512
        cfg.eval_max_new_tokens = 512
        cfg.distill_samples_per_problem = 2
        cfg.assim_epochs = 1
        cfg.student_rl_steps = 4
        cfg.logging_steps = 1
        cfg.checkpoint_every = 3
        cfg.curve_n_distill = 8
        cfg.curve_n_eval = 10
        cfg.probe_n = 12
        return cfg
    raise ValueError(f"unknown preset: {preset}")
