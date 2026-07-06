"""Reward functions for GRPO.

PedagogicalReward implements the product-form reward of the blog post:

    r_ped(x, c, tau) = R(x, c, tau) * G_spike^{theta_S}(tau | x)

R is binary GSM8K correctness of the teacher completion; G_spike is measured
under the FROZEN student policy given the student's (un-privileged) prompt.
The student is recovered from the training model itself by disabling the
teacher's LoRA adapter (OPSD-style single-model setup), so no second copy of
the weights is needed.
"""

import contextlib
import json
import os
import time
from typing import List, Optional

from .data import answers_match, extract_prediction
from .surprisal import score_completions


class PedagogicalReward:
    def __init__(self, tokenizer, cfg, pedagogical: bool = True,
                 log_path: Optional[str] = None):
        self.__name__ = "pedagogical_reward" if pedagogical else "correctness_reward"
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pedagogical = pedagogical
        self.model = None  # attached after the trainer builds/prepares the model
        self.log_path = log_path
        self._n_calls = 0
        self._n_rollouts = 0
        self._t0 = time.time()

    def _log(self, record: dict) -> None:
        """Append one metrics record per reward call — the raw data behind the
        'surprisal decreases as the teacher trains' plot."""
        if self.log_path is None:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def attach(self, model) -> None:
        """Attach the (PEFT-wrapped) policy model used to score surprisal."""
        self.model = model

    def _student_context(self):
        if hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        completion_ids: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> List[float]:
        answers = kwargs["answer"]
        correct = [
            1.0 if answers_match(extract_prediction(c), a) else 0.0
            for c, a in zip(completions, answers)
        ]
        n = len(correct)
        self._n_calls += 1
        self._n_rollouts += n

        if not self.pedagogical:
            self._log({
                "call": self._n_calls,
                "rollouts": self._n_rollouts,
                "acc": sum(correct) / n,
                "mean_reward": sum(correct) / n,
                "elapsed_s": round(time.time() - self._t0, 1),
            })
            return correct

        if self.model is None:
            raise RuntimeError(
                "PedagogicalReward.attach(model) must be called before training."
            )
        student_prompts = kwargs["student_prompt"]
        if completion_ids is None:
            completion_ids = [
                self.tokenizer(c, add_special_tokens=False)["input_ids"]
                for c in completions
            ]
        else:
            # some TRL versions pass tensors; normalize to lists of ints
            completion_ids = [
                c.tolist() if hasattr(c, "tolist") else list(c) for c in completion_ids
            ]

        was_training = self.model.training
        self.model.eval()
        with self._student_context():
            scores = score_completions(
                self.model,
                self.tokenizer,
                student_prompts,
                completion_ids,
                beta=self.cfg.spike_beta,
                lam=self.cfg.spike_lambda,
            )
        if was_training:
            self.model.train()

        rewards = [r * s.g for r, s in zip(correct, scores)]

        self._log({
            "call": self._n_calls,
            "rollouts": self._n_rollouts,
            "acc": sum(correct) / n,
            "mean_g": sum(s.g for s in scores) / n,
            "mean_gap": sum(s.mean_gap for s in scores) / n,
            "mean_max_gap": sum(s.max_gap for s in scores) / n,
            # empty completions carry mean_logp = -inf; keep the log finite
            "mean_logp": (
                sum(s.mean_logp for s in scores if s.n_tokens > 0)
                / max(1, sum(1 for s in scores if s.n_tokens > 0))
            ),
            "mean_reward": sum(rewards) / n,
            "elapsed_s": round(time.time() - self._t0, 1),
        })
        if self._n_calls % 5 == 1:
            print(
                f"[reward] acc={sum(correct)/n:.2f} "
                f"G={sum(s.g for s in scores)/n:.3f} "
                f"max_gap={sum(s.max_gap for s in scores)/n:.2f} "
                f"r_ped={sum(rewards)/n:.3f}"
            )
        return rewards
