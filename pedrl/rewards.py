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
from typing import List, Optional

from .data import answers_match, extract_prediction
from .surprisal import score_completions


class PedagogicalReward:
    def __init__(self, tokenizer, cfg, pedagogical: bool = True):
        self.__name__ = "pedagogical_reward" if pedagogical else "correctness_reward"
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pedagogical = pedagogical
        self.model = None  # attached after the trainer builds/prepares the model
        self._n_calls = 0

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

        if not self.pedagogical:
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

        self._n_calls += 1
        if self._n_calls % 5 == 1:
            n = len(rewards)
            print(
                f"[reward] acc={sum(correct)/n:.2f} "
                f"G={sum(s.g for s in scores)/n:.3f} "
                f"max_gap={sum(s.max_gap for s in scores)/n:.2f} "
                f"r_ped={sum(rewards)/n:.3f}"
            )
        return rewards
