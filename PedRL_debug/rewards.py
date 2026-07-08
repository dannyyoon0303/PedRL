"""GRPO reward for the debugging task.

    r_ped(x, c, tau) = R(x, c, tau) * G_spike^{theta_S}(tau | x)

R: extract the fixed code from the completion and run it against the
problem's verified example tests (all-pass binary, or the pass fraction with
cfg.partial_credit). Because build-verified kept only problems whose buggy
code FAILS at least one test, echoing the buggy code back earns R = 0.

G_spike: measured under the FROZEN student (LoRA adapter disabled) given the
student's un-privileged prompt — identical to pedrl/rewards.py.
"""

import contextlib
import json
import os
import time
from typing import List, Optional

from pedrl.surprisal import score_completions

from .data import extract_code, is_echo
from .verifier import grade_many


class DebugPedagogicalReward:
    def __init__(self, tokenizer, cfg, pedagogical: bool = True,
                 log_path: Optional[str] = None):
        self.__name__ = "debug_pedagogical_reward" if pedagogical else "debug_correctness_reward"
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pedagogical = pedagogical
        self.model = None
        self.log_path = log_path
        self._n_calls = 0
        self._n_rollouts = 0
        self._t0 = time.time()

    def _log(self, record: dict) -> None:
        if self.log_path is None:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def attach(self, model) -> None:
        self.model = model

    def _student_context(self):
        if hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def correctness(self, completions: List[str], kwargs: dict) -> tuple:
        """(rewards R in [0,1], mean pass fraction, echo rate, extract rate)."""
        codes = [extract_code(c) for c in completions]
        problems = [
            {"method": m, "tests": json.loads(tj)}
            for m, tj in zip(kwargs["method"], kwargs["tests_json"])
        ]
        fracs = grade_many(codes, problems, self.cfg.time_limit, self.cfg.verify_workers)
        if self.cfg.partial_credit:
            rewards = list(fracs)
        else:
            rewards = [1.0 if f >= 1.0 else 0.0 for f in fracs]
        n = len(completions)
        echo_rate = sum(
            is_echo(c, b) for c, b in zip(codes, kwargs["buggy_code"])) / n
        extract_rate = sum(c is not None for c in codes) / n
        return rewards, sum(fracs) / n, echo_rate, extract_rate

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        completion_ids: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> List[float]:
        correct, mean_frac, echo_rate, extract_rate = self.correctness(completions, kwargs)
        n = len(correct)
        self._n_calls += 1
        self._n_rollouts += n

        base_record = {
            "call": self._n_calls,
            "rollouts": self._n_rollouts,
            "acc": sum(1.0 for r in correct if r >= 1.0) / n,
            "mean_frac": mean_frac,
            "echo_rate": echo_rate,
            "extract_rate": extract_rate,
            "elapsed_s": round(time.time() - self._t0, 1),
        }

        if not self.pedagogical:
            self._log({**base_record, "mean_reward": sum(correct) / n})
            return correct

        if self.model is None:
            raise RuntimeError(
                "DebugPedagogicalReward.attach(model) must be called before training."
            )
        student_prompts = kwargs["student_prompt"]
        if completion_ids is None:
            completion_ids = [
                self.tokenizer(c, add_special_tokens=False)["input_ids"]
                for c in completions
            ]
        else:
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
            **base_record,
            "mean_g": sum(s.g for s in scores) / n,
            "mean_gap": sum(s.mean_gap for s in scores) / n,
            "mean_max_gap": sum(s.max_gap for s in scores) / n,
            "mean_logp": (
                sum(s.mean_logp for s in scores if s.n_tokens > 0)
                / max(1, sum(1 for s in scores if s.n_tokens > 0))
            ),
            "mean_reward": sum(rewards) / n,
        })
        if self._n_calls % 5 == 1:
            print(
                f"[reward] pass={base_record['acc']:.2f} "
                f"echo={echo_rate:.2f} "
                f"G={sum(s.g for s in scores)/n:.3f} "
                f"r_ped={sum(rewards)/n:.3f}"
            )
        return rewards
