"""Spike-aware learnability score (the "surprisal" measure of the blog post).

For a trajectory tau scored under the STUDENT's context x (no privileged info):

    d_t = log pi_S(a_t^max | x, tau_<t) - log pi_S(tau_t | x, tau_<t)   >= 0

    G_spike(tau | x) = exp( -(lambda/beta) * log( (1/T) * sum_t exp(beta * d_t) ) )

beta -> 0 recovers average surprise, beta -> inf recovers maximum surprise;
intermediate beta penalizes rare implausible "spikes" harder than a uniform
level of mild off-policyness. G is in (0, 1], and equals 1 iff every token is
the student's own greedy choice.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F


@dataclass
class SpikeScore:
    g: float            # G_spike in (0, 1]
    mean_gap: float     # mean d_t
    max_gap: float      # max d_t
    mean_logp: float    # mean student log-prob of the actual tokens
    n_tokens: int


def spike_g_from_gaps(gaps: torch.Tensor, beta: float, lam: float) -> float:
    """gaps: 1-D tensor of d_t for one sequence."""
    t = gaps.numel()
    if t == 0:
        return 0.0
    lse = torch.logsumexp(beta * gaps, dim=0) - math.log(t)
    return float(torch.exp(-(lam / beta) * lse))


@torch.no_grad()
def score_completions(
    model,
    tokenizer,
    student_prompts: Sequence[str],
    completion_ids: Sequence[Sequence[int]],
    beta: float,
    lam: float,
    batch_size: int = 8,
) -> List[SpikeScore]:
    """Score each completion (token ids from generation) under the student policy
    conditioned on the *student* prompt. `model` must already be the student
    (e.g. a PEFT model inside a disable_adapter() context, or the base model).
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_ids = [
        tokenizer(p, add_special_tokens=False)["input_ids"] for p in student_prompts
    ]

    results: List[SpikeScore] = []
    for start in range(0, len(prompt_ids), batch_size):
        p_chunk = prompt_ids[start : start + batch_size]
        c_chunk = [list(c) for c in completion_ids[start : start + batch_size]]

        seqs = [p + c for p, c in zip(p_chunk, c_chunk)]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits

        for i, (p, c) in enumerate(zip(p_chunk, c_chunk)):
            if len(c) == 0:
                results.append(SpikeScore(0.0, 0.0, 0.0, float("-inf"), 0))
                continue
            # logits at position j predict token j+1
            sl = logits[i, len(p) - 1 : len(p) - 1 + len(c)].float()
            logp = F.log_softmax(sl, dim=-1)
            targets = torch.tensor(c, dtype=torch.long, device=device)
            actual = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            best = logp.max(dim=-1).values
            gaps = best - actual
            results.append(
                SpikeScore(
                    g=spike_g_from_gaps(gaps, beta, lam),
                    mean_gap=float(gaps.mean()),
                    max_gap=float(gaps.max()),
                    mean_logp=float(actual.mean()),
                    n_tokens=len(c),
                )
            )
        del logits
    return results
