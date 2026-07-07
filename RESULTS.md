# Experiment log

## Round 1 — dense-reward regime (poc preset, 2026-07)

**Setup:** Qwen2.5-0.5B-Instruct, 256 random GSM8K train problems, answer-only
privilege, single teacher→student round. T4/A100 Colab.

### Headline numbers

| eval (GSM8K test) | pass@1 |
|---|---|
| base student (n=200) | **0.430** |
| PedRL student after assimilation (n=200) | 0.425 |
| vanilla GRPO @ 1920 rollouts (n=100) | **0.500** |
| PedRL curve @ 0/480/960 teacher rollouts (n=100) | 0.450 / 0.450 / 0.430 |

PedRL produced **no eval gain**; vanilla GRPO at the same rollout budget gained
~+7 points. Teacher training itself, however, worked exactly as designed:

### The mechanism worked

From `reward_log_teacher.jsonl` (120 GRPO steps, ~1,920 rollouts):

- `G_spike` **0.19 → ~0.50**, `max_gap` **~5.5 → ~2.8–3.2 nats**, while teacher
  accuracy stayed flat (~0.5–0.6) — correctness held, learnability learned.
- `mean_gap` was **~0.1 nats throughout, flat**: teacher completions were ~97%
  token-identical to the student's distribution *before any training*. Privileged
  off-policyness is concentrated in a few spike tokens at reasoning forks — the
  empirical justification for the *spike-aware* score (a mean-surprisal reward
  would have had almost nothing to optimize).

### Why eval didn't move (diagnosis)

1. **Dense rewards.** The student solves 43% pass@1 / ~0.7 pass@4 of the training
   distribution, so vanilla GRPO gets signal from nearly every rollout group —
   the regime where privileged teaching has no edge. (The blog's setting was ~8%
   pass@1.)
2. **Redundant corpus.** Corpus stats: kept 212/256, teacher pass@4 = 0.83,
   mean demo `G_spike` = 0.59, assimilation CE ≈ 0.23–0.29 → the student already
   assigned ~75–80% per-token probability to the demos before training on them.
   Best-of-G selection + the token gate make assimilation deliberately
   conservative; on easy problems that converges to training the student on its
   own outputs.
3. **Weak privileged "kick" at 0.5B.** The teacher boxed a *different* answer
   than the hint in ~45% of single samples (weak hint compliance), and step-0
   teacher–student trajectory disagreement was near zero (mean gap ~0.1).
   Answer-only privilege requires backward planning the 0.5B model largely
   lacks. Untrained-teacher pass@4 (0.77) barely differs from trained (0.83).
4. **Compounding trend:** across teacher checkpoints (step 0/30/60), demo `G`
   rose 0.38 → 0.53 while assimilation loss *fell* 0.50 → 0.37 — on easy
   problems, "more learnable" converges to "less informative."

Also relevant: the blog used Llama-3.2-3B **explicitly because** Qwen models are
"aggressively mid-trained for math" — our 0.5B Qwen pick was anti-showcase on
both axes (too strong for its size on GSM8K → dense rewards; too small to
exploit an answer hint via in-context learning).

### Decisions for round 2

- **`filter-hard` stage**: restrict training to problems the base student fails
  0/k — reconstructs the sparse-reward regime; vanilla GRPO should stall there.
- **Held-out hard eval slice** (`eval_*_hard.json`) — where the effect must
  appear first.
- **`hard` preset**: Llama-3.2-3B-Instruct on A100 (faithful to the blog),
  answer-only privilege kept as default; `--set privileged=solution` as the
  small-model fallback (substitutes for missing ICL ability).
- **Direct hint prompt**: state the answer plainly and instruct the teacher to
  land on it; leave "don't leak" enforcement to the surprisal reward.

## Round 2 — sparse-reward regime (hard preset)

**Premise check first** (`probe-hard`, before any training): on the hard set,
the same base model sampled k times blind vs k times with the hint. The
method's motivation predicts blind `pass@1 ≈ 0` / near-zero learnable GRPO
groups, and substantially higher reward density with privilege. If privileged
sampling *also* starves, the hint form is insufficient → escalate to
`privileged=solution` before spending the training budget.

| probe (hard train problems) | pass@1 | ≥1 correct/group | mixed (learnable) groups |
|---|---|---|---|
| blind (student prompt) | *(pending)* | | |
| privileged (hint, untrained) | *(pending)* | | |

*(training results pending)*
