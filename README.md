# PedRL — Pedagogical RL on GSM8K (proof of concept)

A compact, Colab-friendly implementation of **Pedagogical RL**
([blog post by Noah Ziems](https://noahziems.com/pedagogical-rl)), with code
patterns borrowed from [OPSD](https://github.com/siyan-zhao/OPSD).

## The idea

On-policy RL is sample-inefficient because the sampler "explores as if it is
blind": even when privileged information `c` (e.g. the correct answer) exists,
vanilla GRPO ignores it and has to stumble onto correct trajectories. The
obvious fix — let a privileged teacher generate demonstrations — fails in a
different way: the teacher's trajectories are **off-policy** for the student,
full of tokens the student finds so surprising that imitation degrades it.

Pedagogical RL threads the needle by training the teacher to produce
demonstrations that are simultaneously **correct** and **low-surprisal under
the student**:

**1. Spike-aware learnability score.** For a trajectory τ scored under the
student's (un-privileged) context x, define per-token surprise gaps

```
d_t = log π_S(a_t^max | x, τ_<t) − log π_S(τ_t | x, τ_<t)   ≥ 0
```

and aggregate them with a soft-max that punishes rare implausible "spikes"
harder than uniform mild off-policyness:

```
G_spike(τ | x) = exp( −(λ/β) · log( (1/T) Σ_t exp(β·d_t) ) )
```

`β → 0` recovers average surprise, `β → ∞` recovers max surprise. `G ∈ (0, 1]`
and equals 1 iff every token is the student's own greedy choice.

**2. Teacher GRPO with a product-form reward.** The teacher (same base model +
LoRA, prompted *with* the answer) is trained with GRPO to maximize

```
r_ped(x, c, τ) = R(x, c, τ) · G_spike^{θ_S}(τ | x)
```

i.e. correctness × learnability. Crucially, `G_spike` is computed by the same
network **with the LoRA adapter disabled** — the frozen student — so a single
copy of the weights serves both roles (the OPSD trick).

**3. Surprisal-gated assimilation.** Sample the trained teacher, keep correct
demonstrations (per problem, the one with the highest `G_spike`), then distill
into a fresh student LoRA with per-token gated cross-entropy:

```
w_t = σ( κ · (log π_S(τ_t | x, τ_<t) − γ) )
L   = E[ (Σ_t w_t · CE_t) / (Σ_t w_t) ]
```

Tokens the current student already finds plausible get weight ≈ 1; alien
tokens get weight ≈ 0 and can't dominate the update. `--no-gating` turns this
into plain rejection-sampling SFT — the natural ablation.

**(4. Optional)** continue with standard GRPO on the student (`student-rl` stage).

## Setup for the PoC

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (student = frozen base, teacher = base + LoRA) |
| Task | GSM8K (256 train problems, 200 test problems) |
| Privileged info | gold final answer in the teacher's system prompt (`--set privileged=solution` for the full reference solution) |
| Hardware | one Colab GPU — T4 works, A100/L4 is ~4× faster |

## Quickstart

```bash
pip install -r requirements.txt

# ~10 min end-to-end pipeline check on a T4
python run.py all --preset smoke

# the real proof of concept (~2–3 h on T4, ~40 min on A100)
python run.py all --preset poc
```

`run.py all` executes, each in its own process (so GPU memory is released
between stages):

1. `eval-base` — baseline student pass@1
2. `teacher` — GRPO on the privileged teacher with reward `R · G_spike`
3. `corpus` — sample the teacher, filter to correct, rank by `G_spike`
4. `assimilate` — surprisal-gated distillation into a fresh student LoRA
5. `eval-student` — student pass@1 after assimilation

Compare `outputs/eval_base.json` vs `outputs/eval_student.json`.

Useful variations:

```bash
# SFT ablation (no surprisal gate) — is the gating actually doing work?
python run.py assimilate --preset poc --no-gating
python run.py eval-student --preset poc --no-gating

# optional stage 3: standard GRPO on the assimilated student
python run.py student-rl --preset poc

# override anything
python run.py teacher --preset poc --set teacher_steps=200 --set spike_lambda=1.0
```

### Colab

Open `PedRL_colab.ipynb` in Colab (GPU runtime), run the cells top to bottom.
It installs dependencies, pulls this repo (or accepts a zip upload), runs the
smoke test, then the PoC.

## Repo layout

```
run.py                  # CLI: stages + presets + config overrides
pedrl/
├── config.py           # every knob in one dataclass (smoke/poc presets)
├── data.py             # GSM8K, student vs privileged-teacher prompts, answer checking
├── surprisal.py        # d_t gaps and G_spike (scored under the student)
├── rewards.py          # r_ped = correctness × G_spike (GRPO reward function)
├── teacher.py          # stage 1 (and optional stage 3) — TRL GRPOTrainer
├── distill.py          # stage 2 — corpus building + gated assimilation Trainer
├── modeling.py         # loading, LoRA config, batched generation
└── evaluate.py         # greedy pass@1 on GSM8K test
PedRL_colab.ipynb       # Colab driver notebook
```

## Key hyperparameters

| Knob | Default | Meaning |
|---|---|---|
| `spike_beta` | 5.0 | spike-awareness: 0 = mean surprise, ∞ = max surprise |
| `spike_lambda` | 0.5 | strength of the learnability penalty in `G_spike` |
| `gate_kappa` | 2.0 | sharpness of the assimilation token gate |
| `gate_gamma` | −3.5 | log-prob threshold: tokens below ≈ e^γ ≈ 3% get down-weighted |
| `num_generations` | 8 | GRPO group size |
| `teacher_steps` | 120 | GRPO steps for the teacher |
| `privileged` | `answer` | what the teacher sees: `answer` or `solution` |

GRPO normalizes advantages within each group, so only the *relative* ordering
of `r_ped` within a group matters — the absolute scale of `λ` is forgiving.

## What to look for

- **Teacher training logs** (`[reward] acc=… G=… r_ped=…`): accuracy should
  stay high (it has the answer!) while `G` climbs — the teacher is learning to
  reach the right answer *along paths the student finds plausible*.
- **Corpus stats**: `mean G_spike of kept demos` should be higher after teacher
  training than a pre-training teacher would give.
- **Final comparison**: `eval_student.json` vs `eval_base.json` vs the
  `--no-gating` ablation.

## Caveats

This is a proof of concept, not a reproduction: the blog uses Llama-3.2-3B on
a hard MATH subset with iterated teacher/student updates; we use a 0.5B model
on GSM8K with a single teacher→student round to fit free Colab compute. The
blog does not publish all hyperparameters (β = 5 appears in an example; λ, κ,
γ here are chosen to give sensible gate/score ranges for a 0.5B model).

## References

- Noah Ziems, *Pedagogical RL* — https://noahziems.com/pedagogical-rl
- Siyan Zhao et al., *OPSD: On-Policy Self-Distillation* — https://github.com/siyan-zhao/OPSD
- Shao et al., *DeepSeekMath: GRPO* — https://arxiv.org/abs/2402.03300
