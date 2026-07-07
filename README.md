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

## Presets / experimental regimes

| preset | model | problems | regime | hardware |
|---|---|---|---|---|
| `smoke` | Qwen2.5-0.5B-Instruct | 16 | pipeline check (~10 min) | T4 |
| `poc` | Qwen2.5-0.5B-Instruct | 256 random GSM8K | **dense rewards** (student pass@1 ≈ 0.43) | T4 (~2–3 h) |
| `hard` | Llama-3.2-3B-Instruct | GSM8K hard tail (student fails 0/k) | **sparse rewards** — the method's target regime | A100 (~4–5 h) |

Round-1 finding (see [RESULTS.md](RESULTS.md)): in the dense-reward `poc` regime
the *mechanism* works (surprisal falls, `G_spike` ~doubles, correctness holds)
but eval doesn't move and vanilla GRPO wins at matched rollouts — privileged
teaching has no edge when blind sampling already finds correct trajectories.
The `hard` preset reconstructs the blog's sparse regime: `filter-hard` keeps
only problems the base student fails at all of `hard_k` samples, and every eval
also reports a held-out **hard slice** (`eval_*_hard.json`), where the effect
must appear first.

Privileged info: the gold final answer in the teacher's system prompt, stated
directly ("use it as a hint … land on this answer"); `--set privileged=solution`
hands over the full reference solution instead — the fallback when the model is
too small to exploit a bare answer (see RESULTS.md round 1).

> `meta-llama` models are gated: accept the license at
> huggingface.co/meta-llama/Llama-3.2-3B-Instruct and set `HF_TOKEN`.
> Ungated alternative: `--set model_name=Qwen/Qwen2.5-1.5B-Instruct`.

## Quickstart

```bash
pip install -r requirements.txt

# ~10 min end-to-end pipeline check on a T4
python run.py all --preset smoke

# dense-reward proof of concept (~2–3 h on T4, ~40 min on A100)
python run.py all --preset poc

# the real test — sparse-reward hard subset on A100
# (runs filter-hard automatically, writes to outputs_hard/)
python run.py filter-hard --preset hard
python run.py probe-hard --preset hard       # premise check BEFORE training: blind sampling
                                             # must starve while privileged sampling succeeds
python run.py all --preset hard
python run.py baseline-rl --preset hard      # vanilla GRPO should STALL here
python run.py curve-baseline --preset hard   # both curves eval on the hard slice
python run.py curve-pedrl --preset hard
```

`run.py all` executes, each in its own process (so GPU memory is released
between stages):

1. `filter-hard` (hard preset only) — screen problems the base student fails 0/k;
   writes `hard_train.json` + a held-out `hard_test.json` eval slice
2. `eval-base` — baseline student pass@1 (plus hard slice in hard mode)
3. `teacher` — GRPO on the privileged teacher with reward `R · G_spike`
4. `corpus` — sample the teacher, filter to correct, rank by `G_spike`
5. `assimilate` — surprisal-gated distillation into a fresh student LoRA
6. `eval-student` — student pass@1 after assimilation (plus hard slice)

Compare `eval_base.json` vs `eval_student.json` (and `eval_base_hard.json` vs
`eval_student_hard.json` in hard mode) in the preset's output dir
(`outputs/` for smoke/poc, `outputs_hard/` for hard).

### Analysis: surprisal & sample efficiency

Two artifacts are produced automatically during training:

- `outputs/reward_log_teacher.jsonl` — per-reward-batch metrics (`acc`,
  `mean_g`, `mean_gap`, `mean_max_gap`, cumulative `rollouts`). Plot these to
  see **surprisal decreasing** (and `G_spike` rising) as the teacher trains.
- `outputs/teacher_adapter_checkpoints/step_*` — adapter snapshots every
  `checkpoint_every` (default 30) steps.

Then, to measure **how fast eval accuracy improves per rollout** vs vanilla RL:

```bash
# vanilla GRPO on the student (no privileged info), SAME rollout budget
python run.py baseline-rl --preset poc

# learning curves: pass@1 vs rollouts -> outputs/curve_baseline.json / curve_pedrl.json
python run.py curve-baseline --preset poc     # evaluates each baseline checkpoint
python run.py curve-pedrl --preset poc        # per teacher checkpoint: corpus -> assimilate -> eval
```

`curve-pedrl` includes a step-0 point (the *untrained* privileged teacher =
plain rejection-sampling distillation), so the curve isolates what pedagogical
training of the teacher adds. Curve points use reduced sizes
(`curve_n_distill=128`, `curve_n_eval=100`) to keep Colab compute sane.
The notebook plots both analyses.

Other variations:

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
RESULTS.md              # experiment log: round-1 findings, round-2 design
pedrl/
├── config.py           # every knob in one dataclass (smoke/poc/hard presets)
├── data.py             # GSM8K, student vs privileged-teacher prompts, answer checking
├── hardset.py          # filter-hard: build the sparse-reward hard subset
├── surprisal.py        # d_t gaps and G_spike (scored under the student)
├── rewards.py          # r_ped = correctness × G_spike (GRPO reward function)
├── teacher.py          # stage 1 (and baseline-rl / stage 3) — TRL GRPOTrainer
├── distill.py          # stage 2 — corpus building + gated assimilation Trainer
├── modeling.py         # loading, LoRA config, batched generation
└── evaluate.py         # greedy pass@1 on GSM8K test (standard + hard slice)
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
| `teacher_system` | — | custom hint prompt template (`{answer}` / `{solution}` placeholder) |
| `checkpoint_every` | 30 | adapter snapshot cadence for learning curves (0 = off) |
| `curve_n_distill` / `curve_n_eval` | 128 / 100 | reduced sizes for curve points |
| `hard_pool` / `hard_test_pool` | 768 / 400 | problems screened by `filter-hard` |
| `hard_k` | 4 | hard = base student fails all k samples |

GRPO normalizes advantages within each group, so only the *relative* ordering
of `r_ped` within a group matters — the absolute scale of `λ` is forgiving.

## What to look for

- **Surprisal over training** (`reward_log_teacher.jsonl`): `mean_gap` /
  `mean_max_gap` should fall and `mean_g` rise while `acc` stays high — the
  teacher already knows the answer; it is learning *how to say it* in a way the
  student finds plausible.
- **Sample efficiency** (`curve_pedrl.json` vs `curve_baseline.json`): pass@1
  as a function of rollouts. The method's claim is that the PedRL curve rises
  much faster than vanilla GRPO at the same rollout budget.
- **Corpus stats**: `mean G_spike of kept demos` should be higher for trained
  teacher checkpoints than for the step-0 (untrained) teacher.
- **Final comparison**: `eval_student.json` vs `eval_base.json` vs the
  `--no-gating` ablation.

## Caveats

This is a proof of concept, not a reproduction. The blog uses Llama-3.2-3B on a
hard MATH subset with *iterated* teacher/student rounds; our `hard` preset
matches the model and reconstructs the sparse regime from GSM8K's hard tail,
but still runs a single teacher→student round. The `poc` preset (0.5B, random
GSM8K) is deliberately kept as the dense-reward contrast — see RESULTS.md for
why that regime shows the mechanism but not eval gains. The blog does not
publish all hyperparameters (β = 5 appears in an example; λ, κ, γ here are
chosen to give sensible gate/score ranges) and does not report teacher
hint-compliance or vanilla-RL baselines, which we measure.

## References

- Noah Ziems, *Pedagogical RL* — https://noahziems.com/pedagogical-rl
- Siyan Zhao et al., *OPSD: On-Policy Self-Distillation* — https://github.com/siyan-zhao/OPSD
- Shao et al., *DeepSeekMath: GRPO* — https://arxiv.org/abs/2402.03300
