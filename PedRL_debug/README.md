# PedRL_debug — Pedagogical RL on DebugBench

Port of the [PedRL](../README.md) recipe from GSM8K math to **bug fixing** on
[DebugBench](https://github.com/thunlp/DebugBench) (LeetCode problems with
implanted bugs, python3 subset).

## The domain hypothesis

Both the teacher and the student see the same buggy code; the teacher is
additionally conditioned on the dataset's `bug_explanation` — a *witness* for
the bug. The hypothesis: **debugging, unlike math, is reversal-friendly.** In
math, handing the model the final answer often doesn't help it construct the
derivation (see round-1 findings in [../RESULTS.md](../RESULTS.md)). In
debugging, the witness (a diagnosis, or the patch itself) frequently converts
a model that is *unconditionally incapable* of fixing the bug into a capable
one — verification/repair given the witness is much easier than blind search.

If true, the privileged/blind pass-rate gap is large, and that gap is exactly
the signal pedagogical distillation can transfer, so PedRL should beat
vanilla GRPO on sample efficiency more clearly than in the math domain.

The `probe` stage measures this directly — the same untrained base model,
sampled k times blind and k times with the witness:

```bash
python PedRL_debug/run.py build-verified --preset poc
python PedRL_debug/run.py probe --preset poc
```

It prints blind vs privileged pass@1 (plus per-bug-category and
per-difficulty breakdowns) and a verdict on the hypothesis. `--set
privileged=solution` swaps the witness for the full corrected code (the
patch) — the upper bound on privileged information.

## The verifier (there are no tests in DebugBench)

DebugBench ships **no executable unit tests** — the paper graded via the
LeetCode online judge, which can't be called from a training loop. We
reconstruct a local judge ([verifier.py](verifier.py)) from what the dataset
does provide:

1. **Parse the worked examples** in each problem statement
   (`Input: nums = [2,7,11,15], target = 9` / `Output: [0,1]`) into
   keyword arguments and an expected value.
2. **Locate the entry method** of `class Solution` by AST-matching its
   parameter names against the parsed input names.
3. **Execute candidates in a sandboxed subprocess** (per-test SIGALRM time
   limit, address-space cap, stdout swallowed); `None`-returning in-place
   problems are graded on the mutated first argument.

Because example parsing can't be right for every problem, `build-verified`
screens the whole python3 subset with two checks that make mis-parses
self-eliminating:

- the reference `solution` must **pass** all parsed tests — kills mis-parsed
  I/O, "any valid answer" problems, special judges, and unsupported
  signatures (linked lists, trees, design problems);
- the `buggy_code` must **fail** at least one test — kills problems whose
  example tests are too weak to expose the bug (without this check, echoing
  the buggy code back would earn full reward; the reward logs also track an
  `echo_rate` for exactly this failure mode).

On a 714-problem sample this yields **~76% verified problems** (~2.4 tests
each) spanning all four bug categories (syntax / logic / reference /
multiple) and all three difficulty levels. The verified set is written once
to `PedRL_debug/verified_debugbench.json` and shared by every stage; we carve
our own train/eval split from it (DebugBench has only a `test` split).

## Pipeline

Identical shape to the math version — the surprisal machinery
(`G_spike`, gated assimilation) is reused from `pedrl/` unchanged; only the
task layer (data, verifier, reward, probe) is new. Reward:

```
r_ped(x, c, tau) = R(x, c, tau) * G_spike^{theta_S}(tau | x)
```

where `R` = extracted fix passes all example tests (binary; `--set
partial_credit=true` for the pass-fraction variant) and `G_spike` is scored
by the same network with the LoRA adapter disabled, under the student's
witness-free prompt.

```bash
pip install -r requirements.txt   # same deps as the math version

# ~15 min pipeline check on a T4 (Qwen2.5-Coder-0.5B)
python PedRL_debug/run.py all --preset smoke

# dense-reward PoC (Qwen2.5-Coder-1.5B)
python PedRL_debug/run.py build-verified --preset poc
python PedRL_debug/run.py probe --preset poc            # hypothesis check FIRST
python PedRL_debug/run.py all --preset poc

# sparse-reward target regime (Llama-3.2-3B, non-coder model, A100):
# filter-hard keeps problems the blind student fails 0/k
python PedRL_debug/run.py build-verified --preset hard
python PedRL_debug/run.py filter-hard --preset hard
python PedRL_debug/run.py probe --preset hard           # witness gap on the hard set
python PedRL_debug/run.py all --preset hard

# sample-efficiency comparison at matched rollout budgets
python PedRL_debug/run.py baseline-rl --preset hard     # vanilla GRPO should stall
python PedRL_debug/run.py curve-baseline --preset hard
python PedRL_debug/run.py curve-pedrl --preset hard

# ablations
python PedRL_debug/run.py assimilate --preset poc --no-gating   # plain SFT
python PedRL_debug/run.py all --preset poc --set privileged=solution
```

`run.py all` executes, each stage in its own process: `build-verified` →
(`filter-hard` in hard mode) → `eval-base` → `teacher` → `corpus` →
`assimilate` → `eval-student`. Compare `eval_base.json` vs
`eval_student.json` (and the `_hard` variants) in the preset's output dir
(`outputs_debug/`, `outputs_debug_hard/`).

## Official-judge evaluation (optional)

The local judge tests ~2–3 example cases; the DebugBench paper graded on
LeetCode's full hidden suites (~100 tests). The `eval-leetcode` stage
re-grades an existing `eval_<tag>.json` through the official judge via
[leetcode-hard-gym](https://github.com/GammaTauAI/leetcode-hard-gym) — use it
for headline numbers, never for training (15 s cooldown per submission; a
GRPO run grades tens of thousands of rollouts, an eval slice a few dozen):

```bash
pip install git+https://github.com/GammaTauAI/leetcode-hard-gym.git
export LEETCODE_SESSION=...   # browser DevTools -> Application -> Cookies; keep it secret

python PedRL_debug/run.py eval-leetcode --preset poc --set eval_tag=base
python PedRL_debug/run.py eval-leetcode --preset poc --set eval_tag=student
```

By default only completions that already pass the local tests are submitted
(a local fail implies an OJ fail — the examples are part of the hidden
suite), so a 150-problem eval costs at most a few dozen submissions.
Submissions are journaled to `leetcode_<tag>.jsonl` and the stage resumes
after interruption (expired cookie, network). The output
`eval_<tag>_leetcode.json` reports OJ pass@1, local pass@1, and the local
judge's false-positive rate (local pass → OJ fail) — how loose the example
tests actually are.

## What to look for

- **`probe.json`** — the headline number of this variant: privileged pass@1
  far above blind pass@1, and by how much per bug category (syntax errors
  should be nearly free given the witness; logic/multiple errors are the
  interesting middle).
- **`reward_log_teacher.jsonl`** — `acc` high & stable, `mean_g` rising,
  `mean_gap` falling (the teacher keeps fixing bugs but learns to *say the
  fix* in a way the student finds plausible), `echo_rate` near zero.
- **`curve_pedrl.json` vs `curve_baseline.json`** — pass@1 vs rollouts; the
  claim is that the PedRL curve rises much faster in the sparse (hard)
  regime, and more decisively than in the math domain because the witness
  gap is bigger.

## Repo layout

```
PedRL_debug/
├── run.py          # CLI: stages + presets + config overrides
├── config.py       # every knob in one dataclass (smoke/poc/hard presets)
├── verifier.py     # example-test reconstruction + sandboxed judge (stdlib-only)
├── data.py         # verified-set loading, student vs privileged prompts, code extraction
├── probe.py        # the hypothesis check (blind vs witness) + filter-hard
├── rewards.py      # r_ped = tests-pass x G_spike
├── teacher.py      # stage 1 GRPO (reuses pedrl.teacher callback/modeling)
├── distill.py      # stage 2a corpus; 2b reuses pedrl.distill.assimilate as-is
└── evaluate.py     # greedy pass@1 with category/level breakdowns
```

Run everything **from the repo root** (imports reach into `pedrl/`).

## Caveats

- The judge tests each fix on ~2–3 example cases, not LeetCode's full hidden
  suite — pass rates are upper bounds on true correctness. The
  `bug_undetected` screen guarantees the tests can at least distinguish the
  buggy from the reference code, which is the contrast the reward needs.
- Candidate code runs with a time limit and memory cap but **no OS-level
  isolation** — run training in Colab/containers, as with the math version.
- python3 subset only; cpp/java would need per-language toolchains.
