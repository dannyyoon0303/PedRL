"""Official-judge evaluation: re-grade eval outputs on LeetCode's hidden suites.

Role in the pipeline: the LOCAL judge (verifier.py) is the training reward —
fast, parallel, offline. The LeetCode online judge is the FINAL evaluation:
~2-3 example tests per problem is an upper bound on correctness, so headline
eval_base vs eval_student numbers deserve the full hidden suites (99+ tests),
exactly as the DebugBench paper graded. Re-grading also measures the local
judge's false-positive rate (local pass -> OJ fail), i.e. how loose the
upper bound actually is.

Why only eval, never training: submissions go through a real LeetCode
account with a cooldown between calls (default 15 s — please keep it; this
is a courtesy to LeetCode and hammering the endpoint risks the account).
A GRPO run grades tens of thousands of rollouts; an eval slice re-grades at
most a couple hundred completions, and by default only those that already
pass the local tests (a local fail on example tests is an OJ fail with
near certainty, since the examples are part of the hidden suite).

Setup (Colab):
    pip install git+https://github.com/GammaTauAI/leetcode-hard-gym.git
    export LEETCODE_SESSION=<cookie from browser DevTools -> Application -> Cookies>
Never share or commit the cookie — it grants full account access. Set it in
the environment of the running process only (Colab Secrets or os.environ).

Usage:
    python PedRL_debug/run.py eval-student --preset poc          # writes eval_student.json
    python PedRL_debug/run.py eval-leetcode --preset poc --set eval_tag=student
    python PedRL_debug/run.py eval-leetcode --preset poc --set eval_tag=base

Submissions are journaled to leetcode_{tag}.jsonl as they complete, so an
interrupted run (expired cookie, network) resumes where it left off.
"""

import json
import os
import time

from .config import DebugPedRLConfig
from .data import extract_code


def _get_tester(cooldown: float):
    try:
        from leetcode_env.environment import LeetCodeEnv
        from leetcode_env.types import LeetCodeSubmission, ProgrammingLanguage
    except ImportError as e:
        raise SystemExit(
            "leetcode_env is not installed. Run:\n"
            "  pip install git+https://github.com/GammaTauAI/leetcode-hard-gym.git\n"
            f"(import error: {e})"
        )
    if "LEETCODE_SESSION" not in os.environ:
        raise SystemExit(
            "LEETCODE_SESSION is not set. Copy the cookie from your browser "
            "(DevTools -> Application -> Cookies -> leetcode.com) and set it "
            "in the environment of THIS process only. Do not commit or share it."
        )
    env = LeetCodeEnv(cooldown=cooldown)

    def submit(code: str, slug: str) -> dict:
        sub = LeetCodeSubmission(
            code=code, lang=ProgrammingLanguage.PYTHON3, question_slug=slug)
        status, reward, done, result = env.step(sub)
        return {
            "accepted": bool(reward),
            "status_msg": result.get("status_msg"),
            "total_correct": result.get("total_correct"),
            "total_testcases": result.get("total_testcases"),
        }

    return submit


def evaluate_leetcode(cfg: DebugPedRLConfig, tag: str) -> float:
    eval_path = os.path.join(cfg.output_dir, f"eval_{tag}.json")
    if not os.path.exists(eval_path):
        raise SystemExit(f"{eval_path} not found — run the corresponding eval stage first")
    with open(eval_path) as f:
        records = json.load(f)["records"]

    # resume journal: slug -> submission outcome
    journal_path = os.path.join(cfg.output_dir, f"leetcode_{tag}.jsonl")
    done = {}
    if os.path.exists(journal_path):
        with open(journal_path) as f:
            for line in f:
                row = json.loads(line)
                done[row["slug"]] = row
        print(f"[leetcode:{tag}] resuming — {len(done)} submissions already journaled")

    to_submit = []
    for r in records:
        if r["slug"] in done:
            continue
        if not cfg.leetcode_submit_all and r["pass_frac"] < 1.0:
            continue  # local fail on example tests => OJ fail; don't spend a submission
        to_submit.append(r)

    submit = _get_tester(cfg.leetcode_cooldown) if to_submit else None
    n_total = len(to_submit)
    print(f"[leetcode:{tag}] {n_total} submissions to make "
          f"(~{n_total * cfg.leetcode_cooldown / 60:.0f} min at "
          f"{cfg.leetcode_cooldown:.0f}s cooldown)")

    for i, r in enumerate(to_submit):
        code = extract_code(r["completion"])
        entry = {"slug": r["slug"]}
        if not code:
            entry.update({"accepted": False, "status_msg": "no code extracted"})
        else:
            for attempt in (1, 2):
                try:
                    entry.update(submit(code, r["slug"]))
                    break
                except Exception as e:
                    if attempt == 2:
                        entry.update({"accepted": False,
                                      "status_msg": f"submission error: {e!r}"})
                    else:
                        print(f"[leetcode:{tag}] {r['slug']}: {e!r} — retrying once")
                        time.sleep(max(cfg.leetcode_cooldown, 20))
        done[r["slug"]] = entry
        with open(journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[leetcode:{tag}] {i + 1}/{n_total} {r['slug']}: "
              f"{entry.get('status_msg')} "
              f"({entry.get('total_correct')}/{entry.get('total_testcases')})")

    # aggregate: OJ accuracy over ALL eval records (non-submitted = fail)
    n = len(records)
    n_accepted = 0
    n_local_pass, n_local_pass_oj_fail = 0, 0
    by_cat = {}
    out_records = []
    for r in records:
        oj = done.get(r["slug"], {"accepted": False, "status_msg": "not submitted (local fail)"})
        accepted = bool(oj.get("accepted"))
        n_accepted += accepted
        if r["pass_frac"] >= 1.0:
            n_local_pass += 1
            n_local_pass_oj_fail += not accepted
        c, t = by_cat.get(r["category"], (0, 0))
        by_cat[r["category"]] = (c + accepted, t + 1)
        out_records.append({
            "slug": r["slug"], "category": r["category"], "level": r["level"],
            "local_pass": r["pass_frac"] >= 1.0, "oj_accepted": accepted,
            "oj_status": oj.get("status_msg"),
            "oj_tests": f"{oj.get('total_correct')}/{oj.get('total_testcases')}",
        })

    acc = n_accepted / n
    fp_rate = n_local_pass_oj_fail / max(1, n_local_pass)
    out_path = os.path.join(cfg.output_dir, f"eval_{tag}_leetcode.json")
    with open(out_path, "w") as f:
        json.dump({
            "tag": tag,
            "n": n,
            "accuracy": acc,
            "local_accuracy": n_local_pass / n,
            "local_judge_false_positive_rate": fp_rate,
            "accuracy_by_category": {c: p / t for c, (p, t) in sorted(by_cat.items())},
            "submit_all": cfg.leetcode_submit_all,
            "records": out_records,
        }, f, indent=2)
    print(f"[leetcode:{tag}] OJ pass@1 = {acc:.3f} ({n_accepted}/{n})  "
          f"local pass@1 = {n_local_pass / n:.3f}  "
          f"local->OJ false-positive rate = {fp_rate:.2f}")
    print(f"-> {out_path}")
    return acc
