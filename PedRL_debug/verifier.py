"""Executable verifier for DebugBench (python3 subset).

DebugBench ships NO unit tests — the paper graded via the LeetCode online
judge, which we cannot call from a training loop. We reconstruct a local judge
from what the dataset does provide:

  1. Each problem statement carries worked examples
     ("Input: nums = [2,7,11,15], target = 9  /  Output: [0,1]").
     We parse the Input line into keyword arguments and the Output line into
     an expected value.
  2. We locate the entry-point method of `class Solution` by AST-matching its
     parameter names against the parsed input names (helper methods don't
     match, so they are skipped automatically).
  3. Candidate code is exec'd in a fresh subprocess (per-test SIGALRM time
     limit, address-space cap, stdout swallowed) and the method's return value
     — or, for None-returning in-place problems, the mutated first argument —
     is compared to the expected value after normalization.

The parser cannot be right for every problem, so `build_verified` screens the
whole dataset with two checks that make mis-parses self-eliminating:

  - the reference `solution` must PASS all parsed tests
    (kills mis-parsed inputs/outputs, ambiguous "any valid answer" problems,
    special-judge problems, data-structure signatures we don't support);
  - the `buggy_code` must FAIL at least one test
    (kills problems whose example tests are too weak to expose the bug —
    without this, echoing the buggy code back would earn full reward).

What survives is a set of problems with a trustworthy executable reward.
Everything here is stdlib-only so it can be unit-tested without torch.
"""

import ast
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

# Names LeetCode makes available implicitly; prepended to every candidate.
PRELUDE = """\
import sys as _sys
_sys.setrecursionlimit(20000)
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import collections, math, itertools, heapq, bisect, functools, operator, re, string
from collections import defaultdict, Counter, deque, OrderedDict
from functools import lru_cache, reduce
from itertools import permutations, combinations, accumulate, product, groupby
from heapq import heappush, heappop, heapify, nlargest, nsmallest
from math import inf, gcd, sqrt, ceil, floor, factorial, comb, log2
from bisect import bisect_left, bisect_right, insort
try:
    from functools import cache
except ImportError:
    cache = lru_cache(None)
"""


# ---------------------------------------------------------------------------
# example parsing
# ---------------------------------------------------------------------------

_FAIL = object()


def _split_top_level(s: str) -> List[str]:
    """Split on commas at bracket depth 0, respecting string quotes."""
    parts, cur, depth, quote = [], [], 0, None
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def _parse_value(s: str):
    """Parse a LeetCode-style literal. Returns _FAIL when unparseable."""
    s = s.strip()
    if not s:
        return _FAIL
    s2 = re.sub(r"\bnull\b", "None", s)
    s2 = re.sub(r"\btrue\b", "True", s2)
    s2 = re.sub(r"\bfalse\b", "False", s2)
    try:
        return ast.literal_eval(s2)
    except Exception:
        # bare unquoted string outputs like `Output: abbaca`
        if re.fullmatch(r"[A-Za-z0-9_. \-]+", s):
            return s
        return _FAIL


def parse_example(text: str) -> Optional[Tuple[dict, object]]:
    """'Input: a = 1, b = [2]  Output: 3 [Explanation: ...]' -> ({a:1, b:[2]}, 3)."""
    m = re.search(r"Input:?\s*(.*?)\s*Output:?\s*(.*)", text, re.DOTALL)
    if not m:
        return None
    in_str = m.group(1)
    out_str = re.split(r"\n\s*Explanation\b|Explanation:", m.group(2))[0].strip()

    kwargs = {}
    for part in _split_top_level(in_str):
        name, eq, val = part.partition("=")
        name = name.strip()
        if not eq or not name.isidentifier():
            return None
        v = _parse_value(val)
        if v is _FAIL:
            return None
        kwargs[name] = v
    if not kwargs:
        return None
    expected = _parse_value(out_str)
    if expected is _FAIL:
        return None
    return kwargs, expected


def find_entry_method(solution_code: str, input_names: set) -> Optional[Tuple[str, List[str]]]:
    """(method_name, ordered_params) of the Solution method whose parameters
    match the example's input names exactly."""
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for fn in node.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if fn.name.startswith("_"):
                    continue
                params = [a.arg for a in fn.args.args if a.arg != "self"]
                if set(params) == input_names and len(params) == len(input_names):
                    return fn.name, params
    return None


# ---------------------------------------------------------------------------
# output comparison
# ---------------------------------------------------------------------------

def normalize(x):
    """Canonical form for comparison (and JSON transport): tuples -> lists,
    sets -> sorted lists, floats rounded. Must match _norm in the driver."""
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        return round(x, 5)
    if isinstance(x, (list, tuple)):
        return [normalize(v) for v in x]
    if isinstance(x, set):
        return sorted((normalize(v) for v in x), key=repr)
    if isinstance(x, dict):
        return {str(k): normalize(v) for k, v in x.items()}
    return x


def outputs_match(got, expected) -> bool:
    got, expected = normalize(got), normalize(expected)
    if isinstance(got, float) or isinstance(expected, float):
        try:
            return abs(float(got) - float(expected)) < 1e-4
        except (TypeError, ValueError):
            return False
    if isinstance(got, list) and isinstance(expected, list):
        return len(got) == len(expected) and all(
            outputs_match(a, b) for a, b in zip(got, expected))
    return got == expected


# ---------------------------------------------------------------------------
# sandboxed execution
# ---------------------------------------------------------------------------

# Runs in a fresh python subprocess. Reads {prelude, code, method, tests,
# time_limit} as JSON on stdin, prints one JSON line with per-test results.
_DRIVER = r"""
import sys, json, io, copy, signal, contextlib

def _norm(x):
    if isinstance(x, bool): return x
    if isinstance(x, float): return round(x, 5)
    if isinstance(x, (list, tuple)): return [_norm(v) for v in x]
    if isinstance(x, set): return sorted((_norm(v) for v in x), key=repr)
    if isinstance(x, dict): return {str(k): _norm(v) for k, v in x.items()}
    return x

def main():
    payload = json.loads(sys.stdin.read())
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    except Exception:
        pass
    ns = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(payload["prelude"] + "\n" + payload["code"], ns)
        cls = ns["Solution"]
    except BaseException as e:
        print(json.dumps({"load_error": repr(e)[:300], "results": []}))
        return
    def _alarm(sig, frame):
        raise TimeoutError("time limit exceeded")
    signal.signal(signal.SIGALRM, _alarm)
    results = []
    for args in payload["tests"]:
        try:
            a = copy.deepcopy(args)
            signal.setitimer(signal.ITIMER_REAL, payload["time_limit"])
            with contextlib.redirect_stdout(io.StringIO()):
                ret = getattr(cls(), payload["method"])(*a)
            signal.setitimer(signal.ITIMER_REAL, 0)
            if ret is None and a:
                ret = a[0]  # in-place problems: result lives in the first arg
            results.append({"ok": True, "out": _norm(ret)})
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            results.append({"ok": False, "err": repr(e)[:200]})
    print(json.dumps({"results": results}, default=repr))

main()
"""


def run_tests(code: str, method: str, tests: List[dict], time_limit: float) -> List[bool]:
    """Execute `code` against `tests` in a subprocess; per-test pass/fail."""
    payload = json.dumps({
        "prelude": PRELUDE,
        "code": code,
        "method": method,
        "tests": [t["args"] for t in tests],
        "time_limit": time_limit,
    })
    budget = time_limit * len(tests) + 15  # exec + spawn overhead
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DRIVER],
            input=payload.encode(),
            capture_output=True,
            timeout=budget,
        )
        lines = [l for l in proc.stdout.decode(errors="replace").splitlines() if l.strip()]
        out = json.loads(lines[-1])
    except Exception:
        return [False] * len(tests)
    results = out.get("results", [])
    verdict = []
    for i, t in enumerate(tests):
        r = results[i] if i < len(results) else {"ok": False}
        verdict.append(bool(r.get("ok")) and outputs_match(r.get("out"), t["expected"]))
    return verdict


def grade(code: Optional[str], problem: dict, time_limit: float) -> float:
    """Fraction of tests passed (0.0 when no code could be extracted)."""
    if not code or "class Solution" not in code:
        return 0.0
    verdict = run_tests(code, problem["method"], problem["tests"], time_limit)
    return sum(verdict) / max(1, len(verdict))


def grade_many(codes: List[Optional[str]], problems: List[dict],
               time_limit: float, workers: int = 8) -> List[float]:
    """Parallel grading (each item = one subprocess; threads just wait on them)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda cp: grade(cp[0], cp[1], time_limit), zip(codes, problems)))


# ---------------------------------------------------------------------------
# dataset screening
# ---------------------------------------------------------------------------

def make_problem(row: dict) -> Optional[dict]:
    """Parse one raw DebugBench row into a testable problem (no execution)."""
    if "class Solution" not in row["solution"] or "class Solution" not in row["buggy_code"]:
        return None  # design problems (MinStack etc.) are out of scope
    parsed = [p for p in (parse_example(e) for e in row["examples"]) if p]
    if not parsed:
        return None
    names = set(parsed[0][0])
    parsed = [p for p in parsed if set(p[0]) == names]
    found = find_entry_method(row["solution"], names)
    if not found:
        return None
    method, params = found
    return {
        "slug": row["slug"],
        "level": row["level"],
        "category": row["category"],
        "subtype": row["subtype"],
        "question": row["question"],
        "examples": row["examples"],
        "constraints": row["constraints"],
        "buggy_code": row["buggy_code"],
        "solution": row["solution"],
        "bug_explanation": row["bug_explanation"],
        "method": method,
        "tests": [
            {"args": [normalize(kw[p]) for p in params], "expected": normalize(exp)}
            for kw, exp in parsed
        ],
    }


def screen_rows(rows: List[dict], time_limit: float = 4.0, workers: int = 8,
                verbose: bool = True) -> Tuple[List[dict], dict]:
    """Parse + execute-screen raw rows. Returns (verified_problems, stats)."""
    stats = {"total": len(rows), "unparseable": 0, "ref_fails": 0,
             "bug_undetected": 0, "verified": 0}
    problems = []
    for row in rows:
        p = make_problem(row)
        if p is None:
            stats["unparseable"] += 1
        else:
            problems.append(p)

    def _screen(p):
        ref = run_tests(p["solution"], p["method"], p["tests"], time_limit)
        if not all(ref):
            return "ref_fails"
        bug = run_tests(p["buggy_code"], p["method"], p["tests"], time_limit)
        if all(bug):
            return "bug_undetected"
        return "ok"

    verified = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (p, verdict) in enumerate(zip(problems, pool.map(_screen, problems))):
            if verdict == "ok":
                p["id"] = len(verified)
                verified.append(p)
            else:
                stats[verdict] += 1
            if verbose and (i + 1) % 100 == 0:
                print(f"[verify] screened {i + 1}/{len(problems)} "
                      f"(verified so far: {len(verified)})")
    stats["verified"] = len(verified)
    return verified, stats


def build_verified(cfg) -> str:
    """Stage: screen the DebugBench python3 subset into cfg.verified_path."""
    import os

    if os.path.exists(cfg.verified_path):
        print(f"[verify] {cfg.verified_path} already exists — skipping (delete to re-screen)")
        return cfg.verified_path

    from datasets import load_dataset
    ds = load_dataset(cfg.dataset_name, split="test")
    rows = [r for r in ds if r["language"] == cfg.language]
    print(f"[verify] {len(rows)} {cfg.language} problems; parsing + execution screening...")

    verified, stats = screen_rows(rows, time_limit=cfg.time_limit, workers=cfg.verify_workers)
    print(f"[verify] {stats}")
    by_cat = {}
    for p in verified:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    print(f"[verify] verified by category: {by_cat}")

    os.makedirs(os.path.dirname(cfg.verified_path) or ".", exist_ok=True)
    with open(cfg.verified_path, "w") as f:
        json.dump({"stats": stats, "problems": verified}, f)
    print(f"[verify] -> {cfg.verified_path}")
    return cfg.verified_path
