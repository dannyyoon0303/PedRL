"""Verified-problem loading, prompt construction, and code extraction.

Two views of every problem (mirrors pedrl/data.py):
  - student prompt: problem + buggy code only (the context x that surprisal
    and evaluation are measured under)
  - teacher prompt: same user turn, plus a system message carrying the
    privileged information c — the bug_explanation (the witness) or the full
    corrected solution (the patch).

The verified problem set comes from verifier.build_verified: only problems
whose example tests the reference passes and the buggy code fails. We carve
our own train/eval split out of it (DebugBench has a single 'test' split).
"""

import json
import random
import re
from typing import List, Optional

STUDENT_SYSTEM = (
    "You are an expert Python programmer. You find and fix bugs in code."
)

USER_TEMPLATE = (
    "The following Python 3 solution to a programming problem contains one or "
    "more bugs. Fix it.\n\n"
    "### Problem\n{question}\n\n"
    "### Examples\n{examples}\n\n"
    "### Constraints\n{constraints}\n\n"
    "### Buggy code\n```python\n{buggy_code}\n```\n\n"
    "Briefly explain what is wrong, then output the complete corrected "
    "solution as a single ```python code block. Keep the class name "
    "`Solution` and the method signature unchanged."
)

# Direct, task-shaped hint prompts (see pedrl round-1 findings: maximize USE of
# the privileged info in the prompt; sounding natural is the surprisal
# reward's job, not the prompt's).
TEACHER_SYSTEM_EXPLANATION = (
    "You are an expert Python programmer. You find and fix bugs in code. "
    "Privileged information — a correct diagnosis of the bug(s) in the code "
    "you will be shown:\n---\n{explanation}\n---\n"
    "Use this diagnosis: locate each described bug and correct it, keeping "
    "the rest of the code unchanged. Do not mention that you were given the "
    "diagnosis."
)

TEACHER_SYSTEM_SOLUTION = (
    "You are an expert Python programmer. You find and fix bugs in code. "
    "Privileged information — a correct version of the code you will be "
    "shown:\n---\n{solution}\n---\n"
    "Use it as a reference to fix the buggy code with a minimal change. "
    "Do not mention that you were given the corrected code."
)


# ---------------------------------------------------------------------------
# code extraction from model completions
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python3?|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> Optional[str]:
    """Pull the fixed solution out of a completion: last fenced block that
    defines class Solution, tolerating an unterminated final fence."""
    blocks = _FENCE_RE.findall(text)
    for b in reversed(blocks):
        if "class Solution" in b:
            return b.strip()
    m = re.search(r"```(?:python3?|py)?[ \t]*\n([^`]*)$", text, re.DOTALL)
    if m and "class Solution" in m.group(1):
        return m.group(1).strip()
    if blocks:
        return blocks[-1].strip()
    idx = text.find("class Solution")
    if idx != -1:
        return text[idx:].strip()
    return None


def is_echo(code: Optional[str], buggy_code: str) -> bool:
    """True when the 'fix' is the buggy code verbatim (modulo whitespace)."""
    if not code:
        return False
    return "".join(code.split()) == "".join(buggy_code.split())


# ---------------------------------------------------------------------------
# prompt construction + loading
# ---------------------------------------------------------------------------

def build_prompts(tokenizer, problem: dict, privileged: str, teacher_system: str = ""):
    """Returns (student_prompt, teacher_prompt) as chat-templated strings."""
    examples = "\n\n".join(
        f"Example {i + 1}:\n{e.strip()}" for i, e in enumerate(problem["examples"])
    )
    user_msg = {
        "role": "user",
        "content": USER_TEMPLATE.format(
            question=problem["question"].strip(),
            examples=examples,
            constraints=problem["constraints"].strip(),
            buggy_code=problem["buggy_code"].strip(),
        ),
    }

    student_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": STUDENT_SYSTEM}, user_msg],
        tokenize=False,
        add_generation_prompt=True,
    )

    if teacher_system:
        sys_content = teacher_system.format(
            explanation=problem["bug_explanation"].strip(),
            solution=problem["solution"].strip(),
        )
    elif privileged == "solution":
        sys_content = TEACHER_SYSTEM_SOLUTION.format(solution=problem["solution"].strip())
    else:
        sys_content = TEACHER_SYSTEM_EXPLANATION.format(
            explanation=problem["bug_explanation"].strip())
    teacher_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_content}, user_msg],
        tokenize=False,
        add_generation_prompt=True,
    )
    return student_prompt, teacher_prompt


def load_split(cfg, split: str) -> List[dict]:
    """Deterministic train/test carve-out of the verified problem set."""
    import os

    if not os.path.exists(cfg.verified_path):
        raise FileNotFoundError(
            f"{cfg.verified_path} not found — run `python PedRL_debug/run.py build-verified` first"
        )
    with open(cfg.verified_path) as f:
        problems = json.load(f)["problems"]
    rng = random.Random(cfg.seed)
    rng.shuffle(problems)
    test = problems[: cfg.n_test_holdout]
    train = problems[cfg.n_test_holdout:]
    return train if split == "train" else test


def load_debugbench(cfg, tokenizer, split: str, n: int,
                    filter_path: Optional[str] = None):
    """Returns a datasets.Dataset with columns:
    id, slug, category, level, prompt (teacher), student_prompt, buggy_code,
    method, tests_json (JSON string — keeps Arrow away from ragged nesting).

    filter_path: optional hard-set file (from filter-hard) holding verified
    problem ids; restricts loading to those problems.
    """
    import os

    from datasets import Dataset

    problems = load_split(cfg, split)

    if filter_path:
        if not os.path.exists(filter_path):
            raise FileNotFoundError(
                f"hard-set file {filter_path} not found — run "
                f"`python PedRL_debug/run.py filter-hard` first"
            )
        with open(filter_path) as f:
            keep = set(json.load(f)["ids"])
        problems = [p for p in problems if p["id"] in keep]

    rows, n_too_long = [], 0
    for p in problems:
        student_prompt, teacher_prompt = build_prompts(
            tokenizer, p, cfg.privileged, teacher_system=cfg.teacher_system)
        n_tok = max(
            len(tokenizer(student_prompt, add_special_tokens=False)["input_ids"]),
            len(tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"]),
        )
        if n_tok > cfg.max_prompt_length:
            n_too_long += 1
            continue
        rows.append({
            "id": p["id"],
            "slug": p["slug"],
            "category": p["category"],
            "level": p["level"],
            "prompt": teacher_prompt,
            "student_prompt": student_prompt,
            "buggy_code": p["buggy_code"],
            "method": p["method"],
            "tests_json": json.dumps(p["tests"]),
        })
        if n > 0 and len(rows) >= n:
            break

    if n_too_long:
        print(f"[data] dropped {n_too_long} problems over max_prompt_length="
              f"{cfg.max_prompt_length} while collecting {len(rows)}")
    return Dataset.from_list(rows)


def row_problem(row: dict) -> dict:
    """Minimal problem dict the verifier needs, from a dataset row."""
    return {"method": row["method"], "tests": json.loads(row["tests_json"])}
