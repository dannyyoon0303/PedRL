"""GSM8K loading, prompt construction, and answer checking.

Two views of every problem:
  - student prompt: question only (this is the context x the surprisal is measured under)
  - teacher prompt: same user turn, plus a system message carrying the privileged
    information c (the final answer, or the full reference solution).
"""

import re
from typing import Optional

from datasets import load_dataset

STUDENT_SYSTEM = (
    "You are a helpful assistant that solves math problems step by step."
)

USER_TEMPLATE = (
    "Solve the following math problem. Reason step by step, and put your final "
    "answer in \\boxed{{}}.\n\n{question}"
)

# Direct, task-shaped hint prompts. The prompt maximizes USE of the privileged
# information; sounding natural / not leaking it is the surprisal reward's job,
# not the prompt's (see round-1 findings in RESULTS.md).
TEACHER_SYSTEM_ANSWER = (
    "The correct final answer to this problem is: {answer}. Use it as a hint. "
    "Solve the problem step by step, choosing steps that genuinely lead to this "
    "answer, and end with the final answer in \\boxed{{}}. Do not mention that "
    "you were given the answer."
)

TEACHER_SYSTEM_SOLUTION = (
    "Here is a correct solution to this problem:\n---\n{solution}\n---\n"
    "Use it as a hint. Write your own step-by-step solution in your natural "
    "style, following the same reasoning, and end with the final answer in "
    "\\boxed{{}}. Do not mention that you were given a solution."
)


def gsm8k_gold_answer(solution: str) -> str:
    """GSM8K gold answers follow '#### <answer>'."""
    return solution.split("####")[-1].strip().replace(",", "").replace("$", "")


def _last_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out) if depth == 0 else None

_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_prediction(text: str) -> Optional[str]:
    """Pull a final numeric answer out of a completion: \\boxed{} first, else last number."""
    boxed = _last_boxed(text)
    if boxed is not None:
        nums = _NUM_RE.findall(boxed)
        if nums:
            return nums[-1].replace(",", "").replace("$", "")
        return boxed.strip()
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").replace("$", "")
    return None


def answers_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    pred = pred.strip().rstrip(".")
    gold = gold.strip()
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, OverflowError):
        return False


def build_prompts(tokenizer, question: str, answer: str, solution: str, privileged: str,
                  teacher_system: str = ""):
    """Returns (student_prompt, teacher_prompt) as fully chat-templated strings."""
    user_msg = {"role": "user", "content": USER_TEMPLATE.format(question=question)}

    student_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": STUDENT_SYSTEM}, user_msg],
        tokenize=False,
        add_generation_prompt=True,
    )

    if teacher_system:
        sys_content = teacher_system.format(answer=answer, solution=solution)
    elif privileged == "solution":
        sys_content = TEACHER_SYSTEM_SOLUTION.format(solution=solution)
    else:
        sys_content = TEACHER_SYSTEM_ANSWER.format(answer=answer)
    teacher_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_content}, user_msg],
        tokenize=False,
        add_generation_prompt=True,
    )
    return student_prompt, teacher_prompt


def load_gsm8k(cfg, tokenizer, split: str, n: int, seed: Optional[int] = None,
               filter_path: Optional[str] = None):
    """Returns a datasets.Dataset with columns:
    question, answer (gold numeric string), prompt (teacher), student_prompt.

    filter_path: optional hard-set file (from the filter-hard stage) holding
    indices into the seed-shuffled split; restricts loading to those problems.
    """
    import json
    import os

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=split)
    ds = ds.shuffle(seed=cfg.seed if seed is None else seed)
    if filter_path:
        if not os.path.exists(filter_path):
            raise FileNotFoundError(
                f"hard-set file {filter_path} not found — run `python run.py filter-hard` first"
            )
        with open(filter_path) as f:
            ds = ds.select(json.load(f)["indices"])
    if n > 0:
        ds = ds.select(range(min(n, len(ds))))

    def _map(ex):
        gold = gsm8k_gold_answer(ex["answer"])
        student_prompt, teacher_prompt = build_prompts(
            tokenizer, ex["question"], gold, ex["answer"], cfg.privileged,
            teacher_system=cfg.teacher_system,
        )
        return {
            "question": ex["question"],
            "answer": gold,
            "prompt": teacher_prompt,
            "student_prompt": student_prompt,
        }

    return ds.map(_map, remove_columns=ds.column_names)
