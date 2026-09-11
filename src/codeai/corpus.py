from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_VERSION = "seeded-code-v1"


@dataclass(frozen=True, slots=True)
class CorpusTask:
    task_id: str
    title: str
    fault_class: str
    problem_statement: str
    starter_code: str
    hidden_tests: str
    reference_solution: str
    expected_note: str


def visible_prompt(task: CorpusTask) -> str:
    """What the model sees. Hidden tests and reference are never included."""
    return (
        f"{task.problem_statement}\n\n"
        f"Current (buggy) code:\n```python\n{task.starter_code}\n```\n\n"
        "Return the complete corrected module in a single ```python code block. "
        "No explanation outside the block is required."
    )


def extract_code_block(raw_output: str) -> str | None:
    """Best-effort extraction. Raw output stays canonical; None means GENERATED-only."""
    matches = re.findall(r"```(?:python)?\s*\n?(.*?)```", raw_output, re.DOTALL)
    if matches:
        code = max(matches, key=len).strip()
        return code or None
    stripped = raw_output.strip()
    if "def " in stripped or "class " in stripped:
        return stripped
    return None


def materialize_candidate(task: CorpusTask, raw_output: str, workdir: str | Path) -> dict[str, object]:
    """Write candidate.py + hidden_test.py into an isolated dir. No execution here."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    code = extract_code_block(raw_output)
    (workdir / "hidden_test.py").write_text(task.hidden_tests, encoding="utf-8")
    if code is None:
        return {"applies": False, "candidate_path": None, "reason": "no code block found"}
    (workdir / "candidate.py").write_text(code, encoding="utf-8")
    return {"applies": True, "candidate_path": str(workdir / "candidate.py"), "reason": "ok"}


def _t(
    task_id: str,
    title: str,
    fault_class: str,
    problem_statement: str,
    starter_code: str,
    hidden_tests: str,
    reference_solution: str,
    expected_note: str,
) -> CorpusTask:
    return CorpusTask(
        task_id=task_id,
        title=title,
        fault_class=fault_class,
        problem_statement=problem_statement,
        starter_code=starter_code,
        hidden_tests=hidden_tests,
        reference_solution=reference_solution,
        expected_note=expected_note,
    )


def seeded_corpus() -> tuple[CorpusTask, ...]:
    return (
        _t(
            "off-by-one-sum",
            "Inclusive range sum",
            "off-by-one",
            "Write total(n): return the sum 1 + 2 + ... + n for n >= 1.",
            "def total(n):\n    return sum(range(n))\n",
            "from candidate import total\n"
            "assert total(1) == 1\nassert total(5) == 15\nassert total(100) == 5050\n"
            "print('hidden OK')\n",
            "def total(n):\n    return sum(range(n + 1))\n",
            "range(n) excludes n; use range(n + 1).",
        ),
        _t(
            "inverted-comparison-adult",
            "Adulthood check",
            "inverted-comparison",
            "Write is_adult(age): return True when age is 18 or older, else False.",
            "def is_adult(age):\n    return age < 18\n",
            "from candidate import is_adult\n"
            "assert is_adult(18) is True\nassert is_adult(42) is True\n"
            "assert is_adult(17) is False\nassert is_adult(0) is False\n"
            "print('hidden OK')\n",
            "def is_adult(age):\n    return age >= 18\n",
            "Comparison was inverted; use >=.",
        ),
        _t(
            "missing-validation-divide",
            "Safe division",
            "missing-validation",
            "Write divide(a, b): return a / b, but raise ValueError when b is zero.",
            "def divide(a, b):\n    return a / b\n",
            "from candidate import divide\n"
            "assert divide(4, 2) == 2\n"
            "try:\n    divide(1, 0)\nexcept ValueError:\n    pass\n"
            "else:\n    raise AssertionError('expected ValueError')\n"
            "try:\n    divide(1, 0)\nexcept ZeroDivisionError:\n    raise AssertionError('must raise ValueError, not ZeroDivisionError')\n"
            "except ValueError:\n    pass\n"
            "print('hidden OK')\n",
            "def divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n",
            "Validate b before dividing; raise ValueError, not ZeroDivisionError.",
        ),
        _t(
            "exception-handling-parse",
            "Strict integer parsing",
            "incorrect-exception-handling",
            "Write parse_int(s): return int(s); invalid input must raise ValueError, never return a sentinel.",
            "def parse_int(s):\n    try:\n        return int(s)\n    except ValueError:\n        return -1\n",
            "from candidate import parse_int\n"
            "assert parse_int('42') == 42\n"
            "try:\n    parse_int('abc')\nexcept ValueError:\n    pass\n"
            "else:\n    raise AssertionError('expected ValueError, sentinel return hides the error')\n"
            "print('hidden OK')\n",
            "def parse_int(s):\n    return int(s)\n",
            "Do not swallow ValueError with a sentinel return.",
        ),
        _t(
            "cache-key-omission",
            "Memoized multiplication",
            "cache-key-omission",
            "Write mul(a, b) with a cache: repeated calls with the same arguments must reuse the cache, "
            "and different arguments must never return a stale cached value.",
            "def mul(a, b):\n    if not hasattr(mul, '_c'):\n        mul._c = None\n"
            "    if mul._c is not None:\n        return mul._c\n"
            "    mul._c = a * b\n    return mul._c\n",
            "from candidate import mul\n"
            "assert mul(2, 3) == 6\nassert mul(4, 5) == 20\nassert mul(2, 3) == 6\n"
            "print('hidden OK')\n",
            "def mul(a, b):\n    if not hasattr(mul, '_c'):\n        mul._c = {}\n"
            "    if (a, b) not in mul._c:\n        mul._c[(a, b)] = a * b\n"
            "    return mul._c[(a, b)]\n",
            "Cache key must include the arguments.",
        ),
        _t(
            "wrong-ordering-sort",
            "Sort by second element",
            "wrong-ordering",
            "Write sort_pairs(pairs): return pairs sorted by their second element, ascending.",
            "def sort_pairs(pairs):\n    return sorted(pairs, key=lambda p: p[0])\n",
            "from candidate import sort_pairs\n"
            "assert sort_pairs([(1, 3), (2, 1)]) == [(2, 1), (1, 3)]\n"
            "assert sort_pairs([(5, 0), (1, 9), (3, 4)]) == [(5, 0), (3, 4), (1, 9)]\n"
            "print('hidden OK')\n",
            "def sort_pairs(pairs):\n    return sorted(pairs, key=lambda p: p[1])\n",
            "Sort key must be p[1], not p[0].",
        ),
        _t(
            "stale-state-average",
            "Fresh running average",
            "stale-state-assumption",
            "Write average(xs): return the mean of the current list contents on every call.",
            "def average(xs):\n    if not hasattr(average, '_n'):\n        average._n = len(xs)\n"
            "        average._s = sum(xs)\n"
            "    return average._s / average._n\n",
            "from candidate import average\n"
            "assert average([1, 2, 3]) == 2 or True  # first call seeds stale state\n"
            "import importlib, candidate\nimportlib.reload(candidate)\n"
            "assert candidate.average([2, 4]) == 3\n"
            "assert candidate.average([10]) == 10\n"
            "print('hidden OK')\n",
            "def average(xs):\n    return sum(xs) / len(xs)\n",
            "Never reuse first-call state; recompute from current input.",
        ),
        _t(
            "incorrect-default-greeting",
            "Default greeting",
            "incorrect-default",
            "Write greet(name, greeting='hello'): return f'{greeting}, {name}!'. The default must be 'hello'.",
            "def greet(name, greeting='hi'):\n    return f'{greeting}, {name}!'\n",
            "from candidate import greet\n"
            "assert greet('Ada') == 'hello, Ada!'\n"
            "assert greet('Ada', greeting='hi') == 'hi, Ada!'\n"
            "print('hidden OK')\n",
            "def greet(name, greeting='hello'):\n    return f'{greeting}, {name}!'\n",
            "Default greeting must be 'hello'.",
        ),
        _t(
            "arithmetic-boundary-clamp",
            "Clamp to range",
            "arithmetic-boundary",
            "Write clamp(x, lo, hi): return lo if x < lo, hi if x > hi, else x.",
            "def clamp(x, lo, hi):\n    if x < lo:\n        return hi\n    if x > hi:\n        return lo\n    return x\n",
            "from candidate import clamp\n"
            "assert clamp(-5, 0, 10) == 0\nassert clamp(99, 0, 10) == 10\n"
            "assert clamp(5, 0, 10) == 5\nassert clamp(0, 0, 10) == 0\n"
            "print('hidden OK')\n",
            "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
            "Boundary returns were swapped.",
        ),
        _t(
            "incorrect-branching-fizzbuzz",
            "FizzBuzz labels",
            "incorrect-branching",
            "Write label(n): 'fizzbuzz' if divisible by 15, 'buzz' if by 5, 'fizz' if by 3, else str(n).",
            "def label(n):\n    if n % 3 == 0:\n        return 'fizz'\n"
            "    if n % 5 == 0:\n        return 'buzz'\n"
            "    if n % 15 == 0:\n        return 'fizzbuzz'\n"
            "    return str(n)\n",
            "from candidate import label\n"
            "assert label(15) == 'fizzbuzz'\nassert label(30) == 'fizzbuzz'\n"
            "assert label(3) == 'fizz'\nassert label(5) == 'buzz'\nassert label(7) == '7'\n"
            "print('hidden OK')\n",
            "def label(n):\n    if n % 15 == 0:\n        return 'fizzbuzz'\n"
            "    if n % 5 == 0:\n        return 'buzz'\n"
            "    if n % 3 == 0:\n        return 'fizz'\n"
            "    return str(n)\n",
            "Most-specific branch (15) must come first.",
        ),
        _t(
            "dropped-condition-password",
            "Password policy",
            "dropped-condition",
            "Write is_valid(pw): True only when len(pw) >= 8 AND pw contains at least one digit.",
            "def is_valid(pw):\n    return len(pw) >= 8\n",
            "from candidate import is_valid\n"
            "assert is_valid('abc12345') is True\n"
            "assert is_valid('abcdefgh') is False\n"
            "assert is_valid('ab12') is False\n"
            "print('hidden OK')\n",
            "def is_valid(pw):\n    return len(pw) >= 8 and any(c.isdigit() for c in pw)\n",
            "The digit requirement was dropped; restore it.",
        ),
        _t(
            "wrong-identifier-mapping",
            "Status code text",
            "wrong-identifier-mapping",
            "Write describe(code): 200 -> 'ok', 404 -> 'not found', 500 -> 'error'; anything else -> 'unknown'.",
            "def describe(code):\n    table = {200: 'ok', 404: 'ok', 500: 'error'}\n    return table.get(code, 'unknown')\n",
            "from candidate import describe\n"
            "assert describe(200) == 'ok'\nassert describe(404) == 'not found'\n"
            "assert describe(500) == 'error'\nassert describe(418) == 'unknown'\n"
            "print('hidden OK')\n",
            "def describe(code):\n    table = {200: 'ok', 404: 'not found', 500: 'error'}\n"
            "    return table.get(code, 'unknown')\n",
            "404 must map to 'not found'.",
        ),
    )


def get_task(corpus_task_id: str) -> CorpusTask:
    for task in seeded_corpus():
        if task.task_id == corpus_task_id:
            return task
    raise KeyError(f"unknown corpus task: {corpus_task_id}")
