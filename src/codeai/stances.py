from __future__ import annotations

NORMAL = "normal"
ASSUMPTION_CHALLENGE = "assumption_challenge"
MINIMALITY = "minimality"
COUNTERFACTUAL = "counterfactual"

STANCES: tuple[str, ...] = (NORMAL, ASSUMPTION_CHALLENGE, MINIMALITY, COUNTERFACTUAL)

_SUFFIXES: dict[str, str] = {
    NORMAL: "",
    ASSUMPTION_CHALLENGE: (
        "\n\nBefore modifying the code, identify any existing assumptions, state, "
        "abstractions, caching, ownership or lifecycle decisions that may themselves "
        "be incorrect. Do not assume the current architecture should be preserved."
    ),
    MINIMALITY: (
        "\n\nPrefer the smallest semantic repair. Before adding state, branches, caches, "
        "wrappers or new abstractions, consider whether incorrect machinery should "
        "instead be removed or simplified."
    ),
    COUNTERFACTUAL: (
        "\n\nIgnore the existing implementation strategy for a moment. Starting only from "
        "the externally observable contract and failing behavior, describe the simplest "
        "implementation you would write from scratch. Then compare that to the existing "
        "implementation and make the smallest change required."
    ),
}


def stance_suffix(stance: str) -> str:
    if stance not in _SUFFIXES:
        raise KeyError(f"unknown stance: {stance}")
    return _SUFFIXES[stance]


def stance_prompt(base_prompt: str, stance: str) -> str:
    return base_prompt + stance_suffix(stance)
