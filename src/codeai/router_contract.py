"""Router experiment v1 data contracts. No scheduler or model-derived oracle."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from .scheduler import SchedulerInput

CORPUS_VERSION = "router_cases_v1"
ORACLE_VERSION = "router_oracle_v1"
EXPERIMENT_VERSION = "router-experiment-v1"
OPERATIONS = frozenset({"CALL", "CHECK", "ASK_HUMAN", "STOP"})
STATE_FIELDS = tuple(f.name for f in fields(SchedulerInput))
PATHS = {"R1": ("D", "M-direct"), "R2": ("D+D", "M+D", "M-direct"),
         "C": ("D", "D+D", "M+D", "M-direct")}
THRESHOLDS = {
    "material_delta": .15, "sub_stratum_loss": .05, "median_flip": .10,
    "p95_flip": .20, "refuse": .05, "prompt_change": .10, "auditability": .90,
    "model_repeats": 5, "deterministic_repeats": 3, "low_scorable": 30,
    "ambiguity": 1 / 3, "cost_reporting_band": .05,
}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def state_from_dict(value: dict) -> SchedulerInput:
    if not isinstance(value, dict) or set(value) != set(STATE_FIELDS):
        raise ValueError("explicit five-field state required; no legacy aliases")
    return SchedulerInput(**value)


@dataclass(frozen=True)
class RouterCase:
    case_id: str
    component: str
    sub_stratum: str
    state: dict
    narrative: str = ""
    semantic_validity: str = "review_required"
    validity_reason: str = "Author must review whether the process state is sufficient."
    pair_id: str | None = None
    base_case_id: str | None = None
    distractor_variant: str | None = None
    distractor: str = ""

    def __post_init__(self):
        state_from_dict(self.state)
        if not self.case_id or self.component not in PATHS or not self.sub_stratum:
            raise ValueError("invalid case identity/component")
        if self.semantic_validity not in {"valid", "review_required", "impossible"}:
            raise ValueError("explicit semantic validity required")
        if not self.validity_reason:
            raise ValueError("validity reason required")
        if self.component in {"R2", "C"} and not self.narrative:
            raise ValueError("narrative required")

    @property
    def state_hash(self):
        return digest(self.state)


def validate_corpus(corpus: dict, *, freeze=False) -> list[RouterCase]:
    if corpus.get("version") != CORPUS_VERSION:
        raise ValueError("unknown corpus version")
    cases = [RouterCase(**c) for c in corpus["cases"]]
    by_id = {c.case_id: c for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case")
    pairs = {}
    for c in cases:
        if freeze and c.semantic_validity != "valid":
            raise ValueError("semantic validity requires human review before freeze")
        if c.component == "C":
            if not c.pair_id or c.base_case_id not in by_id or not c.distractor_variant:
                raise ValueError("incomplete distractor relationship")
            pairs.setdefault(c.pair_id, []).append(c)
    for group in pairs.values():
        if len(group) != 2 or len({c.state_hash for c in group}) != 1:
            raise ValueError("pair must contain two identical decision states")
        if len({c.base_case_id for c in group}) != 1:
            raise ValueError("pair base mismatch")
        base = by_id[group[0].base_case_id]
        if base not in group or len({c.narrative for c in group}) != 1:
            raise ValueError("pair must share base narrative")
        if len({c.distractor for c in group}) != 2:
            raise ValueError("distractors must differ")
    if freeze:
        if sum(c.component == "R1" for c in cases) < 24:
            raise ValueError("R1 needs at least 24 cases")
        if not 24 <= sum(c.component == "R2" for c in cases) <= 32:
            raise ValueError("R2 needs 24–32 narratives")
        if len(pairs) < 10:
            raise ValueError("C needs at least ten pairs")
    return cases


def adjudicate(corpus: dict, entries: list[dict]) -> dict:
    """Two distinct human identifiers; disagreement stays ambiguous. No tie breaker."""
    cases = validate_corpus(corpus)
    grouped = {c.case_id: [] for c in cases}
    for e in entries:
        if e["case_id"] not in grouped:
            raise ValueError("unknown adjudicated case")
        if e["decision"] not in OPERATIONS | {"AMBIGUOUS"}:
            raise ValueError("invalid oracle label")
        for key in ("adjudicator_id", "reason", "facts", "version", "timestamp"):
            if not e.get(key):
                raise ValueError(f"missing adjudication {key}")
        datetime.fromisoformat(e["timestamp"])
        grouped[e["case_id"]].append(e)
    labels = {}
    for key, rows in grouped.items():
        if len(rows) != 2 or len({r['adjudicator_id'] for r in rows}) != 2:
            raise ValueError("two independent adjudications required per case")
        labels[key] = rows[0]["decision"] if rows[0]["decision"] == rows[1]["decision"] else "AMBIGUOUS"
    return {"version": ORACLE_VERSION, "corpus_hash": digest(corpus),
            "adjudications": entries, "labels": labels}


def freeze_manifest(corpus: dict, oracle: dict, configuration: dict) -> dict:
    validate_corpus(corpus, freeze=True)
    if oracle != adjudicate(corpus, oracle["adjudications"]):
        raise ValueError("oracle identity or adjudications mismatch")
    for key in ("models", "prompts", "pricing", "parameters", "reason_rating_plan",
                "preregistration_sha256", "source_revision"):
        if not configuration.get(key):
            raise ValueError(f"freeze requires {key}")
    if len(configuration["models"]) < 2:
        raise ValueError("pin primary and alternate model manifests")
    if set(configuration["prompts"]) != {"router-prompt-v1", "router-prompt-v1b"}:
        raise ValueError("pin both prompt versions")
    return {"version": EXPERIMENT_VERSION, "status": "FROZEN",
            "corpus_hash": digest(corpus), "oracle_hash": digest(oracle),
            "configuration": configuration, "thresholds": THRESHOLDS}


def write_once(path: Path, value: dict):
    """Freeze records cannot be silently replaced by the authoring tool."""
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(canonical(value) + "\n")


def deterministic_extract(narrative: str) -> dict | None:
    """Only a documented explicit-state envelope, never guessed prose comprehension."""
    if not narrative.startswith("STATE_JSON\n"):
        return None
    try:
        value = json.loads(narrative.removeprefix("STATE_JSON\n"))
        return asdict(state_from_dict(value))
    except (ValueError, TypeError):
        return None
