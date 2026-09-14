from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from codeai.adapters import FakeCognitionAdapter
from codeai.analysis import export_experiment
from codeai.artifacts import FileArtifactStore
from codeai.corpus import seeded_corpus
from codeai.experiments import (
    ArmDef,
    ExperimentBudget,
    branch_count,
    build_config,
    create_experiment,
    run_arm,
)
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime
from codeai.stances import (
    ASSUMPTION_CHALLENGE,
    COUNTERFACTUAL,
    MINIMALITY,
    NORMAL,
    STANCES,
    stance_prompt,
    stance_suffix,
)


class StubModelConfig:
    def __init__(self, adapter):
        self._adapter = adapter

    def resolve(self, logical_name: str):
        from codeai.modelconfig import ModelMapping

        return ModelMapping(logical_name=logical_name, adapter="fake", model=logical_name)

    def build_adapter(self, logical_name: str):
        return self._adapter


def make_runtime(tmp_path: Path) -> Runtime:
    ledger = SQLiteLedger()
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


def test_stance_texts_are_small_and_normal_is_empty():
    assert stance_suffix(NORMAL) == ""
    for stance in (ASSUMPTION_CHALLENGE, MINIMALITY, COUNTERFACTUAL):
        assert 20 < len(stance_suffix(stance)) < 600, stance
    with pytest.raises(KeyError):
        stance_suffix("persona-theatre")


def test_normal_prompt_is_byte_identical_to_control():
    from codeai.corpus import visible_prompt

    task = seeded_corpus()[0]
    assert stance_prompt(visible_prompt(task), NORMAL) == visible_prompt(task)


def test_branch_count_for_custom_arms():
    assert branch_count(ArmDef(name="P2C", models=("m",), samples=12)) == 12
    assert branch_count(ArmDef(name="P2S", models=("m",), samples=12)) == 12


def test_stance_arms_share_fanout_with_variant_tags(tmp_path):
    runtime = make_runtime(tmp_path)
    task = seeded_corpus()[0]
    suffixes = ("", "_A", "_M", "_C") * 3
    labels = ("normal", "assumption_challenge", "minimality", "counterfactual") * 3
    config = build_config(
        name="p2test",
        hypothesis="h",
        task_ids=(task.task_id,),
        arms=(
            ArmDef(name="P2C", models=("m",), samples=12),
            ArmDef(name="P2S", models=("m",), samples=12,
                   prompt_suffixes=suffixes, stance_labels=labels),
        ),
        budget=ExperimentBudget(),
    )
    create_experiment(runtime, config)
    stub = StubModelConfig(FakeCognitionAdapter(responses=["```python\nx = 1\n```"]))
    run_arm(runtime, config.experiment_id, "P2S", (task,), stub,
            candidates_root=tmp_path / "cand")
    requested = runtime.ledger.events_by_kind(("call.requested",))
    assert len(requested) == 12
    variants = [r.payload["variant"] for r in requested]
    assert {v["prompt_variant"] for v in variants} == set(labels)
    prompts = {r.payload["context"]["prompt"] for r in requested}
    assert len(prompts) == 4  # base + 3 stance framings
    # stance config is immutable ledger state
    stored = next(iter(runtime.ledger.events_by_kind(("experiment.created",))))
    arm = next(a for a in stored.payload["arms"] if a["name"] == "P2S")
    assert len(arm["prompt_suffixes"]) == 12 and len(arm["stance_labels"]) == 12
    assert STANCES == ("normal", "assumption_challenge", "minimality", "counterfactual")


def test_export_carries_the_prompt_variable_of_each_arm(tmp_path):
    # Regression: the export once dropped prompt_suffixes/stance_labels, so a
    # matched prompt replication (normal x12 vs counterfactual x12) exported two
    # arm configs identical except for their names.
    runtime = make_runtime(tmp_path)
    task = seeded_corpus()[0]
    cf = stance_suffix(COUNTERFACTUAL)
    config = build_config(
        name="p3test",
        hypothesis="h",
        task_ids=(task.task_id,),
        arms=(
            ArmDef(name="P3C", models=("m",), samples=2),
            ArmDef(name="P3CF", models=("m",), samples=2,
                   prompt_suffixes=(cf, cf), stance_labels=(COUNTERFACTUAL, COUNTERFACTUAL)),
        ),
        budget=ExperimentBudget(),
    )
    create_experiment(runtime, config)
    stub = StubModelConfig(FakeCognitionAdapter(responses=["```python\nx = 1\n```"]))
    run_arm(runtime, config.experiment_id, "P3CF", (task,), stub,
            candidates_root=tmp_path / "cand")
    exported = export_experiment(runtime, config.experiment_id)
    cf_calls = [c for c in exported["calls"] if c.get("arm") == "P3CF"]
    assert len(cf_calls) == 2
    assert {c["prompt_variant"] for c in cf_calls} == {COUNTERFACTUAL}
    arms = {a["name"]: a for a in exported["experiment"]["arms"]}
    assert arms["P3C"]["prompt_suffixes"] == [] and arms["P3C"]["stance_labels"] == []
    assert arms["P3CF"]["prompt_suffixes"] == [cf, cf]
    assert arms["P3CF"]["stance_labels"] == [COUNTERFACTUAL, COUNTERFACTUAL]
    assert {k: v for k, v in arms["P3C"].items() if k not in ("name", "prompt_suffixes", "stance_labels")} \
        == {k: v for k, v in arms["P3CF"].items() if k not in ("name", "prompt_suffixes", "stance_labels")}


def test_control_arm_uses_unmodified_prompts(tmp_path):
    runtime = make_runtime(tmp_path)
    task = seeded_corpus()[0]
    config = build_config(
        name="p2control",
        hypothesis="h",
        task_ids=(task.task_id,),
        arms=(ArmDef(name="P2C", models=("m",), samples=4),),
        budget=ExperimentBudget(),
    )
    create_experiment(runtime, config)
    stub = StubModelConfig(FakeCognitionAdapter())
    run_arm(runtime, config.experiment_id, "P2C", (task,), stub,
            candidates_root=tmp_path / "cand")
    from codeai.corpus import visible_prompt

    requested = runtime.ledger.events_by_kind(("call.requested",))
    assert len(requested) == 4
    assert all(r.payload["context"]["prompt"] == visible_prompt(task) for r in requested)
    _ = tempfile.gettempdir()
