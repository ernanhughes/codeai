"""R1/R2/C runner. Authoring and execution are separate operations."""
from __future__ import annotations

from dataclasses import asdict

from .adapters import FakeCognitionAdapter
from .domain import Authority, Budget, Directive, Task
from .ledger import Event
from .providers import PRICING_TABLE, PRICING_VERSION
from .router_contract import (
    EXPERIMENT_VERSION, PATHS, canonical, deterministic_extract, digest,
    freeze_manifest, state_from_dict, validate_corpus,
)
from .router_model import PROMPTS, ModelRouter
from .scheduler import POLICY_VERSION, decide_next_step


def build_schedule(corpus, model_keys, *, synthetic=False):
    cases = validate_corpus(corpus)
    rows = []
    for case in cases:
        for path in PATHS[case.component]:
            model = path.startswith("M")
            variations = [(None, None, "baseline")] if not model else [
                (model_keys[0], "router-prompt-v1", "baseline"),
                (model_keys[0], "router-prompt-v1b", "prompt"),
                (model_keys[1], "router-prompt-v1", "model"),
            ]
            # Extraction wording is fixed; sensitivity applies to direct routing only.
            if path == "M+D":
                variations = [v for v in variations if v[2] != "prompt"]
            if model and case.component == "R1":
                variations.append((model_keys[0], "router-prompt-v1", "field_order"))
            for model_key, prompt, variant in variations:
                for repeat in range(1 if synthetic else (5 if model else 3)):
                    rows.append(dict(case_id=case.case_id, component=case.component, path=path,
                                     model_key=model_key, prompt_version=prompt,
                                     variant=variant, repeat=repeat))
    return rows


def run(runtime, corpus, oracle, manifest, routers: dict[str, ModelRouter], *,
        run_id: str, synthetic=False):
    """No implicit freeze, credentials, adapter discovery or downstream operations.

    Synthetic mode accepts only the repository's exact fake adapter type. Production
    mode requires a fully adjudicated, matching freeze record before any call.
    """
    cases = {c.case_id: c for c in validate_corpus(corpus)}
    if len(routers) != 2:
        raise ValueError("provide pinned primary and alternate model routes")
    if synthetic:
        if any(type(r.adapter) is not FakeCognitionAdapter for r in routers.values()):
            raise ValueError("synthetic mode only accepts exact FakeCognitionAdapter")
    else:
        if manifest != freeze_manifest(corpus, oracle, manifest["configuration"]):
            raise ValueError("freeze mismatch")
        configuration = manifest["configuration"]
        actual = {k: asdict(r.actor) for k, r in routers.items()}
        if actual != configuration["models"]:
            raise ValueError("model manifest mismatch")
        if {k: digest(v) for k, v in PROMPTS.items()} != configuration["prompts"]:
            raise ValueError("prompt manifest mismatch")
        if any(r.parameters != configuration["parameters"] for r in routers.values()):
            raise ValueError("parameter manifest mismatch")
    if runtime.artifact_store is None:
        raise ValueError("artifact store required")
    pricing = {"version": PRICING_VERSION, "ordered_rates": list(PRICING_TABLE.items())}
    # Match the existing estimator's ordered-prefix semantics exactly, even if a
    # future pricing correction changes them. Never substitute current prices later.
    if not synthetic and digest(pricing) != digest(manifest["configuration"]["pricing"]):
        raise ValueError("runtime pricing differs from frozen manifest")
    if any(e.stream_id == run_id for e in runtime.ledger.events_by_kind(("router.run_started",))):
        raise ValueError("run id already used; interrupted runs remain incomplete")
    schedule = build_schedule(corpus, list(routers), synthetic=synthetic)
    refs = {key: asdict(runtime.artifact_store.store_text(canonical(value), artifact_type=key))
            for key, value in {"router_corpus": corpus, "router_oracle": oracle,
                               "router_freeze": manifest}.items()}

    def emit(kind, payload, stream=run_id):
        runtime.ledger.append(Event.create(stream_id=stream, kind=kind, actor_id="router-experiment",
                                          correlation_id=run_id, payload=payload))

    emit("router.run_started", dict(version=EXPERIMENT_VERSION, run_id=run_id,
         synthetic=synthetic, corpus_hash=digest(corpus), oracle_hash=digest(oracle),
         manifest_hash=digest(manifest), refs=refs, schedule=schedule, pricing=pricing,
         policy_version=POLICY_VERSION))
    directive_id = run_id + ":directive"
    runtime.open_directive(Directive(directive_id, "Compare operation selection only", (),
                                      Budget(), Authority()))
    for index, row in enumerate(schedule):
        case = cases[row["case_id"]]
        decision_id = f"{run_id}:{index}"
        task_id = decision_id + ":task"
        runtime.create_task(Task(task_id, directive_id, "Select next operation", (), Budget(), Authority()))
        if case.component == "R1":
            input_value = {"state": case.state}
        elif row["path"] == "D":
            input_value = {"state": case.state}  # structural C baseline
        else:
            input_value = {"narrative": case.narrative, "irrelevant_history": case.distractor}
        # canonical JSON normally sorts keys; order sensitivity uses an explicit list
        # of the same field/value pairs, never extra decision information.
        if row["variant"] == "field_order":
            input_value = {"state_fields": list(reversed(list(case.state.items())))}
        request = row | {"decision_id": decision_id, "run_id": run_id,
                         "input": input_value, "input_hash": digest(input_value),
                         "structured_state_hash": case.state_hash}
        emit("router.requested", request, decision_id)
        model_evidence = None
        if row["path"].startswith("M"):
            model_evidence = routers[row["model_key"]].decide(
                input_value, decision_id=decision_id, task_id=task_id, experiment_id=run_id,
                path=row["path"], prompt_version=row["prompt_version"], extract=row["path"] == "M+D")
            parsed = model_evidence["parsed"]
            if row["path"] == "M+D" and "state" in parsed:
                state = parsed["state"]
                result = asdict(decide_next_step(state_from_dict(state))) | {"extracted_state": state}
            else:
                result = parsed
        elif row["path"] == "D+D":
            state = deterministic_extract(case.narrative)
            result = (asdict(decide_next_step(state_from_dict(state))) | {"extracted_state": state}
                      if state is not None else {"operation": "UNSUPPORTED", "reason": "narrative outside explicit-state grammar"})
        else:
            result = asdict(decide_next_step(state_from_dict(case.state)))
        emit("router.decided", request | {"result": result, "model_evidence": model_evidence}, decision_id)
    emit("router.run_completed", {"run_id": run_id, "decision_count": len(schedule)})
