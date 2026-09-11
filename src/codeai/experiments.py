from __future__ import annotations

import difflib
import hashlib
import json
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters import CheckRequest
from .corpus import CORPUS_VERSION, CorpusTask, materialize_candidate, visible_prompt
from .domain import Authority, Budget, Capability, Claim, Directive, Task
from .ledger import Event
from .verifier import LocalCommandVerifier

ARM_C0 = "C0"
ARM_C1 = "C1"
ARM_H1 = "H1"

# Candidate outcome taxonomy. Never collapse failures into one status.
GENERATED = "GENERATED"
APPLIES = "APPLIES"
COMPILES = "COMPILES"
VERIFIER_PASS = "VERIFIER_PASS"
VERIFIER_FAIL = "VERIFIER_FAIL"
EXECUTION_ERROR = "EXECUTION_ERROR"
TIMEOUT = "TIMEOUT"
NOT_RUN = "NOT_RUN"
BUDGET_STOPPED = "BUDGET_STOPPED"


@dataclass(frozen=True, slots=True)
class ExperimentBudget:
    max_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class ArmDef:
    name: str
    models: tuple[str, ...] = ()
    samples: int = 1
    temperature: float | None = None
    seed: str | None = None
    prompt_version: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_id: str
    name: str
    hypothesis: str
    primary_metric: str
    corpus_version: str
    task_ids: tuple[str, ...]
    arms: tuple[ArmDef, ...]
    budget: ExperimentBudget
    stopping_rule: str
    created_at: str
    config_hash: str


def canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_config(
    *,
    experiment_id: str | None = None,
    name: str,
    hypothesis: str,
    primary_metric: str = "verified oracle@k",
    corpus_version: str = CORPUS_VERSION,
    task_ids: tuple[str, ...],
    arms: tuple[ArmDef, ...],
    budget: ExperimentBudget | None = None,
    stopping_rule: str = "stop when budget exhausted; record incomplete arms explicitly",
) -> ExperimentConfig:
    eid = experiment_id or str(uuid.uuid4())
    body = {
        "experiment_id": eid,
        "name": name,
        "hypothesis": hypothesis,
        "primary_metric": primary_metric,
        "corpus_version": corpus_version,
        "task_ids": list(task_ids),
        "arms": [asdict(a) for a in arms],
        "budget": asdict(budget or ExperimentBudget()),
        "stopping_rule": stopping_rule,
    }
    return ExperimentConfig(
        experiment_id=eid,
        name=name,
        hypothesis=hypothesis,
        primary_metric=primary_metric,
        corpus_version=corpus_version,
        task_ids=task_ids,
        arms=arms,
        budget=budget or ExperimentBudget(),
        stopping_rule=stopping_rule,
        created_at=datetime.now(UTC).isoformat(),
        config_hash=canonical_hash(body),
    )


def preregistration_text(config: ExperimentConfig) -> str:
    lines = [
        f"Experiment: {config.name} ({config.experiment_id})",
        f"Hypothesis: {config.hypothesis}",
        f"Primary metric: {config.primary_metric}",
        "Secondary metrics: cost per solve, pairwise failure correlation, unique rescues.",
        f"Task corpus version: {config.corpus_version} ({len(config.task_ids)} tasks)",
        "Arms:",
    ]
    for arm in config.arms:
        lines.append(f"  - {arm.name}: models={list(arm.models)} samples={arm.samples}")
    lines += [
        f"Budget: {asdict(config.budget)}",
        f"Stopping rule: {config.stopping_rule}",
        "No synthesis.",
        f"Config hash: {config.config_hash}",
    ]
    return "\n".join(lines) + "\n"


def _existing_experiment_ids(runtime: Any) -> set[str]:
    return {
        str(e.payload.get("experiment_id", ""))
        for e in runtime.ledger.events_by_kind(("experiment.created",))
    }


def create_experiment(
    runtime: Any,
    config: ExperimentConfig,
    *,
    actor_id: str = "human",
) -> Event:
    """Persist immutable experiment config + preregistration. Duplicate ids rejected."""
    if config.experiment_id in _existing_experiment_ids(runtime):
        raise ValueError(
            f"experiment {config.experiment_id} already exists and is immutable; "
            "create a new experiment instead of modifying it"
        )
    prereg = preregistration_text(config)
    prereg_ref = None
    if runtime.artifact_store is not None:
        ref = runtime.artifact_store.store_text(
            prereg, media_type="text/plain", artifact_type="preregistration"
        )
        prereg_ref = {"artifact_id": ref.artifact_id, "sha256": ref.sha256}
    event = Event.create(
        stream_id=config.experiment_id,
        kind="experiment.created",
        actor_id=actor_id,
        payload={
            "experiment_id": config.experiment_id,
            "name": config.name,
            "hypothesis": config.hypothesis,
            "primary_metric": config.primary_metric,
            "corpus_version": config.corpus_version,
            "task_ids": list(config.task_ids),
            "arms": [asdict(a) for a in config.arms],
            "budget": asdict(config.budget),
            "stopping_rule": config.stopping_rule,
            "created_at": config.created_at,
            "config_hash": config.config_hash,
            "preregistration": prereg,
            "preregistration_artifact": prereg_ref,
        },
        correlation_id=config.experiment_id,
    )
    runtime.ledger.append(event)
    return event


def get_experiment(runtime: Any, experiment_id: str) -> ExperimentConfig | None:
    for event in runtime.ledger.events_by_kind(("experiment.created",)):
        if str(event.payload.get("experiment_id")) == experiment_id:
            p = event.payload
            return ExperimentConfig(
                experiment_id=str(p["experiment_id"]),
                name=str(p["name"]),
                hypothesis=str(p.get("hypothesis", "")),
                primary_metric=str(p.get("primary_metric", "")),
                corpus_version=str(p.get("corpus_version", "")),
                task_ids=tuple(p.get("task_ids", ())),
                arms=tuple(ArmDef(**a) for a in p.get("arms", ())),
                budget=ExperimentBudget(**p.get("budget", {})),
                stopping_rule=str(p.get("stopping_rule", "")),
                created_at=str(p.get("created_at", "")),
                config_hash=str(p.get("config_hash", "")),
            )
    return None


def plan_experiment(
    config: ExperimentConfig,
    model_config: Any,
    *,
    corpus_tasks: tuple[CorpusTask, ...] | None = None,
) -> dict[str, Any]:
    """Dry-run plan. Pure: no ledger writes, no model calls, no side effects."""
    from .modelconfig import missing_credentials

    tasks = [t for t in (corpus_tasks or ()) if t.task_id in set(config.task_ids)] or None
    task_count = len(config.task_ids)
    arm_plans = []
    total_calls = 0
    for arm in config.arms:
        if arm.name == ARM_C0:
            calls = task_count * 1
        else:
            calls = task_count * branch_count(arm)
        total_calls += calls
        models = []
        for logical in arm.models:
            try:
                mapping = model_config.resolve(logical)
                models.append(
                    {
                        "logical": logical,
                        "adapter": mapping.adapter,
                        "model": mapping.model,
                        "base_url": mapping.base_url,
                        "missing": missing_credentials(mapping),
                    }
                )
            except KeyError:
                models.append({"logical": logical, "missing": "not in config"})
        arm_plans.append(
            {
                "arm": arm.name,
                "models": models,
                "samples": arm.samples,
                "calls_planned": calls,
                "prompt_version": arm.prompt_version,
            }
        )
    within_budget = True
    budget_note = "ok"
    if config.budget.max_calls is not None and total_calls > config.budget.max_calls:
        within_budget = False
        budget_note = f"planned {total_calls} calls exceed max_calls {config.budget.max_calls}"
    return {
        "experiment_id": config.experiment_id,
        "name": config.name,
        "tasks": list(config.task_ids),
        "task_count": task_count,
        "arms": arm_plans,
        "total_calls_planned": total_calls,
        "budget": asdict(config.budget),
        "within_budget": within_budget,
        "budget_note": budget_note,
        "verifier": "hidden deterministic tests per corpus task (never shown to model)",
        "side_effects": "none in dry-run",
        "corpus_tasks_resolved": len(tasks) if tasks else 0,
    }


def experiment_usage(runtime: Any, experiment_id: str) -> dict[str, float]:
    calls = [
        e for e in runtime.ledger.events_by_kind(("call.completed",))
        if str(e.payload.get("experiment_id", "")) == experiment_id
    ]
    tokens = sum(int(e.payload.get("input_tokens", 0) or 0) + int(e.payload.get("output_tokens", 0) or 0) for e in calls)
    cost = sum(float(e.payload.get("cost_usd") or 0) for e in calls)
    return {"calls": float(len(calls)), "tokens": float(tokens), "cost_usd": cost}


def _budget_allows(config: ExperimentConfig, usage: dict[str, float], planned_calls: int = 1) -> tuple[bool, str]:
    b = config.budget
    if b.max_calls is not None and usage["calls"] + planned_calls > b.max_calls:
        return False, f"max_calls {b.max_calls} would be exceeded"
    if b.max_tokens is not None and usage["tokens"] >= b.max_tokens:
        return False, f"max_tokens {b.max_tokens} exhausted"
    if b.max_cost_usd is not None and usage["cost_usd"] >= b.max_cost_usd:
        return False, f"max_cost_usd {b.max_cost_usd} exhausted"
    return True, "ok"


def starting_state_hash(task: CorpusTask) -> str:
    return hashlib.sha256(
        f"{CORPUS_VERSION}\0{task.task_id}\0{task.starter_code}\0{task.hidden_tests}".encode()
    ).hexdigest()


def run_arm(
    runtime: Any,
    experiment_id: str,
    arm_name: str,
    corpus_tasks: tuple[CorpusTask, ...],
    model_config: Any,
    *,
    candidates_root: str | Path,
    verifier: Any | None = None,
    timeout_seconds: float | None = None,
    actor_prefix: str = "exp",
) -> dict[str, Any]:
    """Execute one arm: sealed fanout per task + isolated hidden verification.

    PRODUCT/RESEARCH separation: uses ordinary Directive/Task/Call/Check/Claim
    ledger events, but every branch starts from the same isolated starting state
    and only the hidden verifier decides success. No synthesis.
    """
    from .domain import ActorRef

    config = get_experiment(runtime, experiment_id)
    if config is None:
        raise KeyError(f"unknown experiment: {experiment_id}")
    arm = next((a for a in config.arms if a.name == arm_name), None)
    if arm is None:
        raise KeyError(f"unknown arm {arm_name} in experiment {experiment_id}")
    verifier = verifier or LocalCommandVerifier()
    timeout = timeout_seconds or config.budget.timeout_seconds
    candidates_root = Path(candidates_root)

    directive = Directive(
        directive_id=str(uuid.uuid4()),
        objective=f"Experiment {config.name} arm {arm_name}",
        success_criteria=(f"verified {config.primary_metric}",),
        budget=Budget(),
        authority=Authority(frozenset({Capability.READ, Capability.EXECUTE})),
    )
    runtime.open_directive(directive, actor_id="human")

    branch_models = _models_for_arm(arm)
    completed_tasks = 0
    stopped: dict[str, Any] | None = None

    for corpus_task in corpus_tasks:
        planned = branch_count(arm)
        ok, reason = _budget_allows(config, experiment_usage(runtime, experiment_id), planned)
        if not ok:
            stopped = {"reason": BUDGET_STOPPED, "detail": reason,
                       "remaining": [t.task_id for t in corpus_tasks[completed_tasks:]]}
            runtime.ledger.append(
                Event.create(
                    stream_id=experiment_id, kind="experiment.arm_stopped", actor_id="runtime",
                    payload={"experiment_id": experiment_id, "arm": arm_name, **stopped},
                    correlation_id=experiment_id,
                )
            )
            break

        task = Task(
            task_id=str(uuid.uuid4()),
            directive_id=directive.directive_id,
            objective=f"[{experiment_id}/{arm_name}] {corpus_task.title}",
            success_criteria=("hidden verifier passes",),
            budget=Budget(),
            authority=Authority(frozenset({Capability.READ, Capability.EXECUTE})),
        )
        runtime.create_task(task, actor_id="runtime")
        runtime.ledger.append(
            Event.create(
                stream_id=task.task_id, kind="experiment.task_mapped", actor_id="runtime",
                payload={"experiment_id": experiment_id, "arm": arm_name,
                         "task_id": task.task_id, "corpus_task_id": corpus_task.task_id,
                         "starting_state_hash": starting_state_hash(corpus_task)},
                correlation_id=experiment_id,
            )
        )

        branches = []
        for i in range(planned):
            logical = branch_models[i % len(branch_models)]
            mapping = model_config.resolve(logical)
            adapter = model_config.build_adapter(logical)
            branches.append(
                {
                    "actor": ActorRef(actor_id=f"{actor_prefix}-{logical}-{i}", kind="model",
                                      provider=mapping.adapter, model=mapping.model),
                    "adapter": adapter,
                    "adapter_id": mapping.adapter,
                    "variant": {
                        "model": mapping.model, "provider": mapping.adapter,
                        "experiment": experiment_id,
                    },
                    "prompt": visible_prompt(corpus_task),
                    "parameters": _branch_params(arm, i),
                    "experiment_id": experiment_id,
                    "arm": arm_name,
                }
            )
        # Variant must be a Variant instance for sealed_fanout typing; convert.
        from .domain import Variant as VariantCls

        for branch in branches:
            info = branch["variant"]
            assert isinstance(info, dict)
            branch["variant"] = VariantCls(
                model=info["model"], provider=info["provider"], experiment=info["experiment"],
                temperature=arm.temperature, seed=str(arm.seed) if arm.seed else None,
                tags={"arm": arm_name, "logical_model": branch["actor"].model or ""},
            )

        results = runtime.sealed_fanout(
            task_id=task.task_id, base_prompt=visible_prompt(corpus_task), branches=branches,
            directive_id=directive.directive_id, run_id=directive.directive_id,
            objective=corpus_task.problem_statement, prompt_version=arm.prompt_version or CORPUS_VERSION,
            experiment_id=experiment_id, arm=arm_name,
        )
        for result in results:
            _verify_candidate(
                runtime, experiment_id, arm_name, corpus_task, task.task_id,
                result, candidates_root, verifier, timeout,
            )
        completed_tasks += 1

    runtime.ledger.append(
        Event.create(
            stream_id=experiment_id, kind="experiment.arm_completed", actor_id="runtime",
            payload={"experiment_id": experiment_id, "arm": arm_name,
                     "completed_tasks": completed_tasks,
                     "stopped": stopped},
            correlation_id=experiment_id,
        )
    )
    return {"arm": arm_name, "completed_tasks": completed_tasks, "stopped": stopped,
            "directive_id": directive.directive_id}


def _models_for_arm(arm: ArmDef) -> list[str]:
    if arm.name == ARM_C0:
        return [arm.models[0]] if arm.models else ["fake"]
    if arm.name == ARM_C1:
        base = arm.models[0] if arm.models else "fake"
        return [base] * max(1, arm.samples)
    models = list(arm.models) or ["fake"]
    # H1: one branch per model; samples acts as a minimum total (cycles models).
    total = max(len(models), max(1, arm.samples))
    return [models[i % len(models)] for i in range(total)]


def branch_count(arm: ArmDef) -> int:
    if arm.name == ARM_C0:
        return 1
    if arm.name == ARM_C1:
        return max(1, arm.samples)
    models = list(arm.models) or ["fake"]
    return max(len(models), max(1, arm.samples))


def _branch_params(arm: ArmDef, index: int) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if arm.temperature is not None:
        params["temperature"] = arm.temperature
    seed = arm.seed or f"{arm.name}-{index}"
    params["seed"] = seed
    return params


def _verify_candidate(
    runtime: Any, experiment_id: str, arm: str, corpus_task: CorpusTask, task_id: str,
    result: Any, candidates_root: Path, verifier: Any, timeout: float,
) -> str:
    """Materialize in isolation, compile-check, hidden-verify, ledger everything."""
    start_hash = starting_state_hash(corpus_task)
    workdir = candidates_root / experiment_id / arm / corpus_task.task_id / result.call_id
    material = materialize_candidate(corpus_task, result.raw_output or "", workdir)

    candidate_artifact = None
    diff_text = ""
    if material["applies"] and runtime.artifact_store is not None:
        with open(str(material["candidate_path"]), encoding="utf-8") as handle:
            code = handle.read()
        candidate_artifact = runtime.artifact_store.store_text(
            code, media_type="text/plain", artifact_type="candidate_patch")
        diff_text = "\n".join(difflib.unified_diff(
            corpus_task.starter_code.splitlines(), code.splitlines(),
            fromfile="starter.py", tofile="candidate.py", lineterm=""))
        runtime.artifact_store.store_text(
            diff_text, media_type="text/plain", artifact_type="candidate_diff")

    outcome = GENERATED
    check_id: str | None = None
    if material["applies"]:
        compile_check = verifier.run(CheckRequest(
            check_id=str(uuid.uuid4()), task_id=task_id,
            command=(sys.executable, "-m", "py_compile", "candidate.py"),
            cwd=str(workdir), timeout_seconds=timeout))
        if compile_check.exit_code != 0:
            outcome = EXECUTION_ERROR
        else:
            outcome = COMPILES
            # Single authoritative hidden-verifier execution via the durable path.
            recorded = runtime.run_check(CheckRequest(
                check_id=str(uuid.uuid4()), task_id=task_id,
                command=(sys.executable, "hidden_test.py"),
                cwd=str(workdir), timeout_seconds=timeout), verifier=verifier)
            check_id = recorded.check_id
            if recorded.verdict == "PASS":
                outcome = VERIFIER_PASS
            elif recorded.error and "timed out" in recorded.error:
                outcome = TIMEOUT
            elif recorded.verdict == "ERROR":
                outcome = EXECUTION_ERROR
            else:
                outcome = VERIFIER_FAIL
    if result.status == "failed" and outcome == GENERATED:
        outcome = EXECUTION_ERROR if result.error else GENERATED

    claim_id = f"claim-{result.call_id}"
    runtime.record_claim(Claim(
        claim_id=claim_id, task_id=task_id,
        statement=f"candidate {result.call_id} solves {corpus_task.task_id}",
        source_call_id=result.call_id,
        run_id=experiment_id,
        source_artifact_id=candidate_artifact.artifact_id if candidate_artifact else None,
        scope=f"experiment {experiment_id} arm {arm}",
    ))
    if check_id and outcome == VERIFIER_PASS:
        from .domain import ClaimStatus, EvidenceClass

        runtime.ledger.append(Event.create(
            stream_id=claim_id, kind="claim.evidence", actor_id="verifier",
            payload={"claim_id": claim_id, "evidence_class": EvidenceClass.REPRODUCED.value,
                     "check_id": check_id}, correlation_id=task_id))
    elif outcome in (VERIFIER_FAIL, EXECUTION_ERROR, TIMEOUT) and material["applies"]:
        from .domain import ClaimStatus

        runtime.ledger.append(Event.create(
            stream_id=claim_id, kind="claim.status", actor_id="verifier",
            payload={"claim_id": claim_id, "status": ClaimStatus.REFUTED.value,
                     "details": outcome}, correlation_id=task_id))

    runtime.ledger.append(Event.create(
        stream_id=result.call_id, kind="experiment.candidate_verified", actor_id="verifier",
        payload={"experiment_id": experiment_id, "arm": arm,
                 "corpus_task_id": corpus_task.task_id, "task_id": task_id,
                 "call_id": result.call_id, "outcome": outcome,
                 "check_id": check_id, "starting_state_hash": start_hash,
                 "candidate_artifact_id": candidate_artifact.artifact_id if candidate_artifact else None,
                 "diff": diff_text[:8000]},
        correlation_id=experiment_id))
    return outcome
