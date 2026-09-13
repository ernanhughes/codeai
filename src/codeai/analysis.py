from __future__ import annotations

import json
import statistics
from typing import Any

from .experiments import (
    VERIFIER_PASS,
    call_consumption,
    combine_usage,
    get_experiment,
)

TINY_N_THRESHOLD = 30


def collect_experiment(runtime: Any, experiment_id: str) -> dict[str, Any]:
    """Gather raw observations for one experiment from ledger + artifact store."""
    config = get_experiment(runtime, experiment_id)
    if config is None:
        raise KeyError(f"unknown experiment: {experiment_id}")
    mappings = [
        e.payload for e in runtime.ledger.events_by_kind(("experiment.task_mapped",))
        if str(e.payload.get("experiment_id")) == experiment_id
    ]
    task_to_corpus = {str(m["task_id"]): str(m["corpus_task_id"]) for m in mappings}
    task_to_arm = {str(m["task_id"]): str(m["arm"]) for m in mappings}
    calls = [
        e.payload for e in runtime.ledger.events_by_kind(("call.completed",))
        if str(e.payload.get("experiment_id", "")) == experiment_id
    ]
    candidates = [
        e.payload for e in runtime.ledger.events_by_kind(("experiment.candidate_verified",))
        if str(e.payload.get("experiment_id")) == experiment_id
    ]
    experiment_task_ids = set(task_to_corpus)
    checks = [
        e.payload for e in runtime.ledger.events_by_kind(("check.completed",))
        if str(e.correlation_id) in experiment_task_ids
    ]
    stops = [
        e.payload for e in runtime.ledger.events_by_kind(("experiment.arm_stopped",))
        if str(e.payload.get("experiment_id")) == experiment_id
    ]
    completions = [
        e.payload for e in runtime.ledger.events_by_kind(("experiment.arm_completed",))
        if str(e.payload.get("experiment_id")) == experiment_id
    ]
    return {
        "config": config,
        "mappings": mappings,
        "task_to_corpus": task_to_corpus,
        "task_to_arm": task_to_arm,
        "calls": calls,
        "candidates": candidates,
        "checks": checks,
        "stops": stops,
        "completions": completions,
    }


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def arm_metrics(
    arm: str,
    corpus_task_ids: list[str],
    candidates: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    arm_candidates = [c for c in candidates if str(c.get("arm")) == arm]
    arm_calls = [c for c in calls if str(c.get("arm")) == arm]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for task_id in corpus_task_ids:
        by_task[task_id] = [c for c in arm_candidates if str(c.get("corpus_task_id")) == task_id]
    solved = {t for t, cs in by_task.items() if any(c.get("outcome") == VERIFIER_PASS for c in cs)}
    attempted = {t for t, cs in by_task.items() if cs}
    outcome_counts: dict[str, int] = {}
    for c in arm_candidates:
        outcome_counts[str(c.get("outcome"))] = outcome_counts.get(str(c.get("outcome")), 0) + 1
    latencies = [float(c["latency_ms"]) for c in arm_calls if c.get("latency_ms") is not None]
    usage = combine_usage([call_consumption(c) for c in arm_calls])
    tokens = usage.known_tokens
    cost = usage.known_cost_usd
    # Historical cost_known semantics preserved verbatim (any known): changing
    # it to all-known would silently alter frozen P-series report figures.
    cost_known = any(c.get("cost_usd") is not None for c in arm_calls)
    k = max((len(cs) for cs in by_task.values()), default=0)
    oracle = len(solved) / len(attempted) if attempted else 0.0
    return {
        "arm": arm,
        "tasks_attempted": len(attempted),
        "tasks_total": len(corpus_task_ids),
        "calls": len(arm_calls),
        "k": k,
        f"oracle_at_{k}": round(oracle, 4),
        "verified_solve_rate": round(oracle, 4),
        "successful_candidates": outcome_counts.get(VERIFIER_PASS, 0),
        "failures_by_category": outcome_counts,
        "tokens": tokens,
        "tokens_complete": usage.tokens_complete,
        "cost_usd": round(cost, 6),
        "cost_known": cost_known,
        "cost_per_verified_solve": round(cost / len(solved), 6) if solved and cost_known else None,
        "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "median_latency_ms": _median(latencies),
        "solved_tasks": sorted(solved),
    }


def conditional_failure(
    solved_a: set[str], solved_b: set[str], attempted: set[str]
) -> float | None:
    """P(B fails | A fails) over jointly attempted tasks."""
    both = attempted
    a_fail = {t for t in both if t not in solved_a}
    if not a_fail:
        return None
    b_fail_given_a_fail = {t for t in a_fail if t not in solved_b}
    return round(len(b_fail_given_a_fail) / len(a_fail), 4)


def unique_rescues(
    solved_focus: set[str], solved_baseline: set[str]
) -> list[str]:
    return sorted(solved_focus - solved_baseline)


def build_report(runtime: Any, experiment_id: str) -> dict[str, Any]:
    data = collect_experiment(runtime, experiment_id)
    config = data["config"]
    corpus_task_ids = list(config.task_ids)
    arms = [a.name for a in config.arms]
    per_arm = {
        arm: arm_metrics(arm, corpus_task_ids, data["candidates"], data["calls"]) for arm in arms
    }
    solved_by_arm = {arm: set(m["solved_tasks"]) for arm, m in per_arm.items()}
    attempted_by_arm: dict[str, set[str]] = {}
    for arm in arms:
        attempted_by_arm[arm] = {
            str(c.get("corpus_task_id"))
            for c in data["candidates"]
            if str(c.get("arm")) == arm
        }
    pairwise: dict[str, Any] = {}
    for a in arms:
        for b in arms:
            if a == b:
                continue
            joint = attempted_by_arm[a] & attempted_by_arm[b]
            pairwise[f"P({b} fails|{a} fails)"] = conditional_failure(
                solved_by_arm[a], solved_by_arm[b], joint
            )
    rescues = {}
    for arm in arms:
        others = set().union(*(s for other, s in solved_by_arm.items() if other != arm)) if len(arms) > 1 else set()
        rescues[arm] = unique_rescues(solved_by_arm[arm], others)
    premium = None
    if "H1" in per_arm and "C1" in per_arm:
        premium = round(
            per_arm["H1"]["verified_solve_rate"] - per_arm["C1"]["verified_solve_rate"], 4
        )
    budget_table = {
        arm: {"tokens": m["tokens"], "cost_usd": m["cost_usd"], "cost_known": m["cost_known"],
              "calls": m["calls"], "median_latency_ms": m["median_latency_ms"]}
        for arm, m in per_arm.items()
    }
    costs = [m["cost_usd"] for m in per_arm.values() if m["cost_known"] and m["cost_usd"] > 0]
    budget_warning = None
    if costs and max(costs) / min(costs) > 2:
        budget_warning = (
            "monetary costs differ by >2x between arms; compare cost-normalized "
            "(cost per verified solve) rather than raw oracle@k"
        )
    tiny_n = len(corpus_task_ids) < TINY_N_THRESHOLD
    return {
        "experiment_id": experiment_id,
        "name": config.name,
        "hypothesis": config.hypothesis,
        "primary_metric": config.primary_metric,
        "corpus_version": config.corpus_version,
        "task_count": len(corpus_task_ids),
        "tiny_n_warning": (
            f"n={len(corpus_task_ids)} < {TINY_N_THRESHOLD}: do not infer causal superiority"
            if tiny_n else None
        ),
        "arms": per_arm,
        "pairwise_conditional_failure": pairwise,
        "unique_rescues": rescues,
        "heterogeneity_premium": premium,
        "matched_budget": budget_table,
        "budget_warning": budget_warning,
        "incomplete_arms": data["stops"],
        "human_interventions": 0,
        "notes": [
            "oracle@k is best-of-k with an oracle selector, not deployable performance.",
            "agreement != verification: only hidden-verifier outcomes count as evidence.",
        ],
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"Experiment: {report['name']} ({report['experiment_id']})",
        f"Hypothesis: {report['hypothesis']}",
        f"Primary metric: {report['primary_metric']}",
        f"Tasks: {report['task_count']} (corpus {report['corpus_version']})",
    ]
    if report["tiny_n_warning"]:
        lines.append(f"WARNING: {report['tiny_n_warning']}")
    for arm, m in report["arms"].items():
        oracle_key = f"oracle_at_{m['k']}"
        lines.append(
            f"[{arm}] tasks {m['tasks_attempted']}/{m['tasks_total']} calls={m['calls']} "
            f"{oracle_key}={m[oracle_key]} solves={m['successful_candidates']} "
            f"tokens={m['tokens']} cost={m['cost_usd']} "
            f"cost/solve={m['cost_per_verified_solve']} "
            f"latency(p50)={m['median_latency_ms']} failures={m['failures_by_category']}"
        )
    lines.append(f"Heterogeneity premium (H1 - C1): {report['heterogeneity_premium']}")
    lines.append(f"Matched budget: {json.dumps(report['matched_budget'], sort_keys=True)}")
    if report["budget_warning"]:
        lines.append(f"WARNING: {report['budget_warning']}")
    lines.append(f"Pairwise conditional failure: {json.dumps(report['pairwise_conditional_failure'], sort_keys=True)}")
    lines.append(f"Unique rescues: {json.dumps(report['unique_rescues'], sort_keys=True)}")
    if report["incomplete_arms"]:
        lines.append(f"Incomplete arms: {json.dumps(report['incomplete_arms'], sort_keys=True)}")
    lines.append(f"Human interventions: {report['human_interventions']}")
    lines.extend(report["notes"])
    if report["heterogeneity_premium"] is None:
        lines.append("Verdict: inconclusive (need both C1 and H1 arms)")
    elif report["heterogeneity_premium"] > 0:
        lines.append("Verdict: at approximately matched budget, H1 was better than C1.")
    elif report["heterogeneity_premium"] < 0:
        lines.append("Verdict: at approximately matched budget, H1 was worse than C1.")
    else:
        lines.append("Verdict: at approximately matched budget, H1 was equal to C1.")
    return "\n".join(lines) + "\n"


def export_experiment(runtime: Any, experiment_id: str) -> dict[str, Any]:
    """Stable raw-observation export. No secrets: ledger holds no credentials."""

    data = collect_experiment(runtime, experiment_id)
    config = data["config"]
    return {
        "format": "codeai-experiment-export-v1",
        "experiment": {
            "experiment_id": config.experiment_id,
            "name": config.name,
            "hypothesis": config.hypothesis,
            "primary_metric": config.primary_metric,
            "corpus_version": config.corpus_version,
            "task_ids": list(config.task_ids),
            "arms": [
                {"name": a.name, "models": list(a.models), "samples": a.samples,
                 "temperature": a.temperature, "seed": a.seed,
                 "prompt_version": a.prompt_version}
                for a in config.arms
            ],
            "budget": {
                "max_calls": config.budget.max_calls, "max_tokens": config.budget.max_tokens,
                "max_cost_usd": config.budget.max_cost_usd,
                "timeout_seconds": config.budget.timeout_seconds,
            },
            "stopping_rule": config.stopping_rule,
            "created_at": config.created_at,
            "config_hash": config.config_hash,
        },
        "task_mappings": data["mappings"],
        "calls": data["calls"],
        "candidates": data["candidates"],
        "checks": data["checks"],
        "arm_stops": data["stops"],
        "arm_completions": data["completions"],
        "report": build_report(runtime, experiment_id),
    }
