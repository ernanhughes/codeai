from __future__ import annotations

from pathlib import Path

import pytest

from codeai.adapters import FakeCognitionAdapter
from codeai.domain import ActorRef, Authority, Budget, Capability, Directive, Task
from codeai.ledger import SQLiteLedger
from codeai.quality import (
    _FORBIDDEN_INVESTIGATION_PHRASES,
    METRIC_EROSION,
    METRIC_LOC,
    METRIC_VERBOSITY,
    CodeQualityMeasurement,
    MetricTrend,
    QualityDomain,
    QualitySnapshot,
    collect_episode_friction,
    compare_scopes,
    compare_snapshots,
    generate_investigations,
    measure_erosion,
    measure_verbosity,
    quality_snapshots_for_task,
    record_investigations,
    record_quality_snapshot,
    snapshot_directory,
    trajectory_over,
)
from codeai.runtime import Runtime

CLEAN_MODULE = '''"""Clean control module."""


def add(a, b):
    return a + b


def greet(name):
    if name:
        return f"hello {name}"
    return "hello"
'''

COMPLEX_MODULE = '''"""Module with concentrated complexity."""


def route(kind, a, b, c, d):
    if kind == "a":
        if a:
            if b:
                if c:
                    if d:
                        if a > b:
                            if b > c:
                                if c > d:
                                    if d > 0:
                                        if a != c:
                                            if b != d:
                                                return "deep"
        return "shallow"
    elif kind == "b":
        return "b"
    elif kind == "c":
        return "c"
    elif kind == "d":
        return "d"
    elif kind == "e":
        return "e"
    elif kind == "f":
        return "f"
    return "other"
'''

BLOCK = (
    "result_alpha = compute_first(input_value)\n"
    "result_beta = compute_second(result_alpha)\n"
    "result_gamma = combine_parts(result_alpha, result_beta)\n"
    "final_total = aggregate_all(result_gamma)\n"
    "return render_output(final_total)"
)


def _write(path: Path, name: str, content: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(content, encoding="utf-8")


def _snapshot_with(values: dict[str, float], snapshot_id: str) -> QualitySnapshot:
    return QualitySnapshot(
        snapshot_id=snapshot_id,
        target="test",
        repo_head=None,
        created_at="2026-01-01T00:00:00+00:00",
        measurements=tuple(
            CodeQualityMeasurement(
                snapshot_id=snapshot_id,
                metric=metric,
                value=value,
                scope=None,
                source_refs=(),
                tool="codeai-quality",
                tool_version="0.1.0",
                provenance=("tool=codeai-quality@0.1.0",),
            )
            for metric, value in sorted(values.items())
        ),
    )


def _repo_value(snapshot: QualitySnapshot, metric: str) -> float:
    return next(m.value for m in snapshot.measurements if m.metric == metric and m.scope is None)


# ---------------- measurement tests ----------------


def test_loc_deterministic_and_ignores_comments_blanks(tmp_path):
    _write(tmp_path, "a.py", "# comment\n\n\ndef f():\n    return 1  # trailing\n")
    first = snapshot_directory(tmp_path)
    second = snapshot_directory(tmp_path)
    assert _repo_value(first, METRIC_LOC) == 2.0
    assert first.snapshot_id == second.snapshot_id
    assert _repo_value(first, METRIC_LOC) == _repo_value(second, METRIC_LOC)


def test_erosion_formula_threshold_extremes(tmp_path):
    _write(tmp_path, "m.py", CLEAN_MODULE + "\n" + COMPLEX_MODULE)
    hot = measure_erosion(tmp_path, snapshot_id="s", threshold=0)
    cold = measure_erosion(tmp_path, snapshot_id="s", threshold=10_000)
    assert next(m for m in hot if m.scope is None).value == pytest.approx(1.0)
    assert next(m for m in cold if m.scope is None).value == pytest.approx(0.0)


def test_erosion_matches_mass_formula(tmp_path):
    _write(tmp_path, "m.py", COMPLEX_MODULE)
    (repo,) = [m for m in measure_erosion(tmp_path, snapshot_id="s") if m.scope is None]
    assert repo.value == pytest.approx(1.0)  # single hot function dominates
    assert repo.tool == "codeai-quality" and repo.tool_version
    assert any("threshold=10" in p for p in repo.provenance)


def test_erosion_scales_with_added_hot_function(tmp_path):
    _write(tmp_path, "m.py", CLEAN_MODULE)
    before = _repo_value(snapshot_directory(tmp_path), METRIC_EROSION)
    _write(tmp_path, "hot.py", COMPLEX_MODULE)
    after = _repo_value(snapshot_directory(tmp_path), METRIC_EROSION)
    assert after > before


def test_empty_repository_reports_empty_not_clean(tmp_path):
    snapshot = snapshot_directory(tmp_path)
    erosion = next(m for m in snapshot.measurements if m.metric == METRIC_EROSION and m.scope is None)
    verbosity = next(m for m in snapshot.measurements if m.metric == METRIC_VERBOSITY)
    assert erosion.value == 0.0
    assert verbosity.value == 0.0
    assert any("no-functions-found" in p for p in erosion.provenance)
    assert any("no-significant-lines" in p for p in verbosity.provenance)


def test_parser_failure_is_reported_not_invented(tmp_path):
    _write(tmp_path, "good.py", CLEAN_MODULE)
    _write(tmp_path, "broken.py", "def broken(:\n  this is not python\n")
    snapshot = snapshot_directory(tmp_path)
    assert snapshot.coverage["python_files_failed"] == ["broken.py"]
    # LOC is line-based, not parse-based: the broken file's 2 raw lines
    # still count, while its complexity is never invented.
    assert _repo_value(snapshot, METRIC_LOC) == 9.0


def test_verbosity_duplicated_vs_clean(tmp_path):
    clean = tmp_path / "clean"
    dup = tmp_path / "dup"
    _write(clean, "m.py", CLEAN_MODULE)
    body = "def first(x):\n" + "\n".join(f"    {line}" for line in BLOCK.splitlines()) + "\n"
    body += "def second(x):\n" + "\n".join(f"    {line}" for line in BLOCK.splitlines()) + "\n"
    _write(dup, "m.py", body)
    clean_value = measure_verbosity(clean, snapshot_id="s").value
    dup_value = measure_verbosity(dup, snapshot_id="s").value
    assert clean_value == pytest.approx(0.0)
    assert dup_value > 0.3
    dup_measurement = measure_verbosity(dup, snapshot_id="s")
    assert dup_measurement.source_refs  # duplicated lines are located


def test_provenance_stable_across_runs(tmp_path):
    _write(tmp_path, "m.py", CLEAN_MODULE)
    first = snapshot_directory(tmp_path)
    second = snapshot_directory(tmp_path)
    assert [(m.metric, m.scope, m.value) for m in first.measurements] == [
        (m.metric, m.scope, m.value) for m in second.measurements
    ]
    for m in first.measurements:
        assert m.tool == "codeai-quality"


# ---------------- trajectory tests ----------------


def test_trajectory_degrading():
    base = _snapshot_with({METRIC_EROSION: 0.27, METRIC_VERBOSITY: 0.14, METRIC_LOC: 100.0}, "a")
    current = _snapshot_with({METRIC_EROSION: 0.43, METRIC_VERBOSITY: 0.19, METRIC_LOC: 120.0}, "b")
    trajectory = compare_snapshots(base, current)
    by_metric = {d.metric: d for d in trajectory.deltas}
    assert by_metric[METRIC_EROSION].trend == MetricTrend.DEGRADING
    assert by_metric[METRIC_EROSION].delta == pytest.approx(0.16)
    assert by_metric[METRIC_VERBOSITY].trend == MetricTrend.DEGRADING
    # LOC growth is contextual only: never DEGRADING.
    assert by_metric[METRIC_LOC].trend == MetricTrend.STABLE


def test_trajectory_improving_stable_step_and_history():
    improving = compare_snapshots(
        _snapshot_with({METRIC_EROSION: 0.43}, "a"), _snapshot_with({METRIC_EROSION: 0.27}, "b")
    )
    assert improving.deltas[0].trend == MetricTrend.IMPROVING

    stable = compare_snapshots(
        _snapshot_with({METRIC_EROSION: 0.30}, "a"), _snapshot_with({METRIC_EROSION: 0.31}, "b")
    )
    assert stable.deltas[0].trend == MetricTrend.STABLE

    step = compare_snapshots(
        _snapshot_with({METRIC_EROSION: 0.10}, "a"), _snapshot_with({METRIC_EROSION: 0.60}, "b")
    )
    assert step.deltas[0].trend == MetricTrend.STEP_CHANGE

    identical = compare_snapshots(
        _snapshot_with({METRIC_EROSION: 0.30}, "a"), _snapshot_with({METRIC_EROSION: 0.30}, "b")
    )
    assert identical.deltas[0].trend == MetricTrend.STABLE

    no_history = compare_snapshots(_snapshot_with({}, "a"), _snapshot_with({METRIC_EROSION: 0.3}, "b"))
    assert no_history.deltas[0].trend == MetricTrend.INSUFFICIENT_HISTORY


def test_loc_step_change_not_degrading():
    trajectory = compare_snapshots(
        _snapshot_with({METRIC_LOC: 100.0}, "a"), _snapshot_with({METRIC_LOC: 200.0}, "b")
    )
    assert trajectory.deltas[0].trend == MetricTrend.STEP_CHANGE


def test_per_scope_comparison_and_ordering(tmp_path):
    base_dir = tmp_path / "base"
    cur_dir = tmp_path / "cur"
    _write(base_dir, "m.py", CLEAN_MODULE)
    _write(cur_dir, "m.py", CLEAN_MODULE + "\n" + COMPLEX_MODULE)
    base = snapshot_directory(base_dir)
    current = snapshot_directory(cur_dir)
    scoped = compare_scopes(base, current)
    assert scoped == tuple(sorted(scoped, key=lambda d: (d.metric, d.scope or "")))
    file_erosion = [d for d in scoped if d.metric == METRIC_EROSION and d.scope == "m.py"]
    assert file_erosion and file_erosion[0].trend in (MetricTrend.DEGRADING, MetricTrend.STEP_CHANGE)

    trajectory = compare_snapshots(base, current)
    assert [d.metric for d in trajectory.deltas] == sorted(d.metric for d in trajectory.deltas)

    history = [base, current]
    assert len(trajectory_over(history)) == 1


# ---------------- generator tests ----------------


def test_degrading_generates_question_not_verdict(tmp_path):
    _write(tmp_path, "m.py", CLEAN_MODULE)
    base = snapshot_directory(tmp_path)
    _write(tmp_path, "hot.py", COMPLEX_MODULE)
    current = snapshot_directory(tmp_path)
    trajectory = compare_snapshots(base, current)
    investigations = generate_investigations(trajectory, current=current)
    assert investigations, "expected a degrading trajectory to generate a candidate"
    for inv in investigations:
        assert inv.question.rstrip().endswith("?")
        lowered = inv.question.lower()
        for phrase in _FORBIDDEN_INVESTIGATION_PHRASES:
            assert phrase not in lowered
        assert inv.source_refs
        assert inv.signal_refs
        assert inv.evidence_refs
        assert inv.uncertainty


def test_no_degradation_control_generates_nothing(tmp_path):
    _write(tmp_path, "m.py", CLEAN_MODULE)
    base = snapshot_directory(tmp_path)
    current = snapshot_directory(tmp_path)
    trajectory = compare_snapshots(base, current)
    assert generate_investigations(trajectory, current=current) == ()


def test_erosion_maps_to_complexity_domain_and_verbosity_to_duplication():
    base = _snapshot_with({METRIC_EROSION: 0.2, METRIC_VERBOSITY: 0.1}, "a")
    current = _snapshot_with({METRIC_EROSION: 0.5, METRIC_VERBOSITY: 0.4}, "b")
    current = QualitySnapshot(
        snapshot_id=current.snapshot_id,
        target=current.target,
        repo_head=None,
        created_at=current.created_at,
        measurements=current.measurements,
        coverage={},
    )
    trajectory = compare_snapshots(base, current)
    investigations = generate_investigations(trajectory, current=current)
    domains = {inv.domain for inv in investigations}
    assert QualityDomain.COMPLEXITY in domains
    assert QualityDomain.DUPLICATION in domains
    assert QualityDomain.CONTEXT not in domains  # LOC never generates; no SLOP domain exists
    assert all("SLOP" not in d.value for d in domains)


def test_review_loop_retrieval_accepts_investigations():
    runtime = Runtime(SQLiteLedger(":memory:"))
    directive = Directive(
        directive_id="run-1",
        objective="investigate",
        success_criteria=("q",),
        budget=Budget(),
        authority=Authority(frozenset({Capability.READ})),
    )
    task = Task(
        task_id="task-1",
        directive_id="run-1",
        objective="investigate",
        success_criteria=("q",),
        budget=Budget(),
        authority=Authority(frozenset({Capability.READ})),
    )
    runtime.open_directive(directive)
    runtime.create_task(task)
    base = _snapshot_with({METRIC_EROSION: 0.2}, "a")
    current = _snapshot_with({METRIC_EROSION: 0.5}, "b")
    investigations = generate_investigations(compare_snapshots(base, current), current=current)
    record_investigations(runtime, investigations, task_id="task-1", run_id="run-1")
    claims = runtime.claims_for_run("run-1")
    assert len(claims) == len(investigations)
    report = runtime.disagreement_for_run("run-1")
    assert len(report["unresolved"]) == len(investigations)


# ---------------- integration tests ----------------


def test_full_path_repo_to_review_loop(tmp_path):
    base_dir = tmp_path / "base"
    cur_dir = tmp_path / "cur"
    _write(base_dir, "m.py", CLEAN_MODULE)
    _write(cur_dir, "m.py", CLEAN_MODULE)
    _write(cur_dir, "hot.py", COMPLEX_MODULE)

    runtime = Runtime(SQLiteLedger(":memory:"))
    directive = Directive(
        directive_id="run-9",
        objective="quality shadow run",
        success_criteria=("observe",),
        budget=Budget(),
        authority=Authority(frozenset({Capability.READ})),
    )
    task = Task(
        task_id="task-9",
        directive_id="run-9",
        objective="quality shadow run",
        success_criteria=("observe",),
        budget=Budget(),
        authority=Authority(frozenset({Capability.READ})),
    )
    runtime.open_directive(directive)
    runtime.create_task(task)

    base = snapshot_directory(base_dir, target="base")
    current = snapshot_directory(cur_dir, target="current")
    record_quality_snapshot(runtime, base, task_id="task-9", run_id="run-9")
    record_quality_snapshot(runtime, current, task_id="task-9", run_id="run-9")

    rehydrated = quality_snapshots_for_task(runtime, "task-9")
    assert len(rehydrated) == 2
    trajectory = compare_snapshots(rehydrated[0], rehydrated[1])
    investigations = generate_investigations(trajectory, current=rehydrated[1])
    assert investigations
    record_investigations(runtime, investigations, task_id="task-9", run_id="run-9")
    report = runtime.disagreement_for_run("run-9")
    assert report["unresolved"]
    assert any("complexity mass" in c["statement"] for c in report["unresolved"])


def test_friction_projection_uses_existing_telemetry():
    runtime = Runtime(SQLiteLedger(":memory:"))
    directive = Directive(
        directive_id="run-f",
        objective="friction",
        success_criteria=("observe",),
        budget=Budget(),
        authority=Authority(frozenset({Capability.READ, Capability.EXECUTE})),
    )
    task = Task(
        task_id="task-f",
        directive_id="run-f",
        objective="friction",
        success_criteria=("observe",),
        budget=Budget(),
        authority=Authority(frozenset({Capability.READ, Capability.EXECUTE})),
    )
    runtime.open_directive(directive)
    runtime.create_task(task)
    actor = ActorRef(actor_id="m", kind="model", provider="fake-provider", model="fake-model")
    runtime.compile_and_record_context(task_id="task-f", actor=actor, prompt="do work")
    from codeai.domain import CallSpec

    package, _ = runtime.compile_and_record_context(task_id="task-f", actor=actor, prompt="again")
    adapter = FakeCognitionAdapter(responses=["out"], input_tokens=7, output_tokens=11)
    spec = CallSpec(
        call_id="call-1",
        task_id="task-f",
        actor=actor,
        context=package,
        idempotency_key="k1",
        directive_id="run-f",
        run_id="run-f",
    )
    runtime.invoke_call(spec, adapter=adapter)
    friction = collect_episode_friction(runtime, "task-f")
    assert friction.model_calls == 1
    assert friction.model_input_tokens == 7
    assert friction.model_output_tokens == 11
    # Empty compilations genuinely had zero candidates: real zero, not missing.
    assert friction.context_candidates == 0
    assert friction.context_selected == 0
    # Facets without telemetry stay None: never zero-filled.
    assert friction.files_inspected is None
    assert friction.outcome is None


def test_friction_empty_episode():
    runtime = Runtime(SQLiteLedger(":memory:"))
    friction = collect_episode_friction(runtime, "no-such-episode")
    assert friction.model_calls is None


def test_no_monolithic_score_or_gate_exists():
    from codeai import quality

    assert not hasattr(quality, "sloppiness_score")
    assert not hasattr(quality, "SlopCodeBench")
    assert not hasattr(quality, "quality_gate")
    assert "SLOP" not in {d.value for d in QualityDomain}


def test_cli_measure_and_compare(tmp_path, capsys):
    from codeai.cli import main

    base_dir = tmp_path / "base"
    cur_dir = tmp_path / "cur"
    _write(base_dir, "m.py", CLEAN_MODULE)
    _write(cur_dir, "m.py", CLEAN_MODULE + "\n" + COMPLEX_MODULE)
    assert main(["quality", "measure", str(base_dir)]) == 0
    assert "erosion" in capsys.readouterr().out
    assert main(["quality", "compare", str(base_dir), str(cur_dir)]) == 0
    assert "TRAJECTORY" in capsys.readouterr().out
    assert main(["quality", "measure", str(base_dir), "--json"]) == 0
    assert "snapshot_id" in capsys.readouterr().out
