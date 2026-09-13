"""Deterministic structural code-quality measurements and trajectories.

Shadow-mode, observational capability. This module is a thin consumer of
deterministic repository facts (stdlib ``ast`` parsing, line analysis, ledger
projections). It performs no reasoning, makes no model calls, issues no
verdicts, and enforces no gates.

Epistemic contract (codeai-aligned: claims are not evidence, verification
beats consensus):

- A metric is evidence about *where to investigate*, never a verdict about
  what engineering action to take. There is deliberately no monolithic
  score, no ``SLOP`` domain, and no pass/fail threshold.
- Functional correctness and structural maintainability are different
  dimensions. A passing test suite does not imply that the cost of the
  next modification has remained constant.
- Missing evidence is reported as missing (``None`` / explicit coverage
  provenance), never silently converted to zero.
- Every measurement carries provenance sufficient to reproduce it.

Metrics (all deterministic, stdlib only):

- ``loc``: source lines of code (non-blank, non-comment ``.py`` lines).
  Contextual evidence only; growth alone is not a violation.
- ``erosion``: complexity-mass concentration,
  ``sum(mass(f) for CC(f) > threshold) / sum(mass(f))`` with
  ``mass(f) = CC(f) * sqrt(SLOC(f))``. Default threshold 10.
- ``verbosity``: codeai-specific condensability proxy,
  ``duplicated significant lines / significant lines``, where a line is
  significant when it is non-blank, non-comment, and non-import, and a
  line counts as duplicated when it participates in a repeated window of
  ``VERBOSITY_WINDOW`` consecutive normalized significant lines.

Trends are change-based (``IMPROVING`` / ``STABLE`` / ``DEGRADING`` /
``STEP_CHANGE`` / ``INSUFFICIENT_HISTORY``). Absolute repositories are
never classified as good or bad; ``loc`` never degrades or improves, it
is ``STABLE`` or ``STEP_CHANGE`` (contextual only).

Graph/ledger representation: measurements persist as a single
``quality.snapshot`` ledger event per snapshot (evidence record, not one
node per metric), and trajectory-derived investigation candidates persist
as ordinary ``claim.recorded`` events (``E0_ASSERTED`` questions). There
is no graph database in this runtime; the append-only ledger plus claim
projections are the discoverability path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

TOOL_NAME = "codeai-quality"
TOOL_VERSION = "0.1.0"

METRIC_LOC = "loc"
METRIC_EROSION = "erosion"
METRIC_VERBOSITY = "verbosity"

DEFAULT_COMPLEXITY_THRESHOLD = 10
VERBOSITY_WINDOW = 5

# |delta| at or below the stable band counts as STABLE.
_STABLE_BAND = {METRIC_EROSION: 0.03, METRIC_VERBOSITY: 0.03}
# |delta| at or above the step threshold counts as STEP_CHANGE (abrupt shift).
# Calibrated so the canonical 0.27 -> 0.43 erosion drift reads DEGRADING,
# reserving STEP_CHANGE for genuine regime shifts.
_STEP_THRESHOLD = {METRIC_EROSION: 0.30, METRIC_VERBOSITY: 0.30}
# LOC is contextual only: relative change bands, never IMPROVING/DEGRADING.
_LOC_STABLE_REL = 0.10
_LOC_STEP_REL = 0.50

# Words that would turn a question into a verdict or a pre-decided action.
_FORBIDDEN_INVESTIGATION_PHRASES = (
    "must refactor",
    "should refactor",
    "should split",
    "must split",
    "bad complexity",
    "is verbose",
    "is bad",
    "needs refactoring",
    "requires refactoring",
)


class QualityDomain(StrEnum):
    """Existing-style quality domains. No SLOP domain by design."""

    COMPLEXITY = "complexity"
    DUPLICATION = "duplication"
    MAINTAINABILITY = "maintainability"
    ARCHITECTURE = "architecture"
    RESIDUE = "residue"
    CONTEXT = "context"  # LOC lives here: contextual evidence only.


class MetricTrend(StrEnum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    STEP_CHANGE = "STEP_CHANGE"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


@dataclass(frozen=True, slots=True)
class CodeQualityMeasurement:
    """One deterministic measurement bound to a snapshot with provenance."""

    snapshot_id: str
    metric: str
    value: float
    scope: str | None
    source_refs: tuple[str, ...]
    tool: str
    tool_version: str
    provenance: tuple[str, ...]
    previous_value: float | None = None


@dataclass(frozen=True, slots=True)
class QualitySnapshot:
    snapshot_id: str
    target: str
    repo_head: str | None
    created_at: str
    measurements: tuple[CodeQualityMeasurement, ...] = ()
    coverage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricDelta:
    metric: str
    scope: str | None
    previous: float | None
    current: float | None
    delta: float | None
    trend: MetricTrend
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QualityTrajectory:
    base_snapshot_id: str
    current_snapshot_id: str
    target: str
    deltas: tuple[MetricDelta, ...] = ()


@dataclass(frozen=True, slots=True)
class QualityInvestigation:
    """A question, not a verdict. Deterministic, stable-ordered, sourcereffed."""

    question: str
    domain: QualityDomain
    scope: str | None
    source_refs: tuple[str, ...]
    signal_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    uncertainty: str
    related_domains: tuple[QualityDomain, ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeFriction:
    """Analytical view of Agent Modification Friction for one engineering episode.

    Measurement only. ``None`` means the underlying telemetry does not record
    that facet; missing evidence is never converted to zero.
    """

    episode_id: str
    context_candidates: int | None = None
    context_selected: int | None = None
    context_tokens: int | None = None
    model_input_tokens: int | None = None
    model_output_tokens: int | None = None
    model_calls: int | None = None
    tool_calls: int | None = None
    failed_calls: int | None = None
    failed_checks: int | None = None
    review_rounds: int | None = None
    total_cost_usd: float | None = None
    median_latency_ms: float | None = None
    files_inspected: int | None = None
    outcome: str | None = None


# ---------------------------------------------------------------------------
# Source scanning helpers (deterministic, stdlib only)
# ---------------------------------------------------------------------------


def _iter_python_files(root: Path) -> list[Path]:
    files = [
        p
        for p in root.rglob("*.py")
        if ".codeai" not in p.parts and ".git" not in p.parts
    ]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _significant_line(stripped: str, *, include_imports: bool) -> bool:
    if not stripped or stripped.startswith("#"):
        return False
    return include_imports or not stripped.startswith(("import ", "from "))


def _sloc_of_lines(lines: list[str], *, include_imports: bool = True) -> int:
    return sum(1 for line in lines if _significant_line(line.strip(), include_imports=include_imports))


def _read_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def _normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


# ---------------------------------------------------------------------------
# Cyclomatic complexity (deterministic AST visitor, documented subset)
# ---------------------------------------------------------------------------


class _ComplexityVisitor(ast.NodeVisitor):
    """Count decision points. Documented subset (see module docstring limits)."""

    def __init__(self) -> None:
        self.points = 0

    def visit_If(self, node: ast.If) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.points += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.points += max(0, len(node.cases))
        self.generic_visit(node)


def _function_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    visitor = _ComplexityVisitor()
    # Base complexity 1; nested function/class bodies belong to themselves,
    # so do not descend into them when scoring the outer function.
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        visitor.visit(child)
    for decorator in node.decorator_list:
        visitor.visit(decorator)
    return 1 + visitor.points


@dataclass(frozen=True, slots=True)
class _FunctionRecord:
    path: str
    name: str
    lineno: int
    complexity: int
    sloc: int


def _collect_functions(root: Path) -> tuple[list[_FunctionRecord], list[str], list[str]]:
    """Return (functions, parsed_files, failed_files). Never invents values."""
    functions: list[_FunctionRecord] = []
    parsed: list[str] = []
    failed: list[str] = []
    for path in _iter_python_files(root):
        rel = path.relative_to(root).as_posix()
        lines = _read_lines(path)
        if lines is None:
            failed.append(rel)
            continue
        try:
            tree = ast.parse("\n".join(lines), filename=rel)
        except SyntaxError:
            failed.append(rel)
            continue
        parsed.append(rel)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            end = getattr(node, "end_lineno", None) or node.lineno
            span = lines[node.lineno - 1 : end]
            sloc = _sloc_of_lines(span)
            functions.append(
                _FunctionRecord(
                    path=rel,
                    name=node.name,
                    lineno=node.lineno,
                    complexity=_function_complexity(node),
                    sloc=sloc,
                )
            )
    return functions, parsed, failed


def _snapshot_id_for(target: str, repo_head: str | None, digest_payload: str) -> str:
    canonical = "\0".join([target, repo_head or "", digest_payload])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def measure_loc(root: Path, *, snapshot_id: str) -> tuple[CodeQualityMeasurement, ...]:
    """Deterministic SLOC per scope. Contextual evidence only."""
    per_file: Counter[str] = Counter()
    parsed = 0
    failed: list[str] = []
    for path in _iter_python_files(root):
        rel = path.relative_to(root).as_posix()
        lines = _read_lines(path)
        if lines is None:
            failed.append(rel)
            continue
        parsed += 1
        per_file[rel] = _sloc_of_lines(lines)
    total = sum(per_file.values())
    provenance = (
        f"tool={TOOL_NAME}@{TOOL_VERSION}",
        "definition=non-blank-non-comment-.py-lines",
        f"python_files={parsed}",
        f"unreadable_files={len(failed)}",
    )
    if failed:
        provenance += (f"unreadable={','.join(sorted(failed))}",)
    out = [
        CodeQualityMeasurement(
            snapshot_id=snapshot_id,
            metric=METRIC_LOC,
            value=float(total),
            scope=None,
            source_refs=tuple(sorted(per_file)),
            tool=TOOL_NAME,
            tool_version=TOOL_VERSION,
            provenance=provenance,
        )
    ]
    for rel in sorted(per_file):
        out.append(
            CodeQualityMeasurement(
                snapshot_id=snapshot_id,
                metric=METRIC_LOC,
                value=float(per_file[rel]),
                scope=rel,
                source_refs=(rel,),
                tool=TOOL_NAME,
                tool_version=TOOL_VERSION,
                provenance=provenance,
            )
        )
    return tuple(out)


def measure_erosion(
    root: Path, *, snapshot_id: str, threshold: int = DEFAULT_COMPLEXITY_THRESHOLD
) -> tuple[CodeQualityMeasurement, ...]:
    """Complexity-mass concentration over the repo (and per file)."""
    functions, parsed, failed = _collect_functions(root)
    masses = [(f, f.complexity * math.sqrt(f.sloc) if f.sloc > 0 else 0.0) for f in functions]
    total_mass = sum(m for _, m in masses)
    hot_mass = sum(m for f, m in masses if f.complexity > threshold)
    value = (hot_mass / total_mass) if total_mass > 0 else 0.0
    hot = sorted({f.path for f, m in masses if f.complexity > threshold})
    provenance = (
        f"tool={TOOL_NAME}@{TOOL_VERSION}",
        "formula=sum(CC(f)*sqrt(SLOC(f)) for CC(f)>T)/sum(CC(f)*sqrt(SLOC(f)))",
        f"threshold={threshold}",
        f"functions={len(functions)}",
        f"parsed_files={len(parsed)}",
        f"failed_files={len(failed)}",
    )
    if failed:
        provenance += (f"failed={','.join(sorted(failed))}",)
    if not functions:
        provenance += ("note=no-functions-found;value-0.0-means-empty-not-clean",)
    out = [
        CodeQualityMeasurement(
            snapshot_id=snapshot_id,
            metric=METRIC_EROSION,
            value=value,
            scope=None,
            source_refs=tuple(hot),
            tool=TOOL_NAME,
            tool_version=TOOL_VERSION,
            provenance=provenance,
        )
    ]
    by_file: dict[str, list[_FunctionRecord]] = {}
    for f in functions:
        by_file.setdefault(f.path, []).append(f)
    for rel in sorted(by_file):
        members = by_file[rel]
        file_total = sum(f.complexity * math.sqrt(f.sloc) if f.sloc > 0 else 0.0 for f in members)
        file_hot = sum(
            f.complexity * math.sqrt(f.sloc)
            for f in members
            if f.complexity > threshold and f.sloc > 0
        )
        out.append(
            CodeQualityMeasurement(
                snapshot_id=snapshot_id,
                metric=METRIC_EROSION,
                value=(file_hot / file_total) if file_total > 0 else 0.0,
                scope=rel,
                source_refs=(rel,),
                tool=TOOL_NAME,
                tool_version=TOOL_VERSION,
                provenance=provenance
                + (f"file_functions={len(members)}",),
            )
        )
    return tuple(out)


def measure_verbosity(root: Path, *, snapshot_id: str) -> CodeQualityMeasurement:
    """Condensability proxy: share of significant lines in repeated windows."""
    significant: list[tuple[str, str]] = []  # (normalized, source_ref)
    for path in _iter_python_files(root):
        rel = path.relative_to(root).as_posix()
        lines = _read_lines(path)
        if lines is None:
            continue
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not _significant_line(stripped, include_imports=False):
                continue
            significant.append((_normalize_line(stripped), f"{rel}:{lineno}"))
    total = len(significant)
    window_counts: Counter[tuple[str, ...]] = Counter()
    windows: list[tuple[tuple[str, ...], list[int]]] = []
    for i in range(max(0, total - VERBOSITY_WINDOW + 1)):
        key = tuple(norm for norm, _ in significant[i : i + VERBOSITY_WINDOW])
        window_counts[key] += 1
        windows.append((key, list(range(i, i + VERBOSITY_WINDOW))))
    duplicated = [False] * total
    for key, idxs in windows:
        if window_counts[key] > 1:
            for i in idxs:
                duplicated[i] = True
    dup_lines = sum(1 for d in duplicated if d)
    value = (dup_lines / total) if total > 0 else 0.0
    dup_refs = sorted({ref for flag, (_, ref) in zip(duplicated, significant) if flag})
    provenance = (
        f"tool={TOOL_NAME}@{TOOL_VERSION}",
        "formula=significant-lines-in-repeated-windows/significant-lines",
        "significant=non-blank-non-comment-non-import-normalized",
        f"window={VERBOSITY_WINDOW}",
        f"significant_lines={total}",
        f"duplicated_lines={dup_lines}",
    )
    if total == 0:
        provenance += ("note=no-significant-lines;value-0.0-means-empty-not-clean",)
    return CodeQualityMeasurement(
        snapshot_id=snapshot_id,
        metric=METRIC_VERBOSITY,
        value=value,
        scope=None,
        source_refs=tuple(dup_refs[:50]),
        tool=TOOL_NAME,
        tool_version=TOOL_VERSION,
        provenance=provenance,
    )


def snapshot_directory(
    root: Path,
    *,
    target: str | None = None,
    threshold: int = DEFAULT_COMPLEXITY_THRESHOLD,
) -> QualitySnapshot:
    """Measure a working-tree directory deterministically."""
    root = root.resolve()
    label = target or root.as_posix()
    repo_head = _git_head(root)
    # Pre-snapshot id: measure with a fixed placeholder, then bind the real id.
    loc = measure_loc(root, snapshot_id="pending")
    erosion = measure_erosion(root, snapshot_id="pending", threshold=threshold)
    verbosity = measure_verbosity(root, snapshot_id="pending")
    digest = hashlib.sha256(
        json.dumps(
            [
                {"metric": m.metric, "scope": m.scope, "value": m.value}
                for m in (*loc, *erosion, verbosity)
            ],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    snapshot_id = _snapshot_id_for(label, repo_head, digest)
    rebound = [
        CodeQualityMeasurement(
            snapshot_id=snapshot_id,
            metric=m.metric,
            value=m.value,
            scope=m.scope,
            source_refs=m.source_refs,
            tool=m.tool,
            tool_version=m.tool_version,
            provenance=m.provenance,
            previous_value=m.previous_value,
        )
        for m in (*loc, *erosion, verbosity)
    ]
    rebound.sort(key=lambda m: (m.metric, m.scope or ""))
    functions, parsed, failed = _collect_functions(root)
    coverage: dict[str, Any] = {
        "tool": f"{TOOL_NAME}@{TOOL_VERSION}",
        "python_files_seen": len(_iter_python_files(root)),
        "python_files_parsed": len(parsed),
        "python_files_failed": sorted(failed),
        "functions_found": len(functions),
        "complexity_threshold": threshold,
    }
    return QualitySnapshot(
        snapshot_id=snapshot_id,
        target=label,
        repo_head=repo_head,
        created_at=datetime.now(UTC).isoformat(),
        measurements=tuple(rebound),
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Trajectories (change-based; never absolute good/bad)
# ---------------------------------------------------------------------------


def _classify(metric: str, previous: float | None, current: float) -> MetricTrend:
    if previous is None:
        return MetricTrend.INSUFFICIENT_HISTORY
    delta = current - previous
    if metric == METRIC_LOC:
        rel = abs(delta) / max(1.0, abs(previous))
        if rel >= _LOC_STEP_REL:
            return MetricTrend.STEP_CHANGE
        return MetricTrend.STABLE
    band = _STABLE_BAND[metric]
    step = _STEP_THRESHOLD[metric]
    if abs(delta) >= step:
        return MetricTrend.STEP_CHANGE
    if abs(delta) <= band:
        return MetricTrend.STABLE
    if delta > 0:
        return MetricTrend.DEGRADING
    return MetricTrend.IMPROVING


def compare_snapshots(base: QualitySnapshot, current: QualitySnapshot) -> QualityTrajectory:
    """Compare repo-scope measurements across two snapshots. Deterministic order."""
    base_by_metric = {m.metric: m for m in base.measurements if m.scope is None}
    current_by_metric = {m.metric: m for m in current.measurements if m.scope is None}
    deltas: list[MetricDelta] = []
    for metric in sorted(set(base_by_metric) | set(current_by_metric)):
        b = base_by_metric.get(metric)
        c = current_by_metric.get(metric)
        if c is None:
            continue
        previous = b.value if b is not None else None
        delta = (c.value - previous) if previous is not None else None
        trend = _classify(metric, previous, c.value) if metric in _STABLE_BAND or metric == METRIC_LOC else (
            MetricTrend.STABLE if delta == 0 else MetricTrend.INSUFFICIENT_HISTORY
        )
        provenance = c.provenance
        if b is not None:
            provenance += (f"base_snapshot={base.snapshot_id[:12]}",)
        deltas.append(
            MetricDelta(
                metric=metric,
                scope=None,
                previous=previous,
                current=c.value,
                delta=delta,
                trend=trend,
                provenance=provenance,
            )
        )
    return QualityTrajectory(
        base_snapshot_id=base.snapshot_id,
        current_snapshot_id=current.snapshot_id,
        target=current.target,
        deltas=tuple(deltas),
    )


def compare_scopes(base: QualitySnapshot, current: QualitySnapshot) -> tuple[MetricDelta, ...]:
    """Per-scope (file/module) comparison. Deterministic order."""
    base_map = {(m.metric, m.scope): m for m in base.measurements if m.scope is not None}
    current_map = {(m.metric, m.scope): m for m in current.measurements if m.scope is not None}
    out: list[MetricDelta] = []
    for key in sorted(set(base_map) | set(current_map)):
        metric, scope = key
        b = base_map.get(key)
        c = current_map.get(key)
        if c is None:
            continue
        previous = b.value if b is not None else None
        delta = (c.value - previous) if previous is not None else None
        trend = _classify(metric, previous, c.value) if metric in _STABLE_BAND or metric == METRIC_LOC else (
            MetricTrend.STABLE if delta == 0 else MetricTrend.INSUFFICIENT_HISTORY
        )
        out.append(
            MetricDelta(
                metric=metric,
                scope=scope,
                previous=previous,
                current=c.value,
                delta=delta,
                trend=trend,
                provenance=c.provenance + (f"base_snapshot={base.snapshot_id[:12]}",),
            )
        )
    return tuple(out)


def trajectory_over(history: list[QualitySnapshot]) -> list[QualityTrajectory]:
    """Pairwise consecutive comparisons over an ordered snapshot history."""
    return [compare_snapshots(history[i], history[i + 1]) for i in range(len(history) - 1)]


# ---------------------------------------------------------------------------
# Investigation generators (questions, not verdicts)
# ---------------------------------------------------------------------------


def _check_question_shape(question: str) -> None:
    lowered = question.lower()
    assert question.rstrip().endswith("?"), "investigation must be phrased as a question"
    for phrase in _FORBIDDEN_INVESTIGATION_PHRASES:
        assert phrase not in lowered, f"investigation must not assert or prescribe ({phrase})"


def generate_investigations(
    trajectory: QualityTrajectory,
    *,
    current: QualitySnapshot,
    extra_evidence_refs: tuple[str, ...] = (),
) -> tuple[QualityInvestigation, ...]:
    """Turn significant trajectory changes into investigation candidates.

    Deterministic, stable-ordered, no model calls. Only DEGRADING and
    STEP_CHANGE deltas on erosion/verbosity generate candidates; LOC is
    contextual only and never generates one.
    """
    by_metric_scope = {(m.metric, m.scope): m for m in current.measurements}
    out: list[QualityInvestigation] = []
    for delta in trajectory.deltas:
        if delta.trend not in (MetricTrend.DEGRADING, MetricTrend.STEP_CHANGE):
            continue
        key = (delta.metric, delta.scope)
        measurement = by_metric_scope.get(key)
        source_refs = measurement.source_refs if measurement else ()
        if delta.metric == METRIC_EROSION:
            prev = f"{delta.previous:.2f}" if delta.previous is not None else "unknown"
            question = (
                f"Why did complexity mass in {current.target} increase significantly "
                f"in this change ({prev} -> {delta.current:.2f}), and does the added "
                f"responsibility belong in its current component?"
            )
            out.append(
                QualityInvestigation(
                    question=question,
                    domain=QualityDomain.COMPLEXITY,
                    scope=delta.scope,
                    source_refs=source_refs,
                    signal_refs=(f"erosion@{current.snapshot_id[:12]}",),
                    evidence_refs=(f"snapshot:{current.snapshot_id[:12]}", *extra_evidence_refs),
                    uncertainty=(
                        "Complexity concentration can reflect legitimate domain "
                        "complexity; metric change alone does not establish that a "
                        "split or move is warranted."
                    ),
                    related_domains=(QualityDomain.MAINTAINABILITY, QualityDomain.ARCHITECTURE),
                )
            )
        elif delta.metric == METRIC_VERBOSITY:
            prev = f"{delta.previous:.2f}" if delta.previous is not None else "unknown"
            question = (
                f"Why did duplicate/condensable code in {current.target} increase "
                f"during this change ({prev} -> {delta.current:.2f}), and does the "
                f"new code represent distinct behaviour or a parallel implementation "
                f"of an existing capability?"
            )
            out.append(
                QualityInvestigation(
                    question=question,
                    domain=QualityDomain.DUPLICATION,
                    scope=delta.scope,
                    source_refs=source_refs,
                    signal_refs=(f"verbosity@{current.snapshot_id[:12]}",),
                    evidence_refs=(f"snapshot:{current.snapshot_id[:12]}", *extra_evidence_refs),
                    uncertainty=(
                        "Similar-looking code can implement legitimately different "
                        "concepts; duplication signals require contextual "
                        "interpretation before any consolidation."
                    ),
                    related_domains=(QualityDomain.RESIDUE, QualityDomain.MAINTAINABILITY),
                )
            )
        # LOC and unknown metrics: contextual only, no candidate.
    for inv in out:
        _check_question_shape(inv.question)
    out.sort(key=lambda i: (i.domain.value, i.scope or "", i.question))
    return tuple(out)


# ---------------------------------------------------------------------------
# Ledger integration (evidence records, review-loop compatible claims)
# ---------------------------------------------------------------------------


def snapshot_payload(snapshot: QualitySnapshot) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "target": snapshot.target,
        "repo_head": snapshot.repo_head,
        "created_at": snapshot.created_at,
        "tool": f"{TOOL_NAME}@{TOOL_VERSION}",
        "coverage": dict(snapshot.coverage),
        "measurements": [
            {
                "metric": m.metric,
                "value": m.value,
                "scope": m.scope,
                "source_refs": list(m.source_refs),
                "tool": m.tool,
                "tool_version": m.tool_version,
                "provenance": list(m.provenance),
            }
            for m in snapshot.measurements
        ],
    }


def record_quality_snapshot(
    runtime: Any,
    snapshot: QualitySnapshot,
    *,
    task_id: str,
    run_id: str | None = None,
    actor_id: str = "quality",
) -> Any:
    """Persist one ``quality.snapshot`` event (single evidence record)."""
    from .ledger import Event

    payload = snapshot_payload(snapshot)
    payload["task_id"] = task_id
    payload["run_id"] = run_id
    return runtime.ledger.append(
        Event.create(
            stream_id=snapshot.snapshot_id,
            kind="quality.snapshot",
            actor_id=actor_id,
            payload=payload,
            correlation_id=task_id,
        )
    )


def record_investigations(
    runtime: Any,
    investigations: tuple[QualityInvestigation, ...],
    *,
    task_id: str,
    run_id: str | None = None,
    source_call_id: str = "quality-trajectory",
) -> list[Any]:
    """Persist investigations as ordinary E0 question-claims (review compatible)."""
    import uuid as _uuid

    from .domain import Claim, ClaimStatus, EvidenceClass

    events = []
    for inv in investigations:
        claim = Claim(
            claim_id=str(_uuid.uuid4()),
            task_id=task_id,
            statement=inv.question,
            source_call_id=source_call_id,
            evidence_class=EvidenceClass.ASSERTED,
            status=ClaimStatus.UNRESOLVED,
            scope=inv.scope or inv.domain.value,
            anchors=tuple(inv.source_refs),
            run_id=run_id,
            conditions=inv.uncertainty,
        )
        events.append(runtime.record_claim(claim))
    return events


def quality_snapshots_for_task(runtime: Any, task_id: str) -> list[QualitySnapshot]:
    """Rehydrate snapshots previously recorded for a task (ledger is truth)."""
    out: list[QualitySnapshot] = []
    for event in runtime.ledger.events_by_kind(("quality.snapshot",)):
        if event.correlation_id != task_id and str(event.payload.get("task_id", "")) != task_id:
            continue
        payload = event.payload
        measurements = tuple(
            CodeQualityMeasurement(
                snapshot_id=str(payload.get("snapshot_id", "")),
                metric=str(m.get("metric", "")),
                value=float(m.get("value", 0.0)),
                scope=m.get("scope"),
                source_refs=tuple(m.get("source_refs", ())),
                tool=str(m.get("tool", TOOL_NAME)),
                tool_version=str(m.get("tool_version", TOOL_VERSION)),
                provenance=tuple(m.get("provenance", ())),
            )
            for m in payload.get("measurements", [])
        )
        out.append(
            QualitySnapshot(
                snapshot_id=str(payload.get("snapshot_id", "")),
                target=str(payload.get("target", "")),
                repo_head=payload.get("repo_head"),
                created_at=str(payload.get("created_at", "")),
                measurements=measurements,
                coverage=dict(payload.get("coverage", {})),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Modification-friction telemetry (projection over existing ledger telemetry)
# ---------------------------------------------------------------------------


def collect_episode_friction(runtime: Any, episode_id: str) -> EpisodeFriction:
    """Build the analytical friction view for one run/directive/task id.

    Reuses only telemetry the runtime already records: ``call.completed``
    (model calls, tokens, cost, latency), ``context.compiled`` (candidate
    counts, token totals from the compilation trace), ``action.completed``
    (tool calls), ``check.completed`` (failed validations),
    ``fanout.completed`` (review/investigation rounds proxy). Facets with
    no underlying telemetry stay ``None``.
    """
    calls = [
        e.payload
        for e in runtime.ledger.events_by_kind(("call.completed",))
        if e.correlation_id == episode_id
        or str(e.payload.get("run_id", "")) == episode_id
        or str(e.payload.get("directive_id", "")) == episode_id
    ]
    contexts = [
        e.payload
        for e in runtime.ledger.events_by_kind(("context.compiled",))
        if e.correlation_id == episode_id
    ]
    actions = [
        e.payload
        for e in runtime.ledger.events_by_kind(("action.completed",))
        if e.correlation_id == episode_id
    ]
    checks = [
        e.payload
        for e in runtime.ledger.events_by_kind(("check.completed",))
        if e.correlation_id == episode_id
    ]
    fanouts = [
        e.payload
        for e in runtime.ledger.events_by_kind(("fanout.completed",))
        if e.correlation_id == episode_id
    ]
    if not (calls or contexts or actions or checks):
        return EpisodeFriction(episode_id=episode_id)

    def _ints(payloads: list[dict[str, Any]], key: str) -> int | None:
        values = [p.get(key) for p in payloads if isinstance(p.get(key), (int, float))]
        return int(sum(values)) if values else None

    input_tokens = _ints(calls, "input_tokens")
    output_tokens = _ints(calls, "output_tokens")
    costs = [float(p["cost_usd"]) for p in calls if p.get("cost_usd") is not None]
    latencies = [float(p["latency_ms"]) for p in calls if p.get("latency_ms") is not None]
    latencies.sort()
    candidates: int | None = None
    selected: int | None = None
    ctx_tokens: int | None = None
    if contexts:
        # Real zeros stay zero here: an empty compilation genuinely had no
        # candidates. Only a total absence of context events yields None
        # (handled by the early return above).
        candidates = sum(len(p.get("trace", ())) for p in contexts)
        selected = sum(len(p.get("included_ids", ())) for p in contexts)
        totals = [p.get("total_tokens") for p in contexts if isinstance(p.get("total_tokens"), (int, float))]
        # Only top-level totals exist in newer payloads; the trace payload
        # stores per-entry detail. Fall back to None when absent.
        ctx_tokens = int(sum(totals)) if totals else None
    failed_checks = sum(1 for p in checks if str(p.get("verdict", "")) in ("FAIL", "ERROR")) or None
    failed_calls = sum(1 for p in calls if str(p.get("status", "")) == "failed") or None
    return EpisodeFriction(
        episode_id=episode_id,
        context_candidates=candidates,
        context_selected=selected,
        context_tokens=ctx_tokens,
        model_input_tokens=input_tokens,
        model_output_tokens=output_tokens,
        model_calls=len(calls) or None,
        tool_calls=len(actions) or None,
        failed_calls=failed_calls,
        failed_checks=failed_checks,
        review_rounds=len(fanouts) or None,
        total_cost_usd=round(sum(costs), 6) if costs else None,
        median_latency_ms=latencies[len(latencies) // 2] if latencies else None,
        files_inspected=None,  # no file-level execution telemetry exists; not zero-filled.
        outcome=None,  # outcome attribution belongs to verification, not telemetry.
    )


# ---------------------------------------------------------------------------
# Git calibration harness (research only; read-only toward the repo)
# ---------------------------------------------------------------------------


def _git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip() or None


def list_history_commits(root: Path, *, limit: int = 20) -> list[str]:
    """Oldest-first list of recent commit SHAs (empty when not a git repo)."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-list", f"--max-count={limit}", "--reverse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def measure_commit(
    root: Path, commit: str, *, threshold: int = DEFAULT_COMPLEXITY_THRESHOLD
) -> QualitySnapshot:
    """Measure one commit via ``git archive`` into a temp dir (repo untouched)."""
    with tempfile.TemporaryDirectory(prefix="codeai-quality-") as tmp:
        try:
            archive = subprocess.run(
                ["git", "-C", str(root), "archive", commit],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ValueError(f"cannot archive commit {commit}: {exc}") from exc
        import io as _io
        import tarfile

        with tarfile.open(fileobj=_io.BytesIO(archive.stdout)) as tar:
            tar.extractall(path=tmp)
        snapshot = snapshot_directory(Path(tmp), target=f"{root.name}@{commit[:12]}", threshold=threshold)
    return QualitySnapshot(
        snapshot_id=snapshot.snapshot_id,
        target=f"{root.name}@{commit[:12]}",
        repo_head=commit,
        created_at=snapshot.created_at,
        measurements=snapshot.measurements,
        coverage={**dict(snapshot.coverage), "commit": commit},
    )


def render_trajectory_table(
    history: list[QualitySnapshot], trajectories: list[QualityTrajectory]
) -> str:
    lines = ["commit        LOC      verbosity  erosion"]
    for snapshot in history:
        repo = {m.metric: m.value for m in snapshot.measurements if m.scope is None}
        lines.append(
            f"{(snapshot.repo_head or '?')[:12]:<12}  "
            f"{repo.get(METRIC_LOC, float('nan')):>7.0f}  "
            f"{repo.get(METRIC_VERBOSITY, float('nan')):>9.3f}  "
            f"{repo.get(METRIC_EROSION, float('nan')):>7.3f}"
        )
    for trajectory in trajectories:
        for delta in trajectory.deltas:
            if delta.trend in (MetricTrend.DEGRADING, MetricTrend.STEP_CHANGE):
                lines.append(
                    f"  ! {delta.metric} {delta.trend.value}: "
                    f"{delta.previous} -> {delta.current}"
                )
    return "\n".join(lines) + "\n"


def render_quality_report(
    snapshot: QualitySnapshot,
    trajectory: QualityTrajectory | None = None,
    investigations: tuple[QualityInvestigation, ...] = (),
) -> str:
    repo = {m.metric: m.value for m in snapshot.measurements if m.scope is None}
    lines = ["QUALITY SNAPSHOT", f"target: {snapshot.target}"]
    if trajectory is not None:
        lines.append("QUALITY TRAJECTORY")
        lines.append("")
        lines.append(f"{'Metric':<12}{'Previous':<10}{'Current':<10}{'Delta':<10}Trend")
        lines.append("-" * 53)
        for delta in trajectory.deltas:
            prev = f"{delta.previous:.3f}" if delta.previous is not None else "-"
            cur = f"{delta.current:.3f}" if delta.current is not None else "-"
            dlt = f"{delta.delta:+.3f}" if delta.delta is not None else "-"
            lines.append(f"{delta.metric:<12}{prev:<10}{cur:<10}{dlt:<10}{delta.trend.value}")
    else:
        for metric in (METRIC_LOC, METRIC_VERBOSITY, METRIC_EROSION):
            lines.append(f"{metric}: {repo.get(metric)}")
    failed = snapshot.coverage.get("python_files_failed", [])
    if failed:
        lines.append(f"parser coverage: {len(failed)} file(s) unparsed: {failed}")
    if investigations:
        lines.append("")
        lines.append("Evidence:")
        for inv in investigations:
            refs = ", ".join(inv.source_refs[:5]) or "(repo scope)"
            lines.append(f"- [{inv.domain.value}] {refs}")
        lines.append("")
        lines.append(f"Generated investigations: {len(investigations)}")
        for inv in investigations:
            lines.append(f"- {inv.question}")
    lines.append("")
    lines.append(
        "Note: metrics are evidence about where to investigate, not verdicts; "
        "LOC growth alone is not a violation and trends do not establish causality."
    )
    return "\n".join(lines) + "\n"
