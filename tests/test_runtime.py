import sys
from pathlib import Path

from codeai.adapters import ActionRequest, ActionResult, ActionStatus, CheckRequest, CheckVerdict
from codeai.artifacts import FileArtifactStore
from codeai.domain import Authority, Budget, Capability, Directive
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime
from codeai.verifier import LocalCommandVerifier


class FakeExecutionAdapter:
    def __init__(
        self,
        state: list[str],
        *,
        result: ActionResult | None = None,
        error: Exception | None = None,
    ):
        self.calls = 0
        self.state = state
        self.result = result
        self.error = error

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        self.state[0] = "after"
        if self.error is not None:
            raise self.error
        return self.result or ActionResult(
            action_id=request.action_id,
            status=ActionStatus.SUCCEEDED,
            transcript="done",
            stdout="stdout",
            stderr="stderr",
        )


def build_runtime(tmp_path: Path, state: list[str], *, grant: Authority | None = None) -> Runtime:
    ledger = SQLiteLedger()
    artifacts = FileArtifactStore(tmp_path / "artifacts", ledger)
    runtime = Runtime(ledger, artifact_store=artifacts, state_resolver=lambda: state[0])
    # Actions name directive d1, and a named directive is now what authorizes
    # them, so the fixture records the grant instead of asserting it.
    runtime.open_directive(
        Directive(
            directive_id="d1",
            objective="fixture",
            success_criteria=(),
            budget=Budget(),
            authority=grant or Authority(frozenset({Capability.EXECUTE})),
        )
    )
    return runtime


def allow_execute() -> Authority:
    return Authority(frozenset({Capability.EXECUTE}))


def deny_execute() -> Authority:
    return Authority(frozenset({Capability.READ}))


def request(
    *,
    action_id: str = "a1",
    idempotency_key: str = "key1",
    precondition_hash: str | None = "before",
) -> ActionRequest:
    return ActionRequest(
        action_id=action_id,
        directive_id="d1",
        task_id="t1",
        actor_id="human",
        adapter="fake",
        capability=Capability.EXECUTE.value,
        instruction="do work",
        precondition_hash=precondition_hash,
        idempotency_key=idempotency_key,
    )


def test_allowed_action_records_request_and_result(tmp_path: Path):
    state = ["before"]
    runtime = build_runtime(tmp_path, state)
    adapter = FakeExecutionAdapter(state)

    result = runtime.execute_action(request(), authority=allow_execute(), adapter=adapter)

    assert result.status == ActionStatus.SUCCEEDED
    assert adapter.calls == 1
    # The fixture's directive registration comes first; the action follows.
    assert [event.kind for event in runtime.ledger.read_all()][2:] == [
        "action.requested",
        "operation.governance_recorded",  # what decision it acts under, if any
        "action.authorized",  # the grant the record establishes, with its basis
        "action.execution_started",  # committed before the adapter acts
        "action.completed",
    ]


def test_denied_action_does_not_execute(tmp_path: Path):
    # The recorded directive grants READ. The caller passes EXECUTE anyway, and
    # it makes no difference: the runtime decides from the record.
    state = ["before"]
    runtime = build_runtime(tmp_path, state, grant=Authority(frozenset({Capability.READ})))
    adapter = FakeExecutionAdapter(state)

    result = runtime.execute_action(request(), authority=allow_execute(), adapter=adapter)

    assert result.status == ActionStatus.DENIED
    assert "capability denied" in (result.error or "")
    assert adapter.calls == 0
    refusals = runtime.ledger.events_by_kind(("action.authorization_refused",))
    assert len(refusals) == 1
    assert refusals[0].payload["directive_id"] == "d1"
    assert refusals[0].payload["effective_capabilities"] == ["read"]


def test_duplicate_retry_reuses_previous_result(tmp_path: Path):
    state = ["before"]
    runtime = build_runtime(tmp_path, state)
    adapter = FakeExecutionAdapter(state)

    first = runtime.execute_action(
        request(action_id="a1", idempotency_key="same"),
        authority=allow_execute(),
        adapter=adapter,
    )
    second = runtime.execute_action(
        request(action_id="a2", idempotency_key="same"),
        authority=allow_execute(),
        adapter=adapter,
    )

    assert adapter.calls == 1
    assert first.transcript == second.transcript
    assert second.reused_from_action_id == "a1"


def test_stale_precondition_fails_loudly(tmp_path: Path):
    state = ["current"]
    runtime = build_runtime(tmp_path, state)
    adapter = FakeExecutionAdapter(state)

    result = runtime.execute_action(
        request(precondition_hash="expected"),
        authority=allow_execute(),
        adapter=adapter,
    )

    assert result.status == ActionStatus.FAILED
    assert "precondition mismatch" in (result.error or "")
    assert adapter.calls == 0


def test_failed_execution_is_recorded(tmp_path: Path):
    state = ["before"]
    runtime = build_runtime(tmp_path, state)
    adapter = FakeExecutionAdapter(state, error=RuntimeError("boom"))

    result = runtime.execute_action(request(), authority=allow_execute(), adapter=adapter)

    assert result.status == ActionStatus.FAILED
    assert result.error == "boom"


def test_successful_execution_records_artifacts_and_state(tmp_path: Path):
    state = ["before"]
    runtime = build_runtime(tmp_path, state)
    adapter = FakeExecutionAdapter(state)

    result = runtime.execute_action(request(), authority=allow_execute(), adapter=adapter)

    assert result.resulting_state_hash == "after"
    assert len(result.artifacts) == 3
    assert {artifact.artifact_id for artifact in result.artifacts}


def test_runtime_distinguishes_worker_success_but_verifier_failure(tmp_path: Path):
    state = ["before"]
    runtime = build_runtime(tmp_path, state)
    action = runtime.execute_action(
        request(),
        authority=allow_execute(),
        adapter=FakeExecutionAdapter(state),
    )

    check = runtime.run_check(
        CheckRequest(
            check_id="c1",
            directive_id="d1",
            task_id="t1",
            command=(sys.executable, "-c", "import sys; sys.exit(1)"),
            cwd=str(tmp_path),
        ),
        verifier=LocalCommandVerifier(),
    )

    assert action.status == ActionStatus.SUCCEEDED
    assert check.verdict == CheckVerdict.FAIL


def test_runtime_distinguishes_worker_failure_but_verifier_pass(tmp_path: Path):
    state = ["before"]
    runtime = build_runtime(tmp_path, state)
    action = runtime.execute_action(
        request(),
        authority=allow_execute(),
        adapter=FakeExecutionAdapter(state, error=RuntimeError("worker failed")),
    )

    check = runtime.run_check(
        CheckRequest(
            check_id="c2",
            directive_id="d1",
            task_id="t1",
            command=(sys.executable, "-c", "print('ok')"),
            cwd=str(tmp_path),
        ),
        verifier=LocalCommandVerifier(),
    )

    assert action.status == ActionStatus.FAILED
    assert check.verdict == CheckVerdict.PASS


def test_check_error_is_recorded(tmp_path: Path):
    state = ["before"]
    runtime = build_runtime(tmp_path, state)

    check = runtime.run_check(
        CheckRequest(
            check_id="c3",
            directive_id="d1",
            task_id="t1",
            command=("missing-command-for-codeai-tests",),
            cwd=str(tmp_path),
        ),
        verifier=LocalCommandVerifier(),
    )

    assert check.verdict == CheckVerdict.ERROR
    assert check.error


def test_check_timeout_is_recorded(tmp_path: Path):
    state = ["before"]
    runtime = build_runtime(tmp_path, state)

    check = runtime.run_check(
        CheckRequest(
            check_id="c4",
            directive_id="d1",
            task_id="t1",
            command=(sys.executable, "-c", "import time; time.sleep(0.2)"),
            cwd=str(tmp_path),
            timeout_seconds=0.01,
        ),
        verifier=LocalCommandVerifier(),
    )

    assert check.verdict == CheckVerdict.ERROR
    assert "timed out" in (check.error or "")
