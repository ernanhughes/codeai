from codeai.adapters import ActionRequest
from codeai.opencode import OpenCodeExecutionAdapter, OpenCodeSession


class FakeClient:
    def __init__(self):
        self.created = 0
        self.prompts = []

    def create_session(self, *, title=None):
        self.created += 1
        return OpenCodeSession("ses-test", title)

    def prompt(self, session_id, text):
        self.prompts.append((session_id, text))
        return {"parts": [{"type": "text", "text": "done"}]}


def test_opencode_adapter_reuses_durable_session():
    client = FakeClient()
    adapter = OpenCodeExecutionAdapter(client)  # type: ignore[arg-type]
    request = ActionRequest(
        action_id="a1",
        task_id="t1",
        capability="execute",
        instruction="inspect repo",
        precondition_hash=None,
        idempotency_key="i1",
    )

    first = adapter.execute(request)
    second = adapter.execute(request)

    assert first.transcript == "done"
    assert second.state_hash == "ses-test"
    assert client.created == 1
    assert client.prompts == [("ses-test", "inspect repo"), ("ses-test", "inspect repo")]
