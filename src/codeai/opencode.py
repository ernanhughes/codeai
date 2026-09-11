from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .adapters import ActionRequest, ActionResult, ActionStatus, ExecutionAdapter


class OpenCodeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenCodeSession:
    session_id: str
    title: str | None = None


class OpenCodeClient:
    """Minimal HTTP client for an existing `opencode serve` instance."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4096",
        *,
        username: str = "opencode",
        password: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout

    def create_session(self, *, title: str | None = None) -> OpenCodeSession:
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        data = self._request("POST", "/session", body)
        session_id = data.get("id")
        if not session_id:
            raise OpenCodeError("OpenCode create-session response did not contain an id")
        return OpenCodeSession(session_id=str(session_id), title=data.get("title"))

    def prompt(self, session_id: str, text: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/session/{session_id}/message",
            {"parts": [{"type": "text", "text": text}]},
        )

    def abort(self, session_id: str) -> bool:
        return bool(self._request("POST", f"/session/{session_id}/abort", {}))

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        if self.password is not None:
            token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        request = Request(
            f"{self.base_url}{path}", data=encoded, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OpenCodeError(f"OpenCode HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise OpenCodeError(f"cannot reach OpenCode at {self.base_url}: {exc.reason}") from exc
        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))


class OpenCodeExecutionAdapter(ExecutionAdapter):
    """Executes bounded instructions through one durable OpenCode session."""

    def __init__(
        self,
        client: OpenCodeClient,
        *,
        session_id: str | None = None,
        session_title: str = "codeai",
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.session_title = session_title

    def execute(self, request: ActionRequest) -> ActionResult:
        if self.session_id is None:
            self.session_id = self.client.create_session(title=self.session_title).session_id
        instruction = request.instruction or str(request.payload.get("instruction", ""))
        response = self.client.prompt(self.session_id, instruction)
        text_parts = [
            str(part.get("text", ""))
            for part in response.get("parts", [])
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return ActionResult(
            action_id=request.action_id,
            status=ActionStatus.SUCCEEDED,
            transcript="\n".join(part for part in text_parts if part) or json.dumps(response),
            state_hash=self.session_id,
        )
