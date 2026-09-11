from __future__ import annotations

import argparse
import os
import uuid

from .adapters import ActionRequest
from .opencode import OpenCodeClient, OpenCodeExecutionAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeai", description="codeai epistemic runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    oc = sub.add_parser("opencode", help="send one bounded instruction to OpenCode")
    oc.add_argument("instruction")
    oc.add_argument("--url", default=os.getenv("OPENCODE_URL", "http://127.0.0.1:4096"))
    oc.add_argument("--session")
    oc.add_argument("--title", default="codeai")
    oc.add_argument("--username", default=os.getenv("OPENCODE_SERVER_USERNAME", "opencode"))
    oc.add_argument("--password", default=os.getenv("OPENCODE_SERVER_PASSWORD"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "opencode":
        client = OpenCodeClient(args.url, username=args.username, password=args.password)
        adapter = OpenCodeExecutionAdapter(
            client, session_id=args.session, session_title=args.title
        )
        result = adapter.execute(
            ActionRequest(
                action_id=str(uuid.uuid4()),
                task_id="cli",
                capability="execute",
                instruction=args.instruction,
                precondition_hash=None,
                idempotency_key=str(uuid.uuid4()),
            )
        )
        print(result.transcript or "")
        print(f"\n[opencode-session: {result.state_hash}]")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
