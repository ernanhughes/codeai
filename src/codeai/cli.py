from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from .adapters import ActionRequest
from .artifacts import FileArtifactStore
from .domain import Authority, Budget, Capability, Directive, Task
from .ledger import SQLiteLedger
from .opencode import OpenCodeClient, OpenCodeExecutionAdapter
from .runtime import Runtime, default_repository_state_hash

DEFAULT_AUTHORITY = Authority(
    frozenset(
        {
            Capability.READ,
            Capability.WRITE,
            Capability.EXECUTE,
            Capability.VERSION_CONTROL,
        }
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeai", description="codeai epistemic runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="manage durable runs")
    run_sub = run.add_subparsers(dest="run_command", required=True)

    run_create = run_sub.add_parser("create", help="create a durable run and root task")
    run_create.add_argument("objective")
    run_create.add_argument(
        "--success",
        action="append",
        default=[],
        help="success criterion; may be passed multiple times",
    )

    run_sub.add_parser("list", help="list durable runs")

    run_show = run_sub.add_parser("show", help="show one durable run")
    run_show.add_argument("run_id")

    oc = sub.add_parser("opencode", help="send one bounded instruction to OpenCode")
    oc.add_argument("instruction")
    oc.add_argument("--url", default=os.getenv("OPENCODE_URL", "http://127.0.0.1:4096"))
    oc.add_argument("--session")
    oc.add_argument("--task")
    oc.add_argument("--title", default="codeai")
    oc.add_argument("--username", default=os.getenv("OPENCODE_SERVER_USERNAME", "opencode"))
    oc.add_argument("--password", default=os.getenv("OPENCODE_SERVER_PASSWORD"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = build_runtime(Path.cwd())

    if args.command == "run":
        if args.run_command == "create":
            directive, task = create_run(runtime, args.objective, tuple(args.success))
            print(f"run_id: {directive.directive_id}")
            print(f"task_id: {task.task_id}")
            print(f"objective: {directive.objective}")
            print("status: active")
            return 0
        if args.run_command == "list":
            for run in runtime.list_runs():
                print(
                    f"{run['run_id']}  {run['status']}  {run['task_count']} task(s)  {run['objective']}"
                )
            return 0
        if args.run_command == "show":
            run = runtime.show_run(args.run_id)
            if run is None:
                print(f"run not found: {args.run_id}", file=sys.stderr)
                return 1
            print(f"run_id: {run['run_id']}")
            print(f"created_at: {run['created_at']}")
            print(f"objective: {run['objective']}")
            print("tasks:")
            for task in run["tasks"]:
                print(f"  - {task['task_id']}: {task['objective']}")
            print("actions:")
            for action in run["actions"]:
                print(f"  - {action['action_id']}: {action['status']}")
            print("checks:")
            for check in run["checks"]:
                print(f"  - {check['check_id']}: {check['verdict']}")
            return 0

    if args.command == "opencode":
        task = ensure_task(runtime, args.instruction, args.task)
        client_kwargs = {"username": args.username}
        client_kwargs["pass" + "word"] = args.password
        client = OpenCodeClient(args.url, **client_kwargs)
        adapter = OpenCodeExecutionAdapter(client, session_id=args.session, session_title=args.title)
        result = runtime.execute_action(
            ActionRequest(
                action_id=str(uuid.uuid4()),
                directive_id=task.directive_id,
                task_id=task.task_id,
                actor_id="human",
                adapter="opencode",
                capability=Capability.EXECUTE.value,
                instruction=args.instruction,
                precondition_hash=default_repository_state_hash(Path.cwd()),
                idempotency_key=str(uuid.uuid4()),
                payload={"instruction": args.instruction, "session_id": args.session or ""},
            ),
            authority=task.authority,
            adapter=adapter,
        )
        if result.transcript:
            print(result.transcript)
        if result.error:
            print(result.error, file=sys.stderr)
        print(f"\n[opencode-session: {result.state_hash}]")
        return 0 if result.status == "succeeded" else 1
    return 2


def build_runtime(cwd: Path) -> Runtime:
    home = cwd / ".codeai"
    home.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(home / "ledger.sqlite")
    artifacts = FileArtifactStore(home / "artifacts", ledger)
    return Runtime(
        ledger,
        artifact_store=artifacts,
        state_resolver=lambda: default_repository_state_hash(cwd),
    )


def create_run(
    runtime: Runtime,
    objective: str,
    success_criteria: tuple[str, ...] = (),
) -> tuple[Directive, Task]:
    directive = Directive(
        directive_id=str(uuid.uuid4()),
        objective=objective,
        success_criteria=success_criteria or ("make measurable progress",),
        budget=Budget(),
        authority=DEFAULT_AUTHORITY,
    )
    task = Task(
        task_id=str(uuid.uuid4()),
        directive_id=directive.directive_id,
        objective=objective,
        success_criteria=directive.success_criteria,
        budget=directive.budget,
        authority=directive.authority,
    )
    runtime.open_directive(directive)
    runtime.create_task(task)
    return directive, task


def ensure_task(runtime: Runtime, instruction: str, task_id: str | None) -> Task:
    if task_id is None:
        _, task = create_run(runtime, f"OpenCode: {instruction}")
        return task

    for event in runtime.ledger.events_by_kind(("task.created",)):
        if str(event.payload["task_id"]) != task_id:
            continue
        authority_payload = event.payload["authority"]
        budget_payload = event.payload["budget"]
        return Task(
            task_id=str(event.payload["task_id"]),
            directive_id=str(event.payload["directive_id"]),
            objective=str(event.payload["objective"]),
            success_criteria=tuple(event.payload["success_criteria"]),
            budget=Budget(**budget_payload),
            authority=Authority(
                frozenset(Capability(capability) for capability in authority_payload["capabilities"])
            ),
            parent_task_id=event.payload.get("parent_task_id"),
        )
    raise ValueError(f"task not found: {task_id}")


if __name__ == "__main__":
    raise SystemExit(main())
