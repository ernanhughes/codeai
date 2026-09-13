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

    call = sub.add_parser("call", help="record one isolated cognition call (fake adapter)")
    call.add_argument("--task", required=True)
    call.add_argument("--run", default=None)
    call.add_argument("--prompt", required=True)
    call.add_argument("--actor", default="model-a")
    call.add_argument("--provider", default="fake-provider")
    call.add_argument("--model", default="fake-model")
    call.add_argument("--response", default=None)
    call.add_argument("--experiment", default=None)
    call.add_argument("--chamber", default=None, help="logical job name (e.g. deep-review)")
    call.add_argument("--logical-model", default=None, help="logical model key for resolution")
    call.add_argument("--max-attempts", type=int, default=1)

    calls_cmd = sub.add_parser("calls", help="inspect recorded cognition calls")
    calls_sub = calls_cmd.add_subparsers(dest="calls_command", required=True)
    calls_show = calls_sub.add_parser("show", help="show one recorded call with attempts")
    calls_show.add_argument("call_id")

    fanout = sub.add_parser("fanout", help="sealed blind fanout of N independent calls")
    fanout.add_argument("--task", required=True)
    fanout.add_argument("--run", default=None)
    fanout.add_argument("--prompt", required=True)
    fanout.add_argument("--count", type=int, default=2)
    fanout.add_argument("--models", default="fake-model")
    fanout.add_argument("--experiment", default=None)

    claims_cmd = sub.add_parser("claims", help="show claims and concurrence for a run")
    claims_cmd.add_argument("run_id")

    checks_cmd = sub.add_parser("checks", help="list checks for a run")
    checks_cmd.add_argument("run_id")

    context_cmd = sub.add_parser("context", help="inspect compiled context")
    context_sub = context_cmd.add_subparsers(dest="context_command", required=True)
    context_show = context_sub.add_parser("show", help="show one context package by hash")
    context_show.add_argument("hash")

    models_cmd = sub.add_parser("models", help="inspect logical model/provider mapping")
    models_sub = models_cmd.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="show configured models and missing credentials")
    models_init = models_sub.add_parser("init", help="write example .codeai/config.toml")
    models_init.add_argument("--force", action="store_true")

    exp = sub.add_parser("experiment", help="controlled C0/C1/H1 experiments")
    exp_sub = exp.add_subparsers(dest="experiment_command", required=True)

    exp_create = exp_sub.add_parser("create", help="preregister a versioned experiment")
    exp_create.add_argument("--name", required=True)
    exp_create.add_argument("--hypothesis", required=True)
    exp_create.add_argument("--tasks", default="all", help="'all' or comma-separated corpus task ids")
    exp_create.add_argument("--primary-metric", default="verified oracle@k")
    exp_create.add_argument("--c0", default=None, help="logical model for C0 baseline")
    exp_create.add_argument("--c1", default=None, help="logical model for C1 homogeneous arm")
    exp_create.add_argument("--c1-samples", type=int, default=3)
    exp_create.add_argument("--h1", default=None, help="comma-separated logical models for H1")
    exp_create.add_argument("--h1-samples", type=int, default=3)
    exp_create.add_argument("--max-calls", type=int, default=None)
    exp_create.add_argument("--max-cost", type=float, default=None)
    exp_create.add_argument(
        "--allow-unknown-cost",
        action="store_true",
        help="explicit recorded override: continue while cost is unknown (cost stays UNKNOWN)",
    )
    exp_create.add_argument("--unknown-cost-reason", default=None)
    exp_create.add_argument("--timeout", type=float, default=60.0)
    exp_create.add_argument("--dry-run", action="store_true")
    exp_create.add_argument("--corpus", default="seeded-code-v1",
                            choices=["seeded-code-v1", "semantic-repair-v1"])

    exp_run = exp_sub.add_parser("run", help="run one experiment arm")
    exp_run.add_argument("--experiment", required=True)
    exp_run.add_argument("--arm", required=True, help="arm name as preregistered (e.g. C0, C1, H1, P2C, P2S)")
    exp_run.add_argument("--tasks", default=None, help="optional subset of corpus task ids")
    exp_run.add_argument("--candidates-dir", default=None)
    exp_run.add_argument("--dry-run", action="store_true")
    exp_run.add_argument("--corpus", default=None,
                         help="override corpus (default: experiment config version)")

    exp_report = exp_sub.add_parser("report", help="deterministic experiment report")
    exp_report.add_argument("experiment_id")

    exp_export = exp_sub.add_parser("export", help="export raw experiment observations as JSON")
    exp_export.add_argument("experiment_id")
    exp_export.add_argument("--out", default=None)

    exp_sub.add_parser("list", help="list experiments")

    oc = sub.add_parser("opencode", help="send one bounded instruction to OpenCode")
    oc.add_argument("instruction")
    oc.add_argument("--url", default=os.getenv("OPENCODE_URL", "http://127.0.0.1:4096"))
    oc.add_argument("--session")
    oc.add_argument("--task")
    oc.add_argument("--title", default="codeai")
    oc.add_argument("--username", default=os.getenv("OPENCODE_SERVER_USERNAME", "opencode"))
    oc.add_argument("--password", default=os.getenv("OPENCODE_SERVER_PASSWORD"))
    quality = sub.add_parser("quality", help="deterministic quality measurements (observational)")
    quality_sub = quality.add_subparsers(dest="quality_command", required=True)

    quality_measure = quality_sub.add_parser("measure", help="measure one directory snapshot")
    quality_measure.add_argument("target")
    quality_measure.add_argument("--json", action="store_true", dest="as_json")

    quality_compare = quality_sub.add_parser("compare", help="compare two directory snapshots")
    quality_compare.add_argument("base")
    quality_compare.add_argument("current")
    quality_compare.add_argument("--json", action="store_true", dest="as_json")

    quality_trajectory = quality_sub.add_parser(
        "trajectory", help="measure recent git history and show trends (read-only)"
    )
    quality_trajectory.add_argument("target")
    quality_trajectory.add_argument("--commits", type=int, default=10)
    quality_trajectory.add_argument("--json", action="store_true", dest="as_json")
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
            print("calls:")
            for call_payload in run.get("calls", ()):
                print(
                    f"  - {call_payload['call_id']}: {call_payload.get('status')} "
                    f"model={call_payload.get('model')} cost={call_payload.get('cost_usd')}"
                )
            print("claims:")
            for claim in run.get("claims", ()):
                print(f"  - {claim['claim_id']}: {claim['statement'][:80]} [{claim.get('evidence_class')}]")
            return 0

    if args.command == "call":
        from .adapters import CallSpec, FakeCognitionAdapter
        from .context import ContextCompiler
        from .domain import ActorRef, Variant

        task = ensure_task(runtime, args.prompt, args.task)
        actor = ActorRef(
            actor_id=args.actor, kind="model", provider=args.provider, model=args.model
        )
        package, _trace = ContextCompiler().compile_with_trace(
            task_id=task.task_id, actor=actor, prompt=args.prompt, events=()
        )
        adapter = FakeCognitionAdapter(
            responses=[args.response] if args.response else None,
            provider=args.provider,
            model=args.model,
            # The CLI fake observes no provider revision; stay UNKNOWN rather
            # than promoting the fake's stand-in version to observed metadata.
            model_version=None,
        )
        spec = CallSpec(
            call_id=str(uuid.uuid4()),
            task_id=task.task_id,
            actor=actor,
            context=package,
            idempotency_key=str(uuid.uuid4()),
            directive_id=task.directive_id,
            run_id=args.run or task.directive_id,
            adapter_id=args.provider,
            instruction=args.prompt,
            variant=Variant(
                model=args.model, provider=args.provider, experiment=args.experiment
            ),
            chamber=args.chamber,
            logical_model=args.logical_model,
        )
        recorded = runtime.invoke_recorded_call(
            spec, adapter=adapter, max_attempts=args.max_attempts
        )
        result = recorded.attempts[-1] if recorded.attempts else None
        print(f"task_id: {recorded.task_id}")
        print(f"call_id: {recorded.call_id}")
        print(f"chamber: {recorded.chamber}")
        print(f"requested_model: {recorded.manifest.requested_model}")
        print(f"provider: {recorded.manifest.provider}")
        print(f"resolved_model: {recorded.manifest.resolved_model_id}")
        revision = recorded.attempts[-1].provider_revision if recorded.attempts else None
        print(f"provider_revision: {revision or 'UNKNOWN'}")
        print(f"pricing_version: {recorded.manifest.pricing_version}")
        print(f"attempt_count: {len(recorded.attempts)}")
        for attempt in recorded.attempts:
            print(f"attempt {attempt.attempt_index}: id={attempt.attempt_id} status={attempt.status}")
            print(f"  raw_artifact: {attempt.raw_artifact.sha256 if attempt.raw_artifact else None}")
            print(f"  usage_source: {attempt.usage.source.value} cost={attempt.cost_usd}")
            print(f"  effective_parameters: {dict(attempt.effective_parameters)}")
        print(f"call_status: {recorded.status}")
        print("task_status: not automatically completed")
        return 0 if recorded.status == "succeeded" else 1

    if args.command == "calls" and args.calls_command == "show":
        recorded = runtime.get_recorded_call(args.call_id)
        if recorded is None:
            print(f"call not found: {args.call_id}", file=sys.stderr)
            return 1
        print(f"task_id: {recorded.task_id}")
        print(f"call_id: {recorded.call_id}")
        print(f"chamber: {recorded.chamber}")
        print(f"requested_model: {recorded.manifest.requested_model}")
        print(f"provider: {recorded.manifest.provider}")
        print(f"resolved_model: {recorded.manifest.resolved_model_id}")
        print(f"pricing_version: {recorded.manifest.pricing_version}")
        print(f"attempt_count: {len(recorded.attempts)}")
        for attempt in recorded.attempts:
            print(f"attempt {attempt.attempt_index}: id={attempt.attempt_id} status={attempt.status}")
            print(f"  provider_request_id: {attempt.provider_request_id}")
            print(f"  raw_artifact: {attempt.raw_artifact.sha256 if attempt.raw_artifact else None}")
            print(f"  usage_source: {attempt.usage.source.value} cost={attempt.cost_usd}")
            print(f"  effective_parameters: {dict(attempt.effective_parameters)}")
        print(f"call_status: {recorded.status}")
        return 0

    if args.command == "fanout":
        from .adapters import FakeCognitionAdapter
        from .domain import ActorRef, Variant

        task = ensure_task(runtime, args.prompt, args.task)
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        branches = []
        for i in range(args.count):
            model = models[i % len(models)]
            branches.append(
                {
                    "actor": ActorRef(
                        actor_id=f"{model}-{i}", kind="model", provider="fake-provider", model=model
                    ),
                    "adapter": FakeCognitionAdapter(
                        responses=[f"fake output branch {i}"],
                        provider="fake-provider",
                        model=model,
                    ),
                    "adapter_id": "fake-provider",
                    "variant": Variant(model=model, provider="fake-provider", experiment=args.experiment),
                    "prompt": args.prompt,
                }
            )
        results = runtime.sealed_fanout(
            task_id=task.task_id,
            base_prompt=args.prompt,
            branches=branches,
            directive_id=task.directive_id,
            run_id=args.run or task.directive_id,
        )
        for result in results:
            print(f"{result.call_id}: {result.status} model={result.model}")
        return 0

    if args.command == "claims":
        report = runtime.disagreement_for_run(args.run_id)
        print(f"run_id: {args.run_id}")
        print("concurrence (agreement != evidence):")
        for entry in report["concurrence"]:  # type: ignore[index]
            print(f"  - {entry['statement'][:100]} x{entry['concurrence']} sources={entry['sources']}")
        print("unique:")
        for entry in report["unique"]:  # type: ignore[index]
            print(f"  - {entry['statement'][:100]}")
        print("contradicted:")
        for claim in report["contradicted"]:  # type: ignore[index]
            print(f"  - {claim['claim_id']}: {claim['statement'][:100]}")
        print("unresolved:")
        for claim in report["unresolved"]:  # type: ignore[index]
            print(f"  - {claim['claim_id']}: {claim['statement'][:100]}")
        return 0

    if args.command == "checks":
        run = runtime.show_run(args.run_id)
        if run is None:
            print(f"run not found: {args.run_id}", file=sys.stderr)
            return 1
        for check in run["checks"]:
            print(f"  - {check['check_id']}: {check['verdict']}")
        return 0

    if args.command == "context" and args.context_command == "show":
        for event in runtime.ledger.events_by_kind(("context.compiled",)):
            if str(event.payload.get("package_id", "")) != args.hash and str(event.payload.get("trace_id", "")) != args.hash:
                continue
            print(f"package_id: {event.payload['package_id']}")
            print(f"task_id: {event.payload['task_id']}")
            print(f"events: {event.payload['event_ids']}")
            print(f"artifacts: {event.payload['artifact_ids']}")
            print(f"claims: {event.payload['claim_ids']}")
            print("trace:")
            for entry in event.payload["trace"]:
                print(f"  - {entry['candidate_id']}: {entry['decision']} ({entry['reason']})")
            return 0
        print(f"context not found: {args.hash}", file=sys.stderr)
        return 1

    if args.command == "models":
        from .modelconfig import (
            EXAMPLE_CONFIG,
            default_config_path,
            load_model_config,
            missing_credentials,
        )

        if args.models_command == "init":
            path = default_config_path(Path.cwd())
            if path.exists() and not args.force:
                print(f"exists: {path} (use --force to overwrite)", file=sys.stderr)
                return 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
            print(f"wrote {path}")
            return 0
        config = load_model_config()
        if not config.models:
            print("no models configured; run `codeai models init`")
            return 0
        for name in sorted(config.models):
            mapping = config.models[name]
            missing = missing_credentials(mapping)
            status = f"MISSING: {missing}" if missing else "ok"
            print(f"{name}: adapter={mapping.adapter} model={mapping.model} {status}")
        return 0

    if args.command == "experiment":
        return _experiment_command(runtime, args)

    if args.command == "quality":
        return _quality_command(args)

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
                requested_by="human",
                actor_id="opencode",
                adapter="opencode",
                adapter_id="opencode",
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


def _quality_command(args: argparse.Namespace) -> int:
    import json as _json

    from .quality import (
        compare_snapshots,
        generate_investigations,
        list_history_commits,
        measure_commit,
        render_quality_report,
        render_trajectory_table,
        snapshot_directory,
        snapshot_payload,
    )

    if args.quality_command == "measure":
        snapshot = snapshot_directory(Path(args.target))
        if args.as_json:
            print(_json.dumps(snapshot_payload(snapshot), indent=2, sort_keys=True))
        else:
            print(render_quality_report(snapshot), end="")
        return 0

    if args.quality_command == "compare":
        base = snapshot_directory(Path(args.base))
        current = snapshot_directory(Path(args.current))
        trajectory = compare_snapshots(base, current)
        investigations = generate_investigations(trajectory, current=current)
        if args.as_json:
            print(
                _json.dumps(
                    {
                        "base": snapshot_payload(base),
                        "current": snapshot_payload(current),
                        "deltas": [
                            {
                                "metric": d.metric,
                                "previous": d.previous,
                                "current": d.current,
                                "delta": d.delta,
                                "trend": d.trend.value,
                            }
                            for d in trajectory.deltas
                        ],
                        "investigations": [i.question for i in investigations],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_quality_report(current, trajectory, investigations), end="")
        return 0

    if args.quality_command == "trajectory":
        root = Path(args.target)
        commits = list_history_commits(root, limit=args.commits)
        if not commits:
            print(f"no git history found under {root}", file=sys.stderr)
            return 1
        history = [measure_commit(root, sha) for sha in commits]
        trajectories = [
            compare_snapshots(history[i], history[i + 1]) for i in range(len(history) - 1)
        ]
        if args.as_json:
            print(
                _json.dumps(
                    {
                        "snapshots": [snapshot_payload(s) for s in history],
                        "trajectories": [
                            [
                                {
                                    "metric": d.metric,
                                    "previous": d.previous,
                                    "current": d.current,
                                    "delta": d.delta,
                                    "trend": d.trend.value,
                                }
                                for d in t.deltas
                            ]
                            for t in trajectories
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_trajectory_table(history, trajectories), end="")
        return 0
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


def _experiment_command(runtime: Runtime, args: argparse.Namespace) -> int:
    import json as _json

    from .analysis import build_report, export_experiment, render_report
    from .corpus import CORPUS_VERSION, seeded_corpus
    from .corpus_v2 import CORPUS2_VERSION, semantic_corpus
    from .experiments import (
        ArmDef,
        ExperimentBudget,
        build_config,
        create_experiment,
        get_experiment,
        plan_experiment,
        run_arm,
    )
    from .modelconfig import load_model_config

    if args.experiment_command == "list":
        for event in runtime.ledger.events_by_kind(("experiment.created",)):
            print(f"{event.payload['experiment_id']}  {event.payload['name']}")
        return 0

    def _corpus_for(version: str) -> tuple:
        if version == CORPUS2_VERSION:
            return semantic_corpus()
        if version == CORPUS_VERSION:
            return seeded_corpus()
        print(f"unknown corpus version: {version}", file=sys.stderr)
        raise SystemExit(1)

    if args.experiment_command == "create":
        corpus = _corpus_for(args.corpus)
        available = {t.task_id for t in corpus}
        if args.tasks == "all":
            task_ids = tuple(t.task_id for t in corpus)
        else:
            task_ids = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
            unknown = set(task_ids) - available
            if unknown:
                print(f"unknown corpus tasks: {sorted(unknown)}", file=sys.stderr)
                return 1
        arms: list[ArmDef] = []
        if args.c0:
            arms.append(ArmDef(name="C0", models=(args.c0,), samples=1))
        if args.c1:
            arms.append(ArmDef(name="C1", models=(args.c1,), samples=args.c1_samples))
        if args.h1:
            models = tuple(m.strip() for m in args.h1.split(",") if m.strip())
            arms.append(ArmDef(name="H1", models=models, samples=args.h1_samples))
        if not arms:
            print("define at least one arm: --c0, --c1, or --h1", file=sys.stderr)
            return 1
        config = build_config(
            name=args.name,
            hypothesis=args.hypothesis,
            primary_metric=args.primary_metric,
            corpus_version=args.corpus,
            task_ids=task_ids,
            arms=tuple(arms),
            budget=ExperimentBudget(
                max_calls=args.max_calls, max_cost_usd=args.max_cost,
                timeout_seconds=args.timeout,
                allow_unknown_cost=args.allow_unknown_cost,
                unknown_cost_reason=args.unknown_cost_reason),
        )
        model_config = load_model_config()
        plan = plan_experiment(config, model_config, corpus_tasks=corpus)
        if args.dry_run:
            print(_json.dumps(plan, indent=2, sort_keys=True))
            return 0
        create_experiment(runtime, config)
        print(f"experiment_id: {config.experiment_id}")
        print(f"config_hash: {config.config_hash}")
        print(f"tasks: {len(task_ids)} arms: {[a.name for a in arms]}")
        return 0

    if args.experiment_command == "run":
        config = get_experiment(runtime, args.experiment)
        if config is None:
            print(f"unknown experiment: {args.experiment}", file=sys.stderr)
            return 1
        corpus = _corpus_for(args.corpus or config.corpus_version)
        wanted = set(args.tasks.split(",") if args.tasks else config.task_ids)
        tasks = tuple(t for t in corpus if t.task_id in wanted)
        if not tasks:
            print("no matching corpus tasks", file=sys.stderr)
            return 1
        model_config = load_model_config()
        plan = plan_experiment(config, model_config, corpus_tasks=tasks)
        if args.dry_run:
            print(_json.dumps(plan, indent=2, sort_keys=True))
            return 0
        before = len(runtime.ledger.read_all())
        candidates_dir = Path(args.candidates_dir) if args.candidates_dir else Path.cwd() / ".codeai" / "candidates"
        summary = run_arm(
            runtime, args.experiment, args.arm, tasks, model_config,
            candidates_root=candidates_dir)
        print(f"arm: {summary['arm']} completed_tasks: {summary['completed_tasks']}")
        if summary["stopped"]:
            print(f"stopped: {summary['stopped']}")
        print(f"events_added: {len(runtime.ledger.read_all()) - before}")
        return 0

    if args.experiment_command == "report":
        try:
            print(render_report(build_report(runtime, args.experiment_id)), end="")
        except KeyError:
            print(f"unknown experiment: {args.experiment_id}", file=sys.stderr)
            return 1
        return 0

    if args.experiment_command == "export":
        try:
            payload = export_experiment(runtime, args.experiment_id)
        except KeyError:
            print(f"unknown experiment: {args.experiment_id}", file=sys.stderr)
            return 1
        text = _json.dumps(payload, indent=2, sort_keys=True, default=str)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
