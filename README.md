# codeai

**codeai** is an experimental runtime for solving difficult research and engineering problems with heterogeneous AI systems without making a chat transcript the source of truth.

The central idea is simple:

> Models are stochastic cognitive engines. The runtime owns durable epistemic state.

The project starts from a practical problem — manually copying work between ChatGPT, Claude, Qwen, DeepSeek, OpenCode, Codex, and other systems — but the architecture is deliberately provider-neutral.

## Design principles

1. **Conversation is a control surface, not state.** Durable state belongs in an append-only ledger.
2. **Claims are not evidence.** Model agreement never upgrades a claim by itself.
3. **Independence is enforced.** Context packages can carry seals that prevent sibling outputs from contaminating independent calls.
4. **Cognition, execution, and verification are separate boundaries.** OpenCode is an execution adapter, not the architecture.
5. **Capability is not authority.** Permissions and budgets are explicit and must narrow down a directive tree.
6. **Verification beats consensus.** Prefer tests, reproductions, source checks, and other external evidence to model voting.
7. **Everything important is replayable.** Calls are defined by logged context packages, actor versions, and parameters.

## Current v0 foundation

The first slice intentionally stays small:

- `SQLiteLedger`: append-only event store
- `Directive` / `Task`: bounded units of human intent
- `CallSpec`: replayable cognitive invocation
- `ContextCompiler`: deterministic context packaging with isolation seals
- `Claim`: atomic epistemic assertion with evidence class and status
- `Authority` / `Budget`: hard policy boundaries
- `CognitionAdapter`: model-provider seam
- `ExecutionAdapter`: OpenCode / Codex / Claude Code seam
- `VerificationAdapter`: deterministic or external check seam
- `Runtime`: minimal kernel that records directives and tasks

## Architecture

```text
Human / control surface
          |
          v
+---------------------------+
| Runtime kernel            |
| scheduler / policy        |
| context compiler          |
+-----------+---------------+
            |
      +-----+------+----------------+
      |            |                |
      v            v                v
  Cognition     Execution       Verification
  model calls   OpenCode/...    tests/checks
      |            |                |
      +------------+----------------+
                   v
             Append-only ledger
                   |
         +---------+----------+
         v                    v
      claim graph          decisions
      (projection)         (projection)
```

The event ledger is authoritative. Claim graphs, blackboards, dashboards, cost views, and decision graphs are projections.

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest
ruff check .
```

## What comes next

The next implementation slice should be empirical rather than ornamental:

1. content-addressed artifact storage;
2. call/result recording and idempotency;
3. claim extraction as a derived projection;
4. an isolated command verifier;
5. an OpenCode execution adapter;
6. a simple CLI control surface;
7. the first experiment: **same-model repeated sampling vs heterogeneous sampling at matched cost**.

The browser extension belongs on top of this runtime as a control surface. It should transport interaction, display state, and submit directives; it should not own the intellectual history of the work.

## Durable local state

CodeAI now keeps durable local runtime state under:

```text
.codeai/
  ledger.sqlite
  artifacts/
    ab/
      abcdef...
```

Artifacts are content-addressed by SHA-256 and metadata is stored alongside the append-only ledger.

## First OpenCode path

OpenCode currently exposes a headless HTTP server. Start it in the repository you want it to operate on:

```bash
opencode serve --hostname 127.0.0.1 --port 4096
```

If you protect the server, set `OPENCODE_SERVER_PASSWORD` (and optionally `OPENCODE_SERVER_USERNAME`). Then send a bounded instruction through codeai:

```bash
codeai opencode "Inspect the repository and report the failing tests. Do not modify files."
```

The adapter creates one OpenCode session and returns its ID. Continue the same OpenCode session with:

```bash
codeai opencode --session <session-id> "Now propose the smallest fix."
```

This direct CLI path is intentionally only a control surface. The next slice will record each OpenCode action, result, cost, artifacts, preconditions, and verification outcome in the ledger before we automate browser-to-browser loops.

## Durable run CLI

Create and inspect durable runs:

```bash
codeai run create "Fix the flaky writer-runtime test"
codeai run list
codeai run show <run-id>
```

The existing OpenCode path is preserved, but now records append-only action requests and results in the local ledger before returning control to the human.
