# Seam: process state and durable scheduler decisions

*Applied AI* Chapter 28 (what should happen next) owns this concept; Chapters 16, 20 and 29 depend
on it.

## The question it answers

> When the runtime decides what to do next, can a later reader see which facts the decision rested
> on, and re-derive it — without trusting that the caller described the process honestly?

```text
what is true about the process  !=  what the caller says is true
what was decided                !=  what the policy would decide now
```

## What it separates

```text
ProcessState   what is true          projected from recorded events
scheduler      what to do about it   a pure function of that state
runtime        what was decided      appended with the facts and their basis
```

- `codeai/process_state.py` derives `ProcessState` from the ledger and appends nothing. It contains
  no `should_call_model` and no `should_ask_human`: those are policy conclusions, and keeping them
  out is what lets the same state be replayed against a different policy version.
- `Runtime.process_state(task_id)` projects. `Runtime.decide_next_for_task(task_id)` projects,
  decides and records. `Runtime.decide_next(query)` still accepts a raw `SchedulerInput` unchanged,
  so the frozen router corpus and every existing caller keep working.
- `scheduler.decision_recorded` carries the decision id, operation, reason code, policy version,
  projection version, the full state snapshot, `process_state_sha256` over that snapshot, and the
  basis event ids. A decision is reconstructable, not merely repeatable.
- The scheduler stays pure and now reads two more facts: `unresolved_effect` (seam 1) and
  `process_complete`. Policy order is **complete > process budget > unresolved effect > check >
  cognition > human gate > stop**, at `POLICY_VERSION = "epistemic-v3"`.

### Facts the projection refuses to invent

- **Unknown is not exhausted.** `BudgetFact` carries `limit`, `consumed` and `known`. With no
  recorded directive budget, `known` is False and the budget is never reported as exhausted; the
  state says so in `notes`.
- **Declared is not owed.** `criteria_declared` and `check_required` are separate facts. A criterion
  is a standing requirement; a check is owed only once a proposal exists for it to examine, so a
  fresh task decides CALL rather than CHECK against nothing.
- **Separate is not independent.** `independent_proposal_count` counts only proposals collected
  through a recorded blind fan-out (Chapter 23); ordinary repeated calls raise `proposal_count`
  alone.
- **A completion event is not a completion.** `process_complete` comes from
  `project_task_completion`, so a hand-appended `task.completed` with no acceptance behind it
  completes nothing and the process does not stop.
- **Deciding is not doing.** `decide_next_for_task` returning CALL appends no `call.manifest` and no
  `attempt.started`; re-deciding after a crash is free, because no effect boundary was crossed.
  ACTION remains unreachable from every input combination — an effect is authorized on its own path
  against the recorded grant (seam 2).

## Code

- `src/codeai/process_state.py` — `ProcessState`, `BudgetFact`, `project_process_state`,
  `state_snapshot`, `PROCESS_STATE_V1`.
- `src/codeai/scheduler.py` — `unresolved_effect` and `process_complete` inputs, two new rules,
  `epistemic-v3`.
- `src/codeai/runtime.py` — `process_state`, `decide_next_for_task`, `record_scheduler_decision`.
- `src/codeai/router_contract.py` — `V1_STATE_FIELDS` written out literally, so the frozen v1 corpus
  stays valid as the live schema grows.

## Tests

`tests/test_process_state.py` (20): the decision matrix; an exhausted model budget blocks CALL but
not an owed CHECK; an unresolved effect outranks the ordinary next step and clears on reconciliation;
a hand-appended completion does not stop the process; **the projection contradicts a caller who says
no check is owed**; an unknown budget is never exhausted; the state carries the events it was read
from and states what it could not establish; a recorded decision keeps its payload after the world
changes and replays to the same operation; the state hash pins the facts; deciding CALL starts no
call; ACTION is unreachable across all 64 input combinations.

Full suite: **512 passed**. One frozen-evidence collision was resolved without touching evidence:
the v1 router corpus states predate both new facts, so `state_from_dict` accepts the five-field v1
key set explicitly (legacy aliases still rejected). One assertion moved `epistemic-v2` →
`epistemic-v3`.

## Example

`examples/applied_ai/ch28_scheduler.py`, asserted by `tests/test_examples.py`: one task, four
decisions (CALL → CHECK → ASK_HUMAN → STOP), the last reached through a genuine acceptance by an
acceptor the directive never authorized the process to be. Then the point of recording at all —
decision 2 said CHECK, reprojecting the task now yields STOP, and decision 2 still says CHECK and
still replays to CHECK.

## What it still does not establish

1. **A decision is not a plan.** Each call answers "what kind of work next", one step, with no
   lookahead, no queue and no commitment. Nothing schedules the work it names.
2. **Policy is still a hand-written ladder.** The order encodes judgment that has never been
   measured against outcomes; `epistemic-v3` is a label, not a validation.
3. **No decision-standing gating.** A decision recorded on facts that have since moved is not
   refused or re-opened; the projection reports the drift only if someone looks. Linking Chapter 18
   decision standing to scheduling stays separate work.
4. **The projection is whole-ledger and O(n).** `read_all()` on every call. Correct, not scalable,
   and untested beyond small ledgers.
5. **Budgets are coarse.** Tokens are summed from `call.completed` payloads and turns are counted as
   completed operations; neither is priced, and a failed call that consumed tokens outside a
   recorded payload is invisible.
6. **Acceptance authority is read from the directive chain, but acceptance itself still trusts the
   caller's `Authority` object** (`accept_task`). The state can say a person is needed; it cannot
   prove the person who answered was that person. Unchanged limit from seam 2: no authentication.
7. **Single task.** There is no cross-task or directive-level scheduling, and no notion of which of
   several ready tasks should go first.
8. **`independent_proposal_count` is projected but no rule consumes it.** The fan-out seam (wave 2)
   owns the rule that would.
