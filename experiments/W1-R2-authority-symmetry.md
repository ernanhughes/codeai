# W1-R2: authority symmetry — repair of composition audit gap 2

**Cites:** `experiments/W1-composition-results.md` §4 gap 2 (frozen; not edited).
**Recheck:** `experiments/W1-R2-composition-recheck.json`, same harness.
**Scope, as registered:** acceptance authority *resolved from the durable record*. Not
authentication, not signatures, not identity infrastructure.

## The gap

```text
recorded directive -> action authority      ENFORCED
recorded directive -> acceptance authority  RECORDED (the caller's Authority argument)
```

A directive granting only WRITE; a caller passing `Authority({ACCEPT})`; the task completed. And
`task.accepted` carried no directive id, so the acceptance could not even be re-resolved after the
fact.

## The repair

```text
task.created names a directive
      -> resolve that chain, intersected exactly as the action path resolves it
      -> ACCEPT in the effective grant, or the acceptance is refused
```

**The directive comes from the task record, not the request.** A caller who could name the directive
could name a permissive one, which would be the same hole wearing a new field.

`accept_task(request, authority=...)` keeps its parameter for compatibility and **never consults
it**. What the caller passed is recorded as `caller_claimed_capabilities`: visible, attributable, and
inert. `task.accepted` and `task.acceptance_rejected` now both carry `directive_id` and the resolved
`authority_basis` (status, reason, effective capabilities, the grant chain with the event each link
was read from, and the basis event ids).

Authority is decided **before** any prior acceptance is disclosed, matching the action path: a caller
without a current grant learns the refusal, not the history.

### Refusal vocabulary

```text
acceptance_not_granted:<directive_id>                       the chain resolves, ACCEPT is not in it
acceptance_authority_unresolved:task_names_no_directive     the task record names none
acceptance_authority_unresolved:unknown_directive:<id>      no such directive is recorded
acceptance_authority_unresolved:invalid_chain:<id>          cycle, missing parent, widening child
```

Never a fallback to the caller. An unresolvable chain is a refusal, not a reason to trust the API
argument.

## Evidence

`tests/test_acceptance_authority.py` (11), covering the registered cases:

| Case | Result |
|---|---|
| directive grants ACCEPT | accepted |
| directive lacks ACCEPT, caller claims WRITE+ACCEPT | **refused**, claim recorded and inert |
| child narrows ACCEPT away | refused against the child |
| child tries to enlarge the parent | registration refused durably; chain never offers ACCEPT |
| task names no directive | refused, no fallback |
| unknown directive | refused, no fallback |
| ambiguous chain (two registrations) | refused |
| acceptance record | carries directive id, chain, effective capabilities, basis events |
| reopen | same resolution, same chain, same completion |
| actor label | attribution only; anyone may be named, the grant is what is checked |
| authority before disclosure | a broken chain refuses without revealing the prior acceptance |

Existing expectations changed because the behaviour genuinely changed:

- fixtures in `test_task_acceptance.py` and `test_raw_output.py` now record the directive their tasks
  name — the same adjustment seam 2 required of the action fixtures;
- `test_unauthorized_acceptor_cannot_complete` became
  `test_the_caller_argument_is_no_longer_an_authority_source`: a caller *without* ACCEPT now
  completes the task, because the record grants it and the caller was never the authority;
- `authority_basis` on `task.accepted` is the resolved basis, not a list of what the caller passed.

Full suite: **582 passed**.

## Recheck against the frozen baseline

```text
B  directive -> acceptance authority    RECORDED  ->  ENFORCED
```

Every other classification is unchanged. Two things did move in the harness, both disclosed:

1. The happy path and probes I and M now record directives that grant ACCEPT. Without that they
   would measure the authority rule instead of their own subject.
2. The happy path's decision 3 is now `STOP (no epistemic operation required)` rather than
   `ASK_HUMAN`, because its chain grants ACCEPT and nothing is owed.

## Finding raised by this repair (not repaired here)

Probe **N**, new: **when the scheduler says ASK_HUMAN, no human can answer through this API.**

```text
directive grants WRITE only
  scheduler            ASK_HUMAN   "next effect requires human authority"
  a human accepting    refused: acceptance_not_granted:review-gated
  the task ends at     ASK_HUMAN, permanently
```

Before this repair the gate was passable because anyone could claim ACCEPT — which was the gap. Now
the gate is real, and a task whose chain never granted ACCEPT cannot be accepted at all. Either the
grant exists up front or the task cannot complete.

Classified **ABSENT**: there is no explicit, durably attributed override. The author's gap-4 override
shape would fit exactly —

```text
source = human_override / external_request
decision_id = none
governed_by_scheduler = false
```

— but inventing one unasked would re-open the hole this repair closed, so it is recorded for a
decision rather than implemented. `examples/applied_ai/ch28_scheduler.py` now teaches both sides:
two tasks identical but for one recorded capability, diverging at step 3.

## What this repair does not do

1. **No authentication.** `actor_id` is attribution. The record says which grant permitted the
   acceptance, never that the named actor is who they say they are.
2. **No revocation.** A grant cannot be narrowed after the fact; a second registration makes the
   chain ambiguous, which refuses everything under it.
3. **Task-level `Authority` is still carried and still unused** in `create_task`, as before.
4. **The self-acceptance rule is unchanged** — it remains a check on actor labels, which are not
   identities.
5. **No override path**, per the finding above.
