# Seam: recorded-directive authority resolution

*Applied AI* Chapters 20 (capability is not authority) and 29 (composition) own this concept.

## The question it answers

> When CodeAI permits an effect, can a later reader reconstruct exactly which durable grant
> authorized that exact effect, without trusting an `Authority` object supplied by the caller?

```text
caller says it has authority  !=  runtime establishes authority
```

## What it establishes

```text
action.requested
      -> resolve the grant from the recorded directive chain
action.authorized | action.authorization_refused     (basis attached)
      -> precondition check
action.execution_started
      -> the adapter acts
action.completed
```

- `Runtime.directive_authority(directive_id)` walks the recorded chain and intersects the grants, so
  a child can only narrow. It returns a `DirectiveAuthorityStanding`: effective capabilities, the
  chain (each link with the event it was read from), the basis event ids, and the reason.
- `Runtime.authorize_action(request)` decides one capability against that standing. Four outcomes are
  distinguished, because "refused" is not one thing: `GRANTED`, `DENIED`, `UNKNOWN_DIRECTIVE`,
  `INVALID_CHAIN` (missing parent, cycle, or a recorded child that widens its parent).
- `execute_action` decides **before** the replay lookup, so a caller without a current grant learns
  only the denial and never that an operation under that key exists. The replay, when it happens, is
  disclosed under **its own** authorization basis: the original effect and its later disclosure can
  rest on different grants, and both are recorded.
- The decision's durable basis carries the chain, not a boolean, so the answer can be reconstructed
  rather than re-asserted.
- Registration refusals are durable: `directive.registration_requested` is appended first, then
  either `directive.opened` or `directive.registration_refused` carrying the requested capabilities,
  the parent's capabilities, the reason and the basis events. A later reader can tell "nobody
  attempted this" from "someone attempted this and was refused".

An action that names no directive falls back to the caller-supplied grant and is recorded as
`caller_supplied`. That is a named limitation, not a hidden default: name a directive and the runtime
stops trusting the caller.

## Code

- `src/codeai/authority.py` — `DirectiveAuthorityStanding`, `GrantLink`, `AuthorizationDecision`,
  `resolve_directive_authority`, `authorize`, `standing_from_caller`.
- `src/codeai/runtime.py` — `Runtime.directive_authority`, `Runtime.authorize_action`;
  `execute_action` authorizes from the record before disclosure; `open_directive` appends
  `directive.registration_requested` and `directive.registration_refused`.
- `src/codeai/actions.py` — the projection learns the `authorized` and `authorization_refused`
  stages: a crash after authorization but before execution leaves effect NONE, which is safe to start.

## Tests

`tests/test_authority_resolution.py` (14) covers the matrix: root grant permits; child narrowing
refuses what the child gave up; unknown directive refused durably; valid grandchild chain permits with
a three-event basis; **a forged caller grant changes nothing**; no-directive falls back and says so;
widened child refused at registration with evidence; an unauthorized caller learns nothing about the
recorded operation; a replay is disclosed under its own basis; crash-after-authorization is effect
NONE; and an independent projection rebuilds effective authority from ledger events alone, including
a widening chain and a cycle.

Existing expectations updated, each because the behaviour genuinely changed: `test_runtime.py`'s
fixture now records the directive its actions name, its denial test now denies through the record
while the caller passes a broader grant, and the registration tests now assert durable refusals
instead of "appends nothing". Full suite: 491 passed.

## Example

`examples/applied_ai/ch20_authority.py`, asserted by `tests/test_examples.py`.

## Repair W1-R2: acceptance resolves the same way

The Wave 1 composition audit (gap 2) found the other half of the same question answered by a
different standard: `accept_task` took an `Authority` object from its caller.

```text
recorded directive -> action authority      ENFORCED   (this seam)
recorded directive -> acceptance authority  ENFORCED   (W1-R2)
```

The directive is read from `task.created`, never from the acceptance request: a caller who could name
the directive could name a permissive one. `accept_task(authority=...)` keeps its parameter, never
consults it, and records it as `caller_claimed_capabilities`. Both `task.accepted` and
`task.acceptance_rejected` carry the `directive_id` and the resolved basis, so an acceptance can be
re-resolved later rather than merely re-read.

Refusals distinguish *not granted* from *unresolvable*, and an unresolvable chain never falls back to
the caller. Details and the full case matrix: `experiments/W1-R2-authority-symmetry.md`.

**Consequence, recorded rather than smoothed over:** a task whose chain never granted ACCEPT cannot
be accepted by anyone through this API. The scheduler's ASK_HUMAN on such a task names a gate no call
can pass. There is no explicit override path, by design so far.

## What it still does not establish

1. **No authentication.** `requested_by` and `actor_id` remain caller-supplied strings. A recorded
   actor is an in-process principal, not an identity.
2. **No tamper evidence.** The ledger is unsigned and single-writer; anyone who can write the file
   can write a `directive.opened` event.
3. **Capabilities are still coarse.** A grant names an operation class, not a path, a repository, a
   destination or an expiry.
4. **No containment.** The adapter runs with whatever the process can reach.
5. **Directive-less actions still trust the caller**, by design and by record.
6. **Task authority is not resolved.** `create_task` still carries its own authority field and it is
   still unused. Acceptance no longer relies on it (see below); only `create_task` itself does.
7. **Decision-basis gating is deliberately absent.** Refusing an action whose Chapter 18 decision has
   moved is freshness, not authority. It stays in the ledger as separate work so Chapter 20 does not
   absorb Chapter 18.
