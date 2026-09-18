# Pre-registration: Wave 2

**Registered:** 2026-09-18, before any Wave 2 code was written.
**Subject at registration:** CodeAI `6296eb2` (main, Wave 1 complete: 12 ENFORCED, 2 DERIVED).

## How Wave 2 was chosen

Wave 1 was chosen from the manuscript's own admissions. Wave 2 is chosen by a narrower question,
asked after the book was updated with what Wave 1 taught:

> Which claims does *Applied AI* still make that have no concrete implementation or experiment
> behind them?

That is a smaller list than "everything CodeAI could do next", and deliberately so. The runtime is
not short of interesting engineering; the book is short of evidence in four specific places.

## The list, with what each would establish

| # | Item | Book claim it would support | Kind |
|---|---|---|---|
| 1 | **Sealed fan-out final-input provenance** | Ch 23: blindness is proven over the bytes each branch was actually sent, not over the selection record | runtime |
| 2 | **Acceptance names the action it accepts** | Ch 29: the effect→acceptance link is carried by a hash equality that happened to hold | runtime |
| 3 | **Decision-basis gating** | Ch 20 and 18: an action whose decision's claims have moved is not refused; the chapter calls it proposed | runtime |
| 4 | **The model-router challenger** | Ch 28: specified with its falsifier, unrun; the chapter says so and stays honest, but the experiment is still owed | experiment |

Resolver coverage (Ch 19, 21, 29) is deliberately **not** on this list. One hash per runtime is a
stated limit in three chapters, and widening it is configuration work that would change no book
claim.

## Order, and why

1. **Sealed fan-out** first. It is the item Wave 1 explicitly deferred, it is BOOK-CORE for Chapter
   23, and the machinery it needs already exists on the opt-in rendering path — this is threading a
   proof through a path that does not currently use it, not new invention.
2. **Acceptance → action** second. Small, and it closes the last conventional joint on the
   composition matrix that a repair could close.
3. **Decision-basis gating** third, and only if the first two land cleanly. It is the one item that
   could plausibly grow: "has this decision's basis moved?" is Chapter 18 machinery, and wiring it
   into the action path risks Chapter 20 absorbing Chapter 18, which Wave 1 refused twice.
4. **The router challenger** last, and separately. It is an experiment with live model calls, not a
   seam; it needs its own pre-registration, its frozen corpus, and the author's decision about spend.

## W2-1: sealed fan-out final-input provenance

**The gap, in the chapter's words:** *"manifest hashes still stand in for final input bytes on the
`sealed_fanout` path itself."*

`sealed_fanout` compiles a sealed context package per branch and builds a `CallSpec` with no
`context_render`. Rendering — and with it the binding that proves the prepared request carried
exactly the rendered model input — is therefore never invoked on that path. What the record proves
today is that each branch's *selection* excluded its siblings. What it does not prove is what bytes
each branch was sent.

**Governing question:**

> For each blind branch, can a later reader establish the exact bytes that branch was sent, and that
> no sibling's content was among them — from the record alone?

**Design intent (registered before implementation):**

- `sealed_fanout` takes a `context_render` version and threads it per branch, defaulting to `None`
  so every existing caller is unaffected.
- A branch whose adapter cannot `prepare()` is **refused as a branch failure**, never silently
  downgraded to the prompt-string path. A fan-out that cannot prove its inputs must say so.
- The existing `RenderBindingError` check is what does the proving: the prepared body must carry the
  composed rendered input, or the call fails with `render_binding` recorded.
- The seal is asserted **over bytes**: sibling text must be absent from every branch's rendered
  artifact and from every prepared body, not merely absent from the selection.

**Acceptance matrix, frozen here:**

| Case | Expected |
|---|---|
| Two branches, rendering on | Each records its own `rendered_context_sha256` and `request_body_sha256` |
| Sibling output offered to both | Each branch's rendered bytes contain its own material and not the sibling's |
| Shared base material | Present in both, so the seal is not merely excluding everything |
| Branch adapter without `prepare()` | That branch fails; its siblings still complete |
| Rendering off (default) | Unchanged behaviour, no rendered artifact, no new events |
| Reopen | Rendered artifacts and body hashes resolve from the reopened ledger |
| Prepared body altered to drop the rendered input | `render_binding` preparation failure, no provider effect |

**Predictions, registered:**

1. Threading the version through will be small; the machinery exists.
2. The seal will hold over bytes — the compiler already excluded siblings from the package, and
   rendering only resolves what the package selected.
3. The interesting failure, if there is one, will be at the *branch* boundary rather than the byte
   boundary: what a fan-out does when one branch cannot prove its inputs.

## What Wave 2 will not do

- No authentication, signing, sandboxing, concurrency or transactional atomicity. Unchanged from
  Wave 1: these stay named limits.
- No widening of the state resolver.
- No new composition audit. The Wave 1 harness is re-run as a regression after each item, and its
  frozen baseline is not edited.
- No promotion of the router challenger into a seam. It is an experiment or it is nothing.
