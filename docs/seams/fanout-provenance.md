# Seam: sealed fan-out final-input provenance (W2-1)

*Applied AI* Chapter 23 (blind before you compare) owns this concept; Chapter 29 composes it.

## The question it answers

> For each blind branch, can a later reader establish the exact bytes that branch was sent, and that
> no sibling's content was among them — from the record alone?

The gap in the chapter's own words: *"manifest hashes still stand in for final input bytes on the
`sealed_fanout` path itself."*

## What changed

`sealed_fanout` sealed every branch from its siblings at compile time and then built a `CallSpec`
with no renderer, so the binding that ties a prepared request to its rendered input was never
invoked on that path.

```text
selection record   no sibling was chosen         (before)
rendered bytes     no sibling was in the bytes   (with context_render)
```

Four parameters, all defaulting to today's behaviour:

| Parameter | What it decides |
|---|---|
| `context_render` | Whether branches render, and therefore whether the blindness claim rests on bytes |
| `base_artifact_lineage`, `base_claim_lineage` | Who produced the base material, which is how a seal can exclude it |
| `base_required_artifact_ids` | Which base material is demanded; the rest is offered and seal-filtered |

`fanout.requested` now records what the claim rests on: `context_render`, `input_provenance`
(`rendered_bytes` or `selection_record`), how much base material was offered, and how much of it
carried lineage.

**A branch whose adapter cannot `prepare()` fails as a branch.** It is never quietly downgraded to
the unrendered path, because a fan-out that cannot prove its inputs must say so rather than look as
though it proved them.

## Two findings from writing the tests

**1. The seal excludes by identity, so unattributed material is not excluded.** Handing the fan-out a
sibling's output without lineage puts that output in the other branch's bytes. This is Chapter 23's
admitted provenance hole, now demonstrated on this path and *recorded* rather than guessed at: the
fan-out event reports how many artifacts were offered and how many were attributed, so a reader can
see what the blindness claim rested on. The runtime does not invent lineage it was not given.

**2. A branch's compilation refusal was killing its siblings.** `sealed_fanout` promises that "one
branch failing never erases successful siblings", and the per-branch guard covered only the call —
compilation sat outside it. A seal refusal for one branch therefore propagated out of the whole
fan-out. Compilation is now inside the guard: that branch is recorded as failed with
`context compilation refused: …`, and its siblings run. This was a real defect in the primitive's
stated contract, found by a test written for a different property.

## Loud by default

The compiler treats explicitly passed material as *required* unless told otherwise, so a seal
exclusion raises rather than silently producing a smaller package. The fan-out preserves that: pass
a branch material its seal forbids and demand it, and that branch is refused. `base_required_
artifact_ids` is how a caller says "offer this, drop what the seal forbids" — quiet only when asked.

## Evidence

`tests/test_fanout_provenance.py` (11): each branch records its own rendered and prepared digests,
retrievable and hash-checked; the seal holds over both the rendered bytes and the body actually sent,
with shared base material still present so the seal is not merely excluding everything; unattributed
sibling material reaches the other branch and the record says how much was attributed; declared
lineage makes the seal bite; required-but-forbidden material refuses that branch while its sibling
completes; an adapter without `prepare()` fails its branch without being invoked; a prepared body
that drops the rendered input is refused as `render_binding` with nothing sent; rendering off is
byte-for-byte the old path; and a reopened ledger resolves every branch's inputs.

One case is worth keeping in view: with identical material, prompt and model, two branches produce
**byte-identical** prepared requests, and the record proves it. That is Chapter 23's point about
blind ≠ diverse, made mechanical — those branches differ only by sampling.

Full suite: **640 passed**.

## What it still does not establish

1. **Provenance completeness is the caller's, not the runtime's.** The compiler cannot distinguish
   material that never had lineage from material whose lineage was stripped. The record now says how
   much was attributed; it cannot say whether that was all of it.
2. **Semantic duplication is still invisible.** Copied sibling text inside a shared document matches
   no identifier a seal knows.
3. **Selection is not access enforcement.** An adapter that reads the filesystem, the repository or
   the ledger directly is outside this boundary entirely.
4. **Unknown provenance still has no policy.** Allow, exclude-under-seal and explicit UNKNOWN
   classification remain unimplemented, deliberately: a blanket rule would break ordinary
   compilation, and this seam does not flip a global default.
5. **Byte identity is not independence.** Proving two branches were sent different bytes says
   nothing about whether their errors are correlated — which is Chapter 24's measurement, not this
   seam's.
6. **Interrupted branch attempts still leave orphans.** A branch killed mid-attempt records
   `attempt.started` with no observation, and nothing reconciles it.
