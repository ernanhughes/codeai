"""Chapter 21: four verification outcomes, and three different repairs.

Run it:

    python examples/applied_ai/ch21_verification.py

One file, one declared check, four situations:

    PASS          the marker is gone, and the check can see that
    FAIL          the marker is back, and the check can see that
    INCONCLUSIVE  the file exists, but the check cannot read it as either
    ERROR         the target changed before the check ran, so nothing was
                  measured against the state that was requested

The exit-code mapping is declared before the command runs and recorded with the
result, so "2 means it could not tell" is a statement about this command rather
than a convention smuggled into every command:

    pass 0   fail 1   inconclusive 2   anything else -> ERROR

The reduction a chapter can print:

    binding = bind(target)
    if not binding.permits_verification:
        return ERROR                  # nothing was measured
    result = verifier.run(request)

    FAIL         -> investigate the work
    INCONCLUSIVE -> gather better evidence
    ERROR        -> repair the measurement

A PASS here establishes exactly one thing: this declared command, against this
bound target, exited 0. Not that the criterion was the right one.
"""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from codeai.adapters import CheckRequest, CheckVerdict
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime
from codeai.verifier import LocalCommandVerifier

# 0 the marker is gone, 1 the marker is present, 2 the file cannot be read as
# either. The command says which of the three it found; it never guesses.
SCRIPT = """
import sys, pathlib
text = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
if '\\x00' in text:
    sys.stderr.write('binary content: this check cannot read it as prose\\n')
    sys.exit(2)
sys.exit(1 if 'TODO(marker)' in text else 0)
"""

POLICY = {
    "policy_id": "exit-code-v1",
    "pass_codes": [0],
    "fail_codes": [1],
    "inconclusive_codes": [2],
}

RESPONSE = {
    "PASS": "accept the work",
    "FAIL": "investigate the work",
    "INCONCLUSIVE": "gather better evidence",
    "ERROR": "repair the measurement",
}


def main() -> dict[str, object]:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / "paragraph.txt"
        ledger = SQLiteLedger(root / "ledger.sqlite")

        # The runtime owns the reading of the target state. The verifier is
        # never asked what it was looking at.
        def observe() -> str:
            return sha256(target.read_bytes()).hexdigest()

        runtime = Runtime(ledger, state_resolver=observe)
        verifier = LocalCommandVerifier()  # the mapping comes from the request

        def check(check_id: str, *, requested_hash: str | None) -> object:
            return runtime.run_check(
                CheckRequest(
                    check_id=check_id,
                    task_id="repair",
                    command=(sys.executable, "-c", SCRIPT, str(target)),
                    cwd=str(root),
                    target_state_hash=requested_hash,
                    verdict_policy=POLICY,
                ),
                verifier=verifier,
            )

        target.write_text("The cache makes page loads faster.\n", encoding="utf-8")
        passing = check("k-pass", requested_hash=observe())

        target.write_text("TODO(marker) rewrite this paragraph.\n", encoding="utf-8")
        failing = check("k-fail", requested_hash=observe())

        target.write_bytes(b"\x00\x01binary blob")
        inconclusive = check("k-maybe", requested_hash=observe())

        # The request pins a state; the file moves before the check runs.
        stale = observe()
        target.write_text("something else entirely\n", encoding="utf-8")
        errored = check("k-stale", requested_hash=stale)

        for label, result in (
            ("PASS expected     ", passing),
            ("FAIL expected     ", failing),
            ("cannot tell       ", inconclusive),
            ("target moved      ", errored),
        ):
            detail = result.inconclusive_reason or result.error or f"exit {result.exit_code}"
            print(
                f"{label}: {str(result.verdict):12} binding={result.binding_status:11} "
                f"-> {RESPONSE[str(result.verdict)]}"
            )
            print(f"                    {detail}")

        # The distinction that matters: one of these measured nothing.
        print()
        print(f"verifier ran for the inconclusive check : {inconclusive.exit_code == 2}")
        print(f"verifier ran for the errored check      : {errored.exit_code is not None}")

        completed = {
            event.stream_id: event.payload
            for event in ledger.events_by_kind(("check.completed",))
        }
        maybe = completed["k-maybe"]
        print(
            f"recorded policy for k-maybe             : "
            f"{maybe['declared_verdict_policy']['policy_id']} "
            f"inconclusive={maybe['declared_verdict_policy']['inconclusive_codes']}"
        )

        summary = {
            "verdicts": [
                str(passing.verdict),
                str(failing.verdict),
                str(inconclusive.verdict),
                str(errored.verdict),
            ],
            "bindings": [
                passing.binding_status,
                failing.binding_status,
                inconclusive.binding_status,
                errored.binding_status,
            ],
            "inconclusive_reason": inconclusive.inconclusive_reason,
            "error_reason": errored.error,
            "errored_check_ran_nothing": errored.exit_code is None,
            "declared_policy": maybe["declared_verdict_policy"],
            "verifier_identity": maybe["verifier_identity"]["name"],
        }
        ledger.close()
        return summary


if __name__ == "__main__":
    main()
