"""Execution API for the frozen router experiment.

Deliberately no automatic provider construction or executable live default.
After separate execution authorization, supply pinned ModelRouter instances to run.
See router-prereg.md and router-implementation.md. Synthetic tests use test fixtures,
never router_cases_v1.json as a source of experimental observations.
"""
from codeai.router_experiment import build_schedule, run

__all__ = ["build_schedule", "run"]

if __name__ == "__main__":
    raise SystemExit("NOT RUN. Finalize independent adjudication and freeze first; use the explicit run API only after authorization.")
