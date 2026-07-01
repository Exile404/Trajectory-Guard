"""TrajectoryState: the single typed object that flows through every node.

LangGraph passes this dict node to node. Fields without a reducer are
overwritten on each update (last write wins). error_history and tokens_used
use an add reducer so a node returns just its delta and it accumulates.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class TrajectoryState(TypedDict, total=False):
    task_id: str                                       # benchmark task id, for logging
    task: str                                          # the issue or spec to solve
    test: str                                          # test code to run against the candidate
    plan: list[str]                                    # decomposed ordered steps
    code: str                                          # current code candidate
    test_output: str                                   # raw stdout and stderr from the run
    exec_result: dict                                  # execute->observe handoff: passed, returncode, timed_out, duration
    error_history: Annotated[list[str], operator.add]  # every traceback seen, accumulates
    failure_type: str                                  # diagnose label: syntax, import, logic, timeout, assertion
    step_count: int                                    # repair attempts used
    max_steps: int                                     # hard ceiling on repairs
    features: dict                                     # per step trajectory features for the predictor
    feature_log: Annotated[list[dict], operator.add]   # per-step features, accumulates for logging
    failure_risk: float                                # predictor score, 0 to 1
    tokens_used: Annotated[int, operator.add]          # cumulative model tokens, for H3 and features
    status: str                                        # running, passed, failed, aborted
    abort_enabled: bool 

def new_state(task: str, test: str = "", task_id: str = "task", max_steps: int = 6,
              abort_enabled: bool = False) -> TrajectoryState:
    """Fresh state for one task run: empty accumulators, status running."""
    return TrajectoryState(
        task_id=task_id,
        task=task,
        test=test,
        plan=[],
        code="",
        test_output="",
        exec_result={},
        error_history=[],
        failure_type="",
        step_count=0,
        max_steps=max_steps,
        features={},
        feature_log=[],
        failure_risk=0.0,
        tokens_used=0,
        status="running",
        abort_enabled=abort_enabled
    )


if __name__ == "__main__":
    s = new_state("write a function add(a, b) that returns a + b", task_id="demo", max_steps=6)
    print("keys:", sorted(s.keys()))
    print("status:", s["status"], "| max_steps:", s["max_steps"], "| step_count:", s["step_count"])