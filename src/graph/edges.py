"""Routing logic for the graph.

Phase 1 routing after Observe:
  passed                  -> END
  step_count >= max_steps -> END (budget exhausted)
  otherwise               -> diagnose (then repair, then execute again)

The Predict node and its early abort edge land in Phase 4, inserted between
Observe and this decision.
"""

from __future__ import annotations

from langgraph.graph import END

from graph.state import TrajectoryState


def route_after_observe(state: TrajectoryState) -> str:
    if state.get("status") == "passed":
        return END
    if state.get("step_count", 0) >= state.get("max_steps", 6):
        return END
    return "diagnose"


# Phase 4 abort edge. Tune from the predictor's precision/recall table:
# higher threshold = higher precision(doomed) = fewer wrongful aborts (protects pass rate).
ABORT_THRESHOLD = 0.90   # abort only when P(doomed) >= this
ABORT_MIN_STEP = 2       # never abort before this many repair steps (protect early recoverers)


def route_after_predict(state: TrajectoryState) -> str:
    if state.get("status") == "passed":
        return END
    if state.get("step_count", 0) >= state.get("max_steps", 6):
        return END
    if (
        state.get("abort_enabled")
        and state.get("step_count", 0) >= ABORT_MIN_STEP
        and state.get("failure_risk", 0.0) >= ABORT_THRESHOLD
    ):
        return "abort"
    return "diagnose"