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