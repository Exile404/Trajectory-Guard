"""Per step trajectory features for the failure predictor.

compute_features reads the current state and returns a flat dict describing
the trajectory so far. extract_features is the graph node wrapper: it stores
the features and appends them to feature_log so the runner can label every
step by the final outcome.

Lean v1 feature set. Enrichments (code diff size across attempts, failing line
number, traceback novelty) come later if the predictor underperforms.
"""

from __future__ import annotations

from graph.state import TrajectoryState


def compute_features(state: TrajectoryState) -> dict:
    errs = state.get("error_history", [])
    ex = state.get("exec_result", {})
    return {
        "step_count": state.get("step_count", 0),
        "num_errors": len(errs),
        "last_error_len": len(errs[-1]) if errs else 0,
        "error_repeated": int(len(errs) >= 2 and errs[-1] == errs[-2]),
        "code_len": len(state.get("code", "")),
        "tokens_used": state.get("tokens_used", 0),
        "timed_out": int(bool(ex.get("timed_out", False))),
        "test_output_len": len(state.get("test_output", "")),
        "failure_type": state.get("failure_type", ""),
    }


def extract_features(state: TrajectoryState) -> dict:
    f = compute_features(state)
    return {"features": f, "feature_log": [f]}


if __name__ == "__main__":
    from graph.state import new_state

    s = new_state("demo task", test="assert True")
    s["step_count"] = 2
    s["code"] = "def f(x):\n    return x"
    s["error_history"] = ["AssertionError", "AssertionError"]
    s["tokens_used"] = 540
    s["failure_type"] = "assertion"
    s["test_output"] = "AssertionError"
    print(compute_features(s))