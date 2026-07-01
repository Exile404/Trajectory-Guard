"""Graph nodes: plan, generate, execute, observe, diagnose, repair.

Each node takes the TrajectoryState and returns a partial update dict that
LangGraph merges into state. The Predict node arrives in Phase 4.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import TrajectoryState
from models.predictor import get_predictor
from models.provider import get_llm
from tools.sandbox import run_with_tests

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _count_tokens(resp) -> int:
    meta = getattr(resp, "usage_metadata", None)
    if meta and meta.get("total_tokens"):
        return int(meta["total_tokens"])
    rmeta = getattr(resp, "response_metadata", {}) or {}
    return int(rmeta.get("prompt_eval_count", 0)) + int(rmeta.get("eval_count", 0))


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    return (m.group(1) if m else text).strip()


def plan(state: TrajectoryState) -> dict:
    llm = get_llm(temperature=0.0)
    sys = SystemMessage(content=(
        "You are a planning assistant for a coding agent. "
        "Break the task into 2 to 5 short, concrete steps. "
        "Return one step per line. No numbering, no prose."
    ))
    resp = llm.invoke([sys, HumanMessage(content=f"Task:\n{state['task']}")])
    steps = [ln.strip(" -*\t") for ln in resp.content.splitlines() if ln.strip()]
    return {"plan": steps[:5], "tokens_used": _count_tokens(resp)}


def generate(state: TrajectoryState) -> dict:
    llm = get_llm(temperature=0.0)
    sys = SystemMessage(content=(
        "You are a coding agent. Write correct, self contained Python. "
        "Return only the code in a single python block, no explanation."
    ))
    plan_txt = "\n".join(f"- {s}" for s in state.get("plan", []))
    msg = HumanMessage(content=f"Task:\n{state['task']}\n\nPlan:\n{plan_txt}\n\nWrite the solution.")
    resp = llm.invoke([sys, msg])
    return {"code": _extract_code(resp.content), "tokens_used": _count_tokens(resp)}


def execute(state: TrajectoryState) -> dict:
    result = run_with_tests(state.get("code", ""), state.get("test", ""), timeout=10.0)
    combined = (result.stdout + ("\n" + result.stderr if result.stderr else "")).strip()
    return {
        "test_output": combined,
        "exec_result": {
            "passed": result.passed,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration": result.duration,
        },
    }


def observe(state: TrajectoryState) -> dict:
    ex = state.get("exec_result", {})
    if ex.get("passed"):
        return {"status": "passed"}
    lines = state.get("test_output", "").strip().splitlines()
    last = lines[-1] if lines else "unknown error"
    return {"status": "failed", "error_history": [last]}


_DIAGNOSE_RULES = [
    ("timeout", lambda t, ex: ex.get("timed_out")),
    ("syntax", lambda t, ex: "SyntaxError" in t or "IndentationError" in t),
    ("import", lambda t, ex: "ImportError" in t or "ModuleNotFoundError" in t),
    ("assertion", lambda t, ex: "AssertionError" in t),
]


def diagnose(state: TrajectoryState) -> dict:
    t = state.get("test_output", "")
    ex = state.get("exec_result", {})
    for label, rule in _DIAGNOSE_RULES:
        if rule(t, ex):
            return {"failure_type": label}
    return {"failure_type": "logic"}


def repair(state: TrajectoryState) -> dict:
    llm = get_llm(temperature=0.3)
    errs = state.get("error_history", [])
    recent = errs[-3:]
    history = "\n".join(f"- {e[:300]}" for e in recent) or "none"
    sys = SystemMessage(content=(
        "You are a coding agent fixing failing code. "
        "Identify the root cause of the failure, then rewrite the FULL corrected function. "
        "Do not repeat any previous failed approach listed below. "
        "Return only the corrected code in a single python block, no explanation."
    ))
    msg = HumanMessage(content=(
        f"Task:\n{state['task']}\n\n"
        f"Current code:\n{state.get('code', '')}\n\n"
        f"Failure type: {state.get('failure_type', '')}\n"
        f"Recent errors (most recent last):\n{history}\n\n"
        f"Latest test output:\n{state.get('test_output', '')[:1500]}\n\n"
        "Return the corrected code."
    ))
    resp = llm.invoke([sys, msg])
    return {
        "code": _extract_code(resp.content),
        "step_count": state.get("step_count", 0) + 1,
        "tokens_used": _count_tokens(resp),
    }
def predict(state: TrajectoryState) -> dict:
    """Score P(doomed) for the current partial trajectory. Always runs so the
    risk is logged in both arms; the abort decision itself lives in the edge."""
    feats = state.get("features") or {}
    if not feats:
        return {}
    try:
        risk = get_predictor().score(feats)
    except FileNotFoundError:
        return {}  # no artifact -> predictor disabled, run continues normally
    return {"failure_risk": risk}


def abort(state: TrajectoryState) -> dict:
    """Terminal node: predictor judged this run doomed; stop early."""
    return {"status": "aborted"}


if __name__ == "__main__":
    from graph.state import new_state

    task = "Write a function add(a, b) that returns the sum of a and b."
    test = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('all tests passed')"
    s = new_state(task, test=test, task_id="demo")
    total = 0

    upd = plan(s);     total += upd.get("tokens_used", 0); s.update(upd); print("PLAN:", s["plan"])
    upd = generate(s); total += upd.get("tokens_used", 0); s.update(upd); print("CODE:\n", s["code"])
    s.update(execute(s)); print("EXEC passed:", s["exec_result"]["passed"])
    s.update(observe(s)); print("STATUS:", s["status"])
    if s["status"] != "passed":
        s.update(diagnose(s)); print("DIAGNOSE:", s["failure_type"])
        upd = repair(s); total += upd.get("tokens_used", 0); s.update(upd)
        s.update(execute(s)); s.update(observe(s)); print("AFTER REPAIR:", s["status"])
    print("total tokens:", total)