"""Assemble the LangGraph state machine.

Flow:
  plan -> generate -> execute -> observe -> [route]
  route: passed or budget exhausted -> END
         else -> diagnose -> repair -> execute (loop back)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.edges import route_after_observe
from graph.nodes import diagnose, execute, generate, observe, plan, repair
from graph.state import TrajectoryState
from features.extract import extract_features

def build_graph():
    g = StateGraph(TrajectoryState)

    g.add_node("plan", plan)
    g.add_node("generate", generate)
    g.add_node("execute", execute)
    g.add_node("observe", observe)
    g.add_node("extract_features", extract_features)
    g.add_node("diagnose", diagnose)
    g.add_node("repair", repair)

    g.set_entry_point("plan")
    g.add_edge("plan", "generate")
    g.add_edge("generate", "execute")
    g.add_edge("execute", "observe")
    g.add_edge("observe", "extract_features")
    g.add_conditional_edges("extract_features", route_after_observe, {"diagnose": "diagnose", END: END})
    g.add_edge("diagnose", "repair")
    g.add_edge("repair", "execute")

    return g.compile()


if __name__ == "__main__":
    from graph.state import new_state

    task = "Write a function is_prime(n) that returns True if n is a prime number, else False."
    test = (
        "assert is_prime(2) is True\n"
        "assert is_prime(15) is False\n"
        "assert is_prime(17) is True\n"
        "assert is_prime(1) is False\n"
        "print('all tests passed')"
    )
    app = build_graph()
    final = app.invoke(
        new_state(task, test=test, task_id="is_prime", max_steps=6),
        config={"recursion_limit": 50},
    )

    print("STATUS :", final["status"])
    print("STEPS  :", final["step_count"])
    print("TOKENS :", final["tokens_used"])
    print("ERRORS :", len(final.get("error_history", [])))
    print("FEATLOG:", len(final.get("feature_log", [])))
    print("CODE:\n", final["code"])