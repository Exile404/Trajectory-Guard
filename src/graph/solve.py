"""Run the self-healing agent on YOUR OWN task + test — the user entry point.

Give a task in plain English and a test (assertions). The agent plans, writes
code, runs it in a sandbox, and on failure diagnoses and repairs — looping until
it passes, exhausts its budget, or (with --abort) the failure predictor judges
the run doomed and stops early. Prints a live trace and the final solution.

Examples:
  python -m graph.solve "Write add(a, b) that returns a + b" \
      --provider ollama --test "assert add(2, 3) == 5"

  # let the predictor abort a doomed run (unsatisfiable test)
  python -m graph.solve "Write add(a, b) that returns a + b" --abort \
      --provider ollama --test "assert add(2,3)==6
assert add(2,3)==5"

  # run it on a frontier model via AWS Bedrock
  python -m graph.solve "Write is_prime(n)" --provider bedrock \
      --model au.anthropic.claude-opus-4-6-v1 --test "assert is_prime(17)"
"""

from __future__ import annotations

import argparse
import os

from graph.build import build_graph
from graph.state import new_state


def _last_line(text: str, n: int = 90) -> str:
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return lines[-1][:n] if lines else ""


def main():
    ap = argparse.ArgumentParser(description="Run the self-healing agent on one task.")
    ap.add_argument("task", help="the coding task, in plain English")
    ap.add_argument("--test", default="", help="assertion(s) the solution must pass")
    ap.add_argument("--test-file", default=None, help="read the test from a file instead")
    ap.add_argument("--abort", action="store_true", help="let the predictor abort doomed runs")
    ap.add_argument("--max-steps", type=int, default=6, help="max repair attempts")
    ap.add_argument("--provider", default=None, help="ollama | nim | bedrock (overrides .env)")
    ap.add_argument("--model", default=None, help="override the model id")
    a = ap.parse_args()

    if a.provider:
        os.environ["PROVIDER"] = a.provider
    if a.model:
        prov = (a.provider or os.getenv("PROVIDER", "ollama")).lower()
        key = {"ollama": "OLLAMA_MODEL", "nim": "NIM_MODEL", "bedrock": "BEDROCK_MODEL_ID"}.get(prov)
        if key:
            os.environ[key] = a.model

    test = a.test
    if a.test_file:
        with open(a.test_file) as f:
            test = f.read()

    app = build_graph()
    state = new_state(a.task, test=test, task_id="solve", max_steps=a.max_steps,
                      abort_enabled=a.abort)

    print(f"task     : {a.task}")
    print(f"provider : {os.getenv('PROVIDER', 'ollama')}   abort: {'on' if a.abort else 'off'}"
          f"   max_steps: {a.max_steps}\n")

    total_tokens = 0
    final: dict = {}
    for chunk in app.stream(state, config={"recursion_limit": 50}):
        for node, upd in chunk.items():
            final.update(upd)
            total_tokens += upd.get("tokens_used", 0) or 0
            if node == "plan":
                print(f"[plan]     {len(upd.get('plan', []))} steps")
            elif node == "generate":
                print(f"[generate] wrote {len(upd.get('code', '').splitlines())} lines")
            elif node == "execute":
                ex = upd.get("exec_result", {})
                if ex.get("passed"):
                    print("[execute]  tests PASSED")
                else:
                    print(f"[execute]  tests failed   {_last_line(upd.get('test_output', ''))}")
            elif node == "predict":
                r = upd.get("failure_risk")
                if r is not None:
                    print(f"[predict]  doom risk = {r:.2f}")
            elif node == "diagnose":
                print(f"[diagnose] {upd.get('failure_type')}")
            elif node == "repair":
                print(f"[repair]   attempt {upd.get('step_count')}")
            elif node == "abort":
                print("[abort]    predictor judged the run doomed -> stopping early")

    status = final.get("status", "?")
    print(f"\nverdict  : {status.upper()}   steps: {final.get('step_count', 0)}"
          f"   tokens: {total_tokens}")
    if status == "passed":
        print("\n--- solution ---")
        print(final.get("code", ""))
    elif status == "aborted":
        print("\nThe predictor saw a doomed trajectory and quit early, saving the remaining "
              "repair budget instead of burning it.")
    else:
        print("\nRan out of repair attempts without passing.")


if __name__ == "__main__":
    main()