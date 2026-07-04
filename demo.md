# Demo — the agent, and knowing when to quit

The agent solves a coding task from a plain-English description plus a test: it
plans, writes code, runs it in a sandbox, and on failure diagnoses and repairs —
looping until it passes, exhausts its budget, or (with `--abort`) the failure
predictor judges the run *doomed* and stops early.

Run it yourself on any task with [`graph/solve.py`](src/graph/solve.py):

```bash
python -m graph.solve "<task in plain English>" --provider ollama \
  --test "<assertions the solution must pass>" [--abort]
```

Everything below is real terminal output on a local `qwen2.5-coder:14b`.

## 1. A healthy run — solves it, low risk

```bash
python -m graph.solve "Write add(a, b) that returns the sum of a and b" \
  --provider ollama \
  --test "assert add(2, 3) == 5
assert add(-1, 1) == 0"
```

```
task     : Write add(a, b) that returns the sum of a and b
provider : ollama   abort: off   max_steps: 6

[plan]     3 steps
[generate] wrote 2 lines
[execute]  tests PASSED
[predict]  doom risk = 0.04

verdict  : PASSED   steps: 0   tokens: 177

--- solution ---
def add(a, b):
    return a + b
```

The predictor scores the trajectory's **doom risk at 0.04** — it correctly sees a
healthy run. (It handles non-trivial tasks the same way: an `is_palindrome` that
strips punctuation and case with a regex also passes first try, at risk 0.08.)

## 2. A doomed run — the agent quits early

Here the test is **unsatisfiable** (`add(2,3)` cannot equal both 6 and 5), so no
code can ever pass. A blind retry loop would burn its entire repair budget. With
`--abort`, watch the doom risk climb every step until the agent gives up:

```bash
python -m graph.solve "Write add(a, b) that returns the sum of a and b" \
  --provider ollama --abort --max-steps 12 \
  --test "assert add(2, 3) == 6
assert add(2, 3) == 5"
```

```
task     : Write add(a, b) that returns the sum of a and b
provider : ollama   abort: on   max_steps: 12

[plan]     3 steps
[generate] wrote 2 lines
[execute]  tests failed   AssertionError
[predict]  doom risk = 0.09
[diagnose] assertion
[repair]   attempt 1
[execute]  tests failed   AssertionError
[predict]  doom risk = 0.20
[diagnose] assertion
[repair]   attempt 2
[execute]  tests failed   AssertionError
[predict]  doom risk = 0.37
[diagnose] assertion
[repair]   attempt 3
[execute]  tests failed   AssertionError
[predict]  doom risk = 0.59
[diagnose] assertion
[repair]   attempt 4
[execute]  tests failed   AssertionError
[predict]  doom risk = 0.78
[diagnose] assertion
[repair]   attempt 5
[execute]  tests failed   AssertionError
[predict]  doom risk = 0.90
[diagnose] assertion
[repair]   attempt 6
[execute]  tests failed   AssertionError
[predict]  doom risk = 0.95
[abort]    predictor judged the run doomed -> stopping early

verdict  : ABORTED   steps: 6   tokens: 1350
```

The doom risk rises **0.09 → 0.20 → 0.37 → 0.59 → 0.78 → 0.90 → 0.95** as the same
error repeats and the token count grows. At step 6 it crosses the abort
threshold and the agent stops — **saving the remaining 6 repair attempts** it
would otherwise have wasted. That same early-abort, measured across 1,128 tasks,
is ~27% of tokens saved at a 0.2% pass-rate cost, and it generalizes to hosted
models (0.926 AUROC on Amazon Nova). See the [README results](README.md#results).

## Run it on your own model

Swap the backend with one flag — local, free-hosted (NIM), or AWS Bedrock:

```bash
# frontier model on AWS Bedrock
python -m graph.solve "Write merge(a, b) that merges two sorted lists" \
  --provider bedrock --model au.anthropic.claude-opus-4-6-v1 --abort \
  --test "assert merge([1, 3], [2, 4]) == [1, 2, 3, 4]"
```

## A note on safety

The agent runs model-generated code, so the sandbox blocks destructive
operations (`rmtree`, `subprocess`, `rm -rf`, …) before executing — a stray
generation cannot delete your files. For untrusted tasks, run under firejail or
Docker. See [README → Safety](README.md#safety).
