# Self-Healing Code Agent

> An autonomous LangGraph agent that plans, writes, executes, and repairs code, guided by a learned failure predictor that aborts doomed trajectories before they burn compute — cutting ~27% of tokens at a 0.2% pass-rate cost.

Codename: `trajectory-guard`

## Problem

LLM coding agents fail silently. They loop, hallucinate fixes, and waste tokens chasing solutions that were never reachable. Two gaps: no structured self repair, and no early warning. This project closes both. A LangGraph state machine drives a plan, generate, execute, diagnose, repair loop. A classifier trained on the agent's own runs predicts failure early, so the agent can abort before spending the full budget.

## Architecture

A single typed state object flows through every node.

**Nodes:** plan, generate, execute, observe, extract features, predict, diagnose, repair.

**Loop:** `execute` runs the candidate in a sandbox. `observe` parses the result. `extract features` computes trajectory features. `predict` scores failure risk. If risk is high near the step budget, the agent aborts. Otherwise `diagnose` classifies the error and `repair` patches the code, then back to `execute`. The run exits on a pass, on budget exhaustion, or on a predictor abort.

The predict to abort edge is the research contribution. It is what separates this from a blind retry loop.

## The failure predictor

Given a partial trajectory (steps taken, errors seen, repeated failures), the predictor estimates whether the agent will eventually pass. Trajectories are harvested from the agent's own runs and labeled by final outcome — cold start is handled by phase order: run with no predictor first, log every trajectory, label, then train.

The **shipped predictor is a 14-feature logistic regression** over engineered trajectory-dynamics features (error repetition, error-count growth, token growth, failure type). It reaches **0.806 early AUROC** (5-fold, task-split, leak-free).

An honest ablation: a **QLoRA-fine-tuned code LLM** (Qwen2.5-Coder, trained with unsloth) was tested as an alternative across three input regimes (snapshot / +task / +full history) and two model sizes (7B, 14B). It never beat the logistic regression — richer input did not help and 2x model scale made it *worse* — a data-limited regime where engineered features win. See [Results](#results).

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| Tools and prompts | LangChain |
| Local model (dev) | qwen2.5-coder via Ollama |
| Hosted models (eval) | Claude Opus 4.6 via AWS Bedrock (frontier lane, 7/24 SWE-bench); DeepSeek-V4-Pro + 3 others A/B-tested via NVIDIA NIM |
| Predictor | scikit-learn logistic regression (shipped) + unsloth LoRA ablation |
| Sandbox | subprocess with timeout and resource limits |
| Datasets | HumanEval, MBPP, SWE-bench Lite |

## Repository layout

```
src/
  graph/      state, nodes, edges, build (the LangGraph machine)
  tools/      sandbox, test runner, repo checkout/search tools
  models/     provider (ollama, nim, bedrock switch), predictor
  features/   per step trajectory features
  eval/       benchmark runner, SWE-bench patch agent + report summary
predictor/    logged trajectories, LoRA training
results/      tables, figures
```

## Setup

```bash
python3.11 -m venv trajectory-env
source trajectory-env/bin/activate
pip install -e .
ollama pull qwen2.5-coder:14b
cp .env.example .env
```

On Blackwell GPUs (RTX 50 series) install torch from the matching CUDA index before `pip install -e .`.

## Usage

```bash
# run the agent loop on a demo task
python -m graph.build

# run on HumanEval and log trajectories to predictor/data
python -m eval.run_benchmark --limit 5

# build the failure predictor from logged trajectories
python predictor/train_predictor.py

# reproduce the abort-policy measurement (H3)
python predictor/measure_abort.py

# SWE-bench Lite: generate patches (hosted via NIM), score with the official harness
python -m eval.swebench_agent --repos pallets/flask,psf/requests --limit 24 --use-gold-file \
  --provider nim --model deepseek-ai/deepseek-v4-pro --max-file-chars 60000 \
  --out swebench_preds/run.jsonl
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path swebench_preds/run.jsonl --max_workers 4 --run_id my-run --cache_level env
python -m eval.ab_summary v2:deepseek v3:nemotron    # summarize harness reports

# Phase 6 — Bedrock frontier lane + cross-model generalization
python -m eval.swebench_agent --provider bedrock --model au.anthropic.claude-opus-4-6-v1 \
  --repos pallets/flask,psf/requests,pydata/xarray,mwaskom/seaborn,pylint-dev/pylint \
  --limit 24 --use-gold-file --max-file-chars 60000 --out swebench_preds/opus.jsonl
python predictor/measure_transfer.py predictor/data/mbpp_nova.jsonl   # predictor trained on qwen, tested on Nova
python predictor/measure_abort.py --pricing opus                      # abort savings in dollars
```

## Demo

Run the agent on your own task + test with [`graph/solve.py`](src/graph/solve.py) — it writes code, repairs on failure, and (with `--abort`) quits when the predictor judges the run doomed:

```bash
python -m graph.solve "Write is_prime(n) returning True if n is prime" \
  --provider ollama --abort \
  --test "assert is_prime(17) and not is_prime(15)"
```

A healthy run keeps its doom risk near zero and passes; a doomed run (here an unsatisfiable test) shows the risk climb step by step until the agent aborts and saves the rest of its budget:

```
[predict]  doom risk = 0.09
   ...
[predict]  doom risk = 0.95
[abort]    predictor judged the run doomed -> stopping early
verdict  : ABORTED   steps: 6   tokens: 1350
```

Full walkthrough with both real transcripts (healthy pass + doomed abort): **[demo.md](demo.md)**.

## Safety

The agent executes model-generated code, so the sandbox ([src/tools/sandbox.py](src/tools/sandbox.py)) runs it in an isolated temp dir with CPU / memory / file-size limits and a process-group kill on timeout. A **SandboxGuard** additionally refuses to execute any generation containing destructive or system calls (`rmtree`, `os.remove`, `subprocess`, `shutil`, sockets, `rm -rf`), so a stray generation cannot touch your files — override with `TG_SANDBOX_GUARD=0`. This stops accidental damage; for genuinely untrusted tasks, run the process under **firejail** or **Docker** (rlimits and the guard are defense-in-depth, not an escape-proof jail). Before pushing, `bash scripts/check_before_push.sh` verifies no secrets, dataset, or model weights are tracked.

## Results

Harvested **3,556 trajectory steps across 1,128** HumanEval + MBPP tasks on `qwen2.5-coder:14b`.

**H1 — the predictor sees failure early.** Scored on in-progress steps only (predicting "already passed" is trivial and excluded), task-split so no task leaks across folds:

| model | early AUROC (step ≤ 2) | failing AUROC |
|---|---|---|
| **logistic regression (14 features)** | **0.806 ± 0.019** | 0.882 |
| QLoRA 7B — snapshot | 0.775 | — |
| QLoRA 7B — + task | 0.776 | 0.834 |
| QLoRA 7B — + full history | 0.765 | 0.839 |
| QLoRA 14B — + full history | 0.745 | 0.800 |

The fine-tuned LLM never beat the linear model: more input did not help, and doubling model scale *hurt* — textbook overfitting in a data-limited regime. The cheap, fast predictor is the right tool here, so it ships.

**H3 — early abort cuts compute without hurting pass rate.** Leak-free counterfactual replay (out-of-fold scoring) applying the abort policy to every logged trajectory:

| abort threshold | tokens saved | pass-rate change | wrongful aborts |
|---|---|---|---|
| 0.90 | **27.3%** | −0.2% (67.0 → 66.8%) | 2 / 1128 |
| 0.80 | 34.2% | −0.7% | 8 / 1128 |
| 0.70 | 38.0% | −0.9% | 10 / 1128 |

At the conservative threshold the agent cuts **~27% of tokens for a 0.2% pass-rate cost** — two tasks out of 1,128. The predictor pays for itself.

Reproduce: `python predictor/baseline.py` (H1), `python predictor/measure_abort.py` (H3).

**SWE-bench Lite (Phase 5) — hosted 4-model A/B.** A deliberately lean one-shot patcher (gold-file context, SEARCH/REPLACE edits, exact + whitespace-tolerant apply) scored by the **official SWE-bench Docker harness** — n=24 tasks across flask / requests / xarray / seaborn / pylint, reasoning-allowed prompt, fresh `run_id` per experiment:

| model (via NVIDIA NIM free tier) | resolved | valid patches | per-attempt rate |
|---|---|---|---|
| **DeepSeek-V4-Pro** | **5/24 (21%)** | 22/24 | 23% |
| **Nemotron-3-Ultra 550B** | **5/24 (21%)** | 19/24 | 26% |
| Kimi-K2.6 | 4/24 (17%) | 16/24 | 25% |
| Qwen3-Next-80B | 2/24 (8%) | 13/24 | 15% |

DeepSeek and Nemotron **tie**; per-attempt differences are within noise at n=24, so DeepSeek ships as the workhorse on the dimensions that are not noise: zero API errors across every run, the most format-valid patches, and ~6× less wall-clock than the 550B on a free endpoint. Findings worth defending:

- **A benchmark number = model × scaffold × prompt.** Allowing reasoning in the prompt flipped the ranking (DeepSeek went from last to tied-first). The 8,192-token output cap truncated Nemotron on 4 tasks (verified via `finish_reason`) — reasoning models pay scaffold taxes that leaderboard numbers hide.
- **Models solve different tasks.** The 4-model union resolves 8/24 (33%) vs any single model's 5/24 — diversity beats model choice at this scale.
- **The harness caches results by `run_id`** and silently skips already-run instances even if the patch changed. Every experiment gets a fresh `run_id`, or the numbers are fiction.

A local qwen2.5-coder:14b pilot resolved 0/3 — it cannot reliably emit exact-match edits (an Ollama context-truncation bug, since fixed, also contributed).

**SWE-bench Lite (Phase 6) — Bedrock frontier lane.** The same lean patcher, the same 24 instances, the same official harness, now on **Claude Opus 4.6 via AWS Bedrock** — one `PROVIDER` switch, zero graph changes (*same agent, three clouds*: local Ollama → NIM → Bedrock):

| model | resolved | valid patches | per-attempt rate | API errors |
|---|---|---|---|---|
| **Opus 4.6 (Bedrock)** | **7/24 (29%)** | **23/24** | **30%** | 0 |
| DeepSeek-V4-Pro (best of 4 on NIM) | 5/24 (21%) | 22/24 | 23% | 0 |

Opus tops every axis, and **one frontier model alone (7) nearly matches the entire 4-model NIM union (8/24)**. Bedrock also erased the free-tier failure modes that dogged NIM — zero truncation, zero rate-limit storms, zero errors across 24 tasks, the reliability you pay for. The caveat is unchanged: this is the lean one-shot patcher with gold-file localization ("fix quality given the right file"), not the 70%+ that heavy agentic scaffolds reach. (Opus **4.8/4.7** were account-gated on a fresh Bedrock account — AWS Sales qualification — so 4.6 stands in at the same $5/$25 frontier tier.)

**Generalization — does the predictor transfer across models?** Trained *only* on local `qwen2.5-coder:14b` trajectories, the failure predictor scores **Amazon Nova's** doomed MBPP runs at **0.926 AUROC, fully task-disjoint** (every one of the 150 test tasks removed from training; overlap-vs-disjoint AUROC differ by 0.001, so this is model-transfer, not task-recognition). Replayed on the real Nova run it catches **52/52 doomed, zero false aborts, −0% pass rate**, saving 26.6% of tokens at threshold 0.90. The predictor reads trajectory *dynamics*, not one model's fingerprints.

**Cost-aware abort.** Priced through Bedrock's per-token rates, the ~27%-token policy scales with the cost of the model behind it: **$0.05 at Nova rates on a $0.19 benchmark, but $4.87 of a $17.83 run at Opus 4.8 rates.** The more capable the model, the more a "know when to quit" predictor is worth.

## Project status

- [x] Phase 0: scaffold, pinned deps, sandbox runner
- [x] Phase 1: LangGraph self repair loop on Ollama
- [x] Phase 2: trajectory feature logging to JSONL
- [x] Phase 3: failure predictor (feature logreg 0.806 AUROC; LoRA ablation)
- [x] Phase 4: predict and abort edge (H3: ~27% tokens saved)
- [x] Phase 5: SWE-bench Lite via the official harness + hosted 4-model A/B (NIM)
- [x] Phase 6: AWS Bedrock — Opus 4.6 frontier lane (7/24, best single model), cross-model generalization (0.926 AUROC), cost-aware abort

## Hypotheses

- **H1 — met.** Predictor early AUROC **0.806** (target > 0.75), cross-validated and leak-free.
- **H2 — supported.** The repair loop recovers runs that failed on the first attempt, lifting final pass rate over a no-repair baseline.
- **H3 — met.** Early abort cuts **~27% of tokens at a 0.2% pass-rate cost**.
