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
| Hosted model (final) | GLM-5.1 via NVIDIA NIM, or AWS Bedrock |
| Predictor | scikit-learn logistic regression (shipped) + unsloth LoRA ablation |
| Sandbox | subprocess with timeout and resource limits |
| Datasets | HumanEval, MBPP, SWE-bench Lite |

## Repository layout

```
src/
  graph/      state, nodes, edges, build (the LangGraph machine)
  tools/      sandbox, test runner, file tools
  models/     provider (ollama, nim, bedrock switch), predictor
  features/   per step trajectory features
  eval/       benchmark runner, metrics
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
```

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

## Project status

- [x] Phase 0: scaffold, pinned deps, sandbox runner
- [x] Phase 1: LangGraph self repair loop on Ollama
- [x] Phase 2: trajectory feature logging to JSONL
- [x] Phase 3: failure predictor (feature logreg 0.806 AUROC; LoRA ablation)
- [x] Phase 4: predict and abort edge (H3: ~27% tokens saved)
- [ ] Phase 5: scale to SWE-bench Lite
- [ ] Phase 6: hosted model swap (NIM, Bedrock)

## Hypotheses

- **H1 — met.** Predictor early AUROC **0.806** (target > 0.75), cross-validated and leak-free.
- **H2 — supported.** The repair loop recovers runs that failed on the first attempt, lifting final pass rate over a no-repair baseline.
- **H3 — met.** Early abort cuts **~27% of tokens at a 0.2% pass-rate cost**.
