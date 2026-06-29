# Self-Healing Code Agent

> An autonomous LangGraph agent that plans, writes, executes, and repairs code, guided by a failure predictor trained with LoRA that aborts doomed trajectories before they burn compute.

Codename: `trajectory-guard`

## Problem

LLM coding agents fail silently. They loop, hallucinate fixes, and waste tokens chasing solutions that were never reachable. Two gaps: no structured self repair, and no early warning. This project closes both. A LangGraph state machine drives a plan, generate, execute, diagnose, repair loop. A classifier trained on the agent's own runs predicts failure early, so the agent can abort before spending the full budget.

## Architecture

A single typed state object flows through every node.

**Nodes:** plan, generate, execute, observe, extract features, predict, diagnose, repair.

**Loop:** `execute` runs the candidate in a sandbox. `observe` parses the result. `extract features` computes trajectory features. `predict` scores failure risk. If risk is high near the step budget, the agent aborts. Otherwise `diagnose` classifies the error and `repair` patches the code, then back to `execute`. The run exits on a pass, on budget exhaustion, or on a predictor abort.

The predict to abort edge is the research contribution. It is what separates this from a blind retry loop.

## The failure predictor

Given a partial trajectory (steps taken, errors seen, code deltas), the predictor estimates whether the agent will eventually pass. Trained locally with unsloth LoRA on a small open base, over trajectories harvested from the agent's own runs and labeled by final outcome. Cold start is handled by phase order: run the agent with no predictor first, log every trajectory, label, then train.

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| Tools and prompts | LangChain |
| Local model (dev) | qwen2.5-coder via Ollama |
| Hosted model (final) | GLM-5.1 via NVIDIA NIM, or AWS Bedrock |
| Predictor training | unsloth LoRA |
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
```

## Project status

- [x] Phase 0: scaffold, pinned deps, sandbox runner
- [x] Phase 1: LangGraph self repair loop on Ollama
- [x] Phase 2: trajectory feature logging to JSONL
- [ ] Phase 3: train the LoRA failure predictor
- [ ] Phase 4: wire the predict and abort edge
- [ ] Phase 5: scale to SWE-bench Lite
- [ ] Phase 6: hosted model swap (NIM, Bedrock)

## Hypotheses

- H1: predictor AUROC above 0.75 at a 3 step lead time.
- H2: the repair loop lifts pass rate over a baseline with no repair.
- H3: early abort cuts tokens per failed task without hurting pass rate.
