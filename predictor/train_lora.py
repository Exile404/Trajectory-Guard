"""Unsloth LoRA failure predictor (text input, v3: attempt history).

Reads the FULL repair history of a trajectory (task + every prior error + a
[code unchanged] marker when a repair changed nothing) and predicts whether the
agent will eventually pass. v1/v2 showed only the current step's snapshot and
lost to the feature baseline (0.775/0.776 vs 0.806) because the signal lives in
the cross-step dynamics (repeated errors, unchanged code) the snapshot hides.
Trained on in-progress (failing) steps, task-split, compared by AUROC.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from dataset import load_records

warnings.filterwarnings("ignore")

MAX_SEQ = 1536
BASE = "unsloth/Qwen2.5-Coder-14B-Instruct"
SEED = 42


def task_text_map():
    m = {}
    for it in load_dataset("openai/openai_humaneval")["test"]:
        m[it["task_id"]] = it["prompt"]
    mb = load_dataset("google-research-datasets/mbpp", "full")
    for split in ("train", "test", "validation"):
        for it in mb[split]:
            m[f"MBPP/{it['task_id']}"] = it["text"]
    return m


def index_by_task(recs):
    """task_id -> {step: record}, built from ALL records so history lookups
    can reach earlier in-progress steps of eventually-passing trajectories."""
    by = defaultdict(dict)
    for r in recs:
        by[r["task_id"]][int(r["step"])] = r
    return by


def _first_line(s, n=160):
    lines = (s or "").strip().splitlines()
    return (lines[0] if lines else "")[:n]


def build_prompt(r, by_task, tmap):
    task = (tmap.get(r["task_id"], "") or "")[:500]
    seq = by_task[r["task_id"]]
    k = int(r["step"])

    hist, prev_code = [], None
    for j in range(k + 1):
        rj = seq.get(j)
        if rj is None:
            continue
        codej = rj.get("raw_code") or ""
        errj = _first_line(rj.get("raw_last_error") or rj.get("raw_test_output") or "")
        same = " [code unchanged]" if prev_code is not None and codej == prev_code else ""
        hist.append(f"Attempt {j}{same}: {errj or 'no error text'}")
        prev_code = codej
    history = "\n".join(hist)

    code = (r.get("raw_code") or "")[:900]
    err = (r.get("raw_test_output") or r.get("raw_last_error") or "")[:400]
    return (
        "A coding agent is repairing a failing solution. Given the FULL attempt "
        "history, decide if it will EVENTUALLY pass. Repeated errors or "
        "[code unchanged] markers mean it is stuck.\n\n"
        f"Task:\n{task}\n\n"
        f"Attempt history (each line = one attempt and its error):\n{history}\n\n"
        f"Current code:\n{code}\n\n"
        f"Current error:\n{err}\n\n"
        f"Failure type: {r.get('failure_type', '')}. Attempts so far: {k}.\n\n"
        "Will it eventually pass? Answer PASS or FAIL.\nAnswer:"
    )


def main():
    tmap = task_text_map()
    all_recs = load_records()
    by_task = index_by_task(all_recs)

    recs = [r for r in all_recs if not (int(r["label"]) == 1 and r["step"] == r["traj_len"] - 1)]
    groups = np.array([r["task_id"] for r in recs])
    y = np.array([int(r["label"]) for r in recs])

    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(recs, y, groups))
    tr = [recs[i] for i in tr_idx]
    te = [recs[i] for i in te_idx]

    rec_tr = [r for r in tr if int(r["label"]) == 1]
    doom_tr = [r for r in tr if int(r["label"]) == 0]
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(doom_tr), size=min(len(doom_tr), 2 * len(rec_tr)), replace=False)
    tr_bal = rec_tr + [doom_tr[i] for i in keep]
    rng.shuffle(tr_bal)
    print(f"train balanced: {len(tr_bal)} (recover={len(rec_tr)}, doom={len(keep)})   test: {len(te)}")

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )

    eos = tokenizer.eos_token
    texts = [build_prompt(r, by_task, tmap) + (" PASS" if int(r["label"]) == 1 else " FAIL") + eos for r in tr_bal]
    ds = Dataset.from_dict({"text": texts})

    from trl import SFTConfig, SFTTrainer
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            warmup_steps=5, num_train_epochs=3, learning_rate=2e-4,
            logging_steps=10, optim="adamw_8bit", seed=SEED,
            output_dir="predictor/lora_out", report_to="none",
            max_seq_length=MAX_SEQ, dataset_text_field="text",
        ),
    )
    trainer.train()

    FastLanguageModel.for_inference(model)
    pass_id = tokenizer(" PASS", add_special_tokens=False).input_ids[0]
    fail_id = tokenizer(" FAIL", add_special_tokens=False).input_ids[0]

    scores, ytrue, steps = [], [], []
    for r in te:
        ids = tokenizer(build_prompt(r, by_task, tmap), return_tensors="pt",
                        truncation=True, max_length=MAX_SEQ).to("cuda")
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        p = torch.softmax(torch.stack([logits[fail_id], logits[pass_id]]), dim=0)
        scores.append(float(p[0]))
        ytrue.append(1 - int(r["label"]))
        steps.append(int(r["step"]))

    scores, ytrue, steps = np.array(scores), np.array(ytrue), np.array(steps)
    early = steps <= 2
    print(f"\nLoRA 14B AUROC failing={roc_auc_score(ytrue, scores):.3f}   "


          f"early(step<=2)={roc_auc_score(ytrue[early], scores[early]):.3f}   "
          f"(n_te={len(ytrue)}, n_early={int(early.sum())})")
    print("baseline: failing~0.882, early~0.806   |   7B: v1=0.775 v2=0.776 v3=0.765")


if __name__ == "__main__":
    main()