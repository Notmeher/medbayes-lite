"""Sensitivity sweeps requiring GPU inference (MedMCQA / PubMedBERT, pretrained):
  - MC-sample sweep M in {5,10,15,20,50}  (one MC=50 pass, subsampled)
  - dropout-rate sweep p in {0.1,0.2,0.3,0.5}  (MC=15 each)
Reports the MC-dropout predictor's ECE / harmful@0.8 / AURC / coverage@0.7 / selective-acc@0.7.
Saves CSVs under results/metrics/. No results in the main paper are changed.
"""
import os
import numpy as np
import pandas as pd

from medbayes_eval import config
from medbayes_eval.pipeline import get_train_eval
from medbayes_eval.models.encoder import EncoderModel
from medbayes_eval.models import uq_methods as uq
from medbayes_eval.buckets.rules import assign_buckets, sample_severity
from medbayes_eval.metrics.core import (expected_calibration_error,
    harmful_overconfident_error_rate, aurc, selective_at_threshold)

MET = os.path.join(config.RESULTS_DIR, "metrics")
DATASET, MODEL = "medmcqa", "pubmedbert_medmcqa"

train, calib, test = get_train_eval(DATASET)
labels = np.array([s.label for s in test])
sev = np.array([sample_severity(assign_buckets(s)) for s in test])
spec = config.MODELS[MODEL]
model = EncoderModel(spec.hf_id, mcq=True, device="cuda", max_len=320)


def metrics_from_probs(probs):
    pred = probs.argmax(1); correct = (pred == labels); conf = probs.max(1)
    cov, sa, _, _ = selective_at_threshold(conf, correct, sev, 0.7)
    return dict(accuracy=round(float(correct.mean()), 4),
                ece=round(expected_calibration_error(conf, correct), 4),
                harmful=round(harmful_overconfident_error_rate(conf, correct, sev, 0.8), 4),
                aurc=round(aurc(conf, correct), 4),
                cov07=round(cov, 4), selacc07=round(sa if sa == sa else float("nan"), 4))


# ---- MC-sample sweep (one MC=50 pass, subsample first M) ----
print("[sens] MC=50 pass ...", flush=True)
mc50 = model.mc_logits(test, 50, mc_p=0.2)
probs_s = uq.softmax(mc50, axis=-1)
rowsM = [{"M": M, **metrics_from_probs(probs_s[:M].mean(0))} for M in [5, 10, 15, 20, 50]]
pd.DataFrame(rowsM).to_csv(os.path.join(MET, "sens_mc_samples.csv"), index=False)
print("[sens] MC-sample sweep:\n", pd.DataFrame(rowsM).to_string(index=False), flush=True)

# ---- dropout-rate sweep (MC=15 each p) ----
rowsP = []
for p in [0.1, 0.2, 0.3, 0.5]:
    print(f"[sens] dropout p={p} ...", flush=True)
    mcp = uq.softmax(model.mc_logits(test, 15, mc_p=p), axis=-1).mean(0)
    rowsP.append({"dropout_p": p, **metrics_from_probs(mcp)})
pd.DataFrame(rowsP).to_csv(os.path.join(MET, "sens_dropout_rate.csv"), index=False)
print("[sens] dropout-rate sweep:\n", pd.DataFrame(rowsP).to_string(index=False), flush=True)
print("[sens] GPU sweeps DONE", flush=True)
