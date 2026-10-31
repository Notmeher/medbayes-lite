"""R3:5 - reproducibility manifest and leakage checks."""
from __future__ import annotations

import glob
import json
import os
import platform
import sys
from importlib import metadata

import pandas as pd

from .. import config
from . import paths
from .jobs import load_npz

PKGS = ["torch", "transformers", "datasets", "tokenizers", "huggingface-hub", "accelerate",
        "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "pyarrow"]

PROTOCOL = {
    "splits": "Calibration slice carved from the evaluation split with "
              "numpy default_rng(0) permutation (MedMCQA 500, MedQA 300, PubMedQA 150 items); "
              "remaining items are the test set. MedMCQA 'test' = HF validation split "
              "(official test labels are hidden). PubMedQA pqa_labeled (1000) -> 500 eval "
              "(rng 0) / 500 fine-tuning items.",
    "calibration": "Temperature fit by NLL minimization (LBFGS, lr=0.01, max_iter=200) on "
                   "the calibration slice only. Temperature scaling acts on logits; "
                   "MedBayes-Lite fits T on log of the MC-averaged probabilities and applies "
                   "it as p^(1/T)/Z (temperature on averaged probabilities). For MedQA the "
                   "*_srcT variants fit T on the MedMCQA calibration slice (true shift).",
    "fine_tuning": "PubMedQA encoders: AdamW lr 3e-5, wd 0.01, batch 16, 2 epochs, linear "
                   "warm-up 6%, fp16, max_len 512; fixed schedule - no early stopping and no "
                   "validation data used for model selection.",
    "thresholds": config.THRESHOLD_PROTOCOL,
    "mc_dropout": "M=15 passes, dropout p=0.2 on all nn.Dropout modules (LayerNorm in eval), "
                  "torch seed 10000+seed (test) / 11000+seed (calibration).",
    "gates": "maxprob: calibrated max probability >= tau; entropy: C(p)=1-H(p)/log K >= tau.",
    "bootstrap": "Item-level (uid) resampling, paired across methods; seeds averaged within "
                 "each draw; percentile 95% CI; two-sided p with +1 correction "
                 "(min p = 2/(B+1)); B=10000 for primary tests, 2000 otherwise; Holm within "
                 "comparison family, BH across all.",
    "pretrained_checkpoints_note": "RUI525 MedMCQA checkpoints are third-party; their "
                                   "training data are reported as the MedMCQA train split "
                                   "(the evaluation uses the validation split).",
}


def _versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PKGS:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            v[p] = None
    try:
        import torch
        v["cuda"] = torch.version.cuda
        v["cudnn"] = torch.backends.cudnn.version()
        v["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return v


def _resolved_revisions():
    out = {}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for hf_id in config.DATASET_REVISIONS:
            out[hf_id] = api.dataset_info(hf_id, revision=config.dataset_revision(hf_id)).sha
        for hf_id in config.MODEL_REVISIONS:
            out[hf_id] = api.model_info(hf_id, revision=config.model_revision(hf_id)).sha
    except Exception as e:  # offline
        out["error"] = repr(e)
    return out


def leakage_checks():
    from ..data.loaders import load_dataset_samples
    from .items import load_items
    rows = []
    norm = lambda s: " ".join(str(s).lower().split())
    for ds in ("medmcqa", "medqa", "pubmedqa"):
        try:
            it = load_items(ds, None)
        except FileNotFoundError:
            continue
        roles = {r: set(sub.uid) for r, sub in it.groupby("role")}
        test, cal = roles.get("test", set()), roles.get("calibration", set())
        rows.append({"dataset": ds, "check": "calibration ∩ test (uid)", "n": len(test & cal)})
        rows.append({"dataset": ds, "check": "train ∩ test (uid)",
                     "n": len(roles.get("train", set()) & test)})
        tq = it[it.role == "test"].question.map(norm)
        rows.append({"dataset": ds, "check": "duplicate question text within test",
                     "n": int(tq.duplicated().sum())})
        if ds in ("medmcqa", "medqa"):
            train = load_dataset_samples(ds, split="train")
            trq = {norm(s.question) for s in train}
            rows.append({"dataset": ds, "check": "test questions found verbatim in HF train split",
                         "n": int(tq.isin(trq).sum())})
    return pd.DataFrame(rows)


def write_manifest(temps: pd.DataFrame | None = None, run_leakage: bool = True) -> str:
    caches = []
    for f in sorted(glob.glob(os.path.join(paths.CACHE, "**", "*.npz"), recursive=True)):
        meta, _ = load_npz(f)
        meta.pop("texts", None)
        caches.append({"file": os.path.relpath(f, paths.ROOT), **meta})
    man = {"versions": _versions(), "pinned_dataset_revisions": config.DATASET_REVISIONS,
           "pinned_model_revisions": config.MODEL_REVISIONS,
           "resolved_revisions": _resolved_revisions(), "protocol": PROTOCOL,
           "seeds": sorted({c.get("seed") for c in caches if c.get("seed") is not None}),
           "caches": caches}
    p = paths.out("manifest", "reproducibility_manifest.json")
    with open(p, "w") as f:
        json.dump(man, f, indent=1, default=str)
    pd.DataFrame(caches).to_csv(paths.out("manifest", "cache_provenance.csv"), index=False)
    if temps is not None:
        temps.to_csv(paths.out("manifest", "fitted_temperatures.csv"), index=False)
    if run_leakage:
        leakage_checks().to_csv(paths.out("manifest", "leakage_checks.csv"), index=False)
    return p
