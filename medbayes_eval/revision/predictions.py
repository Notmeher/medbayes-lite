"""Rebuild every method's predictive distribution from the logit caches (CPU).

Methods (encoders):
  base            softmax of the deterministic logits (MSP)
  temp_scaling    softmax(logits / T), T fit by NLL on the calibration slice (LBFGS)
  mc_dropout      mean of M MC-dropout softmaxes (no calibration)
  medbayes_lite   MC mean, then temperature-calibrated by powering p^(1/T) / Z with T fit
                  on log of the calibration MC mean (= temperature on averaged probabilities)
  conformal       split-conformal (LAC) confidence on the deterministic logits
  deep_ensemble   mean softmax over the per-seed fine-tuned checkpoints (PubMedQA only)
  *_srcT          MedQA only: temperature fit on the MedMCQA (source) calibration slice
Gates are applied later (fastmetrics.make_pred), so every method is scored under both.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..models import uq_methods as uq
from . import paths
from .fastmetrics import Pred, make_pred
from .items import load_items
from .jobs import load_npz


@dataclass
class Run:
    dataset: str
    model: str
    method: str
    seed: int
    uids: np.ndarray
    labels: np.ndarray
    probs: np.ndarray
    conf_override: Optional[np.ndarray] = None


def _fit_T(calib_det, calib_mc, labels):
    T_ts = uq.fit_temperature(calib_det, labels)
    T_mbl = uq.fit_temperature_on_probs(uq.softmax(calib_mc, -1).mean(0), labels)
    return T_ts, T_mbl


def _encoder_runs(meta, a, T_ts, T_mbl, suffix=""):
    ds, model, seed = meta["dataset"], meta["model"], meta["seed"]
    u, y = a["test_uids"], a["test_labels"]
    det, mc = a["test_det"], a["test_mc"]
    runs = []
    if not suffix:
        runs.append(Run(ds, model, "base", seed, u, y, uq.softmax(det)))
        runs.append(Run(ds, model, "mc_dropout", seed, u, y, uq.softmax(mc, -1).mean(0)))
        probs, conf, _ = uq.predict_conformal(det, a["calib_det"], a["calib_labels"])
        runs.append(Run(ds, model, "conformal", seed, u, y, probs, conf))
    runs.append(Run(ds, model, "temp_scaling" + suffix, seed, u, y, uq.softmax(det / T_ts)))
    runs.append(Run(ds, model, "medbayes_lite" + suffix, seed, u, y,
                    uq.predict_medbayes_lite(mc, T_mbl)[0]))
    return runs


def load_encoder_runs(source_ood=True):
    runs: List[Run] = []
    temps = []
    caches = {}
    for f in sorted(glob.glob(os.path.join(paths.LOGITS, "*.npz"))):
        meta, a = load_npz(f)
        caches[(meta["dataset"], meta["model"], meta["seed"])] = (meta, a)
    for (ds, model, seed), (meta, a) in caches.items():
        T_ts, T_mbl = _fit_T(a["calib_det"], a["calib_mc"], a["calib_labels"])
        temps.append({"dataset": ds, "model": model, "seed": seed, "calibrated_on": ds,
                      "T_temp_scaling": T_ts, "T_medbayes_lite": T_mbl,
                      "n_calib": len(a["calib_labels"]),
                      "objective": "NLL, LBFGS lr=0.01 max_iter=200; TS on logits, "
                                   "MBL on log of MC-averaged probabilities"})
        runs += _encoder_runs(meta, a, T_ts, T_mbl)
        src = caches.get(("medmcqa", model, seed))
        if source_ood and ds == "medqa" and src is not None:
            s_meta, s_a = src
            sT_ts, sT_mbl = _fit_T(s_a["calib_det"], s_a["calib_mc"], s_a["calib_labels"])
            temps.append({"dataset": ds, "model": model, "seed": seed,
                          "calibrated_on": "medmcqa", "T_temp_scaling": sT_ts,
                          "T_medbayes_lite": sT_mbl, "n_calib": len(s_a["calib_labels"])})
            runs += _encoder_runs(meta, a, sT_ts, sT_mbl, suffix="_srcT")
    # deep ensemble across seeds of locally fine-tuned checkpoints
    by = {}
    for (ds, model, seed), (meta, a) in caches.items():
        if meta.get("kind") == "finetuned":
            by.setdefault((ds, model), []).append(a)
    for (ds, model), arrs in by.items():
        if len(arrs) > 1:
            probs = np.mean([uq.softmax(a["test_det"]) for a in arrs], 0)
            runs.append(Run(ds, model, "deep_ensemble", -1, arrs[0]["test_uids"],
                            arrs[0]["test_labels"], probs))
    return runs, pd.DataFrame(temps)


def load_generative_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(paths.GEN, "*__qwen2_5_7b.parquet"))):
        df = pd.read_parquet(f)
        for (ds, method), sub in df.groupby(["dataset", "method"]):
            probs = np.array([np.array(eval_json(p)) for p in sub["probs"]])
            runs.append(Run(ds, "qwen2_5_7b", method, int(sub["seed"].iloc[0]),
                            sub["uid"].to_numpy(), sub["label"].to_numpy(), probs,
                            sub["confidence"].to_numpy(float)))
    return runs


def eval_json(s):
    import json
    return json.loads(s)


class Store:
    """Indexes runs and serves seed lists of aligned Pred objects per gate."""

    def __init__(self, runs: List[Run], ruleset: str = "default"):
        self.runs = runs
        self.ruleset = ruleset
        self._items: Dict[str, pd.DataFrame] = {}

    def items(self, dataset):
        """Test + calibration items (generative subsets can fall in either slice)."""
        if dataset not in self._items:
            df = load_items(dataset, role=None)
            self._items[dataset] = df[df.role != "train"].drop_duplicates("uid").set_index("uid")
        return self._items[dataset]

    def keys(self):
        return sorted({(r.dataset, r.model, r.method) for r in self.runs})

    def get(self, dataset, model, method) -> List[Run]:
        return sorted([r for r in self.runs if (r.dataset, r.model, r.method) ==
                       (dataset, model, method)], key=lambda r: r.seed)

    def severity(self, run: Run, sev_override=None):
        if sev_override is not None:
            return sev_override(run)
        col = "severity" if self.ruleset == "default" else "severity_strict"
        return self.items(run.dataset).loc[run.uids, col].to_numpy(float)

    def preds(self, dataset, model, method, gate="maxprob", sev_override=None) -> List[Pred]:
        out = []
        ref = None
        for r in self.get(dataset, model, method):
            if ref is None:
                ref = r.uids
            assert np.array_equal(ref, r.uids), "seed runs are not uid-aligned"
            conf = r.conf_override if gate == "maxprob" else None
            out.append(make_pred(r.probs, r.labels, self.severity(r, sev_override), gate, conf))
        return out


def load_store(include_generative=True, ruleset="default"):
    runs, temps = load_encoder_runs()
    if include_generative:
        runs += load_generative_runs()
    return Store(runs, ruleset), temps
