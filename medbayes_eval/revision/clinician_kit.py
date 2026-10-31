"""R3:3 / R1:3 - kit for blinded clinician validation of the rule-based severity layer.

export_severity_rating_kit()  writes a blinded, stratified rating sheet (no model output,
                              no rule labels), a separate answer key, and instructions.
analyze_clinician_ratings()   once rated sheets come back: inter-rater agreement,
                              rule-vs-clinician agreement per category and for the
                              high-severity cutoff, and harmful@0.8 recomputed with
                              clinician labels on the rated subset.
"""
from __future__ import annotations

import glob
import json
import os
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from .. import config
from . import paths
from .items import load_items

SCALE = ("1 = negligible, 2 = minor, 3 = moderate, 4 = serious, 5 = critical "
         "(potential patient harm if a clinician acted on a WRONG answer to this item)")
CATS = [b for b in config.BUCKETS if b not in ("low_severity",)]

INSTRUCTIONS = f"""Clinician severity rating - instructions
==========================================

You will rate {{n}} exam-style clinical questions. Model predictions and any automatic
labels are deliberately hidden.

For each row:
  1. severity_if_wrong  - integer 1-5. {SCALE}
  2. For each category column (cat_*), enter 1 if the question clearly belongs to that
     category, otherwise leave blank or 0. Several categories may apply.
  3. comments - optional.

Rate independently; do not discuss items with other raters until all ratings are in.
Save the file as rater_<initials>.csv in the same folder.
"""


def export_severity_rating_kit(n_per_category: int = 15, n_low: int = 30,
                               datasets: Sequence[str] = ("medmcqa", "medqa"),
                               seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    frames = []
    for ds in datasets:
        it = load_items(ds, "test").copy()
        it["bl"] = it.buckets.map(json.loads)
        frames.append(it)
    it = pd.concat(frames, ignore_index=True)
    picked = set()
    for b in CATS:
        pool = it.index[it.bl.map(lambda x, b=b: b in x)].to_numpy()
        pool = np.setdiff1d(pool, list(picked))
        picked.update(rng.permutation(pool)[:n_per_category].tolist())
    low = np.setdiff1d(it.index[it.bl.map(lambda x: x == ["low_severity"])].to_numpy(),
                       list(picked))
    picked.update(rng.permutation(low)[:n_low].tolist())
    sel = it.loc[rng.permutation(sorted(picked))].reset_index(drop=True)
    sel["rating_id"] = [f"R{i:04d}" for i in range(len(sel))]
    opts = sel.options.map(lambda s: "\n".join(f"{chr(65 + i)}. {o}"
                                               for i, o in enumerate(json.loads(s))) if s else "")
    sheet = pd.DataFrame({"rating_id": sel.rating_id, "question": sel.question,
                          "options": opts, "severity_if_wrong": ""})
    for c in CATS:
        sheet[f"cat_{c}"] = ""
    sheet["comments"] = ""
    key = sel[["rating_id", "dataset", "uid", "buckets", "severity", "buckets_strict",
               "severity_strict", "triggers"]]
    d = os.path.dirname(paths.out("clinician_kit", "x"))
    out = {"sheet": os.path.join(d, "severity_rating_sheet_BLINDED.csv"),
           "key": os.path.join(d, "severity_rating_key_DO_NOT_SHARE.csv"),
           "instructions": os.path.join(d, "RATER_INSTRUCTIONS.txt")}
    sheet.to_csv(out["sheet"], index=False)
    key.to_csv(out["key"], index=False)
    with open(out["instructions"], "w") as f:
        f.write(INSTRUCTIONS.format(n=len(sheet)))
    return out


def _kappa(a, b, weights=None):
    from sklearn.metrics import cohen_kappa_score
    ok = ~(pd.isna(a) | pd.isna(b))
    return float(cohen_kappa_score(np.asarray(a)[ok], np.asarray(b)[ok], weights=weights)) \
        if ok.sum() > 1 else np.nan


def analyze_clinician_ratings(rater_glob: str | None = None, clinician_high: int = 4,
                              store=None) -> dict:
    """Agreement analysis for returned sheets (clinician_kit/rater_*.csv)."""
    from scipy.stats import spearmanr
    d = os.path.dirname(paths.out("clinician_kit", "x"))
    files = sorted(glob.glob(rater_glob or os.path.join(d, "rater_*.csv")))
    if not files:
        print("[clinician] no rater_*.csv files found - nothing to analyze yet.")
        return {}
    key = pd.read_csv(os.path.join(d, "severity_rating_key_DO_NOT_SHARE.csv"))
    raters = {os.path.basename(f)[6:-4]: pd.read_csv(f).set_index("rating_id") for f in files}
    sev = pd.DataFrame({r: df["severity_if_wrong"] for r, df in raters.items()})
    out = {}
    inter = []
    for r1, r2 in combinations(sev.columns, 2):
        inter.append({"rater_a": r1, "rater_b": r2,
                      "weighted_kappa_quadratic": _kappa(sev[r1], sev[r2], "quadratic"),
                      "kappa_high_severity": _kappa(sev[r1] >= clinician_high,
                                                    sev[r2] >= clinician_high)})
    out["inter_rater"] = pd.DataFrame(inter)
    consensus = sev.median(axis=1)
    k = key.set_index("rating_id").loc[consensus.index]
    rule_hi = k.severity >= config.HIGH_SEVERITY_CUTOFF
    clin_hi = consensus >= clinician_high
    tp = int((rule_hi & clin_hi).sum())
    rows = [{"comparison": "rule severity vs clinician consensus",
             "spearman_rho": float(spearmanr(k.severity, consensus, nan_policy="omit")[0]),
             "high_sev_kappa": _kappa(rule_hi, clin_hi),
             "high_sev_precision (rule PPV)": tp / max(rule_hi.sum(), 1),
             "high_sev_recall (rule sensitivity)": tp / max(clin_hi.sum(), 1),
             "rule_false_positive_rate": float((rule_hi & ~clin_hi).sum() / max((~clin_hi).sum(), 1)),
             "n_items": int(consensus.notna().sum())}]
    for cutoff in config.SEVERITY_CUTOFFS:
        rh = k.severity >= cutoff
        rows.append({"comparison": f"rule cutoff {cutoff} vs clinician >= {clinician_high}",
                     "high_sev_kappa": _kappa(rh, clin_hi),
                     "high_sev_precision (rule PPV)": float((rh & clin_hi).sum() / max(rh.sum(), 1)),
                     "high_sev_recall (rule sensitivity)": float((rh & clin_hi).sum() / max(clin_hi.sum(), 1))})
    out["rule_vs_clinician"] = pd.DataFrame(rows)
    cat_rows = []
    bl = k.buckets.map(json.loads)
    for c in CATS:
        votes = pd.DataFrame({r: pd.to_numeric(df.get(f"cat_{c}"), errors="coerce").fillna(0)
                              for r, df in raters.items()}).loc[k.index]
        clin = votes.mean(axis=1) >= 0.5
        rule = bl.map(lambda x, c=c: c in x)
        tp = int((rule & clin).sum())
        cat_rows.append({"category": c, "n_rule": int(rule.sum()), "n_clinician": int(clin.sum()),
                         "precision (rule)": tp / max(rule.sum(), 1),
                         "recall (rule)": tp / max(clin.sum(), 1), "kappa": _kappa(rule, clin)})
    out["per_category"] = pd.DataFrame(cat_rows)
    if store is not None:
        out["harmful_clinician_labels"] = _harm_with_clinician(store, k, consensus, clinician_high)
    for name, df in out.items():
        df.to_csv(paths.out("clinician_kit", "results", f"{name}.csv"), index=False)
    return out


def _harm_with_clinician(store, key, consensus, clinician_high):
    rows = []
    for ds in key.dataset.unique():
        kd = key[key.dataset == ds]
        hi_clin = dict(zip(kd.uid, consensus.loc[kd.index] >= clinician_high))
        for model in ("pubmedbert_medmcqa", "biobert_medmcqa"):
            for method in ("base", "medbayes_lite"):
                for gate in config.GATES:
                    preds = store.preds(ds, model, method, gate)
                    if not preds:
                        continue
                    uids = store.get(ds, model, method)[0].uids
                    m = np.isin(uids, kd.uid)
                    lab = np.array([hi_clin.get(u, False) for u in uids[m]])
                    rule = preds[0].sev[m] >= config.HIGH_SEVERITY_CUTOFF
                    for name, hi in (("rule", rule), ("clinician", lab)):
                        h = np.mean([((~p.correct[m]) & (p.score[m] >= 0.8) & hi).mean()
                                     for p in preds])
                        rows.append({"dataset": ds, "model": model, "method": method,
                                     "gate": gate, "severity_labels": name,
                                     "n_rated_items": int(m.sum()), "harmful@0.8": h})
    return pd.DataFrame(rows)
