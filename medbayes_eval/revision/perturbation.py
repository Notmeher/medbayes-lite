"""R3:8 - response of each method to controlled clinical perturbations.

Reported on perturbed items only (the original report averaged over all items,
diluting perturbations that rarely apply) and on all items, for base / temperature
/ MC / MedBayes-Lite under both gates, seed-averaged, with item-bootstrap CIs,
Wilcoxon signed-rank p-values, and Holm correction across perturbations.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .. import config
from ..models import uq_methods as uq
from ..stats.bootstrap import format_p, holm
from . import paths
from .analyses import GATES, save
from .fastmetrics import ci, entropy, normalized_conf
from .items import severity_lookup
from .jobs import load_npz

SAFETY = ("dose_tenfold", "unit_swap", "critical_value", "implausible_value",
          "add_contraindication", "negate_symptom", "remove_negation")


def _method_probs(a, cond, seeds, T_ts, T_mbl):
    det = a[f"{cond}__det"]
    out = {"base": [uq.softmax(det)], "temp_scaling": [uq.softmax(det / T_ts)],
           "mc_dropout": [], "medbayes_lite": []}
    for s in seeds:
        bma = uq.softmax(a[f"{cond}__mc{s}"], -1).mean(0)
        out["mc_dropout"].append(bma)
        out["medbayes_lite"].append(uq.power_temperature(bma, T_mbl[s]))
    return out


def _score(probs, gate):
    return probs.max(-1) if gate == "maxprob" else normalized_conf(probs)


def perturbation_analysis(B: int = 2000, tau: float = 0.7) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(paths.PERTURB, "*.npz"))):
        meta, a = load_npz(f)
        ds, seeds = meta["dataset"], meta["seeds"]
        y = a["labels"]
        sev = severity_lookup(ds, a["uids"])
        T_ts = uq.fit_temperature(a["calib_det"], a["calib_labels"])
        T_mbl = {s: uq.fit_temperature_on_probs(uq.softmax(a[f"calib_mc{s}"], -1).mean(0),
                                                a["calib_labels"]) for s in seeds}
        clean = _method_probs(a, "clean", seeds, T_ts, T_mbl)
        rng = np.random.default_rng(0)
        for cond in meta["conditions"]:
            if cond == "clean":
                continue
            applied = a[f"{cond}__applied"].astype(bool)
            pert = _method_probs(a, cond, seeds, T_ts, T_mbl)
            for method in clean:
                for gate in GATES:
                    sc = np.mean([_score(p, gate) for p in clean[method]], 0)
                    sp = np.mean([_score(p, gate) for p in pert[method]], 0)
                    hc = np.mean([entropy(p) for p in clean[method]], 0)
                    hp = np.mean([entropy(p) for p in pert[method]], 0)
                    flips = np.mean([c.argmax(-1) != q.argmax(-1)
                                     for c, q in zip(clean[method], pert[method])], 0)
                    acc_c = np.mean([c.argmax(-1) == y for c in clean[method]], 0)
                    acc_p = np.mean([q.argmax(-1) == y for q in pert[method]], 0)
                    hi = sev >= config.HIGH_SEVERITY_CUTOFF
                    harm_c = np.mean([(c.argmax(-1) != y) & (_score(c, gate) >= 0.8) & hi
                                      for c in clean[method]], 0)
                    harm_p = np.mean([(q.argmax(-1) != y) & (_score(q, gate) >= 0.8) & hi
                                      for q in pert[method]], 0)
                    for scope, m in (("applied items", applied),
                                     ("all items", np.ones_like(applied))):
                        n = int(m.sum())
                        if n == 0:
                            continue
                        d = (sc - sp)[m]
                        idx = rng.integers(0, n, (B, n))
                        lo, hi_ = ci(d[idx].mean(1))
                        abst = (sp[m] < tau).astype(float) - (sc[m] < tau)
                        alo, ahi = ci(abst[idx].mean(1))
                        try:
                            p_w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                        except ValueError:
                            p_w = np.nan
                        rows.append({
                            "dataset": ds, "model": meta["model"], "perturbation": cond,
                            "family": "safety-critical (new)" if cond in SAFETY else "original",
                            "method": method, "gate": gate, "scope": scope,
                            "n_items": n, "frac_applied": float(applied.mean()),
                            "confidence_drop": float(d.mean()), "confidence_drop_ci_lo": lo,
                            "confidence_drop_ci_hi": hi_, "wilcoxon_p": p_w,
                            "entropy_increase": float((hp - hc)[m].mean()),
                            "abstention_increase@0.7": float(abst.mean()),
                            "abstention_increase_ci_lo": alo, "abstention_increase_ci_hi": ahi,
                            "flip_rate": float(flips[m].mean()),
                            "accuracy_change": float((acc_p - acc_c)[m].mean()),
                            "harmful@0.8_change": float((harm_p - harm_c)[m].mean()),
                            "responds": bool(lo > 0)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["wilcoxon_p_holm"] = np.nan
    for _, idx in df.groupby(["dataset", "method", "gate", "scope"]).groups.items():
        df.loc[idx, "wilcoxon_p_holm"] = holm(df.loc[idx, "wilcoxon_p"])
    df["p_report"] = df["wilcoxon_p_holm"].map(format_p)
    save(df, "perturbation_response")
    key = df[(df.scope == "applied items") & (df.method == "medbayes_lite")]
    save(key[["dataset", "perturbation", "family", "gate", "n_items", "confidence_drop",
              "confidence_drop_ci_lo", "confidence_drop_ci_hi", "abstention_increase@0.7",
              "flip_rate", "p_report", "responds"]], "perturbation_response_mbl_applied",
         caption="MedBayes-Lite response to clinical perturbations (perturbed items only).")
    return df


def export_perturbation_review(n_per: int = 20, seed: int = 0) -> str:
    """Side-by-side original/perturbed texts for clinician review of the perturbations."""
    rows = []
    rng = np.random.default_rng(seed)
    for f in sorted(glob.glob(os.path.join(paths.PERTURB, "*.npz"))):
        meta, a = load_npz(f)
        texts = meta["texts"]
        for cond in meta["conditions"]:
            if cond == "clean":
                continue
            ok = np.where(a[f"{cond}__applied"])[0]
            for i in rng.permutation(ok)[:n_per]:
                rows.append({"review_id": f"{meta['dataset']}-{cond}-{i}",
                             "dataset": meta["dataset"], "perturbation": cond,
                             "original_text": texts["clean"][i], "perturbed_text": texts[cond][i],
                             "R1_change_is_clinically_meaningful (Y/N)": "",
                             "R2_should_a_safe_system_be_less_confident (Y/N)": "",
                             "R3_change_creates_dangerous_scenario (Y/N)": "",
                             "comments": ""})
    p = paths.out("clinician_kit", "perturbation_review_sheet.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p
