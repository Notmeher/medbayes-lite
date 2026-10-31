"""Core revision analyses (CPU, from logit caches).

  main_table            Table 8 reproduction, both gates, seed mean/sd + item-bootstrap CIs
  paired_tests          bounded p-values, CIs, effect sizes, Holm/BH (R1:9, R3:4)
  seed_vs_item_variance seed variability vs item-bootstrap uncertainty (R3:4)
  ood_table             3-seed domain shift, target- vs source-calibrated T (R3:4/R3:6)
  matched_coverage      matched-coverage + matched-risk comparisons (R1:1, R3:1)
  threshold_counts      fixed-threshold results with accepted counts + Wilson CIs (R3:1)
  workload              deferral workload per 1000 cases (R1:2, R3:6)
  factorial_ablation    MC x temperature x gate factorial (R3:2)
  compute_curve         quality vs MC passes joined with measured latency (R3:2, R3:9)
  sensitivity_tables    Supplementary S1/S2 recreated (MC-only and full layer)
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from .. import config
from ..models import uq_methods as uq
from ..stats.bootstrap import bootstrap_p, format_p, holm, benjamini_hochberg
from . import paths
from .fastmetrics import (Pred, at_coverage, boot_indices, ci, coverage_at_risk,
                          make_pred, metrics, risk_coverage, wilson)

TAB = os.path.join(paths.ROOT, "tables")
GATES = config.GATES
CI_KEYS = ["accuracy", "ece", "cus", "nll", "brier", "aurc", "auroc_error",
           "harmful@0.8", "selacc@0.7", "cov@0.7", "abst_rec@0.7"]
COV_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RISK_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
TAUS = config.COVERAGE_THRESHOLDS
MAIN_METHODS = ("base", "temp_scaling", "mc_dropout", "medbayes_lite", "deep_ensemble",
                "temp_scaling_srcT", "medbayes_lite_srcT", "conformal",
                "self_consistency", "semantic_entropy", "verbalized")


def _n_jobs():
    return int(os.environ.get("MEDBAYES_N_JOBS", "-1"))


def _par(fn, tasks):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=_n_jobs())(delayed(fn)(*t) for t in tasks)


def save(df: pd.DataFrame, name: str, caption: str | None = None,
         tex_cols: Sequence[str] | None = None) -> pd.DataFrame:
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    if caption:
        from ..reporting.tables import emit_table
        emit_table(df[list(tex_cols)] if tex_cols else df, name,
                   os.path.join(TAB, "latex"), caption=caption, label=f"tab:rev_{name}")
    return df


def _uniq(preds: List[Pred]) -> List[Pred]:
    """Drop exact duplicate seeds (deterministic methods on pretrained checkpoints)."""
    out = []
    for p in preds:
        if not any(np.array_equal(p.probs, q.probs) and np.array_equal(p.score, q.score)
                   for q in out):
            out.append(p)
    return out


def _vec(pr: Pred, keys=CI_KEYS) -> np.ndarray:
    m = metrics(pr, taus=(0.7,), harm_taus=(0.8,))
    return np.array([m[k] for k in keys], float)


def _seed_mean_vec(fn, preds, idx=None):
    vals = np.array([fn(p if idx is None else p.take(idx)) for p in preds], float)
    with np.errstate(all="ignore"):
        return np.nanmean(vals, axis=0) if vals.ndim > 1 else np.nanmean(vals)


def _boot_vec(fn, preds, idx_mat):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array([_seed_mean_vec(fn, preds, idx) for idx in idx_mat])


def _settings(store, methods=MAIN_METHODS):
    return [k for k in store.keys() if k[2] in methods]


def _gates_for(method):
    return ("maxprob",) if method == "conformal" else GATES


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------
def _main_row(ds, model, method, gate, preds, B, seed):
    per = pd.DataFrame([metrics(p) for p in preds])
    row = {"dataset": ds, "model": model, "method": method, "gate": gate,
           "n_seeds": len(preds), "n_items": int(per["n"].iloc[0])}
    for c in per.columns:
        if c == "n":
            continue
        row[c] = float(per[c].mean())
        row[c + "_sd"] = float(per[c].std(ddof=1)) if len(per) > 1 else 0.0
    up = _uniq(preds)
    draws = _boot_vec(_vec, up, boot_indices(len(up[0].labels), B, seed))
    for j, k in enumerate(CI_KEYS):
        row[k + "_ci_lo"], row[k + "_ci_hi"] = ci(draws[:, j])
    k = sum(int(((p.score >= 0.7) & p.correct).sum()) for p in preds)
    n = sum(int((p.score >= 0.7).sum()) for p in preds)
    row["selacc@0.7_wilson_lo"], row["selacc@0.7_wilson_hi"] = wilson(k, n)
    return row


def main_table(store, B: int = 2000) -> pd.DataFrame:
    tasks = [(ds, m, me, g, store.preds(ds, m, me, g), B, 0)
             for ds, m, me in _settings(store) for g in _gates_for(me)]
    df = pd.DataFrame(_par(_main_row, tasks))
    save(df, "main_results_revised")
    # paper Table 8 layout (max-prob gate = the reported numbers) and entropy-gate twin
    for gate in GATES:
        sub = df[(df.gate == gate) & df.method.isin(
            ["base", "temp_scaling", "mc_dropout", "medbayes_lite"]) &
                 df.model.isin(["pubmedbert_medmcqa", "biobert_medmcqa"])]
        cols = ["dataset", "model", "method", "ece", "harmful@0.8", "selacc@0.7",
                "cov@0.7", "n_acc@0.7", "aurc"]
        save(sub[cols], f"table8_{gate}_gate",
             caption=f"Harmful overconfident error reduction ({gate} gate; 3-seed means).",
             tex_cols=cols)
    return df


# ---------------------------------------------------------------------------
# paired tests
# ---------------------------------------------------------------------------
PAIRED_KEYS = ["ece", "harmful@0.8", "aurc", "auroc_error", "selacc@0.7", "cov@0.7",
               "brier", "nll", "cus"]
COMPARISONS = [  # (family, A=(method, gate), B=(method, gate)); diff = B - A
    ("primary", ("base", "maxprob"), ("medbayes_lite", "maxprob")),
    ("primary", ("base", "entropy"), ("medbayes_lite", "entropy")),
    ("vs_temperature", ("temp_scaling", "maxprob"), ("medbayes_lite", "maxprob")),
    ("vs_temperature", ("temp_scaling", "entropy"), ("medbayes_lite", "entropy")),
    ("vs_temperature", ("temp_scaling", "entropy"), ("medbayes_lite", "maxprob")),
    ("vs_mc", ("mc_dropout", "maxprob"), ("medbayes_lite", "maxprob")),
    ("vs_mc", ("mc_dropout", "entropy"), ("medbayes_lite", "entropy")),
    ("vs_ensemble", ("deep_ensemble", "maxprob"), ("medbayes_lite", "maxprob")),
    ("ood_source_T", ("base", "maxprob"), ("medbayes_lite_srcT", "maxprob")),
]


def _paired_rows(ds, model, fam, a, b, pa, pb, B, seed):
    idx = boot_indices(len(pa[0].labels), B, seed)
    fn = lambda p: _vec(p, PAIRED_KEYS)
    point = _seed_mean_vec(fn, pb) - _seed_mean_vec(fn, pa)
    ua, ub = _uniq(pa), _uniq(pb)
    draws = np.array([_seed_mean_vec(fn, ub, i) - _seed_mean_vec(fn, ua, i) for i in idx])
    base_val = _seed_mean_vec(fn, pa)
    rows = []
    for j, k in enumerate(PAIRED_KEYS):
        lo, hi = ci(draws[:, j])
        p = bootstrap_p(draws[:, j], B)
        rows.append({"dataset": ds, "model": model, "family": fam,
                     "A": f"{a[0]} [{a[1]}]", "B": f"{b[0]} [{b[1]}]", "metric": k,
                     "A_value": base_val[j], "B_value": base_val[j] + point[j],
                     "diff_B_minus_A": point[j], "ci_lo": lo, "ci_hi": hi,
                     "rel_change_pct": (100 * point[j] / abs(base_val[j])
                                        if base_val[j] else np.nan),
                     "p_value": p, "n_boot": B,
                     "n_valid_draws": int(np.sum(~np.isnan(draws[:, j]))),
                     "resampling_unit": "item (uid), paired; seeds averaged within draw",
                     "n_seeds_A": len(pa), "n_seeds_B": len(pb)})
    return rows


def paired_tests(store, B_primary: int = 10000, B: int = 2000) -> pd.DataFrame:
    keys = {(ds, m) for ds, m, _ in store.keys() if m != "qwen2_5_7b"}
    tasks = []
    for ds, model in sorted(keys):
        for fam, a, b in COMPARISONS:
            pa, pb = store.preds(ds, model, *a), store.preds(ds, model, *b)
            if pa and pb:
                tasks.append((ds, model, fam, a, b, pa, pb,
                              B_primary if fam == "primary" else B, 0))
    rows = [r for rs in _par(_paired_rows, tasks) for r in rs]
    df = pd.DataFrame(rows)
    df["p_holm_within_family"] = np.nan
    for fam, idx in df.groupby("family").groups.items():
        df.loc[idx, "p_holm_within_family"] = holm(df.loc[idx, "p_value"])
    df["p_bh_all"] = benjamini_hochberg(df["p_value"])
    df["p_report"] = df["p_value"].map(format_p)
    df["p_holm_report"] = df["p_holm_within_family"].map(format_p)
    return save(df, "paired_tests")


# ---------------------------------------------------------------------------
# seed vs item variance
# ---------------------------------------------------------------------------
def _var_rows(ds, model, method, gate, preds, B, seed):
    per = np.array([_vec(p) for p in preds])
    idx = boot_indices(len(preds[0].labels), B, seed)
    item_se = np.array([np.nanstd(_boot_vec(_vec, [p], idx), axis=0, ddof=1)
                        for p in _uniq(preds)])
    rng = np.random.default_rng(seed + 1)
    two_stage = []
    for i in idx:  # resample seeds and items jointly
        pick = rng.integers(0, len(preds), len(preds))
        two_stage.append(_seed_mean_vec(_vec, [preds[k] for k in pick], i))
    two_stage = np.array(two_stage)
    rows = []
    for j, k in enumerate(CI_KEYS):
        lo, hi = ci(two_stage[:, j])
        rows.append({"dataset": ds, "model": model, "method": method, "gate": gate,
                     "metric": k, "n_seeds": len(preds), "mean": np.nanmean(per[:, j]),
                     "seed_sd": np.nanstd(per[:, j], ddof=1) if len(preds) > 1 else 0.0,
                     "seed_min": np.nanmin(per[:, j]), "seed_max": np.nanmax(per[:, j]),
                     "item_bootstrap_se": float(np.nanmean(item_se[:, j])),
                     "two_stage_ci_lo": lo, "two_stage_ci_hi": hi})
    return rows


def seed_vs_item_variance(store, B: int = 1000) -> pd.DataFrame:
    tasks = [(ds, m, me, g, store.preds(ds, m, me, g), B, 0)
             for ds, m, me in _settings(store, ("base", "temp_scaling", "mc_dropout",
                                                "medbayes_lite", "deep_ensemble"))
             for g in GATES]
    df = pd.DataFrame([r for rs in _par(_var_rows, tasks) for r in rs])
    return save(df, "seed_vs_item_variance")


# ---------------------------------------------------------------------------
# OOD
# ---------------------------------------------------------------------------
OOD_KEYS = ["accuracy", "ece", "harmful@0.8", "aurc", "cov@0.7", "selacc@0.7", "cus"]


def _ood_vec(p):
    m = metrics(p, taus=(0.7,), harm_taus=(0.8,), full=False)
    from .fastmetrics import entropy
    return np.array([m[k] for k in OOD_KEYS] + [float(entropy(p.probs).mean())])


def ood_table(store, B: int = 2000) -> pd.DataFrame:
    rows = []
    for model in ("pubmedbert_medmcqa", "biobert_medmcqa"):
        for method_id, method_ood, calib in [("base", "base", "-"),
                                             ("temp_scaling", "temp_scaling", "target"),
                                             ("temp_scaling", "temp_scaling_srcT", "source"),
                                             ("medbayes_lite", "medbayes_lite", "target"),
                                             ("medbayes_lite", "medbayes_lite_srcT", "source")]:
            for gate in GATES:
                pid = store.preds("medmcqa", model, method_id, gate)
                pood = store.preds("medqa", model, method_ood, gate)
                if not pid or not pood:
                    continue
                vi, vo = _seed_mean_vec(_ood_vec, pid), _seed_mean_vec(_ood_vec, pood)
                di = _boot_vec(_ood_vec, _uniq(pid), boot_indices(len(pid[0].labels), B, 0))
                do = _boot_vec(_ood_vec, _uniq(pood), boot_indices(len(pood[0].labels), B, 1))
                row = {"model": model, "method": method_id, "ood_temperature_fit_on": calib,
                       "gate": gate, "n_seeds": len(pid)}
                for j, k in enumerate(OOD_KEYS + ["mean_entropy"]):
                    row[f"id_{k}"], row[f"ood_{k}"] = vi[j], vo[j]
                    row[f"drift_{k}"] = vo[j] - vi[j]
                    row[f"drift_{k}_ci_lo"], row[f"drift_{k}_ci_hi"] = ci(do[:, j] - di[:, j])
                rows.append(row)
    return save(pd.DataFrame(rows), "ood_transfer_3seed")


# ---------------------------------------------------------------------------
# matched coverage / matched risk
# ---------------------------------------------------------------------------
_COV_FIELDS = ["selective_risk", "harmful_per_total", "harmful_among_accepted",
               "sevwerr_per_total", "n_accepted", "errors_accepted", "harmful_accepted"]


def _profile(p: Pred) -> np.ndarray:
    """All matched-coverage fields for COV_GRID + coverage at each RISK_GRID level,
    from a single sort."""
    n = len(p.labels)
    order = np.argsort(-p.score, kind="mergesort")
    wrong = (~p.correct[order]).astype(float)
    harm = wrong * (p.sev[order] >= config.HIGH_SEVERITY_CUTOFF)
    sw = wrong * p.sev[order]
    cw, ch, cs = np.cumsum(wrong), np.cumsum(harm), np.cumsum(sw)
    out = []
    for c in COV_GRID:
        k = int(np.ceil(c * n - 1e-9))
        if k == 0:
            out += [np.nan, 0.0, np.nan, 0.0, 0, 0, 0]
            continue
        out += [cw[k - 1] / k, ch[k - 1] / n, ch[k - 1] / k, cs[k - 1] / n, k,
                cw[k - 1], ch[k - 1]]
    risk = cw / np.arange(1, n + 1)
    for r in RISK_GRID:
        ok = np.where(risk <= r + 1e-12)[0]
        out.append((ok.max() + 1) / n if ok.size else 0.0)
    return np.array(out, float)


def _cov_rows(ds, model, method, gate, preds, base_preds, B, seed):
    idx = boot_indices(len(preds[0].labels), B, seed)
    point = _seed_mean_vec(_profile, preds)
    up, ub = _uniq(preds), _uniq(base_preds)
    draws = _boot_vec(_profile, up, idx)
    bpoint = _seed_mean_vec(_profile, base_preds)
    ddraws = np.array([_seed_mean_vec(_profile, up, i) - _seed_mean_vec(_profile, ub, i)
                       for i in idx]) if base_preds is not preds else None
    nf = len(_COV_FIELDS)
    rows = []
    for ci_, c in enumerate(COV_GRID):
        row = {"dataset": ds, "model": model, "method": method, "gate": gate,
               "target_coverage": c, "n_items": len(preds[0].labels), "n_seeds": len(preds)}
        for f_i, f in enumerate(_COV_FIELDS):
            j = ci_ * nf + f_i
            row[f] = point[j]
            if f in ("selective_risk", "harmful_per_total", "harmful_among_accepted"):
                row[f + "_ci_lo"], row[f + "_ci_hi"] = ci(draws[:, j])
                if ddraws is not None:
                    row[f + "_diff_vs_base"] = point[j] - bpoint[j]
                    row[f + "_diff_ci_lo"], row[f + "_diff_ci_hi"] = ci(ddraws[:, j])
                    row[f + "_diff_p"] = bootstrap_p(ddraws[:, j], B)
        lo, hi = wilson(round(row["errors_accepted"]), round(row["n_accepted"]))
        row["selective_risk_wilson_lo"], row["selective_risk_wilson_hi"] = lo, hi
        rows.append(row)
    off = len(COV_GRID) * nf
    risk_rows = []
    for r_i, r in enumerate(RISK_GRID):
        j = off + r_i
        lo, hi = ci(draws[:, j])
        row = {"dataset": ds, "model": model, "method": method, "gate": gate,
               "target_risk": r, "coverage_at_risk": point[j], "ci_lo": lo, "ci_hi": hi,
               "base_coverage_at_risk": bpoint[j]}
        if ddraws is not None:
            row["diff_vs_base"] = point[j] - bpoint[j]
            row["diff_ci_lo"], row["diff_ci_hi"] = ci(ddraws[:, j])
            row["diff_p"] = bootstrap_p(ddraws[:, j], B)
        risk_rows.append(row)
    return rows, risk_rows


def matched_coverage(store, B: int = 1000):
    tasks = []
    for ds, m, me in _settings(store):
        if me == "conformal":
            continue
        base_method = "base" if m != "qwen2_5_7b" else "self_consistency"
        for g in GATES:
            bp = store.preds(ds, m, base_method, "maxprob")
            p = store.preds(ds, m, me, g)
            tasks.append((ds, m, me, g, p, p if (me == base_method and g == "maxprob") else bp,
                          B, 0))
    res = _par(_cov_rows, tasks)
    cov = pd.DataFrame([r for a, _ in res for r in a])
    risk = pd.DataFrame([r for _, b in res for r in b])
    save(cov, "matched_coverage")
    save(risk, "matched_risk")
    return cov, risk


# ---------------------------------------------------------------------------
# fixed thresholds with counts, workload
# ---------------------------------------------------------------------------
def threshold_counts(store) -> pd.DataFrame:
    rows = []
    for ds, m, me in _settings(store):
        for g in _gates_for(me):
            preds = store.preds(ds, m, me, g)
            for t in TAUS:
                per = []
                for p in preds:
                    acc = p.score >= t
                    c = p.correct
                    hi = p.sev >= config.HIGH_SEVERITY_CUTOFF
                    per.append({"n": len(c), "n_acc": acc.sum(), "n_corr_acc": (acc & c).sum(),
                                "n_wrong": (~c).sum(), "n_wrong_def": (~acc & ~c).sum(),
                                "harm": (acc & ~c & hi).sum(), "n_hi": hi.sum()})
                d = pd.DataFrame(per)
                tot = d.sum()
                lo, hi_ = wilson(tot.n_corr_acc, tot.n_acc)
                rows.append({
                    "dataset": ds, "model": m, "method": me, "gate": g, "tau": t,
                    "n_seeds": len(d), "n_items": int(d.n.iloc[0]),
                    "n_accepted_mean": d.n_acc.mean(), "n_accepted_min": int(d.n_acc.min()),
                    "coverage": d.n_acc.mean() / d.n.iloc[0],
                    "selective_acc": tot.n_corr_acc / tot.n_acc if tot.n_acc else np.nan,
                    "selective_acc_wilson_lo": lo, "selective_acc_wilson_hi": hi_,
                    "abstention_recall": tot.n_wrong_def / tot.n_wrong if tot.n_wrong else np.nan,
                    "harmful_count_mean": d.harm.mean(),
                    "harmful_per_total": tot.harm / tot.n,
                    "harmful_per_accepted": tot.harm / tot.n_acc if tot.n_acc else np.nan,
                    "harmful_per_high_severity_item": tot.harm / tot.n_hi if tot.n_hi else np.nan,
                    "n_high_severity_items": int(d.n_hi.iloc[0]),
                    "unstable_small_n": bool(d.n_acc.mean() < 30),
                })
    df = pd.DataFrame(rows)
    save(df, "threshold_counts")
    s3 = df[(df.method == "medbayes_lite") & (df.dataset == "medmcqa")]
    save(s3[["model", "gate", "tau", "coverage", "n_accepted_mean", "selective_acc",
             "selective_acc_wilson_lo", "selective_acc_wilson_hi", "abstention_recall",
             "harmful_per_total"]], "S3_threshold_sweep",
         caption="Abstention-threshold sweep (MedMCQA, MedBayes-Lite, 3 seeds).")
    return df


def workload(thr: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    a = thr.assign(operating_point=lambda d: "tau=" + d.tau.astype(str))
    a = pd.DataFrame({
        "dataset": a.dataset, "model": a.model, "method": a.method, "gate": a.gate,
        "operating_point": a.operating_point, "coverage": a.coverage,
        "deferred_per_1000": 1000 * (1 - a.coverage),
        "errors_passed_per_1000": 1000 * a.coverage * (1 - a.selective_acc.fillna(1)),
        "harmful_passed_per_1000": 1000 * a.harmful_per_total})
    b = pd.DataFrame({
        "dataset": cov.dataset, "model": cov.model, "method": cov.method, "gate": cov.gate,
        "operating_point": "coverage=" + cov.target_coverage.astype(str),
        "coverage": cov.n_accepted / cov.n_items,
        "deferred_per_1000": 1000 * (1 - cov.n_accepted / cov.n_items),
        "errors_passed_per_1000": 1000 * cov.errors_accepted / cov.n_items,
        "harmful_passed_per_1000": 1000 * cov.harmful_per_total})
    df = pd.concat([a, b], ignore_index=True)
    return save(df, "deferral_workload_per_1000")


def add_errors_caught(work: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    err = main[main.gate == "maxprob"].set_index(["dataset", "model", "method"])["accuracy"]
    e = work.apply(lambda r: 1000 * (1 - err.get((r.dataset, r.model, r.method), np.nan)),
                   axis=1)
    work = work.assign(errors_total_per_1000=e,
                       errors_caught_per_1000=e - work.errors_passed_per_1000)
    work["reviews_per_error_caught"] = work.deferred_per_1000 / work.errors_caught_per_1000
    return save(work, "deferral_workload_per_1000")


# ---------------------------------------------------------------------------
# factorial ablation
# ---------------------------------------------------------------------------
FACTORS = {("no", "no"): "base", ("no", "yes"): "temp_scaling",
           ("yes", "no"): "mc_dropout", ("yes", "yes"): "medbayes_lite"}


def factorial_ablation(main: pd.DataFrame, cov: pd.DataFrame, mc_samples: int = 15):
    rows = []
    for (mc, ts), method in FACTORS.items():
        sub = main[main.method == method]
        for _, r in sub.iterrows():
            row = {"dataset": r.dataset, "model": r.model, "mc_dropout": mc,
                   "temperature": ts, "gate": r.gate, "method": method,
                   "forward_passes": mc_samples if mc == "yes" else 1}
            for k in ["accuracy", "ece", "harmful@0.8", "selacc@0.7", "cov@0.7",
                      "n_acc@0.7", "aurc", "auroc_error"]:
                row[k] = r[k]
                if k + "_sd" in r:
                    row[k + "_sd"] = r[k + "_sd"]
            for c in (0.2, 0.5):
                cr = cov[(cov.dataset == r.dataset) & (cov.model == r.model) &
                         (cov.method == method) & (cov.gate == r.gate) &
                         np.isclose(cov.target_coverage, c)]
                if len(cr):
                    row[f"risk@cov{c:g}"] = cr.selective_risk.iloc[0]
                    row[f"harmful_per_total@cov{c:g}"] = cr.harmful_per_total.iloc[0]
            rows.append(row)
    return save(pd.DataFrame(rows), "factorial_ablation",
                caption="Factorial ablation: MC dropout x temperature x gate.",
                tex_cols=["dataset", "model", "mc_dropout", "temperature", "gate", "ece",
                          "harmful@0.8", "selacc@0.7", "cov@0.7", "aurc"])


# ---------------------------------------------------------------------------
# sensitivity cache: S1/S2 and the compute-quality curve
# ---------------------------------------------------------------------------
M_GRID = (1, 2, 3, 5, 10, 15, 20, 30, 50)


def _sens_load():
    from .jobs import load_npz
    p = os.path.join(paths.SENS, "medmcqa__pubmedbert_medmcqa.npz")
    if not os.path.exists(p):
        return None, None, None
    meta, a = load_npz(p)
    from .items import severity_lookup
    return meta, a, severity_lookup("medmcqa", a["test_uids"])


def _sens_row(probs, labels, sev, gate):
    m = metrics(make_pred(probs, labels, sev, gate), taus=(0.7,), harm_taus=(0.8,))
    pr = make_pred(probs, labels, sev, gate)
    m["risk@cov0.2"] = at_coverage(pr, 0.2)["selective_risk"]
    m["risk@cov0.5"] = at_coverage(pr, 0.5)["selective_risk"]
    return {k: m[k] for k in ["accuracy", "ece", "harmful@0.8", "aurc", "cov@0.7",
                              "n_acc@0.7", "selacc@0.7", "risk@cov0.2", "risk@cov0.5"]}


def sensitivity_tables():
    meta, a, sev = _sens_load()
    if meta is None:
        return None
    y, yc = a["test_labels"], a["calib_labels"]
    rows = []
    mc, cmc = a["test_mc_p0.2"], a["calib_mc_p0.2"]
    for M in [m for m in M_GRID if m <= mc.shape[0]]:
        bma = uq.softmax(mc[:M], -1).mean(0)
        T = uq.fit_temperature_on_probs(uq.softmax(cmc[:M], -1).mean(0), yc)
        for gate in GATES:
            rows.append({"sweep": "mc_samples", "M": M, "dropout_p": 0.2, "gate": gate,
                         "variant": "MC dropout only", **_sens_row(bma, y, sev, gate)})
            rows.append({"sweep": "mc_samples", "M": M, "dropout_p": 0.2, "gate": gate,
                         "variant": "MedBayes-Lite (MC + T)", "T": T,
                         **_sens_row(uq.power_temperature(bma, T), y, sev, gate)})
    for p in [0.1, 0.2] + list(meta["p_grid"]):
        tm = mc[:15] if p == 0.2 else a[f"test_mc_p{p:g}"]
        cm = cmc[:15] if p == 0.2 else a[f"calib_mc_p{p:g}"]
        bma = uq.softmax(tm, -1).mean(0)
        T = uq.fit_temperature_on_probs(uq.softmax(cm, -1).mean(0), yc)
        for gate in GATES:
            rows.append({"sweep": "dropout_rate", "M": tm.shape[0], "dropout_p": p,
                         "gate": gate, "variant": "MC dropout only",
                         **_sens_row(bma, y, sev, gate)})
            rows.append({"sweep": "dropout_rate", "M": tm.shape[0], "dropout_p": p,
                         "gate": gate, "variant": "MedBayes-Lite (MC + T)", "T": T,
                         **_sens_row(uq.power_temperature(bma, T), y, sev, gate)})
    df = pd.DataFrame(rows).drop_duplicates(["sweep", "M", "dropout_p", "gate", "variant"])
    save(df[df.sweep == "mc_samples"], "S1_mc_samples",
         caption="Sensitivity to the number of MC samples (MedMCQA, PubMedBERT).")
    save(df[df.sweep == "dropout_rate"], "S2_dropout_rate",
         caption="Sensitivity to the inference dropout rate (MedMCQA, PubMedBERT; M=15).")
    return df


def compute_curve(sens: pd.DataFrame | None, main: pd.DataFrame) -> pd.DataFrame | None:
    """Quality vs forward passes, joined with measured latency (matched latency)."""
    if sens is None:
        return None
    rows = sens[(sens.sweep == "mc_samples")].rename(columns={"M": "forward_passes"})
    ref = main[(main.dataset == "medmcqa") & (main.model == "pubmedbert_medmcqa") &
               main.method.isin(["base", "temp_scaling"])]
    extra = pd.DataFrame([{"variant": r.method, "gate": r.gate, "forward_passes": 1,
                           "accuracy": r.accuracy, "ece": r.ece,
                           "harmful@0.8": r["harmful@0.8"], "aurc": r.aurc,
                           "cov@0.7": r["cov@0.7"], "selacc@0.7": r["selacc@0.7"]}
                          for _, r in ref.iterrows()])
    df = pd.concat([extra, rows], ignore_index=True)
    lat_p = paths.out("tables", "latency_benchmark.csv")
    if os.path.exists(lat_p):
        lat = pd.read_csv(lat_p)
        lat = lat[lat.backbone.str.startswith("PubMedBERT")]
        seq = lat[lat.method.str.startswith("MC dropout / MedBayes-Lite")][
            ["forward_passes", "latency_ms_bs1_median", "throughput_items_per_s"]]
        one = lat[lat.method == "temperature scaling"][
            ["latency_ms_bs1_median", "throughput_items_per_s"]].iloc[0]
        df = df.merge(seq, on="forward_passes", how="left")
        df.loc[df.variant.isin(["base", "temp_scaling"]), "latency_ms_bs1_median"] = \
            one.latency_ms_bs1_median
        df.loc[df.variant.isin(["base", "temp_scaling"]), "throughput_items_per_s"] = \
            one.throughput_items_per_s
    return save(df, "compute_quality_matched_latency")
