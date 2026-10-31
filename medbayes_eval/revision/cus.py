"""R3:7 / R1:4 - CUS validity with dependence-aware uncertainty.

Groups follow the original analysis: (dataset, model, method) runs, seeds pooled,
evaluated overall and within each of the 13 buckets. Because groups share items
(the same item appears in several buckets, methods and models), group-level
correlations are not independent. We therefore report, besides the naive
correlation: an item-cluster bootstrap (items resampled once per draw and all
groups recomputed, preserving cross-group dependence), a group bootstrap, subset
analyses without overlapping overall/bucket rows, the partial correlation given
ECE, cluster-robust incremental regressions, and alternative weighting schemes
(including weights decoupled from the harmful-error definition).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .. import config
from ..stats.bootstrap import bootstrap_p
from .analyses import save
from .fastmetrics import aurc, ci, ece
from .severity import _agg, _bucket_lists

TARGETS = ("harmful@0.8", "sevwerr@0.8", "abst_rec@0.7", "aurc", "overconf@0.8")
METHODS = ("base", "temp_scaling", "conformal", "mc_dropout", "medbayes_lite",
           "deep_ensemble", "self_consistency", "semantic_entropy", "verbalized")
BUCKETS = config.BUCKETS


class _Run:
    def __init__(self, ds, model, method, preds, buckets):
        self.key = (ds, model, method)
        self.p = np.stack([q.p for q in preds])            # (S, N)
        self.correct = np.stack([q.correct for q in preds])
        self.sev = preds[0].sev
        self.member = np.array([[b in bl for b in BUCKETS] for bl in buckets])  # (N, 13)
        self.uids = None


def build_runs(store):
    runs = []
    for ds, model, method in store.keys():
        if method not in METHODS:
            continue
        preds = store.preds(ds, model, method, "maxprob")
        uids = store.get(ds, model, method)[0].uids
        r = _Run(ds, model, method, preds, _bucket_lists(ds).loc[uids].tolist())
        r.uids = np.asarray(uids)
        runs.append(r)
    return runs


def _group_metrics(run: _Run, idx=None, sev=None):
    """Rows: overall + each bucket; cols: cus, ece, targets, n."""
    p, c = run.p, run.correct
    s = run.sev if sev is None else sev
    mem = run.member
    if idx is not None:
        p, c, s, mem = p[:, idx], c[:, idx], s[idx], mem[idx]
    S = p.shape[0]
    out = []
    masks = [np.ones(len(s), bool)] + [mem[:, j] for j in range(len(BUCKETS))]
    for m in masks:
        n = int(m.sum())
        if n == 0:
            out.append([np.nan] * 8)
            continue
        pp, cc, ss = p[:, m].ravel(), c[:, m].ravel(), np.tile(s[m], S)
        wrong = ~cc
        acc7, acc8 = pp >= 0.7, pp >= 0.8
        out.append([ece(pp, cc, ss), ece(pp, cc),
                    float(np.mean(wrong & acc8 & (ss >= config.HIGH_SEVERITY_CUTOFF))),
                    float(np.mean(ss * wrong * acc8)),
                    float((~acc7 & wrong).sum() / wrong.sum()) if wrong.any() else np.nan,
                    aurc(pp, cc), float(np.mean(wrong & acc8)), n])
    return np.array(out, float)


COLS = ["cus", "ece", "harmful@0.8", "sevwerr@0.8", "abst_rec@0.7", "aurc",
        "overconf@0.8", "n_items"]


def group_table(runs, sev_fn=None):
    rows = []
    for r in runs:
        g = _group_metrics(r, sev=None if sev_fn is None else sev_fn(r))
        for j, name in enumerate(["overall"] + list(BUCKETS)):
            rows.append(dict(zip(["dataset", "model", "method"], r.key), bucket=name,
                             **dict(zip(COLS, g[j]))))
    return pd.DataFrame(rows)


def _corr(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return np.nan, np.nan, len(d)
    return pearsonr(d[x], d[y])[0], spearmanr(d[x], d[y])[0], len(d)


def _fisher_ci(r, n):
    if np.isnan(r) or n < 4:
        return np.nan, np.nan
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def _partial(df, x, y, ctrl="ece"):
    d = df[[x, y, ctrl]].dropna()
    if len(d) < 4:
        return np.nan
    A = np.c_[np.ones(len(d)), d[ctrl]]
    rx = d[x] - A @ np.linalg.lstsq(A, d[x], rcond=None)[0]
    ry = d[y] - A @ np.linalg.lstsq(A, d[y], rcond=None)[0]
    return float(pearsonr(rx, ry)[0])


SUBSETS = {
    "original (overall + bucket rows)": lambda d: d,
    "bucket rows only": lambda d: d[d.bucket != "overall"],
    "overall rows only": lambda d: d[d.bucket == "overall"],
    "encoders only": lambda d: d[d.model != "qwen2_5_7b"],
    "medmcqa only": lambda d: d[d.dataset == "medmcqa"],
    "medqa only": lambda d: d[d.dataset == "medqa"],
    "pubmedqa only": lambda d: d[d.dataset == "pubmedqa"],
}


def cus_correlations(runs, B_items: int = 500, B_groups: int = 2000):
    base = group_table(runs)
    save(base, "cus_group_metrics")
    # item-cluster bootstrap: items are resampled once per dataset per draw (over the union
    # of item ids, so runs on different subsets share weights) and all groups recomputed
    by_ds = {}
    for r in runs:
        by_ds.setdefault(r.key[0], []).append(r)
    pos = {}
    for ds, rs in by_ds.items():
        universe = {u: i for i, u in enumerate(np.unique(np.concatenate([r.uids for r in rs])))}
        pos[ds] = (len(universe), [np.array([universe[u] for u in r.uids]) for r in rs])
    rng = np.random.default_rng(0)
    item_draws = []
    for _ in range(B_items):
        frames = []
        for ds, rs in by_ds.items():
            n_u, run_pos = pos[ds]
            counts = np.bincount(rng.integers(0, n_u, n_u), minlength=n_u)
            for r, rp in zip(rs, run_pos):
                idx = np.repeat(np.arange(len(rp)), counts[rp])
                if idx.size == 0:
                    continue
                g = _group_metrics(r, idx)
                for j, name in enumerate(["overall"] + list(BUCKETS)):
                    frames.append(dict(zip(["dataset", "model", "method"], r.key),
                                       bucket=name, **dict(zip(COLS, g[j]))))
        item_draws.append(pd.DataFrame(frames))
    rows = []
    for sname, sub_fn in SUBSETS.items():
        d0 = sub_fn(base)
        for tgt in TARGETS:
            for pred in ("cus", "ece"):
                r, rho, n = _corr(d0, pred, tgt)
                if n < 3:
                    continue
                f_lo, f_hi = _fisher_ci(r, n)
                gidx = rng.integers(0, len(d0), (B_groups, len(d0)))
                g_draws = [(_corr(d0.iloc[i], pred, tgt)[0]) for i in gidx]
                i_draws = [_corr(sub_fn(dd), pred, tgt)[0] for dd in item_draws]
                rows.append({"subset": sname, "predictor": pred, "target": tgt,
                             "n_groups": n, "pearson_r": r, "spearman_r": rho,
                             "fisher_ci_lo": f_lo, "fisher_ci_hi": f_hi,
                             "group_boot_ci_lo": ci(g_draws)[0],
                             "group_boot_ci_hi": ci(g_draws)[1],
                             "item_cluster_boot_ci_lo": ci(i_draws)[0],
                             "item_cluster_boot_ci_hi": ci(i_draws)[1],
                             "partial_r_given_ece": _partial(d0, pred, tgt)
                             if pred == "cus" else np.nan})
    df = pd.DataFrame(rows)
    save(df, "cus_correlations_dependence_aware")
    return base, df, item_draws


def cus_incremental(base: pd.DataFrame, item_draws, B: int = 2000):
    """Nested OLS target ~ ECE vs ECE + CUS: naive, cluster-robust, cluster/item bootstrap,
    leave-one-model-out."""
    import statsmodels.api as sm
    rows = []
    d_all = base[base.bucket != "overall"]

    def fit(d, tgt, cols):
        X = sm.add_constant(d[list(cols)].to_numpy(float), has_constant="add")
        return sm.OLS(d[tgt].to_numpy(float), X)

    def dr2(d, tgt):
        d = d[[tgt, "ece", "cus"]].dropna()
        if len(d) < 5:
            return np.nan
        return fit(d, tgt, ["ece", "cus"]).fit().rsquared - fit(d, tgt, ["ece"]).fit().rsquared

    for tgt in ("harmful@0.8", "sevwerr@0.8"):
        d = d_all[[tgt, "ece", "cus", "dataset", "model"]].dropna()
        clusters = (d.dataset + "|" + d.model).astype("category").cat.codes.to_numpy()
        full, red = fit(d, tgt, ["ece", "cus"]).fit(), fit(d, tgt, ["ece"]).fit()
        rob = fit(d, tgt, ["ece", "cus"]).fit(cov_type="cluster",
                                             cov_kwds={"groups": clusters})
        rng = np.random.default_rng(0)
        uc = np.unique(clusters)
        cb = []
        for _ in range(B):
            pick = rng.choice(uc, len(uc), replace=True)
            dd = pd.concat([d[clusters == k] for k in pick])
            cb.append(dr2(dd, tgt))
        ib = [dr2(x[x.bucket != "overall"], tgt) for x in item_draws]
        lomo = [dr2(d[d.model != m], tgt) for m in d.model.unique()]
        rows.append({"target": tgt, "n_groups": len(d), "n_clusters": len(uc),
                     "r2_ece": red.rsquared, "r2_ece_cus": full.rsquared,
                     "delta_r2": full.rsquared - red.rsquared,
                     "partial_F_p": float(full.compare_f_test(red)[1]),
                     "cus_coef": float(full.params[2]),
                     "cus_p_naive": float(full.pvalues[2]),
                     "cus_p_cluster_robust": float(rob.pvalues[2]),
                     "delta_r2_cluster_boot_ci_lo": ci(cb)[0],
                     "delta_r2_cluster_boot_ci_hi": ci(cb)[1],
                     "delta_r2_item_boot_ci_lo": ci(ib)[0],
                     "delta_r2_item_boot_ci_hi": ci(ib)[1],
                     "delta_r2_leave_one_model_out_min": float(np.nanmin(lomo)),
                     "delta_r2_leave_one_model_out_max": float(np.nanmax(lomo))})
    return save(pd.DataFrame(rows), "cus_incremental_dependence_aware")


def cus_weight_schemes(runs, n_random: int = 20):
    """CUS under alternative weights vs harmful@0.8 under the ORIGINAL definition.
    Random permutations of the bucket weights decouple CUS from the severity
    structure used by the target (circularity check). Recreates Supplementary S4."""
    W = config.SEVERITY_WEIGHTS
    rng = np.random.default_rng(0)
    schemes = {"original weights": W,
               "uniform (CUS = ECE)": {b: 1.0 for b in W},
               "binary (3 if >= 2.5 else 1)": {b: 3.0 if w >= 2.5 else 1.0 for b, w in W.items()},
               "rank-based": {b: float(r + 1) for r, b in enumerate(sorted(W, key=W.get))}}
    for i in range(n_random):
        schemes[f"random +/-25% #{i}"] = {b: w * rng.uniform(0.75, 1.25) for b, w in W.items()}
    vals = list(W.values())
    for i in range(n_random):
        schemes[f"permuted weights #{i}"] = dict(zip(W, rng.permutation(vals)))
    base = group_table(runs)
    rows = []
    bl_cache = {}
    for name, w in schemes.items():
        def sev_fn(r, w=w):
            key = (r.key[0], id(r))
            if key not in bl_cache:
                bl_cache[key] = [[b for b, m in zip(BUCKETS, row) if m] for row in r.member]
            return np.array([_agg(b, w, "max") for b in bl_cache[key]])
        g = group_table(runs, sev_fn)
        # target keeps the original severity definition
        g["harmful@0.8"] = base["harmful@0.8"].to_numpy()
        for sname in ("original (overall + bucket rows)", "bucket rows only"):
            d = SUBSETS[sname](g)
            r_, rho, n = _corr(d, "cus", "harmful@0.8")
            rows.append({"scheme": name, "subset": sname, "pearson_r": r_,
                         "spearman_r": rho, "n_groups": n})
    df = pd.DataFrame(rows)
    save(df, "cus_weight_schemes")
    summ = []
    for fam in ("random +/-25%", "permuted weights"):
        for sname, sub in df[df.scheme.str.startswith(fam)].groupby("subset"):
            summ.append({"scheme": f"{fam} (mean over {n_random})", "subset": sname,
                         "pearson_r": sub.pearson_r.mean(), "min": sub.pearson_r.min(),
                         "max": sub.pearson_r.max(), "n_groups": int(sub.n_groups.iloc[0])})
    fixed = df[~df.scheme.str.contains("#")].assign(min=np.nan, max=np.nan)
    s4 = pd.concat([fixed, pd.DataFrame(summ)], ignore_index=True)
    save(s4, "S4_cus_weight_sensitivity",
         caption="CUS severity-weight sensitivity (Pearson r with harmful@0.8).")
    return df
