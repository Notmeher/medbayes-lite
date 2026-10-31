"""Incremental-value analysis for the Clinical Uncertainty Score (CUS).

The correlation analysis (metrics/core.correlation_analysis) shows CUS tracks harmful
overconfidence. This module asks a stricter question: does CUS explain variance in
clinical failure *beyond* standard ECE? We nest two regressions per target,

    target ~ ECE            (base)
    target ~ ECE + CUS      (full)

and report R2, adjusted R2, the increment Delta-R2, the CUS coefficient with its OLS
p-value, a partial-F test for adding CUS, and a bootstrap CI for Delta-R2. Because CUS
is a severity-weighted calibration error it is collinear with ECE, so the partial-F and
the Delta-R2 bootstrap CI are the honest evidence, not the (inflated-variance) CUS
coefficient alone.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _fit(df: pd.DataFrame, target: str, predictors: Sequence[str]):
    X = sm.add_constant(df[list(predictors)].to_numpy(dtype=float), has_constant="add")
    y = df[target].to_numpy(dtype=float)
    return sm.OLS(y, X).fit()


def _r2(df, target, predictors):
    return _fit(df, target, predictors).rsquared


def incremental_one(df: pd.DataFrame, target: str, base=("ece",), add="cus",
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Nested-regression incremental value of `add` over `base` for `target`."""
    sub = df[[target, *base, add]].dropna()
    n = len(sub)
    if n < len(base) + 3:
        return {"n_groups": n, "insufficient": True}
    m_base = _fit(sub, target, base)
    m_full = _fit(sub, target, list(base) + [add])
    # partial F test for adding `add`
    ftest = m_full.compare_f_test(m_base)  # (F, p, df_diff)
    delta_r2 = m_full.rsquared - m_base.rsquared
    # coefficient of `add` is the last slope (after const + base predictors)
    add_idx = 1 + len(base)
    add_coef = float(m_full.params[add_idx])
    add_p = float(m_full.pvalues[add_idx])
    # bootstrap CI for Delta-R2
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    idx_all = np.arange(n)
    for b in range(n_boot):
        bi = rng.choice(idx_all, size=n, replace=True)
        s = sub.iloc[bi]
        try:
            boots[b] = _r2(s, target, list(base) + [add]) - _r2(s, target, base)
        except Exception:
            boots[b] = np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {
        "n_groups": n,
        "r2_base": float(m_base.rsquared),
        "adj_r2_base": float(m_base.rsquared_adj),
        "r2_full": float(m_full.rsquared),
        "adj_r2_full": float(m_full.rsquared_adj),
        "delta_r2": float(delta_r2),
        "delta_r2_ci_lo": float(lo),
        "delta_r2_ci_hi": float(hi),
        "cus_coef": add_coef,
        "cus_p": add_p,
        "partial_F": float(ftest[0]),
        "partial_F_p": float(ftest[1]),
        "insufficient": False,
    }


def run_incremental(df: pd.DataFrame,
                    targets=("harmful@0.8", "sevwerr@0.8"),
                    base=("ece",), add="cus",
                    per_model: bool = True, seed: int = 0) -> pd.DataFrame:
    """Pooled and (optionally) per-model incremental-value table."""
    rows = []
    for target in targets:
        r = incremental_one(df, target, base=base, add=add, seed=seed)
        r.update({"target": target, "scope": "pooled (all groups)"})
        rows.append(r)
        if per_model:
            for model, sub in df.groupby("model"):
                r = incremental_one(sub, target, base=base, add=add, seed=seed)
                r.update({"target": target, "scope": f"model={model}"})
                rows.append(r)
    cols = ["target", "scope", "n_groups", "r2_base", "adj_r2_base", "r2_full",
            "adj_r2_full", "delta_r2", "delta_r2_ci_lo", "delta_r2_ci_hi",
            "cus_coef", "cus_p", "partial_F", "partial_F_p", "insufficient"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]]


def verdict(row: pd.Series) -> str:
    """Honest one-line verdict for a pooled result."""
    if row.get("insufficient"):
        return "insufficient groups"
    sig = (row["delta_r2_ci_lo"] > 0) and (row["partial_F_p"] < 0.05)
    if sig and row["delta_r2"] >= 0.02:
        return "CUS adds explanatory power beyond ECE"
    if sig:
        return "CUS adds small but significant variance beyond ECE"
    return "CUS adds no variance beyond ECE (behaves as severity-weighted ECE)"


if __name__ == "__main__":
    import os
    from .. import config
    src = os.path.join(config.RESULTS_DIR, "metrics", "metrics_by_bucket.csv")
    df = pd.read_csv(src)
    tbl = run_incremental(df)
    print(tbl.to_string(index=False))
    for _, r in tbl[tbl.scope.str.startswith("pooled")].iterrows():
        print(f"\n[{r['target']}] {verdict(r)}")
