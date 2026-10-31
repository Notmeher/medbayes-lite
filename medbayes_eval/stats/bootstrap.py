"""Bootstrap and rank-test utilities for comparing UQ methods.

Every metric in this package reduces a per-sample record DataFrame to a scalar.
To attach uncertainty to those scalars (and to compare two methods fairly), we
resample with replacement and report percentile confidence intervals. Paired
comparisons align two methods on shared `uid`s so the same items are resampled
together, which removes item-difficulty variance from the difference.

All randomness flows through ``numpy.random.default_rng(seed)`` so results are
reproducible across runs and platforms.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values, stat: Callable[[np.ndarray], float] = np.mean,
                 n: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a statistic of a 1-D sample.

    Parameters
    ----------
    values : array-like
        The observations to resample.
    stat : callable
        Maps a 1-D array to a scalar (default: mean).
    n : int
        Number of bootstrap resamples.
    alpha : float
        Two-sided miscoverage; the CI spans the [alpha/2, 1 - alpha/2] quantiles.
    seed : int
        Seed for the random generator.

    Returns
    -------
    (point, lo, hi) : tuple of float
        Statistic on the original sample and the percentile interval bounds.
        Bounds are NaN when the sample is empty.
    """
    values = np.asarray(values, dtype=float)
    point = float(stat(values)) if values.size else float("nan")
    if values.size == 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    boot = np.array([stat(values[row]) for row in idx], dtype=float)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def bootstrap_metric(df: pd.DataFrame, metric_fn: Callable[[pd.DataFrame], float],
                     n: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a metric computed over record ROWS.

    Unlike :func:`bootstrap_ci`, the resampled unit is a full DataFrame row, so
    the metric may use any columns (e.g. confidence and correctness jointly).

    Returns
    -------
    (point, lo, hi) : tuple of float
        Metric on the original rows and the percentile interval bounds. Bounds
        are NaN when ``df`` is empty.
    """
    point = float(metric_fn(df)) if len(df) else float("nan")
    if len(df) == 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_rows = len(df)
    boot = np.empty(n, dtype=float)
    for b in range(n):
        idx = rng.integers(0, n_rows, size=n_rows)
        boot[b] = metric_fn(df.iloc[idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     metric_fn: Callable[[pd.DataFrame], float],
                     n: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired percentile bootstrap of the metric difference (A minus B).

    The two frames are inner-joined on `uid` so only shared items contribute,
    and both are reindexed to that shared order. Each bootstrap draw selects a
    set of positions and applies it to *both* frames, keeping the pairing intact.

    Returns
    -------
    dict
        ``diff`` (point estimate metric(A) - metric(B)), ``ci_lo``, ``ci_hi``
        (percentile interval of the bootstrap difference), and ``p_value``
        (two-sided ``2 * min(P(diff <= 0), P(diff >= 0))``). All NaN when there
        is no shared item.
    """
    shared = pd.Index(df_a["uid"]).intersection(pd.Index(df_b["uid"]))
    a = df_a.set_index("uid").loc[shared].reset_index()
    b = df_b.set_index("uid").loc[shared].reset_index()

    n_rows = len(shared)
    if n_rows == 0:
        return {"diff": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value": float("nan")}

    diff = float(metric_fn(a) - metric_fn(b))

    rng = np.random.default_rng(seed)
    boot = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, n_rows, size=n_rows)
        boot[i] = metric_fn(a.iloc[idx]) - metric_fn(b.iloc[idx])

    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p_le = float(np.mean(boot <= 0))
    p_ge = float(np.mean(boot >= 0))
    p_value = min(1.0, 2.0 * min(p_le, p_ge))
    return {"diff": diff, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": p_value}


def wilcoxon_seeds(values_a, values_b) -> dict:
    """Paired Wilcoxon signed-rank test across seeds (or any paired runs).

    Degenerate inputs (fewer than one pair, or all paired differences zero) make
    the test undefined; we return ``p_value=nan`` instead of raising.

    Returns
    -------
    dict
        ``statistic`` and ``p_value`` from :func:`scipy.stats.wilcoxon`.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diffs = a - b
    if diffs.size < 1 or np.allclose(diffs, 0.0):
        return {"statistic": float("nan"), "p_value": float("nan")}

    res = stats.wilcoxon(a, b)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def seed_aggregate(metrics_df: pd.DataFrame, group_cols, value_cols) -> pd.DataFrame:
    """Aggregate per-seed metrics into mean/std summaries.

    Parameters
    ----------
    metrics_df : DataFrame
        One row per (group, seed) with the metric columns to summarize.
    group_cols : list[str]
        Columns identifying a configuration (e.g. dataset, model, method).
    value_cols : list[str]
        Metric columns to average across seeds.

    Returns
    -------
    DataFrame
        One row per group with ``f"{col}_mean"`` and ``f"{col}_std"`` for each
        value column, plus ``n_seeds`` (the number of rows aggregated).
    """
    group_cols = list(group_cols)
    value_cols = list(value_cols)

    agg = {col: ["mean", "std"] for col in value_cols}
    grouped = metrics_df.groupby(group_cols, dropna=False)
    out = grouped.agg(agg)
    out.columns = [f"{col}_{kind}" for col, kind in out.columns]
    out["n_seeds"] = grouped.size()
    return out.reset_index()


def format_mean_std(mean: float, std: float, fmt: str = "{:.3f}") -> str:
    """Render a mean +/- std pair, e.g. ``"0.123 ± 0.011"``."""
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


if __name__ == "__main__":
    # Synthetic smoke test: two methods share uids; 'mbl' is more often correct.
    rng = np.random.default_rng(42)
    n_items = 400
    uids = [f"u{i}" for i in range(n_items)]

    base_correct = rng.random(n_items) < 0.62
    # mbl keeps base's correct items and flips some of base's mistakes to correct.
    flips = (~base_correct) & (rng.random(n_items) < 0.30)
    mbl_correct = base_correct | flips

    def _records(method, correct):
        return pd.DataFrame({
            "uid": uids,
            "method": method,
            "correct": correct.astype(bool),
        })

    df_base = _records("base", base_correct)
    df_mbl = _records("mbl", mbl_correct)

    metric_fn = lambda d: d.correct.mean()

    point, lo, hi = bootstrap_ci(df_base["correct"].to_numpy(dtype=float))
    print(f"bootstrap_ci (base accuracy): {point:.3f} "
          f"[{lo:.3f}, {hi:.3f}]")

    point_m, lo_m, hi_m = bootstrap_metric(df_mbl, metric_fn)
    print(f"bootstrap_metric (mbl accuracy): {point_m:.3f} "
          f"[{lo_m:.3f}, {hi_m:.3f}]")

    paired = paired_bootstrap(df_mbl, df_base, metric_fn)
    print(f"paired_bootstrap (mbl - base): diff={paired['diff']:.3f} "
          f"[{paired['ci_lo']:.3f}, {paired['ci_hi']:.3f}] "
          f"p={paired['p_value']:.4f}")

    # Per-seed accuracies for two methods; mbl consistently higher.
    seeds_base = np.array([0.610, 0.605, 0.620, 0.615, 0.608])
    seeds_mbl = np.array([0.690, 0.685, 0.700, 0.695, 0.688])
    wil = wilcoxon_seeds(seeds_mbl, seeds_base)
    print(f"wilcoxon_seeds (mbl vs base): statistic={wil['statistic']:.1f} "
          f"p={wil['p_value']:.4f}")

    # seed_aggregate / format_mean_std demonstration.
    metrics_df = pd.DataFrame({
        "method": ["base"] * 5 + ["mbl"] * 5,
        "seed": list(range(5)) * 2,
        "acc": np.concatenate([seeds_base, seeds_mbl]),
    })
    summary = seed_aggregate(metrics_df, ["method"], ["acc"])
    print("seed_aggregate:")
    print(summary.to_string(index=False))
    for _, row in summary.iterrows():
        print(f"  {row['method']}: "
              f"{format_mean_std(row['acc_mean'], row['acc_std'])}")
