"""Plotting suite for the MedBayes clinical-LLM evaluation package.

Every public function takes a data object (a records DataFrame or the tidy
``metrics_df``) plus an output path, renders a figure with a non-interactive
backend, saves a ~150 dpi PNG, and returns the saved path. All functions are
defensive against tiny / degenerate inputs (single group, all-correct, empty
buckets) so a benign edge case never raises.

Inputs
------
records : per-sample DataFrame with columns described in ``schema.RECORD_COLUMNS``
    (notably ``confidence``, ``correct``, ``severity``, ``uncertainty``,
    ``buckets`` and ``probs`` JSON columns).
metrics_df : tidy table keyed by some of [dataset, model, method, bucket] with
    metric columns (accuracy, ece, nll, brier, aurc, eaurc, cus, zti) and
    threshold columns such as ``harmful@0.8`` / ``cov@0.7`` / ``sevwerr@0.8``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive; never call plt.show()
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")

_DPI = 150


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _save(fig, path: str) -> str:
    """Save ``fig`` to ``path`` at the package DPI (plus a vector PDF twin) and close it."""
    _ensure_parent(path)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _groups(df: pd.DataFrame, group_col: str):
    """Yield (label, subframe) for ``group_col``; falls back to one group."""
    if group_col in df.columns and df[group_col].nunique(dropna=False) >= 1:
        for key, sub in df.groupby(group_col, sort=True, dropna=False):
            yield str(key), sub
    else:
        yield "all", df


def _legend_if_labeled(ax, **kwargs) -> None:
    """Add a legend only when the axes has labeled artists (avoids warnings)."""
    handles, lbls = ax.get_legend_handles_labels()
    if handles and lbls:
        ax.legend(**kwargs)


def _palette(n: int):
    n = max(int(n), 1)
    return sns.color_palette("tab10", n_colors=max(n, 3))[:n]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r robust to NaNs / constant input; returns nan when undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    xs, ys = x[m], y[m]
    if np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


# ---------------------------------------------------------------------------
# 1. risk-coverage curve
# ---------------------------------------------------------------------------
def risk_coverage_curve(records: pd.DataFrame, out: str,
                        group_col: str = "method") -> str:
    """Selective risk (cumulative error among accepted) vs coverage.

    For each group, samples are sorted by descending confidence and the
    cumulative error rate of the most-confident ``k`` predictions is plotted
    against coverage ``k / N``. One line per group.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [lbl for lbl, _ in _groups(records, group_col)]
    colors = _palette(len(labels))

    for color, (label, sub) in zip(colors, _groups(records, group_col)):
        sub = sub.dropna(subset=["confidence", "correct"])
        if sub.empty:
            continue
        order = np.argsort(-sub["confidence"].to_numpy(dtype=float))
        wrong = (~sub["correct"].to_numpy(dtype=bool))[order]
        n = len(wrong)
        coverage = np.arange(1, n + 1) / n
        risk = np.cumsum(wrong) / np.arange(1, n + 1)
        ax.plot(coverage, risk, label=label, color=color, lw=2)

    ax.set_xlabel("Coverage (fraction accepted)")
    ax.set_ylabel("Selective risk (error among accepted)")
    ax.set_title("Risk-coverage curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    _legend_if_labeled(ax, title=group_col, loc="best")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 2. bucket-wise harmful bar
# ---------------------------------------------------------------------------
def bucketwise_harmful_bar(metrics_df: pd.DataFrame, out: str,
                           harmful_col: str = "harmful@0.8",
                           bucket_col: str = "bucket",
                           group_col: str = "method") -> str:
    """Grouped bar chart of harmful-overconfident-error rate per bucket.

    Bars are grouped by ``bucket_col`` and hued by ``group_col`` (method).
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    if harmful_col not in metrics_df.columns or bucket_col not in metrics_df.columns:
        ax.text(0.5, 0.5, f"missing '{harmful_col}' or '{bucket_col}'",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _save(fig, out)

    cols = [bucket_col, harmful_col]
    hue = group_col if group_col in metrics_df.columns else None
    if hue:
        cols = [bucket_col, group_col, harmful_col]
    data = metrics_df[cols].dropna(subset=[harmful_col]).copy()

    if data.empty:
        ax.text(0.5, 0.5, "no harmful-rate data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return _save(fig, out)

    sns.barplot(data=data, x=bucket_col, y=harmful_col, hue=hue, ax=ax,
                palette="tab10", errorbar=None)
    ax.set_xlabel("Clinical risk category")
    ax.set_ylabel(f"Harmful overconfident-error rate ({harmful_col})")
    ax.set_title("Harmful overconfident errors by clinical risk category")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    if hue:
        ax.legend(title=group_col, loc="best")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 3. reliability diagram
# ---------------------------------------------------------------------------
def reliability_diagram(records: pd.DataFrame, out: str,
                        group_col: str = "method", n_bins: int = 15) -> str:
    """Calibration reliability diagram: bin accuracy vs mean confidence.

    Each group is overlaid as a line of empirical accuracy per equal-width
    confidence bin against the identity (perfect-calibration) diagonal.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1.5,
            label="perfect calibration")

    edges = np.linspace(0.0, 1.0, int(max(n_bins, 1)) + 1)
    labels = [lbl for lbl, _ in _groups(records, group_col)]
    colors = _palette(len(labels))

    for color, (label, sub) in zip(colors, _groups(records, group_col)):
        sub = sub.dropna(subset=["confidence", "correct"])
        if sub.empty:
            continue
        conf = sub["confidence"].to_numpy(dtype=float)
        acc = sub["correct"].to_numpy(dtype=float)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = idx == b
            if m.any():
                xs.append(conf[m].mean())
                ys.append(acc[m].mean())
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=color, label=label)

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    _legend_if_labeled(ax, title=group_col, loc="best")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 4. confidence histogram (correct vs incorrect)
# ---------------------------------------------------------------------------
def confidence_histogram(records: pd.DataFrame, out: str) -> str:
    """Overlaid confidence histograms for correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = records.dropna(subset=["confidence", "correct"])

    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return _save(fig, out)

    correct = sub.loc[sub["correct"].astype(bool), "confidence"].to_numpy(float)
    wrong = sub.loc[~sub["correct"].astype(bool), "confidence"].to_numpy(float)
    bins = np.linspace(0, 1, 26)

    if correct.size:
        ax.hist(correct, bins=bins, alpha=0.6, color="#2ca02c",
                label=f"correct (n={correct.size})", density=True)
    if wrong.size:
        ax.hist(wrong, bins=bins, alpha=0.6, color="#d62728",
                label=f"incorrect (n={wrong.size})", density=True)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence distribution: correct vs incorrect")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 5. abstention precision-recall curve
# ---------------------------------------------------------------------------
def abstention_pr_curve(records: pd.DataFrame, out: str,
                        group_col: str = "method") -> str:
    """Abstention PR curve swept over confidence threshold ``tau``.

    A prediction is abstained when ``confidence < tau``. Precision is
    ``P(wrong | abstained)`` and recall is ``P(abstained | wrong)``; the curve
    traces both as ``tau`` sweeps ``[0, 1]``. One line per group.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    taus = np.linspace(0.0, 1.0, 101)
    labels = [lbl for lbl, _ in _groups(records, group_col)]
    colors = _palette(len(labels))

    for color, (label, sub) in zip(colors, _groups(records, group_col)):
        sub = sub.dropna(subset=["confidence", "correct"])
        if sub.empty:
            continue
        conf = sub["confidence"].to_numpy(dtype=float)
        wrong = (~sub["correct"].to_numpy(dtype=bool))
        total_wrong = wrong.sum()
        precision, recall = [], []
        for tau in taus:
            abst = conf < tau
            n_abst = abst.sum()
            prec = (wrong & abst).sum() / n_abst if n_abst > 0 else 1.0
            rec = (wrong & abst).sum() / total_wrong if total_wrong > 0 else 0.0
            precision.append(prec)
            recall.append(rec)
        ax.plot(recall, precision, label=label, color=color, lw=2)

    ax.set_xlabel("Recall  P(abstained | wrong)")
    ax.set_ylabel("Precision  P(wrong | abstained)")
    ax.set_title("Abstention precision-recall (sweep over tau)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _legend_if_labeled(ax, title=group_col, loc="best")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 6. severity-weighted error vs coverage
# ---------------------------------------------------------------------------
def severity_weighted_error_vs_coverage(records: pd.DataFrame, out: str,
                                        group_col: str = "method") -> str:
    """Severity-weighted error among accepted vs coverage (sweep over tau).

    Acceptance is ``confidence >= tau``; the y-value is the mean of
    ``severity * wrong`` over accepted samples, plotted against coverage
    ``n_accepted / N``. One line per group.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    taus = np.linspace(0.0, 1.0, 101)
    labels = [lbl for lbl, _ in _groups(records, group_col)]
    colors = _palette(len(labels))

    for color, (label, sub) in zip(colors, _groups(records, group_col)):
        sub = sub.dropna(subset=["confidence", "correct"])
        if sub.empty:
            continue
        conf = sub["confidence"].to_numpy(dtype=float)
        wrong = (~sub["correct"].to_numpy(dtype=bool)).astype(float)
        sev = sub["severity"].fillna(1.0).to_numpy(dtype=float)
        n = len(conf)
        cov, swerr = [], []
        for tau in taus:
            acc = conf >= tau
            k = acc.sum()
            if k == 0:
                continue
            cov.append(k / n)
            swerr.append(float(np.mean(sev[acc] * wrong[acc])))
        if cov:
            order = np.argsort(cov)
            ax.plot(np.asarray(cov)[order], np.asarray(swerr)[order],
                    label=label, color=color, lw=2)

    ax.set_xlabel("Coverage (fraction accepted)")
    ax.set_ylabel("Severity-weighted error among accepted")
    ax.set_title("Severity-weighted error vs coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    _legend_if_labeled(ax, title=group_col, loc="best")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 7. OOD uncertainty distribution
# ---------------------------------------------------------------------------
def ood_uncertainty_distribution(records_id: pd.DataFrame,
                                 records_ood: pd.DataFrame, out: str) -> str:
    """Overlaid uncertainty distributions for in-distribution vs OOD records."""
    fig, ax = plt.subplots(figsize=(7, 5))

    id_u = (records_id["uncertainty"].dropna().to_numpy(float)
            if "uncertainty" in records_id.columns else np.array([]))
    ood_u = (records_ood["uncertainty"].dropna().to_numpy(float)
             if "uncertainty" in records_ood.columns else np.array([]))

    if id_u.size == 0 and ood_u.size == 0:
        ax.text(0.5, 0.5, "no uncertainty data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return _save(fig, out)

    def _draw(values, label, color):
        if values.size == 0:
            return
        # KDE needs variance and >1 point; otherwise fall back to a histogram.
        if values.size > 1 and np.std(values) > 1e-9:
            try:
                sns.kdeplot(values, ax=ax, fill=True, alpha=0.35,
                            color=color, label=f"{label} (n={values.size})")
                return
            except Exception:
                pass
        ax.hist(values, bins=20, alpha=0.5, color=color, density=True,
                label=f"{label} (n={values.size})")

    _draw(id_u, "in-distribution", "#1f77b4")
    _draw(ood_u, "OOD", "#ff7f0e")

    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty distribution: in-distribution vs OOD")
    _legend_if_labeled(ax, loc="best")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 8. CUS / ZTI vs clinical-metric scatter grid
# ---------------------------------------------------------------------------
def cus_zti_correlation(metrics_df: pd.DataFrame, out: str,
                        x_cols: Sequence[str] = ("harmful@0.8", "aurc", "sevwerr@0.8"),
                        y_cols: Sequence[str] = ("cus", "zti")) -> str:
    """Scatter grid of CUS / ZTI (rows) against clinical metrics (cols).

    Each panel scatters one ``y_col`` against one ``x_col`` across groups and
    annotates the Pearson correlation ``r`` in the panel title.
    """
    x_cols = [c for c in x_cols if c in metrics_df.columns]
    y_cols = [c for c in y_cols if c in metrics_df.columns]

    if not x_cols or not y_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no matching x/y metric columns", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _save(fig, out)

    nrows, ncols = len(y_cols), len(x_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
                             squeeze=False)

    for i, yc in enumerate(y_cols):
        for j, xc in enumerate(x_cols):
            ax = axes[i][j]
            sub = metrics_df[[xc, yc]].dropna()
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                xv = sub[xc].to_numpy(float)
                yv = sub[yc].to_numpy(float)
                ax.scatter(xv, yv, s=36, alpha=0.75, color="#4c72b0",
                           edgecolor="white", linewidth=0.5)
                r = _pearson(xv, yv)
                # least-squares trend line when well-defined
                if np.isfinite(r) and len(xv) >= 2 and np.std(xv) > 0:
                    coef = np.polyfit(xv, yv, 1)
                    xl = np.array([xv.min(), xv.max()])
                    ax.plot(xl, coef[0] * xl + coef[1], color="#c44e52",
                            lw=1.4, ls="--")
                r_txt = "n/a" if not np.isfinite(r) else f"{r:.2f}"
                ax.set_title(f"{yc} vs {xc}  (r={r_txt})", fontsize=10)
            ax.set_xlabel(xc)
            ax.set_ylabel(yc)

    fig.suptitle("CUS / ZTI vs clinical metrics", y=1.0, fontsize=13)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def make_all_plots(records: pd.DataFrame, metrics_df: pd.DataFrame,
                   outdir: str) -> Dict[str, str]:
    """Render every plot with sensible defaults into ``outdir``.

    Each plot is guarded so a single failure (or absent column) does not abort
    the remainder. Returns a ``name -> saved-path`` dict for the plots that
    succeeded.
    """
    os.makedirs(outdir, exist_ok=True)
    produced: Dict[str, str] = {}

    def _run(name: str, fn, *args, **kwargs):
        path = os.path.join(outdir, f"{name}.png")
        try:
            produced[name] = fn(*args, path, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[plots] {name} failed: {type(exc).__name__}: {exc}")

    _run("risk_coverage_curve", risk_coverage_curve, records)
    _run("bucketwise_harmful_bar", bucketwise_harmful_bar, metrics_df)
    _run("reliability_diagram", reliability_diagram, records)
    _run("confidence_histogram", confidence_histogram, records)
    _run("abstention_pr_curve", abstention_pr_curve, records)
    _run("severity_weighted_error_vs_coverage",
         severity_weighted_error_vs_coverage, records)
    # OOD plot needs two frames; split by dataset if available, else reuse.
    if "dataset" in records.columns and records["dataset"].nunique() >= 2:
        ds = sorted(records["dataset"].unique())
        rec_id = records[records["dataset"] == ds[0]]
        rec_ood = records[records["dataset"] == ds[-1]]
    else:
        rec_id, rec_ood = records, records
    _run("ood_uncertainty_distribution", ood_uncertainty_distribution,
         rec_id, rec_ood)
    _run("cus_zti_correlation", cus_zti_correlation, metrics_df)

    return produced


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _synth_records(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthesize a plausible records DataFrame for smoke testing."""
    rng = np.random.default_rng(seed)
    bucket_pool = [
        ["medication_drug_interaction", "high_severity"],
        ["diagnosis_differential"],
        ["lab_biomarker_interpretation"],
        ["low_severity"],
        ["ambiguous_symptoms", "missing_evidence"],
    ]
    rows = []
    for i in range(n):
        method = "medbayes_lite" if i % 2 == 0 else "base"
        dataset = "pubmedqa" if i % 3 == 0 else "medmcqa"
        n_classes = 3 if dataset == "pubmedqa" else 4
        # medbayes_lite is better calibrated: lower confidence on errors
        base_acc = 0.78 if method == "medbayes_lite" else 0.7
        correct = rng.random() < base_acc
        if correct:
            conf = float(np.clip(rng.beta(6, 2), 0, 1))
        else:
            shape = (2, 4) if method == "medbayes_lite" else (4, 3)
            conf = float(np.clip(rng.beta(*shape), 0, 1))
        # build a probs vector consistent with confidence
        rest = (1.0 - conf)
        others = rng.dirichlet(np.ones(n_classes - 1)) * rest
        probs = np.concatenate([[conf], others])
        rng.shuffle(probs)
        pred = int(np.argmax(probs))
        label = pred if correct else (pred + 1) % n_classes
        uncertainty = float(-np.sum(probs * np.log(probs + 1e-12)))
        buckets = bucket_pool[i % len(bucket_pool)]
        from medbayes_eval.config import severity_for_buckets
        severity = severity_for_buckets(buckets)
        rows.append({
            "dataset": dataset, "model": "biobert_medmcqa", "method": method,
            "seed": 0, "uid": f"u{i}", "label": label, "pred": pred,
            "n_classes": n_classes, "confidence": conf,
            "uncertainty": uncertainty, "probs": json.dumps(probs.tolist()),
            "correct": bool(correct), "buckets": json.dumps(buckets),
            "severity": float(severity),
        })
    return pd.DataFrame(rows)


def _synth_metrics(records: pd.DataFrame) -> pd.DataFrame:
    """Build a small per-(method, bucket) tidy metrics_df from records."""
    rng = np.random.default_rng(1)
    rows = []
    buckets = ["medication_drug_interaction", "diagnosis_differential",
               "lab_biomarker_interpretation", "low_severity",
               "ambiguous_symptoms"]
    for method in sorted(records["method"].unique()):
        better = method == "medbayes_lite"
        for b in buckets:
            rows.append({
                "method": method,
                "bucket": b,
                "accuracy": float(rng.uniform(0.6, 0.85)),
                "ece": float(rng.uniform(0.02, 0.12) * (0.6 if better else 1.0)),
                "aurc": float(rng.uniform(0.05, 0.2) * (0.7 if better else 1.0)),
                "harmful@0.8": float(rng.uniform(0.01, 0.1) * (0.5 if better else 1.0)),
                "sevwerr@0.8": float(rng.uniform(0.1, 0.4) * (0.7 if better else 1.0)),
                "cus": float(rng.uniform(0.4, 0.9) + (0.1 if better else 0.0)),
                "zti": float(rng.uniform(0.3, 0.8) + (0.1 if better else 0.0)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_dir = "/tmp/plot_test"
    recs = _synth_records()
    mdf = _synth_metrics(recs)
    paths = make_all_plots(recs, mdf, out_dir)

    print(f"\nProduced {len(paths)} plots in {out_dir}:")
    for name, p in sorted(paths.items()):
        exists = os.path.isfile(p)
        size = os.path.getsize(p) if exists else 0
        print(f"  [{'OK' if exists else 'MISSING'}] {name:38s} {p} ({size} bytes)")

    missing = [n for n, p in paths.items() if not os.path.isfile(p)]
    if missing:
        print(f"\nWARNING: missing files for: {missing}")
    else:
        print("\nAll produced files exist on disk.")
