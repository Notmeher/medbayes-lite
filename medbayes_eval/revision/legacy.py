"""Legacy-format outputs from the revision caches: per-sample records (both gates),
the original metric tables, and figures (PNG + vector PDF)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..metrics import bucket_metrics_table, correlation_analysis, metrics_table
from . import paths
from .fastmetrics import normalized_conf

PAPER_METHODS = ("base", "temp_scaling", "mc_dropout", "medbayes_lite", "deep_ensemble",
                 "conformal", "self_consistency", "semantic_entropy", "verbalized")


def records_frame(store) -> pd.DataFrame:
    frames = []
    for r in store.runs:
        if r.method not in PAPER_METHODS:
            continue
        it = store.items(r.dataset).loc[r.uids]
        conf = r.probs.max(-1) if r.conf_override is None else r.conf_override
        pred = r.probs.argmax(-1)
        frames.append(pd.DataFrame({
            "dataset": r.dataset, "model": r.model, "method": r.method, "seed": r.seed,
            "uid": r.uids, "label": r.labels, "pred": pred, "n_classes": r.probs.shape[1],
            "confidence": conf, "conf_entropy_gate": normalized_conf(r.probs),
            "uncertainty": -(np.clip(r.probs, 1e-12, 1) * np.log(np.clip(r.probs, 1e-12, 1))).sum(-1),
            "probs": [json.dumps([round(float(x), 6) for x in p]) for p in r.probs],
            "correct": pred == r.labels, "buckets": it.buckets.to_numpy(),
            "severity": it.severity.to_numpy(float)}))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(paths.out("raw_predictions", "all_records.parquet"), index=False)
    return df


def legacy_tables(df: pd.DataFrame):
    out = os.path.join(paths.ROOT, "legacy_metrics")
    os.makedirs(out, exist_ok=True)
    overall = metrics_table(df, group_cols=("dataset", "model", "method"))
    bucket = bucket_metrics_table(df, group_cols=("dataset", "model", "method"))
    overall.to_csv(os.path.join(out, "metrics_overall.csv"), index=False)
    bucket.to_csv(os.path.join(out, "metrics_by_bucket.csv"), index=False)
    corr = correlation_analysis(pd.concat([overall, bucket], ignore_index=True))
    corr.to_csv(os.path.join(out, "cus_zti_correlation.csv"), index=False)
    return overall, bucket


def figures(df: pd.DataFrame, overall: pd.DataFrame, bucket: pd.DataFrame, cov=None):
    from ..plotting import plots as P
    main = df[(df.dataset == "medmcqa") & (df.model == "pubmedbert_medmcqa") &
              df.method.isin(["base", "temp_scaling", "mc_dropout", "medbayes_lite"])]
    out = os.path.join(paths.ROOT, "figures", "maxprob_gate")
    P.make_all_plots(main, pd.concat([overall, bucket], ignore_index=True), out)
    ent = main.assign(confidence=main.conf_entropy_gate)
    out_e = os.path.join(paths.ROOT, "figures", "entropy_gate")
    for name, fn in [("risk_coverage_curve", P.risk_coverage_curve),
                     ("abstention_pr_curve", P.abstention_pr_curve),
                     ("severity_weighted_error_vs_coverage", P.severity_weighted_error_vs_coverage)]:
        fn(ent, os.path.join(out_e, f"{name}.png"))
    if cov is not None:
        matched_coverage_figure(cov)
    return os.path.join(paths.ROOT, "figures")


def matched_coverage_figure(cov: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["base", "temp_scaling", "mc_dropout", "medbayes_lite"]
    sets = [(d, m) for d, m in [("medmcqa", "pubmedbert_medmcqa"), ("medmcqa", "biobert_medmcqa"),
                                ("medqa", "pubmedbert_medmcqa"), ("medqa", "biobert_medmcqa")]
            if ((cov.dataset == d) & (cov.model == m)).any()]
    if not sets:
        return
    for metric in ("selective_risk", "harmful_per_total"):
        fig, axes = plt.subplots(2, len(sets), figsize=(4 * len(sets), 6.5), squeeze=False)
        for gi, gate in enumerate(("maxprob", "entropy")):
            for si, (d, m) in enumerate(sets):
                ax = axes[gi, si]
                for me in methods:
                    s = cov[(cov.dataset == d) & (cov.model == m) & (cov.method == me) &
                            (cov.gate == gate)].sort_values("target_coverage")
                    if s.empty:
                        continue
                    ax.plot(s.target_coverage, s[metric], marker="o", ms=3, label=me)
                    ax.fill_between(s.target_coverage, s[metric + "_ci_lo"],
                                    s[metric + "_ci_hi"], alpha=0.15)
                ax.set_title(f"{d} / {m.split('_')[0]} ({gate} gate)", fontsize=9)
                ax.set_xlabel("coverage (matched)")
                ax.set_ylabel(metric.replace("_", " "))
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        p = paths.out("figures", f"matched_coverage_{metric}.png")
        fig.savefig(p, dpi=150)
        fig.savefig(p[:-4] + ".pdf")
        plt.close(fig)
