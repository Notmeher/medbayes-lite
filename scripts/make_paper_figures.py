"""Vector (PDF) figures for the revised manuscript, built from the revision outputs.

  MEDBAYES_RESULTS_DIR=<repo> python scripts/make_paper_figures.py <out_dir>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MEDBAYES_RESULTS_DIR", Path(__file__).resolve().parents[1])) / "revision"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = {"base": "#4d4d4d", "temp_scaling": "#1f77b4", "mc_dropout": "#2ca02c", "medbayes_lite": "#d62728",
       "deep_ensemble": "#9467bd"}
LAB = {"base": "Base (single pass)", "temp_scaling": "Temperature scaling", "mc_dropout": "MC dropout",
       "medbayes_lite": "MedBayes-Lite", "deep_ensemble": "Deep ensemble"}
SETTINGS = [("medmcqa", "pubmedbert_medmcqa", "MedMCQA\nPubMedBERT"),
            ("medmcqa", "biobert_medmcqa", "MedMCQA\nBioBERT"),
            ("medqa", "pubmedbert_medmcqa", "MedQA-USMLE\nPubMedBERT"),
            ("medqa", "biobert_medmcqa", "MedQA-USMLE\nBioBERT")]
METHODS = ["base", "temp_scaling", "mc_dropout", "medbayes_lite"]
HIGH_SEV = 2.5

rec = pd.read_parquet(ROOT / "raw_predictions" / "all_records.parquet")


def runs(ds, model, method):
    d = rec[(rec.dataset == ds) & (rec.model == model) & (rec.method == method)]
    return [g for _, g in d.groupby("seed")]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def sweep(g, conf_col="confidence"):
    """Coverage grid -> (selective risk, harmful errors per total item) when accepting the most confident items."""
    order = np.argsort(-g[conf_col].to_numpy(), kind="stable")
    wrong = ~g.correct.to_numpy()[order]
    harm = wrong & (g.severity.to_numpy()[order] >= HIGH_SEV)
    n = len(g)
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(wrong) / k, np.cumsum(harm) / n


def interp_mean(gs, grid, which):
    ys = []
    for g in gs:
        cov, risk, harm = sweep(g)
        ys.append(np.interp(grid, cov, risk if which == "risk" else harm))
    return np.mean(ys, axis=0)


# Figure: reliability diagrams --------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
bins = np.linspace(0, 1, 16)
for ax, (ds, model, title) in zip(axes, SETTINGS[:1] + SETTINGS[2:3]):
    ax.plot([0, 1], [0, 1], ls=":", c="k", lw=0.8)
    for m in METHODS:
        g = pd.concat(runs(ds, model, m))
        idx = np.clip(np.digitize(g.confidence, bins) - 1, 0, 14)
        x, y = [], []
        for b in range(15):
            s = idx == b
            if s.sum() >= 20:
                x.append(g.confidence[s].mean()); y.append(g.correct[s].mean())
        ax.plot(x, y, marker="o", ms=2.5, c=COL[m], label=LAB[m])
    ax.set(title=title.replace("\n", ", "), xlabel="Mean confidence (max probability)", xlim=(0.2, 1), ylim=(0, 1))
axes[0].set_ylabel("Empirical accuracy")
axes[0].legend(loc="upper left", frameon=False)
save(fig, "fig_reliability")

# Figure: risk-coverage and harmful-per-total vs coverage, with tau=0.7 operating points -----
grid = np.linspace(0.02, 1, 400)
fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), sharex=True)
for j, (ds, model, title) in enumerate(SETTINGS):
    for m in METHODS:
        gs = runs(ds, model, m)
        axes[0, j].plot(grid, interp_mean(gs, grid, "risk"), c=COL[m], label=LAB[m])
        axes[1, j].plot(grid, interp_mean(gs, grid, "harm"), c=COL[m])
        cov = np.mean([(g.confidence >= 0.7).mean() for g in gs])
        hpt = np.mean([((g.confidence >= 0.7) & ~g.correct & (g.severity >= HIGH_SEV)).mean() for g in gs])
        axes[1, j].plot([cov], [hpt], marker="D", ms=4, c=COL[m], mec="white", mew=0.5, zorder=5)
    axes[0, j].set_title(title)
fig.supxlabel("Coverage (fraction of items accepted); diamonds = operating point at the fixed threshold $\\tau=0.7$",
              fontsize=7.5)
axes[0, 0].set_ylabel("Selective risk\n(error rate among accepted)")
axes[1, 0].set_ylabel("Harmful errors per\ntotal item (sev. $\\geq$2.5)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
save(fig, "fig_matched_coverage")

# Figure: deferral workload per 1000 cases -------------------------------------------------
wl = pd.read_csv(ROOT / "tables" / "deferral_workload_per_1000.csv")
wl = wl[(wl.operating_point == "tau=0.7")]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
for ax, (ds, model, title) in zip(axes, SETTINGS):
    s = wl[(wl.dataset == ds) & (wl.model == model) & (wl.gate == "maxprob")].set_index("method")
    ms = [m for m in ["base", "temp_scaling", "medbayes_lite"] if m in s.index]
    x = np.arange(len(ms))
    ax.bar(x, s.loc[ms, "deferred_per_1000"], color=[COL[m] for m in ms], alpha=0.35, label="Deferred to reviewer")
    ax.bar(x, s.loc[ms, "errors_passed_per_1000"], color=[COL[m] for m in ms], label="Errors passed through")
    ax.bar(x, s.loc[ms, "harmful_passed_per_1000"], color="none", edgecolor="k", hatch="////", lw=0.5,
           label="High-severity errors passed")
    ax.set_xticks(x, ["Base", "TS", "MBL"])
    ax.set_title(title)
axes[0].set_ylabel("Cases per 1000 ($\\tau=0.7$)")
from matplotlib.patches import Patch
fig.legend([Patch(color="#999999", alpha=0.35), Patch(color="#999999"),
            Patch(facecolor="none", edgecolor="k", hatch="////")],
           ["Deferred to human review", "Errors passed through (accepted, wrong)",
            "High-severity errors passed through"], ncol=3, frameon=False, loc="lower center",
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
save(fig, "fig_workload")

# Figure: compute-quality curve ------------------------------------------------------------
cc = pd.read_csv(ROOT / "tables" / "compute_quality_matched_latency.csv")
lat = pd.read_csv(ROOT / "tables" / "latency_benchmark.csv")
lat = lat[lat.backbone.str.startswith("PubMedBERT")]
packed = lat[lat.method.str.startswith("MC dropout packed")].set_index("forward_passes").latency_ms_bs1_median
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
for variant, c in [("MC dropout only", COL["mc_dropout"]), ("MedBayes-Lite (MC + T)", COL["medbayes_lite"])]:
    s = cc[(cc.variant == variant) & (cc.gate == "maxprob")].sort_values("forward_passes")
    axes[0].plot(s.forward_passes, s.ece, marker="o", ms=3, c=c, label=variant)
    axes[1].plot(s.latency_ms_bs1_median, s.ece, marker="o", ms=3, c=c, label=f"{variant}, sequential")
    axes[1].plot(packed.reindex(s.forward_passes).to_numpy(), s.ece, marker="s", ms=3, c=c, ls="--",
                 mfc="white", label=f"{variant}, packed batch")
for m in ["base", "temp_scaling"]:
    r = cc[(cc.variant == m) & (cc.gate == "maxprob")].iloc[0]
    axes[0].axhline(r.ece, c=COL[m], ls=":", lw=1, label=LAB[m])
    axes[1].plot([r.latency_ms_bs1_median], [r.ece], marker="*", ms=8, c=COL[m], ls="none", label=LAB[m])
axes[0].set(xscale="log", xlabel="Monte Carlo passes $M$", ylabel="ECE (MedMCQA, PubMedBERT)")
axes[0].set_xticks([1, 2, 3, 5, 10, 15, 20, 30, 50], ["1", "2", "3", "5", "10", "15", "20", "30", "50"])
axes[1].set(xscale="log", xlabel="Median latency per item, ms (T4, batch 1)")
axes[1].set_xticks([10, 20, 50, 100, 200, 500], ["10", "20", "50", "100", "200", "500"])
axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].legend(frameon=False, fontsize=6)
axes[1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
save(fig, "fig_compute")

# Figure: harmful errors by heuristic evaluation group --------------------------------------
GROUPS = ["high_severity", "diagnosis_differential", "treatment_recommendation", "medication_drug_interaction",
          "risk_prognosis", "rare_disease", "lab_biomarker_interpretation", "numeric_vital_lab_heavy",
          "contradictory_evidence", "negation_heavy", "low_severity"]
NICE = {"high_severity": "High-severity cue", "diagnosis_differential": "Diagnosis", "treatment_recommendation":
        "Treatment", "medication_drug_interaction": "Medication", "risk_prognosis": "Risk / prognosis",
        "rare_disease": "Rare condition", "lab_biomarker_interpretation": "Lab / biomarker",
        "numeric_vital_lab_heavy": "Numeric-heavy", "contradictory_evidence": "Contradictory cue",
        "negation_heavy": "Negation-heavy", "low_severity": "Routine (fallback)"}
import json
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, (ds, model, title) in zip(axes, [SETTINGS[0], SETTINGS[2]]):
    w = 0.27
    for k, m in enumerate(["base", "temp_scaling", "medbayes_lite"]):
        g = pd.concat(runs(ds, model, m))
        b = g.buckets.map(json.loads)
        vals = []
        for grp in GROUPS:
            s = b.map(lambda x: grp in x)
            vals.append(((g.confidence >= 0.8) & ~g.correct)[s].mean() if s.sum() >= 10 else np.nan)
        ax.barh(np.arange(len(GROUPS)) + (k - 1) * w, vals, height=w, color=COL[m], label=LAB[m])
    ax.set_yticks(np.arange(len(GROUPS)), [NICE[x] for x in GROUPS])
axes[0].invert_yaxis()
for ax, (ds, model, title) in zip(axes, [SETTINGS[0], SETTINGS[2]]):
    ax.set(title=title.replace("\n", ", "))
fig.supxlabel("Confident-error rate within group (confidence $\\geq$0.8 and wrong)", fontsize=7.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
save(fig, "fig_groups")

# Figure: domain shift with source- vs target-fitted temperature ----------------------------
ood = pd.read_csv(ROOT / "tables" / "ood_transfer_3seed.csv")
ood = ood[ood.gate == "maxprob"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
rows = [("base", "-", "Base"), ("temp_scaling", "source", "TS (T from source)"),
        ("temp_scaling", "target", "TS (T from target)"), ("medbayes_lite", "source", "MBL (T from source)"),
        ("medbayes_lite", "target", "MBL (T from target)")]
for ax, model, title in [(axes[0], "pubmedbert_medmcqa", "PubMedBERT"), (axes[1], "biobert_medmcqa", "BioBERT")]:
    s = ood[ood.model == model]
    for k, (m, fit, lab) in enumerate(rows):
        r = s[(s.method == m) & (s.ood_temperature_fit_on.fillna("-") == fit)]
        if r.empty:
            continue
        r = r.iloc[0]
        c = COL[m]
        ax.errorbar([k], [r.drift_ece], yerr=[[r.drift_ece - r.drift_ece_ci_lo], [r.drift_ece_ci_hi - r.drift_ece]],
                    fmt="o" if fit != "target" else "s", c=c, mfc=c if fit != "target" else "white", capsize=2, ms=4)
        ax.annotate(f"{r['ood_harmful@0.8']:.3f}", (k, r.drift_ece_ci_hi), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(range(len(rows)), [r[2] for r in rows], rotation=30, ha="right")
    ax.set(title=f"MedMCQA $\\to$ MedQA-USMLE, {title}", ylim=(-0.03, 0.16), xlim=(-0.5, len(rows) - 0.5))
axes[0].set_ylabel("ECE drift (OOD $-$ ID), 95% CI")
fig.text(0.5, 1.0, "Numbers above bars: OOD harmful@0.8 (harmful errors per total MedQA-USMLE item)",
         ha="center", fontsize=7)
fig.tight_layout()
save(fig, "fig_shift")
