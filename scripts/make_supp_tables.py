"""Supplementary-table LaTeX bodies for the revised manuscript, generated from the revision CSVs.

  MEDBAYES_RESULTS_DIR=<repo> python scripts/make_supp_tables.py <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MEDBAYES_RESULTS_DIR", Path(__file__).resolve().parents[1])) / "revision"
TAB = ROOT / "tables"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "supp_tables")
OUT.mkdir(parents=True, exist_ok=True)

SET = {("medmcqa", "pubmedbert_medmcqa"): "MedMCQA, PubMedBERT", ("medmcqa", "biobert_medmcqa"): "MedMCQA, BioBERT",
       ("medqa", "pubmedbert_medmcqa"): "MedQA-USMLE, PubMedBERT", ("medqa", "biobert_medmcqa"): "MedQA-USMLE, BioBERT"}
METH = {"base": "Base", "temp_scaling": "TS", "mc_dropout": "MC dropout", "medbayes_lite": "MedBayes-Lite",
        "medbayes_lite_srcT": "MedBayes-Lite (src $T$)", "temp_scaling_srcT": "TS (src $T$)",
        "deep_ensemble": "Deep ensemble", "conformal": "Conformal (LAC)"}
GATE = {"maxprob": "max-prob", "entropy": "entropy"}
BUCK = {"medication_drug_interaction": "Medication", "diagnosis_differential": "Diagnosis",
        "treatment_recommendation": "Treatment", "risk_prognosis": "Risk / prognosis", "rare_disease": "Rare condition",
        "lab_biomarker_interpretation": "Lab / biomarker", "numeric_vital_lab_heavy": "Numeric-heavy",
        "ambiguous_symptoms": "Ambiguous", "missing_evidence": "Missing evidence",
        "contradictory_evidence": "Contradictory", "negation_heavy": "Negation-heavy",
        "high_severity": "High-severity cue", "low_severity": "Routine", "ALL (severity changes)": "Any high-severity"}
MCQ_ORDER = list(SET)


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}".replace("-", "$-$") if x < 0 else f"{x:.{d}f}"


def ci(lo, hi, d=3):
    return f"[{f(lo, d)}, {f(hi, d)}]"


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def p_fmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, body):
    body = (body.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\textwidth}\begin{tabular}")
            .replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
            .replace("\\\\[6pt]\n", "\\par\\medskip\n").replace("}\\\\[2pt]\n", "}\\par\\smallskip\n"))
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", OUT / f"{name}.tex")


def mcq(df):
    df = df[df.apply(lambda r: (r.dataset, r.model) in SET, axis=1)].copy()
    df["_o"] = df.apply(lambda r: MCQ_ORDER.index((r.dataset, r.model)), axis=1)
    return df


# S1 / S2 MC samples and dropout rate -------------------------------------------------------
for name, key in (("S1", "M"), ("S2", "dropout_p")):
    d = pd.read_csv(TAB / ("S1_mc_samples.csv" if name == "S1" else "S2_dropout_rate.csv"))
    rows = []
    for v, g in d.groupby(key):
        def get(variant, gate, col):
            s = g[(g.variant == variant) & (g.gate == gate)]
            return s[col].iloc[0] if len(s) else np.nan
        mc, mbl = "MC dropout only", "MedBayes-Lite (MC + T)"
        rows.append(" & ".join([f"{v:g}", f(get(mc, "maxprob", "ece")), f(get(mc, "maxprob", "harmful@0.8"), 4),
                                f(get(mc, "maxprob", "cov@0.7")), f(get(mbl, "maxprob", "accuracy")),
                                f(get(mbl, "maxprob", "ece")), f(get(mbl, "maxprob", "harmful@0.8"), 4),
                                f"{f(get(mbl, 'maxprob', 'cov@0.7'))} ({get(mbl, 'maxprob', 'n_acc@0.7'):.0f})",
                                f(get(mbl, "maxprob", "selacc@0.7")), f(get(mbl, "maxprob", "aurc")),
                                f"{f(get(mbl, 'entropy', 'cov@0.7'))} ({get(mbl, 'entropy', 'n_acc@0.7'):.0f})",
                                f(get(mbl, "maxprob", "T"), 2)]) + r" \\")
    head = ("$M$" if name == "S1" else "Dropout $p$")
    write(name, "\n".join([
        r"\begin{tabular}{lccc|cccccccc}", r"\toprule",
        r" & \multicolumn{3}{c|}{\textbf{MC dropout only (max-prob)}} & \multicolumn{8}{c}{\textbf{MedBayes-Lite}} \\",
        f"\\textbf{{{head}}} & ECE & harm@0.8 & Cov@0.7 & Acc & ECE & harm@0.8 & Cov@0.7 ($n$) & SelAcc@0.7 & AURC & Cov@0.7 entropy ($n$) & $T$ \\\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))

# S3 threshold sweep -----------------------------------------------------------------------
d = pd.read_csv(TAB / "S3_threshold_sweep.csv")
rows = []
for (model, gate), g in d.groupby(["model", "gate"], sort=False):
    for _, r in g.iterrows():
        rows.append(" & ".join([SET[("medmcqa", model)].split(", ")[1], GATE[gate], f"{r.tau:.1f}", f(r.coverage),
                                f"{r.n_accepted_mean:.1f}", f"{f(r.selective_acc)} {ci(r.selective_acc_wilson_lo, r.selective_acc_wilson_hi)}",
                                f(r.abstention_recall), f(r.harmful_per_total, 4)]) + r" \\")
    rows.append(r"\addlinespace")
write("S3", "\n".join([r"\begin{tabular}{llcccccc}", r"\toprule",
                       r"\textbf{Model} & \textbf{Gate} & $\tau$ & \textbf{Coverage} & $n_{\mathrm{acc}}$ & \textbf{SelAcc [Wilson 95\% CI]} & \textbf{Abst.\ recall} & \textbf{Harm/total} \\",
                       r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}"]))

# S4 CUS dependence-aware -----------------------------------------------------------------
c = pd.read_csv(TAB / "cus_correlations_dependence_aware.csv")
rows = []
for sub in ["original (overall + bucket rows)", "bucket rows only", "overall rows only", "encoders only",
            "medmcqa only", "medqa only", "pubmedqa only"]:
    for tgt in ["harmful@0.8", "sevwerr@0.8", "overconf@0.8", "aurc"]:
        for pred in ["cus", "ece"]:
            r = c[(c.subset == sub) & (c.target == tgt) & (c.predictor == pred)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(" & ".join([esc(sub) if (tgt == "harmful@0.8" and pred == "cus") else "",
                                    esc(tgt) if pred == "cus" else "", pred.upper(), f"{r.n_groups:.0f}",
                                    f(r.pearson_r), ci(r.fisher_ci_lo, r.fisher_ci_hi), ci(r.group_boot_ci_lo, r.group_boot_ci_hi),
                                    ci(r.item_cluster_boot_ci_lo, r.item_cluster_boot_ci_hi),
                                    f(r.partial_r_given_ece) if pred == "cus" else ""]) + r" \\")
    rows.append(r"\addlinespace")
inc = pd.read_csv(TAB / "cus_incremental_dependence_aware.csv").set_index("target")
irows = []
for tgt, r in inc.iterrows():
    irows.append(" & ".join([esc(tgt), f"{r.n_groups:.0f}", f"{r.n_clusters:.0f}", f(r.r2_ece), f(r.r2_ece_cus), f(r.delta_r2),
                             ci(r.delta_r2_cluster_boot_ci_lo, r.delta_r2_cluster_boot_ci_hi),
                             ci(r.delta_r2_item_boot_ci_lo, r.delta_r2_item_boot_ci_hi),
                             f"{f(r.delta_r2_leave_one_model_out_min)}--{f(r.delta_r2_leave_one_model_out_max)}",
                             p_fmt(r.cus_p_cluster_robust)]) + r" \\")
w = pd.read_csv(TAB / "S4_cus_weight_sensitivity.csv")
wrows = []
for _, r in w.iterrows():
    rng = f"{f(r['min'])}--{f(r['max'])}" if not np.isnan(r["min"]) else "--"
    wrows.append(" & ".join([esc(r.scheme), esc(r.subset), f(r.pearson_r), f(r.spearman_r), f"{r.n_groups:.0f}", rng]) + r" \\")
write("S4", "\n".join([
    r"\textbf{A. Correlation with clinical failure metrics (95\% CIs: Fisher; group bootstrap; item-cluster bootstrap).}\\[2pt]",
    r"\begin{tabular}{lllcccccc}", r"\toprule",
    r"\textbf{Subset} & \textbf{Target} & \textbf{Pred.} & $n$ & $r$ & \textbf{Fisher CI} & \textbf{Group boot.} & \textbf{Item-cluster boot.} & \textbf{Partial $r\mid$ECE} \\",
    r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{B. Incremental variance explained by CUS over ECE (per-group rows; clusters = model--dataset pairs).}\\[2pt]",
    r"\begin{tabular}{lccccccccc}", r"\toprule",
    r"\textbf{Target} & $n$ & \textbf{Clusters} & $R^2$(ECE) & $R^2$(+CUS) & $\Delta R^2$ & \textbf{Cluster boot.} & \textbf{Item boot.} & \textbf{LOMO range} & \textbf{Cluster-robust $p$} \\",
    r"\midrule", *irows, r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{C. Alternative severity weightings (Pearson / Spearman $r$ between CUS and harmful@0.8).}\\[2pt]",
    r"\begin{tabular}{llcccc}", r"\toprule",
    r"\textbf{Weighting} & \textbf{Subset} & \textbf{Pearson} & \textbf{Spearman} & $n$ & \textbf{Range over 20 draws} \\",
    r"\midrule", *wrows, r"\bottomrule", r"\end{tabular}"]))

# S5 matched coverage ----------------------------------------------------------------------
m = mcq(pd.read_csv(TAB / "matched_coverage.csv"))
m = m[m.target_coverage.isin([0.1, 0.2, 0.5]) & m.method.isin(["base", "temp_scaling", "mc_dropout", "medbayes_lite"])]
m["_m"] = m.method.map(["base", "temp_scaling", "mc_dropout", "medbayes_lite"].index)
m["_g"] = m.gate.map(["maxprob", "entropy"].index)
rows, prev = [], None
for _, r in m.sort_values(["_o", "_g", "target_coverage", "_m"]).iterrows():
    key = (r.dataset, r.model, r.gate)
    if prev is not None and key != prev:
        rows.append(r"\midrule")
    prev = key
    rows.append(" & ".join([SET[(r.dataset, r.model)], GATE[r.gate], f"{r.target_coverage:.1f}", f"{r.n_accepted:.0f}",
                            METH[r.method], f"{f(r.selective_risk)} {ci(r.selective_risk_ci_lo, r.selective_risk_ci_hi)}",
                            f"{f(r.harmful_per_total, 4)} {ci(r.harmful_per_total_ci_lo, r.harmful_per_total_ci_hi, 4)}",
                            f(r.harmful_per_total_diff_vs_base, 4) if r.method != "base" else "--",
                            p_fmt(r.harmful_per_total_diff_p) if r.method != "base" else "--"]) + r" \\")
write("S5", "\n".join([
    r"\begin{longtable}{lllclcccc}",
    r"\caption{Matched-coverage comparison, all methods and both gates (mean of three MC seeds; 95\% item-bootstrap CIs). Each method accepts its top $n$ items by its own score. Harm/total: heuristically high-severity errors among accepted items per total test item. $\Delta$ and $p$: method minus base on Harm/total (paired bootstrap, unadjusted).}\label{tab:S5}\\",
    r"\toprule", r"\textbf{Setting} & \textbf{Gate} & \textbf{Cov.} & $n$ & \textbf{Method} & \textbf{Selective risk} & \textbf{Harm/total} & $\Delta$ & $p$ \\", r"\midrule\endfirsthead",
    r"\toprule", r"\textbf{Setting} & \textbf{Gate} & \textbf{Cov.} & $n$ & \textbf{Method} & \textbf{Selective risk} & \textbf{Harm/total} & $\Delta$ & $p$ \\", r"\midrule\endhead",
    r"\bottomrule\endfoot", *rows, r"\end{longtable}"]))

# S6 matched risk -------------------------------------------------------------------------
m = mcq(pd.read_csv(TAB / "matched_risk.csv"))
m = m[m.method.isin(["base", "temp_scaling", "mc_dropout", "medbayes_lite"]) & m.target_risk.isin([0.3, 0.4, 0.5])]
m["_m"] = m.method.map(["base", "temp_scaling", "mc_dropout", "medbayes_lite"].index)
m["_g"] = m.gate.map(["maxprob", "entropy"].index)
rows, prev = [], None
for _, r in m.sort_values(["_o", "_g", "target_risk", "_m"]).iterrows():
    key = (r.dataset, r.model, r.gate)
    if prev is not None and key != prev:
        rows.append(r"\midrule")
    prev = key
    rows.append(" & ".join([SET[(r.dataset, r.model)], GATE[r.gate], f"{r.target_risk:.1f}", METH[r.method],
                            f"{f(r.coverage_at_risk)} {ci(r.ci_lo, r.ci_hi)}",
                            f(r.diff_vs_base) if r.method != "base" else "--",
                            p_fmt(r.diff_p) if r.method != "base" else "--"]) + r" \\")
write("S6", "\n".join([
    r"\begin{longtable}{lllllcc}",
    r"\caption{Matched-risk comparison: largest coverage at which selective risk does not exceed the target (mean of three MC seeds; 95\% bootstrap CI). $\Delta$ and $p$: method minus base coverage (paired bootstrap, unadjusted). Negative $\Delta$ means the method must defer more to reach the same risk.}\label{tab:S6}\\",
    r"\toprule", r"\textbf{Setting} & \textbf{Gate} & \textbf{Target risk} & \textbf{Method} & \textbf{Coverage at risk [95\% CI]} & $\Delta$ & $p$ \\", r"\midrule\endfirsthead",
    r"\toprule", r"\textbf{Setting} & \textbf{Gate} & \textbf{Target risk} & \textbf{Method} & \textbf{Coverage at risk [95\% CI]} & $\Delta$ & $p$ \\", r"\midrule\endhead",
    r"\bottomrule\endfoot", *rows, r"\end{longtable}"]))

# S7 fixed threshold, both gates (factorial) ------------------------------------------------
fa = mcq(pd.read_csv(TAB / "factorial_ablation.csv"))
fa = fa[fa.method.isin(["base", "temp_scaling", "mc_dropout", "medbayes_lite"])]
fa["_m"] = fa.method.map(["base", "temp_scaling", "mc_dropout", "medbayes_lite"].index)
fa["_g"] = fa.gate.map(["maxprob", "entropy"].index)
rows, prev = [], None
for _, r in fa.sort_values(["_o", "_m", "_g"]).iterrows():
    key = (r.dataset, r.model)
    if prev is not None and key != prev:
        rows.append(r"\midrule")
    prev = key
    rows.append(" & ".join([SET[key] if r.method == "base" and r.gate == "maxprob" else "", METH[r.method],
                            "\\checkmark" if r.mc_dropout == "yes" else "--", "\\checkmark" if r.temperature == "yes" else "--",
                            GATE[r.gate], f(r.accuracy), f(r.ece), f(r["harmful@0.8"], 4),
                            f"{f(r['cov@0.7'])} ({r['n_acc@0.7']:.1f})", f(r["selacc@0.7"]), f(r.aurc), f(r.auroc_error),
                            f(r["risk@cov0.2"]), f(r["harmful_per_total@cov0.2"], 4)]) + r" \\")
write("S7", "\n".join([
    r"\begin{tabular}{llcclccccccccc}", r"\toprule",
    r"\textbf{Setting} & \textbf{Method} & \textbf{MC} & $T$ & \textbf{Gate} & \textbf{Acc} & \textbf{ECE} & \textbf{harm@0.8} & \textbf{Cov@0.7 ($n$)} & \textbf{SelAcc@0.7} & \textbf{AURC} & \textbf{AUROC$_{\mathrm{err}}$} & \textbf{Risk@cov0.2} & \textbf{Harm@cov0.2} \\",
    r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))

# S8 paired tests -------------------------------------------------------------------------
pt = mcq(pd.read_csv(TAB / "paired_tests.csv"))
NICE_M = {"ece": "ECE", "harmful@0.8": "harm@0.8", "aurc": "AURC", "auroc_error": "AUROC$_{\\mathrm{err}}$",
          "selacc@0.7": "SelAcc@0.7", "brier": "Brier"}
pt = pt[pt.metric.isin(NICE_M)]
pt["_f"] = pt.family.map({"primary": 0, "vs_temperature": 1, "ood_source_T": 2})
pt["_k"] = pt.metric.map(list(NICE_M).index)


def lab(s):
    meth, gate = s.split(" [")
    return f"{METH[meth]} [{GATE[gate.rstrip(']')]}]"


rows, prev = [], None
for _, r in pt.sort_values(["_o", "_f", "A", "B", "_k"]).iterrows():
    key = (r.dataset, r.model, r.A, r.B)
    if prev is not None and key[:2] != prev[:2]:
        rows.append(r"\midrule")
    elif prev is not None and key != prev:
        rows.append(r"\addlinespace")
    first = key != prev
    prev = key
    rows.append(" & ".join([SET[key[:2]] if first else "", f"{lab(r.B)} $-$ {lab(r.A)}" if first else "", NICE_M[r.metric],
                            f(r.A_value), f(r.B_value), f"{f(r.diff_B_minus_A)} {ci(r.ci_lo, r.ci_hi)}",
                            esc(r.p_report) if isinstance(r.p_report, str) else "--",
                            esc(r.p_holm_report) if isinstance(r.p_holm_report, str) else "--", f"{r.n_boot:.0f}"]) + r" \\")
write("S8", "\n".join([
    r"\begin{longtable}{p{2.5cm}p{4.2cm}lccccccc}",
    r"\caption{All paired item-bootstrap contrasts on the multiple-choice tasks (B minus A). Families: primary (MedBayes-Lite vs base, $B=10{,}000$), versus temperature scaling ($B=2{,}000$; includes temperature scaling with the entropy gate against MedBayes-Lite with either gate), and source-calibrated shift (MedQA-USMLE). $p_{\mathrm{Holm}}$ is adjusted within family; its lower bound is the family size times $2/(B+1)$.}\label{tab:S8}\\",
    r"\toprule", r"\textbf{Setting} & \textbf{Contrast (B $-$ A)} & \textbf{Metric} & \textbf{A} & \textbf{B} & \textbf{$\Delta$ [95\% CI]} & $p$ & $p_{\mathrm{Holm}}$ & $B$ \\", r"\midrule\endfirsthead",
    r"\toprule", r"\textbf{Setting} & \textbf{Contrast (B $-$ A)} & \textbf{Metric} & \textbf{A} & \textbf{B} & \textbf{$\Delta$ [95\% CI]} & $p$ & $p_{\mathrm{Holm}}$ & $B$ \\", r"\midrule\endhead",
    r"\bottomrule\endfoot", *rows, r"\end{longtable}"]))

# S9 seed vs item variance ------------------------------------------------------------------
sv = mcq(pd.read_csv(TAB / "seed_vs_item_variance.csv"))
sv = sv[(sv.gate == "maxprob") & sv.method.isin(["base", "temp_scaling", "mc_dropout", "medbayes_lite"])
        & sv.metric.isin(["ece", "harmful@0.8", "aurc", "selacc@0.7"])]
sv["_m"] = sv.method.map(["base", "temp_scaling", "mc_dropout", "medbayes_lite"].index)
sv["_k"] = sv.metric.map(["ece", "harmful@0.8", "aurc", "selacc@0.7"].index)
rows, prev = [], None
for _, r in sv.sort_values(["_o", "_m", "_k"]).iterrows():
    key = (r.dataset, r.model)
    if prev is not None and key != prev:
        rows.append(r"\midrule")
    prev = key
    rows.append(" & ".join([SET[key], METH[r.method], NICE_M.get(r.metric, r.metric), f(r["mean"], 4), f(r.seed_sd, 4),
                            f(r.item_bootstrap_se, 4), ci(r.two_stage_ci_lo, r.two_stage_ci_hi, 4)]) + r" \\")
write("S9", "\n".join([
    r"\begin{longtable}{lllcccc}",
    r"\caption{Seed variability versus item-sampling uncertainty (max-probability gate). Seed SD: standard deviation across the three MC seeds; item SE: bootstrap standard error over test items with seeds averaged; two-stage CI resamples seeds and items. Deterministic methods use fixed third-party checkpoints and have no seed variance.}\label{tab:S9}\\",
    r"\toprule", r"\textbf{Setting} & \textbf{Method} & \textbf{Metric} & \textbf{Mean} & \textbf{Seed SD} & \textbf{Item SE} & \textbf{Two-stage 95\% CI} \\", r"\midrule\endfirsthead",
    r"\toprule", r"\textbf{Setting} & \textbf{Method} & \textbf{Metric} & \textbf{Mean} & \textbf{Seed SD} & \textbf{Item SE} & \textbf{Two-stage 95\% CI} \\", r"\midrule\endhead",
    r"\bottomrule\endfoot", *rows, r"\end{longtable}"]))

# S10 severity sensitivity ------------------------------------------------------------------
ss = mcq(pd.read_csv(TAB / "severity_sensitivity.csv"))
ss = ss[ss.gate == "maxprob"]
rows, prev = [], None
for _, r in ss.sort_values(["_o"], kind="stable").iterrows():
    key = (r.dataset, r.model)
    if prev is not None and key != prev:
        rows.append(r"\midrule")
    prev = key
    rows.append(" & ".join([SET[key], esc(r.scheme), f"{r.cutoff:.1f}", f(r.frac_items_high_severity), f(r.harmful_base, 4),
                            f(r.harmful_temp_scaling, 4), f(r.harmful_medbayes_lite, 4),
                            f"{f(r.reduction, 4)} {ci(r.reduction_ci_lo, r.reduction_ci_hi, 4)}"]) + r" \\")
rw = mcq(pd.read_csv(TAB / "severity_sensitivity_random_weights_summary.csv"))
rrows = [" & ".join([SET[(r.dataset, r.model)], GATE[r.gate], r"\multicolumn{2}{c}{" + f(r.reduction_mean, 4) + "}",
                     f(r.reduction_min, 4), f(r.reduction_max, 4),
                     r"\multicolumn{2}{c}{" + ("yes" if r.all_ci_exclude_zero else "no") + "}"]) + r" \\"
         for _, r in rw.sort_values(["_o", "gate"]).iterrows()]
write("S10", "\n".join([
    r"\begin{longtable}{llcccccc}",
    r"\caption{Sensitivity of harmful@0.8 to the severity definition (max-probability gate; the entropy gate is in the repository). Schemes vary the high-severity cutoff, the rule for combining the weights of an item's groups (maximum, mean, sum), and the rule set (default substring matching versus strict word boundaries); ``uniform'' sets every weight to 1 (all confident errors count). Reduction: base minus MedBayes-Lite, 95\% item-bootstrap CI. The last block summarises 20 random $\pm25\%$ perturbations of the group weights (both gates).}\label{tab:S10}\\",
    r"\toprule", r"\textbf{Setting} & \textbf{Scheme} & \textbf{Cutoff} & \textbf{Frac.\ high-sev.} & \textbf{Base} & \textbf{TS} & \textbf{MBL} & \textbf{Reduction [95\% CI]} \\", r"\midrule\endfirsthead",
    r"\toprule", r"\textbf{Setting} & \textbf{Scheme} & \textbf{Cutoff} & \textbf{Frac.\ high-sev.} & \textbf{Base} & \textbf{TS} & \textbf{MBL} & \textbf{Reduction [95\% CI]} \\", r"\midrule\endhead",
    r"\bottomrule\endfoot", *rows, r"\midrule",
    r"\multicolumn{8}{l}{\textbf{Random $\pm25\%$ weight perturbations (20 draws)}} \\",
    r"\textbf{Setting} & \textbf{Gate} & \multicolumn{2}{c}{\textbf{Mean reduction}} & \textbf{Min} & \textbf{Max} & \multicolumn{2}{c}{\textbf{All CIs exclude 0}} \\",
    *rrows,
    r"\end{longtable}"]))

# S11 denominators, overlap, trigger audit, strict rules ------------------------------------
dn = mcq(pd.read_csv(TAB / "harmful_denominators.csv"))
dn = dn[dn.gate == "maxprob"]
drows = [" & ".join([SET[(r.dataset, r.model)], METH[r.method], f"{r.n_items:.0f}", f"{r.n_high_severity_items:.0f}",
                     f"{r.n_accepted_mean:.1f}", f"{r.harmful_count_mean:.1f}", f(r.harmful_per_total, 4),
                     f(r.harmful_per_high_severity_item, 4), f(r.harmful_per_accepted, 4)]) + r" \\"
         for _, r in dn.sort_values(["_o", "method"]).iterrows()]
mc_ = pd.read_csv(TAB / "multi_category_distribution.csv")
mrows = []
for ds, g in mc_.groupby("dataset", sort=False):
    mrows.append(" & ".join([{"medmcqa": "MedMCQA", "medqa": "MedQA-USMLE", "pubmedqa": "PubMedQA"}[ds]] +
                            [f"{r.n_items:.0f} ({r.frac:.3f})" for _, r in g.iterrows()]) + r" \\")
hn = mcq(pd.read_csv(TAB / "harmful_by_n_categories.csv"))
hn = hn[(hn.gate == "maxprob") & hn.method.isin(["base", "medbayes_lite"])]
hrows = []
for (o, ds, model, meth), g in hn.groupby(["_o", "dataset", "model", "method"]):
    g = g.set_index("n_categories")
    hrows.append(" & ".join([SET[(ds, model)], METH[meth]] +
                            [f"{f(g.loc[k, 'harmful@0.8_within_group'])} ({g.loc[k, 'n_items']:.0f})" if k in g.index else "--"
                             for k in ["1", "2", "3", "4+"]]) + r" \\")
tr = pd.read_csv(TAB / "trigger_audit_false_positive_candidates.csv")
tr = tr[tr.substring_only_items >= 5].sort_values("substring_only_items", ascending=False).head(14)
trows = [" & ".join([{"medmcqa": "MedMCQA", "medqa": "MedQA-USMLE", "pubmedqa": "PubMedQA"}[r.dataset], BUCK.get(r.bucket, esc(r.bucket)),
                     esc(r.cue), f"{r.items_substring_match:.0f}", f"{r.items_word_match:.0f}", f(r.substring_only_frac, 2),
                     esc(", ".join(str(r.top_substring_only_tokens).split(", ")[:3]))]) + r" \\" for _, r in tr.iterrows()]
ra = pd.read_csv(TAB / "ruleset_agreement_default_vs_strict.csv")
ra = ra[ra.dataset.isin(["medmcqa", "medqa"])]
piv = ra.pivot(index="bucket", columns="dataset", values=["n_default", "n_strict", "jaccard"])
srows = []
for b in BUCK:
    if b not in piv.index:
        continue
    r = piv.loc[b]
    srows.append(" & ".join([BUCK[b], f"{r[('n_default', 'medmcqa')]:.0f}", f"{r[('n_strict', 'medmcqa')]:.0f}",
                             f(r[('jaccard', 'medmcqa')], 2) if not np.isnan(r[('jaccard', 'medmcqa')]) else "--",
                             f"{r[('n_default', 'medqa')]:.0f}", f"{r[('n_strict', 'medqa')]:.0f}",
                             f(r[('jaccard', 'medqa')], 2) if not np.isnan(r[('jaccard', 'medqa')]) else "--"]) + r" \\")
write("S11", "\n".join([
    r"\textbf{A. Denominators of harmful@0.8 (max-probability gate, threshold 0.8).}\\[2pt]",
    r"\begin{tabular}{llccccccc}", r"\toprule",
    r"\textbf{Setting} & \textbf{Method} & $N$ & \textbf{High-sev.\ items} & \textbf{Accepted} & \textbf{Harmful} & \textbf{per total} & \textbf{per high-sev.\ item} & \textbf{per accepted} \\",
    r"\midrule", *drows, r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{B. Number of heuristic groups per item: items (fraction).}\\[2pt]",
    r"\begin{tabular}{lccccc}", r"\toprule", r"\textbf{Dataset} & 1 & 2 & 3 & 4 & 5+ \\", r"\midrule", *mrows,
    r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{C. Harmful@0.8 within items assigned to 1, 2, 3, or 4+ groups (number of items).}\\[2pt]",
    r"\begin{tabular}{llcccc}", r"\toprule", r"\textbf{Setting} & \textbf{Method} & 1 & 2 & 3 & 4+ \\", r"\midrule", *hrows,
    r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{D. Trigger audit: cues that match mostly inside other words (substring-only matches $\geq 5$).}\\[2pt]",
    r"\begin{tabular}{lllcccl}", r"\toprule",
    r"\textbf{Dataset} & \textbf{Group} & \textbf{Cue} & \textbf{Substring} & \textbf{Word} & \textbf{Frac.\ substring-only} & \textbf{Examples} \\",
    r"\midrule", *trows, r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{E. Group membership under default versus strict word-boundary rules (items; Jaccard overlap).}\\[2pt]",
    r"\begin{tabular}{lcccccc}", r"\toprule",
    r" & \multicolumn{3}{c}{\textbf{MedMCQA}} & \multicolumn{3}{c}{\textbf{MedQA-USMLE}} \\",
    r"\textbf{Group} & Default & Strict & Jaccard & Default & Strict & Jaccard \\", r"\midrule", *srows,
    r"\bottomrule", r"\end{tabular}"]))

# S12 LLM-assisted agreement -------------------------------------------------------------
K = ROOT / "clinician_kit" / "results"
pc = pd.read_csv(K / "per_category.csv")
prow = [" & ".join([BUCK.get(r.category, esc(r.category)), f"{r.n_rule:.0f}", f"{r.n_clinician:.0f}", f(r["precision (rule)"], 2),
                    f(r["recall (rule)"], 2), f(r.kappa, 2)]) + r" \\" for _, r in pc.iterrows()]
rv = pd.read_csv(K / "rule_vs_clinician.csv")
vrows = [" & ".join([esc(r.comparison.replace("clinician consensus", "LLM consensus").replace("clinician", "LLM consensus"))
                     .replace(">=", r"$\geq$"), f(r.spearman_rho), f(r.high_sev_kappa),
                     f(r["high_sev_precision (rule PPV)"]), f(r["high_sev_recall (rule sensitivity)"]),
                     f(r.rule_false_positive_rate)]) + r" \\" for _, r in rv.iterrows()]
hc = pd.read_csv(K / "harmful_clinician_labels.csv")
hcp = hc.pivot_table(index=["dataset", "model", "method", "gate", "n_rated_items"], columns="severity_labels",
                     values="harmful@0.8").reset_index()
hcp["_o"] = hcp.apply(lambda r: MCQ_ORDER.index((r.dataset, r.model)), axis=1)
crows = [" & ".join([SET[(r.dataset, r.model)], METH[r.method], GATE[r.gate], f"{r.n_rated_items:.0f}", f(r["rule"]),
                     f(r["clinician"])]) + r" \\" for _, r in hcp.sort_values(["_o", "method", "gate"], ascending=[True, True, False]).iterrows()]
ir = pd.read_csv(K / "inter_rater.csv").iloc[0]
write("S12", "\n".join([
    f"\\textbf{{A. Agreement between rule-based groups and LLM-assisted ratings (210 items). LLM--LLM agreement on severity: quadratic-weighted $\\kappa={ir.weighted_kappa_quadratic:.2f}$; high-severity $\\kappa={ir.kappa_high_severity:.2f}$.}}\\\\[2pt]",
    r"\begin{tabular}{lccccc}", r"\toprule",
    r"\textbf{Comparison} & \textbf{Spearman} & $\kappa$ & \textbf{PPV} & \textbf{Sensitivity} & \textbf{FP rate} \\",
    r"\midrule", *vrows, r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{B. Per-group agreement (rule membership versus membership flagged by at least one LLM rater).}\\[2pt]",
    r"\begin{tabular}{lccccc}", r"\toprule",
    r"\textbf{Group} & \textbf{$n$ rule} & \textbf{$n$ LLM} & \textbf{Precision} & \textbf{Recall} & $\kappa$ \\",
    r"\midrule", *prow, r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{C. Harmful@0.8 on the rated items using rule severity versus LLM-consensus severity ($\geq 4$).}\\[2pt]",
    r"\begin{tabular}{lllccc}", r"\toprule",
    r"\textbf{Setting} & \textbf{Method} & \textbf{Gate} & \textbf{Rated items} & \textbf{Rule labels} & \textbf{LLM labels} \\",
    r"\midrule", *crows, r"\bottomrule", r"\end{tabular}"]))

# S13 perturbations -----------------------------------------------------------------------
pr = pd.read_csv(TAB / "perturbation_response.csv")
pr = pr[(pr.scope == "applied items") & (pr.gate == "maxprob") & pr.method.isin(["base", "temp_scaling", "medbayes_lite"])]
pr["_m"] = pr.method.map(["base", "temp_scaling", "medbayes_lite"].index)
rows, prev = [], None
for _, r in pr.sort_values(["dataset", "family", "perturbation", "_m"], ascending=[False, True, True, True]).iterrows():
    key = (r.dataset, r.perturbation)
    if prev is not None and key[0] != prev[0]:
        rows.append(r"\midrule")
    first = key != prev
    prev = key
    rows.append(" & ".join([{"medmcqa": "MedMCQA", "medqa": "MedQA-USMLE"}[r.dataset] if first else "",
                            (esc(r.perturbation) + (" (safety)" if r.family.startswith("safety") else "")) if first else "",
                            f"{r.n_items:.0f}" if first else "", METH[r.method],
                            f"{f(r.confidence_drop, 4)} {ci(r.confidence_drop_ci_lo, r.confidence_drop_ci_hi, 4)}",
                            f(r["abstention_increase@0.7"]), f(r.flip_rate), esc(r.p_report)]) + r" \\")
rr = pd.read_csv(K / "perturbation_review_rates_by_perturbation.csv")
rr = rr.pivot_table(index="perturbation", columns=["rater", "question"], values="frac_yes")
rrows = [" & ".join([esc(p)] + [f(rr.loc[p, (rt, q)], 2) for rt in ["llm-claude", "llm-gpt"] for q in ["R1", "R2", "R3"]]) + r" \\"
         for p in rr.index]
mv = pd.read_csv(K / "perturbation_review_model_response_vs_rater_consensus.csv")
mv = mv[mv.gate == "maxprob"]
mrows2 = [" & ".join([METH[r.method], esc(r.rater_consensus.replace("raters: ", "")), f"{r.n_items:.0f}",
                      f(r.mean_confidence_drop, 4), f(r["frac_newly_abstains@0.7"])]) + r" \\" for _, r in mv.iterrows()]
write("S13", "\n".join([
    r"\begin{longtable}{lllclccc}",
    r"\caption{Response to controlled perturbations (PubMedBERT trained on MedMCQA; 300 items per dataset; items where the edit applied; max-probability gate; mean of three MC seeds). Confidence drop: clean minus edited score, 95\% bootstrap CI. $p$: Wilcoxon signed-rank, Holm-adjusted across perturbations.}\label{tab:S13}\\",
    r"\toprule", r"\textbf{Dataset} & \textbf{Edit} & $n$ & \textbf{Method} & \textbf{Confidence drop} & \textbf{$\Delta$ abstention@0.7} & \textbf{Flip rate} & $p$ \\", r"\midrule\endfirsthead",
    r"\toprule", r"\textbf{Dataset} & \textbf{Edit} & $n$ & \textbf{Method} & \textbf{Confidence drop} & \textbf{$\Delta$ abstention@0.7} & \textbf{Flip rate} & $p$ \\", r"\midrule\endhead",
    r"\bottomrule\endfoot", *rows, r"\end{longtable}",
    r"\textbf{B. LLM review of edits: fraction answering yes (R1 clinically meaningful; R2 a safe system should be less confident; R3 dangerous or implausible).}\\[2pt]",
    r"\begin{tabular}{lcccccc}", r"\toprule",
    r" & \multicolumn{3}{c}{\textbf{Claude Opus 5.5}} & \multicolumn{3}{c}{\textbf{GPT-5}} \\",
    r"\textbf{Edit} & R1 & R2 & R3 & R1 & R2 & R3 \\", r"\midrule", *rrows, r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{C. Model response by rater consensus (majority; ties count as yes).}\\[2pt]",
    r"\begin{tabular}{llccc}", r"\toprule",
    r"\textbf{Method} & \textbf{Consensus} & $n$ & \textbf{Mean confidence drop} & \textbf{Newly deferred at 0.7} \\",
    r"\midrule", *mrows2, r"\bottomrule", r"\end{tabular}"]))

# S14 latency -----------------------------------------------------------------------------
lt = pd.read_csv(TAB / "latency_benchmark.csv")
rows, prev = [], None
for _, r in lt.iterrows():
    if prev is not None and r.backbone != prev:
        rows.append(r"\midrule")
    first = r.backbone != prev
    prev = r.backbone
    rows.append(" & ".join([esc(r.backbone) if first else "", esc(r.method), f"{r.forward_passes:.0f}", f"{r.params / 1e6:.0f}M",
                            f"{r.latency_ms_bs1_median:.1f}", f"{r.latency_ms_bs1_p90:.1f}", f"{r.throughput_items_per_s:.1f}",
                            f"{r.peak_gpu_mem_gb:.2f}"]) + r" \\")
write("S14", "\n".join([
    r"\begin{longtable}{p{3.2cm}p{5.0cm}cccccc}",
    r"\caption{Latency, throughput, and peak GPU memory on one NVIDIA T4 (batch size 1 for latency; median and 90th percentile of 30 repetitions). Packed batch: the $M$ MC passes are run as one batch of $M$ copies. SWAG cost proxy: 10 weight samples of the same backbone.}\label{tab:S14}\\",
    r"\toprule", r"\textbf{Backbone} & \textbf{Method} & \textbf{Passes} & \textbf{Params} & \textbf{Median (ms)} & \textbf{P90 (ms)} & \textbf{Items/s} & \textbf{Mem (GB)} \\", r"\midrule\endfirsthead",
    r"\toprule", r"\textbf{Backbone} & \textbf{Method} & \textbf{Passes} & \textbf{Params} & \textbf{Median (ms)} & \textbf{P90 (ms)} & \textbf{Items/s} & \textbf{Mem (GB)} \\", r"\midrule\endhead",
    r"\bottomrule\endfoot", *rows, r"\end{longtable}"]))

# S15 reproducibility manifest -------------------------------------------------------------
man = json.loads((ROOT / "manifest" / "reproducibility_manifest.json").read_text())
vrows = [f"{esc(k)} & {esc(v if not isinstance(v, list) else ', '.join(map(str, v)))} \\\\" for k, v in man["versions"].items()]
rrows = [f"\\texttt{{{esc(k)}}} & \\texttt{{{v}}} \\\\" for k, v in {**man["pinned_dataset_revisions"], **man["pinned_model_revisions"]}.items()]
ft = pd.read_csv(ROOT / "manifest" / "fitted_temperatures.csv")
ft = ft[ft.objective.notna()]
trows = []
for (ds, model, on), g in ft.groupby(["dataset", "model", "calibrated_on"], sort=False):
    trows.append(" & ".join([{"medmcqa": "MedMCQA", "medqa": "MedQA-USMLE", "pubmedqa": "PubMedQA"}[ds], esc(model), f"{g.n_calib.iloc[0]:.0f}",
                             ", ".join(f"{t:.3f}" for t in g.T_temp_scaling), ", ".join(f"{t:.3f}" for t in g.T_medbayes_lite)]) + r" \\")
lk = pd.read_csv(ROOT / "manifest" / "leakage_checks.csv")
lrows = [f"{ {'medmcqa': 'MedMCQA', 'medqa': 'MedQA-USMLE', 'pubmedqa': 'PubMedQA'}[r.dataset]} & {esc(r.check).replace('∩', '$\\cap$')} & {r.n:.0f} \\\\"
         for _, r in lk.iterrows()]
write("S15", "\n".join([
    r"\textbf{A. Software and hardware.}\\[2pt]", r"\begin{tabular}{ll}", r"\toprule", *vrows, r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{B. Pinned Hugging Face revisions (datasets and models).}\\[2pt]", r"{\footnotesize\begin{tabular}{ll}", r"\toprule",
    r"\textbf{Repository} & \textbf{Commit} \\", r"\midrule", *rrows, r"\bottomrule", r"\end{tabular}}", r"\\[6pt]",
    r"\textbf{C. Fitted temperatures per seed (seeds 0, 1, 2), NLL objective, LBFGS (lr 0.01, at most 200 iterations), calibration slice only.}\\[2pt]",
    r"\begin{tabular}{llccc}", r"\toprule", r"\textbf{Calibrated on} & \textbf{Model} & $n_{\mathrm{calib}}$ & $T$ (TS) & $T$ (MedBayes-Lite) \\",
    r"\midrule", *trows, r"\bottomrule", r"\end{tabular}", r"\\[6pt]",
    r"\textbf{D. Leakage checks (number of offending items).}\\[2pt]", r"\begin{tabular}{llc}", r"\toprule",
    r"\textbf{Dataset} & \textbf{Check} & $n$ \\", r"\midrule", *lrows, r"\bottomrule", r"\end{tabular}"]))
