"""Reproducible orchestrator for the clinical uncertainty governance evaluation.

Subcommands (python -m medbayes_eval.run <cmd>):
  run        fine-tune/load models, run all UQ methods, save per-sample records
  ood        domain-shift transfer (MedMCQA-trained -> MedQA): ID vs OOD drift
  perturb    controlled clinical perturbations: confidence/abstention response
  analyze    metrics tables, CUS/ZTI correlations, stats, plots, LaTeX/CSV tables
  all        run + ood + perturb + analyze end to end

Everything is config-driven (configs/default.yaml). Outputs land under results/.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd

from . import config
from .config import EvalConfig
from .pipeline import run_encoder_dataset, save_predictions, get_train_eval
from .metrics import (metrics_table, bucket_metrics_table, correlation_analysis)
from .metrics.core import compute_record_metrics

RAW = os.path.join(config.RESULTS_DIR, "raw_predictions")
MET = os.path.join(config.RESULTS_DIR, "metrics")
PLT = os.path.join(config.RESULTS_DIR, "plots")
TAB = os.path.join(config.RESULTS_DIR, "tables")
STT = os.path.join(config.RESULTS_DIR, "statistical_tests")
BKT = os.path.join(config.RESULTS_DIR, "bucket_assignments")
for d in (RAW, MET, PLT, TAB, STT, BKT):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
def _models_for(dataset: str, cfg_raw: dict):
    if dataset == "pubmedqa":
        return cfg_raw.get("pubmedqa_models", ["pubmedbert_pubmedqa", "bert_base"])
    return cfg_raw.get("models", ["pubmedbert_medmcqa", "biobert_medmcqa"])


def cmd_run(ec: EvalConfig, cfg_raw: dict) -> pd.DataFrame:
    """Run all (dataset, model) combinations -> save + return concatenated records."""
    frames = []
    for dataset in ec.datasets:
        for model_name in _models_for(dataset, cfg_raw):
            spec = config.MODELS.get(model_name)
            if spec is None or dataset not in spec.datasets:
                print(f"[skip] {model_name} not valid for {dataset}")
                continue
            print(f"[run] {dataset} x {model_name} seeds={ec.seeds}", flush=True)
            df = run_encoder_dataset(
                dataset, model_name, seeds=tuple(ec.seeds), methods=tuple(ec.methods),
                mc_samples=ec.mc_samples, mc_p=cfg_raw.get("mc_p", 0.2),
                max_eval=ec.max_samples, device=ec.device)
            save_predictions(df, dataset, model_name)
            frames.append(df)
    allrecs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    allrecs.to_parquet(os.path.join(RAW, "all_records.parquet"), index=False)
    _save_bucket_assignments(allrecs)
    print(f"[run] total records: {len(allrecs)}")
    return allrecs


def _save_bucket_assignments(df: pd.DataFrame):
    if df.empty:
        return
    sub = df.drop_duplicates(["dataset", "uid"])[["dataset", "uid", "buckets", "severity"]]
    sub.to_parquet(os.path.join(BKT, "bucket_assignments.parquet"), index=False)
    # bucket frequency summary
    import json as _json
    from collections import Counter
    c = Counter()
    for b in sub["buckets"]:
        c.update(_json.loads(b))
    pd.DataFrame(sorted(c.items(), key=lambda x: -x[1]),
                 columns=["bucket", "count"]).to_csv(
        os.path.join(BKT, "bucket_frequency.csv"), index=False)


# ---------------------------------------------------------------------------
def cmd_ood(ec: EvalConfig, cfg_raw: dict) -> pd.DataFrame:
    """For each OOD pair, eval the model on ID and OOD; report drift per metric."""
    rows = []
    for pair in cfg_raw.get("ood_pairs", []):
        model_name, id_ds, ood_ds = pair["model"], pair["id"], pair["ood"]
        out = {}
        for tag, ds in [("id", id_ds), ("ood", ood_ds)]:
            df = run_encoder_dataset(ds, model_name, seeds=(0,),
                                     methods=("base", "medbayes_lite"),
                                     mc_samples=ec.mc_samples,
                                     mc_p=cfg_raw.get("mc_p", 0.2),
                                     max_eval=ec.max_samples, device=ec.device)
            for method in ["base", "medbayes_lite"]:
                m = compute_record_metrics(df[df.method == method])
                out[(tag, method)] = m
        for method in ["base", "medbayes_lite"]:
            mi, mo = out[("id", method)], out[("ood", method)]
            rows.append({
                "model": model_name, "transfer": f"{id_ds}->{ood_ds}", "method": method,
                "id_acc": mi["accuracy"], "ood_acc": mo["accuracy"],
                "id_ece": mi["ece"], "ood_ece": mo["ece"],
                "ece_drift": mo["ece"] - mi["ece"],
                "id_harmful@0.8": mi["harmful@0.8"], "ood_harmful@0.8": mo["harmful@0.8"],
                "harmful_drift": mo["harmful@0.8"] - mi["harmful@0.8"],
                "id_mean_unc": mi["mean_uncertainty"], "ood_mean_unc": mo["mean_uncertainty"],
                "unc_increase": mo["mean_uncertainty"] - mi["mean_uncertainty"],
                "id_aurc": mi["aurc"], "ood_aurc": mo["aurc"],
            })
    odf = pd.DataFrame(rows)
    odf.to_csv(os.path.join(MET, "ood_transfer.csv"), index=False)
    print(f"[ood] {len(odf)} transfer rows")
    return odf


# ---------------------------------------------------------------------------
def cmd_perturb(ec: EvalConfig, cfg_raw: dict, n: int = 300) -> pd.DataFrame:
    """Apply each perturbation to a test subset and measure UQ response."""
    from .models.encoder import EncoderModel
    from .models.encoding import is_mcq
    from .models import uq_methods as uq
    from .perturb.perturbations import PERTURBATIONS, apply_perturbation, perturbation_report
    from .buckets.rules import assign_buckets, sample_severity
    from .schema import make_record, records_to_frame
    from .pipeline import _resolve_source
    from .models.finetune import TrainConfig

    pair = cfg_raw.get("ood_pairs", [{"model": "pubmedbert_medmcqa", "id": "medmcqa"}])[0]
    model_name, dataset = pair["model"], pair["id"]
    train, calib, test = get_train_eval(dataset, max_eval=None)
    rng = np.random.default_rng(0)
    test = [test[i] for i in rng.permutation(len(test))[:n]]
    ckpt, mcq, _ = _resolve_source(model_name, dataset, train, test[0].n_classes, 0,
                                   TrainConfig())
    model = EncoderModel(ckpt, mcq=mcq, device=ec.device,
                         max_len=320 if mcq else 512)

    def records_for(samples, tag):
        mcl = model.mc_logits(samples, ec.mc_samples, mc_p=cfg_raw.get("mc_p", 0.2))
        probs, conf, unc = uq.predict_medbayes_lite(mcl, 1.0)
        recs = []
        for i, s in enumerate(samples):
            b = assign_buckets(s)
            recs.append(make_record(s, model=model_name, method=tag, seed=0,
                                    probs=probs[i], confidence=float(conf[i]),
                                    uncertainty=float(unc[i]), buckets=b,
                                    severity=sample_severity(b)))
        return records_to_frame(recs)

    clean = records_for(test, "clean")
    reports = []
    for pname in PERTURBATIONS:
        perturbed = [apply_perturbation(s, pname, rng) for s in test]
        pr_df = records_for(perturbed, pname)
        rep = perturbation_report(clean.assign(perturbation="clean"),
                                  pr_df.assign(perturbation=pname), tau=ec.tau)
        rep["perturbation"] = pname
        reports.append(rep)
    out = pd.concat(reports, ignore_index=True)
    out.to_csv(os.path.join(MET, "perturbation_report.csv"), index=False)
    print(f"[perturb] {len(out)} perturbation rows")
    return out


# ---------------------------------------------------------------------------
def cmd_analyze(ec: EvalConfig):
    """Compute metric tables, correlations, stats, plots, and LaTeX/CSV tables."""
    from .stats.bootstrap import paired_bootstrap, seed_aggregate
    from .reporting import tables as T
    from .plotting import plots as P

    path = os.path.join(RAW, "all_records.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("run `python -m medbayes_eval.run run` first")
    df = pd.read_parquet(path)
    # fold in the generative model records if present
    gen_path = os.path.join(RAW, "generative_records.parquet")
    if os.path.exists(gen_path):
        df = pd.concat([df, pd.read_parquet(gen_path)], ignore_index=True)
        print(f"[analyze] including {gen_path}")

    # per (dataset, model, method) and per-seed
    overall = metrics_table(df, group_cols=("dataset", "model", "method"))
    perseed = metrics_table(df, group_cols=("dataset", "model", "method", "seed"))
    bucket = bucket_metrics_table(df, group_cols=("dataset", "model", "method"))
    overall.to_csv(os.path.join(MET, "metrics_overall.csv"), index=False)
    perseed.to_csv(os.path.join(MET, "metrics_per_seed.csv"), index=False)
    bucket.to_csv(os.path.join(MET, "metrics_by_bucket.csv"), index=False)

    # CUS/ZTI validation against clinical failure metrics
    corr = correlation_analysis(pd.concat([overall, bucket], ignore_index=True))
    corr.to_csv(os.path.join(MET, "cus_zti_correlation.csv"), index=False)

    # seed aggregation (mean +/- std) for headline metrics
    value_cols = [c for c in ["accuracy", "ece", "nll", "brier", "aurc", "eaurc",
                              "harmful@0.8", "cov@0.7", "selacc@0.7", "zti", "cus"]
                  if c in perseed.columns]
    agg = seed_aggregate(perseed, ["dataset", "model", "method"], value_cols)
    agg.to_csv(os.path.join(MET, "metrics_seed_aggregated.csv"), index=False)

    # paired bootstrap: base vs medbayes_lite per (dataset, model) on key metrics
    stat_rows = []
    for (ds, mdl), sub in df.groupby(["dataset", "model"]):
        a = sub[sub.method == "base"]
        b = sub[sub.method == "medbayes_lite"]
        if a.empty or b.empty:
            continue
        for metric, fn in [
            ("harmful@0.8", lambda d: compute_record_metrics(d)["harmful@0.8"]),
            ("aurc", lambda d: compute_record_metrics(d)["aurc"]),
            ("ece", lambda d: compute_record_metrics(d)["ece"]),
            ("selacc@0.7", lambda d: compute_record_metrics(d)["selacc@0.7"]),
        ]:
            try:
                res = paired_bootstrap(a, b, fn, n=config.BOOTSTRAP_N, seed=0)
                res.update({"dataset": ds, "model": mdl, "metric": metric,
                            "comparison": "base - medbayes_lite (positive = MedBayes-Lite lowers the metric)"})
                stat_rows.append(res)
            except Exception as e:  # noqa
                print(f"[stats] skip {ds}/{mdl}/{metric}: {e}")
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(STT, "paired_bootstrap_base_vs_mbl.csv"), index=False)

    # LaTeX/CSV tables
    T.main_results_table(overall, TAB, caption="Main clinical UQ results.",
                         label="tab:main")
    T.bucket_table(bucket, TAB, metric="harmful@0.8",
                   caption="Harmful overconfident error per clinical bucket.",
                   label="tab:bucket_harmful")
    if os.path.exists(os.path.join(MET, "ood_transfer.csv")):
        T.ood_table(pd.read_csv(os.path.join(MET, "ood_transfer.csv")), TAB,
                    caption="OOD domain transfer (MedMCQA -> MedQA).", label="tab:ood")
    T.correlation_table(corr, TAB, caption="CUS/ZTI correlation with clinical metrics.",
                        label="tab:corr")

    # plots
    P.make_all_plots(df, pd.concat([overall, bucket], ignore_index=True), PLT)
    print("[analyze] metrics, correlations, stats, tables, and plots written.")
    return overall, corr


# ---------------------------------------------------------------------------
def load_config(path: str | None) -> tuple[EvalConfig, dict]:
    import yaml
    raw = {}
    if path and os.path.exists(path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    ec = EvalConfig(
        datasets=tuple(raw.get("datasets", EvalConfig.datasets)),
        models=tuple(raw.get("models", EvalConfig.models)),
        methods=tuple(raw.get("methods", EvalConfig.methods)),
        seeds=tuple(raw.get("seeds", EvalConfig.seeds)),
        mc_samples=raw.get("mc_samples", 20),
        tau=raw.get("tau", 0.7),
        max_samples=raw.get("max_eval", None),
        device=raw.get("device", "cuda"),
    )
    return ec, raw


def main():
    ap = argparse.ArgumentParser(description="MedBayes-Lite clinical UQ evaluation")
    ap.add_argument("command", choices=["run", "ood", "perturb", "analyze", "all"])
    ap.add_argument("--config", default=os.path.join(config.REPO_ROOT, "configs", "default.yaml"))
    ap.add_argument("--max-eval", type=int, default=None,
                    help="override: cap eval items per dataset (quick runs)")
    args = ap.parse_args()
    ec, raw = load_config(args.config)
    if args.max_eval is not None:
        ec.max_samples = args.max_eval

    if args.command in ("run", "all"):
        cmd_run(ec, raw)
    if args.command in ("ood", "all"):
        cmd_ood(ec, raw)
    if args.command in ("perturb", "all"):
        cmd_perturb(ec, raw)
    if args.command in ("analyze", "all"):
        cmd_analyze(ec)


if __name__ == "__main__":
    main()
