"""CLI for the revision experiments.

  python -m medbayes_eval.revision.reproduce prefetch     # download data + weights
  python -m medbayes_eval.revision.reproduce gpu          # encoder/sensitivity/perturb caches
  python -m medbayes_eval.revision.reproduce latency      # latency benchmark (1 GPU)
  python -m medbayes_eval.revision.reproduce generative   # Qwen2.5-7B (all GPUs)
  python -m medbayes_eval.revision.reproduce analyze      # every CPU analysis
  python -m medbayes_eval.revision.reproduce table8       # Table 8 only (runs GPU jobs if missing)
  python -m medbayes_eval.revision.reproduce all

Add --smoke for a tiny end-to-end check (40 eval items, 2 MC samples, 1 seed).
"""
from __future__ import annotations

import argparse
import time

from . import plan


def prefetch(include_generative: bool = True):
    from huggingface_hub import snapshot_download
    from .. import config
    from .items import prepare_items
    prepare_items()
    for ds in ("medmcqa", "medqa"):  # train splits are used by the leakage checks
        from ..data.loaders import load_dataset_samples
        load_dataset_samples(ds, split="train", max_samples=1)
    ids = [config.MODELS[m].hf_id for m in plan.MCQ_MODELS] + \
          [config.FINETUNE_BACKBONES[m] for m in plan.PUBMEDQA_MODELS]
    if include_generative:
        ids.append(config.MODELS["qwen2_5_7b"].hf_id)
    for hf_id in ids:
        snapshot_download(hf_id, revision=config.model_revision(hf_id),
                          ignore_patterns=["*.msgpack", "*.h5", "*.ckpt*", "*.pbtxt", "*.onnx"])
        print("[prefetch]", hf_id, flush=True)


def gpu(smoke=False, seeds=(0, 1, 2), n_gpus=None):
    from .multigpu import run_parallel
    kw = dict(seeds=(0,), mc_samples=2, max_eval=40, max_train=32, perturb_n=12) if smoke \
        else dict(seeds=tuple(seeds))
    return run_parallel(plan.encoder_jobs(**kw), n_gpus=n_gpus, queue_name="encoders")


def latency(smoke=False):
    from .multigpu import run_parallel
    return run_parallel(plan.latency_jobs(reps=3 if smoke else 30), gpu_groups=["0"],
                        queue_name="latency")


def generative(smoke=False, n_gpus=None, model_id=None):
    import torch
    from .multigpu import run_parallel
    n = torch.cuda.device_count() if n_gpus is None else n_gpus
    jobs = plan.generative_jobs(n=6 if smoke else 200, n_samples=3 if smoke else 10)
    if model_id:
        jobs[0]["args"]["model_id"] = model_id
    groups = [",".join(str(i) for i in range(n))] if n else None
    return run_parallel(jobs, gpu_groups=groups, queue_name="generative")


def analyze(smoke=False, include_generative=True):
    """Run every CPU analysis; returns a dict of result tables."""
    from . import analyses as A, cus, legacy, perturbation, severity
    from .clinician_kit import analyze_clinician_ratings, export_severity_rating_kit
    from .items import export_split_ids, prepare_items
    from .manifest import write_manifest
    from .predictions import load_store
    B1, B, Bi = (200, 100, 20) if smoke else (10000, 2000, 500)
    t0 = time.time()
    res = {}

    def step(name, fn, *a, **k):
        import traceback
        t = time.time()
        try:
            res[name] = fn(*a, **k)
            print(f"[analyze] {name:28s} {time.time() - t:6.0f}s", flush=True)
        except Exception:
            res[name] = None
            print(f"[analyze] {name} FAILED:\n{traceback.format_exc()}", flush=True)
        return res[name]

    prepare_items()
    store, temps = load_store(include_generative=include_generative)
    main = step("main_table", A.main_table, store, B=B)
    step("paired_tests", A.paired_tests, store, B_primary=B1, B=B)
    step("seed_vs_item_variance", A.seed_vs_item_variance, store, B=B // 2)
    step("ood", A.ood_table, store, B=B)
    cov = (step("matched_coverage", A.matched_coverage, store, B=B // 2) or (None, None))[0]
    thr = step("threshold_counts", A.threshold_counts, store)
    if thr is not None and cov is not None and main is not None:
        work = A.workload(thr, cov)
        step("workload", A.add_errors_caught, work, main)
        step("factorial_ablation", A.factorial_ablation, main, cov)
    sens = step("sensitivity", A.sensitivity_tables)
    if main is not None:
        step("compute_curve", A.compute_curve, sens, main)
    step("severity_sensitivity", severity.severity_sensitivity, store, B=B)
    if thr is not None:
        step("denominators", severity.denominators, thr)
    step("multi_category", severity.multi_category, store)
    step("trigger_audit", severity.trigger_audit)
    runs = step("cus_runs", cus.build_runs, store)
    if runs:
        base, _, item_draws = (step("cus_correlations", cus.cus_correlations, runs, B_items=Bi)
                               or (None, None, None))
        if base is not None:
            step("cus_incremental", cus.cus_incremental, base, item_draws, B=B)
        step("cus_weight_schemes", cus.cus_weight_schemes, runs)
    step("perturbation", perturbation.perturbation_analysis, B=B)
    step("perturbation_review", perturbation.export_perturbation_review)
    step("clinician_kit", export_severity_rating_kit)
    step("clinician_ratings", analyze_clinician_ratings, store=store)
    df = step("records", legacy.records_frame, store)
    if df is not None:
        overall, bucket = step("legacy_tables", legacy.legacy_tables, df) or (None, None)
        if overall is not None:
            step("figures", legacy.figures, df, overall, bucket, cov)
    step("split_ids", export_split_ids)
    step("manifest", write_manifest, temps)
    print(f"[analyze] all done in {time.time() - t0:.0f}s")
    return res


def analyze_cus(smoke=False, include_generative=True):
    """Re-run only the CUS analyses (R3:7) from the existing caches."""
    from . import cus
    from .predictions import load_store
    B, Bi = (100, 20) if smoke else (2000, 500)
    store, _ = load_store(include_generative=include_generative)
    runs = cus.build_runs(store)
    base, corr, item_draws = cus.cus_correlations(runs, B_items=Bi)
    inc = cus.cus_incremental(base, item_draws, B=B)
    w = cus.cus_weight_schemes(runs)
    return {"cus_correlations": corr, "cus_incremental": inc, "cus_weight_schemes": w}


def table8(smoke=False):
    """Reproduce Table 8 (main harmful-error table) from scratch or from caches."""
    import os
    from . import analyses as A
    from .multigpu import run_parallel
    from .predictions import load_store
    from .items import prepare_items
    jobs = [j for j in plan.encoder_jobs(include_sensitivity=False, include_perturb=False,
                                         **(dict(seeds=(0,), mc_samples=2, max_eval=40)
                                            if smoke else {}))
            if "pubmedqa" not in j["id"]]
    run_parallel(jobs, queue_name="table8")
    prepare_items()
    store, _ = load_store(include_generative=False)
    df = A.main_table(store, B=100 if smoke else 2000)
    print(df[(df.gate == "maxprob") & df.method.isin(
        ["base", "temp_scaling", "mc_dropout", "medbayes_lite"])][
        ["dataset", "model", "method", "ece", "harmful@0.8", "selacc@0.7", "cov@0.7",
         "aurc"]].round(3).to_string(index=False))
    print("written to", os.path.join(A.TAB, "table8_maxprob_gate.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prefetch", "gpu", "latency", "generative", "analyze",
                                      "table8", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-generative", action="store_true")
    a = ap.parse_args()
    if a.stage in ("prefetch", "all"):
        prefetch(include_generative=not a.no_generative)
    if a.stage in ("gpu", "all"):
        gpu(a.smoke)
    if a.stage in ("latency", "all"):
        latency(a.smoke)
    if a.stage in ("generative", "all") and not a.no_generative:
        generative(a.smoke)
    if a.stage in ("analyze", "all"):
        analyze(a.smoke, include_generative=not a.no_generative)
    if a.stage == "table8":
        table8(a.smoke)


if __name__ == "__main__":
    main()
