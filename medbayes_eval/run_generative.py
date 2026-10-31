"""Run the local generative model (Qwen2.5-7B-Instruct) on medical QA with
API-compatible UQ (self-consistency, semantic entropy, verbalized confidence).

Separate entry point because it loads a 7B model. Designed to run on a second GPU
in parallel with the encoder sweep. Outputs land in the same results/ layout so
`analyze` can pick them up alongside the encoder records.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from . import config
from .data.loaders import load_dataset_samples
from .buckets.rules import assign_buckets, sample_severity
from .schema import make_record, records_to_frame
from .models.generative import GenerativeModel
from .metrics.core import compute_record_metrics

RAW = os.path.join(config.RESULTS_DIR, "raw_predictions")
MET = os.path.join(config.RESULTS_DIR, "metrics")


def run(datasets, n_per_dataset=200, n_samples=10, temperature=0.7,
        model_id="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
    model = GenerativeModel(model_id, device=device)
    frames = []
    for ds in datasets:
        split = "validation" if ds == "medmcqa" else "test"
        samples = load_dataset_samples(ds, split=split, max_samples=n_per_dataset)
        t0 = time.time()
        recs = []
        for i, s in enumerate(samples):
            buckets = assign_buckets(s)
            sev = sample_severity(buckets)
            out = model.predict_shared(s, n_samples=n_samples, temperature=temperature)
            for method, (probs, conf, unc) in out.items():
                recs.append(make_record(s, model="qwen2_5_7b", method=method, seed=0,
                                        probs=probs, confidence=conf, uncertainty=unc,
                                        buckets=buckets, severity=sev))
            if (i + 1) % 50 == 0:
                print(f"  [{ds}] {i+1}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)
        df = records_to_frame(recs)
        df.to_parquet(os.path.join(RAW, f"{ds}__qwen2_5_7b.parquet"), index=False)
        frames.append(df)
        for method in df.method.unique():
            m = compute_record_metrics(df[df.method == method])
            print(f"[gen] {ds} {method:18s} acc={m['accuracy']:.3f} ece={m['ece']:.3f} "
                  f"aurc={m['aurc']:.3f} harm@0.8={m['harmful@0.8']:.3f} "
                  f"cov@0.7={m['cov@0.7']:.2f} selacc@0.7={m['selacc@0.7']:.3f}", flush=True)
    allg = pd.concat(frames, ignore_index=True)
    allg.to_parquet(os.path.join(RAW, "generative_records.parquet"), index=False)
    print(f"[gen] total generative records: {len(allg)}")
    return allg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pubmedqa", "medmcqa", "medqa"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    run(args.datasets, n_per_dataset=args.n, n_samples=args.samples, model_id=args.model)


if __name__ == "__main__":
    main()
