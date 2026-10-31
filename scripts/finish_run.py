"""Finish the interrupted full run: run the 2 remaining encoder combos (biobert,
clinicalbert on PubMedQA), rebuild all_records.parquet from every saved per-combo
parquet, then run ood + perturb + analyze. Reuses the 6 already-saved combos."""
import glob
import os

import pandas as pd

from medbayes_eval import config
from medbayes_eval.config import EvalConfig
from medbayes_eval.pipeline import run_encoder_dataset, save_predictions
from medbayes_eval.run import cmd_ood, cmd_perturb, cmd_analyze, load_config, RAW

ec, raw = load_config(os.path.join(config.REPO_ROOT, "configs", "default.yaml"))

# 1. remaining encoder combos
for model_name in ["biobert", "clinicalbert"]:
    out = os.path.join(RAW, f"pubmedqa__{model_name}.parquet")
    if os.path.exists(out):
        print(f"[finish] {model_name} already present, skip")
        continue
    print(f"[finish] running pubmedqa x {model_name}", flush=True)
    df = run_encoder_dataset("pubmedqa", model_name, seeds=tuple(ec.seeds),
                             methods=tuple(ec.methods), mc_samples=ec.mc_samples,
                             mc_p=raw.get("mc_p", 0.2), device=ec.device)
    save_predictions(df, "pubmedqa", model_name)

# 2. rebuild all_records from every encoder per-combo parquet (exclude qwen/aggregates)
parts = []
for p in sorted(glob.glob(os.path.join(RAW, "*.parquet"))):
    base = os.path.basename(p)
    if "qwen" in base or base in ("all_records.parquet", "generative_records.parquet"):
        continue
    parts.append(pd.read_parquet(p))
allrecs = pd.concat(parts, ignore_index=True)
allrecs.to_parquet(os.path.join(RAW, "all_records.parquet"), index=False)
print(f"[finish] all_records rebuilt: {len(allrecs)} rows from {len(parts)} combos")
print("  models:", sorted(allrecs.model.unique()))
print("  datasets:", sorted(allrecs.dataset.unique()))
print("  methods:", sorted(allrecs.method.unique()))

# bucket assignment export
from medbayes_eval.run import _save_bucket_assignments
_save_bucket_assignments(allrecs)

# 3. downstream analyses
cmd_ood(ec, raw)
cmd_perturb(ec, raw)
cmd_analyze(ec)
print("[finish] DONE")
