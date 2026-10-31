"""Item tables (split roles, clinical buckets under both rulesets, triggers) and the
split-ID export used for reproducibility (R3:5)."""
from __future__ import annotations

import json
import os
from typing import Sequence

import pandas as pd

from ..buckets.rules import assign_buckets, explain_buckets, sample_severity
from ..pipeline import get_train_eval
from . import paths

DATASETS = ("medmcqa", "medqa", "pubmedqa")


def items_path(dataset: str) -> str:
    return os.path.join(paths.ITEMS, f"{dataset}.parquet")


def _rows(samples, dataset, role):
    rows = []
    for s in samples:
        b = assign_buckets(s)
        bs = assign_buckets(s, strict=True)
        rows.append({
            "dataset": dataset, "uid": s.uid, "role": role, "label": s.label,
            "question": s.question, "context": s.context or "",
            "options": json.dumps(s.options) if s.options else "",
            "buckets": json.dumps(b), "severity": sample_severity(b), "n_buckets": len(b),
            "buckets_strict": json.dumps(bs), "severity_strict": sample_severity(bs),
            "triggers": json.dumps(explain_buckets(s)),
        })
    return rows


def prepare_items(datasets: Sequence[str] = DATASETS, max_eval=None, force=False):
    """Write cache/items/<dataset>.parquet for every dataset (CPU, idempotent)."""
    out = {}
    for ds in datasets:
        p = items_path(ds)
        if os.path.exists(p) and not force:
            out[ds] = p
            continue
        train, calib, test = get_train_eval(ds, max_eval=max_eval,
                                            load_train=(ds == "pubmedqa"))
        rows = _rows(test, ds, "test") + _rows(calib, ds, "calibration")
        rows += [{"dataset": ds, "uid": s.uid, "role": "train", "label": s.label}
                 for s in train]
        pd.DataFrame(rows).to_parquet(p, index=False)
        out[ds] = p
    return out


def load_items(dataset: str, role: str = "test") -> pd.DataFrame:
    df = pd.read_parquet(items_path(dataset))
    return df[df.role == role] if role else df


def severity_lookup(dataset: str, uids, col: str = "severity"):
    """Per-uid severity from the eval items (test + calibration roles)."""
    df = load_items(dataset, None)
    df = df[df.role != "train"].drop_duplicates("uid").set_index("uid")
    return df.loc[list(uids), col].to_numpy(float)


def export_split_ids():
    """splits/split_ids.csv: dataset, role, uid for every item (train/calib/test)."""
    frames = [pd.read_parquet(items_path(ds))[["dataset", "role", "uid"]]
              for ds in DATASETS if os.path.exists(items_path(ds))]
    gen = [os.path.join(paths.GEN, f) for f in os.listdir(paths.GEN)
           if f.endswith("__qwen2_5_7b.parquet")]
    for g in gen:
        d = pd.read_parquet(g, columns=["dataset", "uid"]).drop_duplicates()
        frames.append(d.assign(role="generative_subset"))
    df = pd.concat(frames, ignore_index=True)
    p = paths.out("splits", "split_ids.csv")
    df.to_csv(p, index=False)
    df.groupby(["dataset", "role"]).size().rename("n").reset_index().to_csv(
        paths.out("splits", "split_sizes.csv"), index=False)
    return p
