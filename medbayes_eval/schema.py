"""Shared data contracts: the Sample (model input) and the per-sample
PredictionRecord (the linchpin row that every metric/plot/stat consumes).

Keeping the record schema explicit and flat lets the whole downstream stack
(metrics, buckets, plots, stats) operate on a single tidy DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json
import os

import numpy as np
import pandas as pd


@dataclass
class Sample:
    """A single evaluation item, dataset-agnostic."""
    uid: str
    dataset: str
    question: str
    label: int
    n_classes: int
    options: Optional[list] = None      # list[str] for MCQ; None for pure classification
    context: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def display_text(self) -> str:
        parts = [self.question]
        if self.context:
            parts.append(self.context)
        if self.options:
            parts.append(" ".join(self.options))
        return " ".join(parts)


# Canonical columns of a per-sample prediction record. `probs` is stored as a JSON
# string of the calibrated predictive distribution; acceptance / overconfident /
# harmful flags are derived downstream so thresholds stay configurable.
RECORD_COLUMNS = [
    "dataset", "model", "method", "seed", "uid",
    "label", "pred", "n_classes",
    "confidence",      # max calibrated prob (or 1 - H/logK; see method)
    "uncertainty",     # method-specific scalar (predictive entropy / std / etc.)
    "probs",           # JSON list[float]
    "correct",         # bool
    "buckets",         # JSON list[str]
    "severity",        # float (max bucket severity)
]


def make_record(sample: Sample, *, model: str, method: str, seed: int,
                probs: np.ndarray, confidence: float, uncertainty: float,
                buckets, severity: float) -> dict:
    probs = np.asarray(probs, dtype=float).ravel()
    pred = int(np.argmax(probs))
    return {
        "dataset": sample.dataset,
        "model": model,
        "method": method,
        "seed": int(seed),
        "uid": sample.uid,
        "label": int(sample.label),
        "pred": pred,
        "n_classes": int(sample.n_classes),
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "probs": json.dumps([round(float(p), 6) for p in probs]),
        "correct": bool(pred == int(sample.label)),
        "buckets": json.dumps(list(buckets)),
        "severity": float(severity),
    }


def records_to_frame(records) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=RECORD_COLUMNS)
    return df


def probs_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack the JSON `probs` column into an (N, K) array (ragged-safe)."""
    rows = [json.loads(p) for p in df["probs"].tolist()]
    k = max(len(r) for r in rows)
    out = np.full((len(rows), k), np.nan)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def buckets_series(df: pd.DataFrame):
    return df["buckets"].apply(json.loads)


def save_records(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_records(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
