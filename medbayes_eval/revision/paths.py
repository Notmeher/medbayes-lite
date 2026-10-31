"""Output layout for the revision runs (all under config.RESULTS_DIR/revision)."""
from __future__ import annotations

import os

from .. import config

ROOT = os.path.join(config.RESULTS_DIR, "revision")
CACHE = os.path.join(ROOT, "cache")
LOGITS = os.path.join(CACHE, "logits")
SENS = os.path.join(CACHE, "sensitivity")
PERTURB = os.path.join(CACHE, "perturb")
ITEMS = os.path.join(CACHE, "items")
GEN = os.path.join(CACHE, "generative")
QUEUE = os.path.join(CACHE, "queue")
LOGS = os.path.join(ROOT, "logs")


def out(*parts: str) -> str:
    """Path under revision/<parts>, creating the parent directory."""
    p = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def logits_path(dataset: str, model: str, seed: int) -> str:
    return os.path.join(LOGITS, f"{dataset}__{model}__seed{seed}.npz")


for _d in (LOGITS, SENS, PERTURB, ITEMS, GEN, QUEUE, LOGS):
    os.makedirs(_d, exist_ok=True)
