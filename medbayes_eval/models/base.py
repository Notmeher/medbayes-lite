"""UQ model interface. A UQModel maps a list of Samples to per-sample predictive
distributions plus a scalar uncertainty, for one UQ method.

Concrete implementations live in models/encoder.py and models/generative.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from ..schema import Sample


@dataclass
class Prediction:
    probs: np.ndarray      # (K,) calibrated predictive distribution
    confidence: float      # scalar confidence in [0,1]
    uncertainty: float     # scalar uncertainty (method-specific)


class UQModel:
    """Base class. Subclasses implement `predict_batch`."""

    #: human-readable method name written into records
    method: str = "base"

    def predict_batch(self, samples: List[Sample], seed: int = 0) -> List[Prediction]:
        raise NotImplementedError

    # convenience
    def predict(self, samples: List[Sample], seed: int = 0) -> List[Prediction]:
        return self.predict_batch(samples, seed=seed)


def entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def normalized_confidence(probs: np.ndarray, eps: float = 1e-12) -> float:
    """C(p) = 1 - H(p)/log K  (paper's confidence-guided shaping score)."""
    probs = np.asarray(probs, dtype=float)
    k = len(probs)
    if k <= 1:
        return 1.0
    return float(1.0 - entropy(probs, eps) / np.log(k))
