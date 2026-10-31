"""Inference backbone for fine-tuned encoders, with MC-dropout support and an
optional uncertainty-weighted attention hook.

This object produces logits; the UQ methods in uq_methods.py turn logits into
calibrated predictive distributions + scalar uncertainties.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForMultipleChoice,
                          AutoModelForSequenceClassification)

from ..schema import Sample
from .encoding import build_mcq_features, build_seqcls_features, is_mcq, load_tokenizer


def _enable_mc_dropout(model: nn.Module, p: float | None = None):
    """Put ONLY dropout layers into train mode (keep LayerNorm in eval). Optionally
    override the dropout probability, useful for checkpoints trained with a small
    default rate (e.g. 0.1) where stronger inference-time stochasticity is desired."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            if p is not None:
                m.p = p
            m.train()


class EncoderModel:
    """Loads a fine-tuned checkpoint and exposes logit prediction."""

    def __init__(self, ckpt_dir: str, mcq: bool, device: str = "cuda",
                 max_len: int = 320, batch_size: int = 32):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mcq = mcq
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = load_tokenizer(ckpt_dir)
        if mcq:
            self.model = AutoModelForMultipleChoice.from_pretrained(ckpt_dir)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        self.model.to(self.device).eval()

    def _features(self, samples: List[Sample]):
        if self.mcq:
            feats, _ = build_mcq_features(self.tokenizer, samples, self.max_len)
        else:
            feats = build_seqcls_features(self.tokenizer, samples, self.max_len)
        return feats

    @torch.no_grad()
    def logits(self, samples: List[Sample], mc_dropout: bool = False,
               mc_p: Optional[float] = None) -> np.ndarray:
        """Single forward pass -> (N, K) logits. If mc_dropout, dropout stays active."""
        if mc_dropout:
            _enable_mc_dropout(self.model, p=mc_p)
        else:
            self.model.eval()
        feats = self._features(samples)
        keys = list(feats.keys())
        n = len(samples)
        all_logits = []
        for start in range(0, n, self.batch_size):
            batch = {k: feats[k][start:start + self.batch_size].to(self.device) for k in keys}
            with torch.autocast("cuda", enabled=self.device == "cuda"):
                out = self.model(**batch)
            all_logits.append(out.logits.float().cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    @torch.no_grad()
    def mc_logits(self, samples: List[Sample], n_samples: int = 20,
                  mc_p: Optional[float] = None) -> np.ndarray:
        """Stack of MC-dropout logits -> (S, N, K)."""
        stack = [self.logits(samples, mc_dropout=True, mc_p=mc_p) for _ in range(n_samples)]
        self.model.eval()
        return np.stack(stack, axis=0)


def load_encoder_for(ckpt_dir: str, sample: Sample, **kw) -> EncoderModel:
    mcq = is_mcq(sample)
    return EncoderModel(ckpt_dir, mcq=mcq, **kw)
