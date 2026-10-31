"""Tokenization helpers shared by training and inference.

Two task formats:
  - multiple-choice (MedQA/MedMCQA/MediQ): each option is paired with the
    question(+context); model scores the 4 choices -> softmax over choices.
  - sequence classification (PubMedQA 3-way, MIMIC binary): question[+context]
    encoded once -> softmax over classes.
"""
from __future__ import annotations

from typing import List
import torch

from ..schema import Sample


def load_tokenizer(name_or_path: str):
    """Robustly load a tokenizer. Some BERT checkpoints (e.g. BioBERT v1.1) ship a
    config the AutoTokenizer backend cannot instantiate; fall back to the explicit
    fast/slow BERT tokenizer which reads vocab.txt directly."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except Exception:
        from transformers import BertTokenizerFast, BertTokenizer
        try:
            return BertTokenizerFast.from_pretrained(name_or_path)
        except Exception:
            return BertTokenizer.from_pretrained(name_or_path)


def is_mcq(sample: Sample) -> bool:
    return sample.options is not None and len(sample.options) > 0


def build_mcq_features(tokenizer, samples: List[Sample], max_len: int = 320):
    """Return dict of tensors shaped (N, C, L) for AutoModelForMultipleChoice."""
    first_parts, second_parts = [], []
    n_choices = max(len(s.options) for s in samples)
    for s in samples:
        ctx = (s.context + " ") if s.context else ""
        stem = f"{ctx}{s.question}"
        opts = list(s.options) + [""] * (n_choices - len(s.options))
        for opt in opts:
            first_parts.append(stem)
            second_parts.append(opt)
    enc = tokenizer(first_parts, second_parts, truncation="only_first",
                    max_length=max_len, padding="max_length", return_tensors="pt")
    out = {}
    for k, v in enc.items():
        out[k] = v.view(len(samples), n_choices, -1)
    return out, n_choices


def build_seqcls_features(tokenizer, samples: List[Sample], max_len: int = 512):
    """Return dict of tensors shaped (N, L) for AutoModelForSequenceClassification."""
    texts_a, texts_b = [], []
    for s in samples:
        texts_a.append(s.question)
        texts_b.append(s.context if s.context else "")
    has_b = any(t for t in texts_b)
    if has_b:
        enc = tokenizer(texts_a, texts_b, truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
    else:
        enc = tokenizer(texts_a, truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
    return enc


def labels_tensor(samples: List[Sample]) -> torch.Tensor:
    return torch.tensor([s.label for s in samples], dtype=torch.long)
