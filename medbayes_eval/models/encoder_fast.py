"""Fast encoder inference for the revision runs.

Same predictions as models/encoder.py but (1) tokenizes each sample list once,
(2) batches length-sorted items with dynamic padding (attention-masked, so results
match max-length padding up to fp16 noise) and (3) seeds every MC-dropout stack so
reruns are reproducible.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..schema import Sample
from .encoder import _enable_mc_dropout


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer
    rev = config.model_revision(name_or_path)
    try:
        return AutoTokenizer.from_pretrained(name_or_path, revision=rev)
    except Exception:
        from transformers import BertTokenizerFast
        return BertTokenizerFast.from_pretrained(name_or_path, revision=rev)


def load_encoder(name_or_path: str, mcq: bool, num_labels: Optional[int] = None):
    from transformers import (AutoModelForMultipleChoice,
                              AutoModelForSequenceClassification,
                              BertForMultipleChoice, BertForSequenceClassification)
    rev = config.model_revision(name_or_path)
    kw = {"revision": rev}
    if not mcq and num_labels is not None:
        kw["num_labels"] = num_labels
    auto, bert = ((AutoModelForMultipleChoice, BertForMultipleChoice) if mcq else
                  (AutoModelForSequenceClassification, BertForSequenceClassification))
    try:
        return auto.from_pretrained(name_or_path, **kw)
    except (ValueError, KeyError):
        # some BERT checkpoints (BioBERT v1.1) ship a config without `model_type`
        return bert.from_pretrained(name_or_path, **kw)


class FastEncoder:
    def __init__(self, ckpt: str, mcq: bool, device: str = "cuda", max_len: int = 320,
                 batch_size: int = 64, fp16: bool = True):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mcq, self.max_len, self.batch_size = mcq, max_len, batch_size
        self.fp16 = fp16 and self.device == "cuda"
        self.tokenizer = load_tokenizer(ckpt)
        self.model = load_encoder(ckpt, mcq).to(self.device).eval()

    # -- tokenization -------------------------------------------------------
    def _encode(self, samples: List[Sample]):
        """Per-sample lists of token ids (and token_type_ids), no padding."""
        if self.mcq:
            n_ch = max(len(s.options) for s in samples)
            a, b = [], []
            for s in samples:
                stem = ((s.context + " ") if s.context else "") + s.question
                for opt in list(s.options) + [""] * (n_ch - len(s.options)):
                    a.append(stem)
                    b.append(opt)
            enc = self.tokenizer(a, b, truncation="only_first", max_length=self.max_len)
            per = {k: [v[i * n_ch:(i + 1) * n_ch] for i in range(len(samples))]
                   for k, v in enc.items()}
            lengths = [max(len(x) for x in ids) for ids in per["input_ids"]]
        else:
            a = [s.question for s in samples]
            b = [s.context or "" for s in samples]
            enc = (self.tokenizer(a, b, truncation=True, max_length=self.max_len)
                   if any(b) else self.tokenizer(a, truncation=True, max_length=self.max_len))
            per = dict(enc)
            lengths = [len(x) for x in per["input_ids"]]
        return per, lengths

    def prepare(self, samples: List[Sample]):
        """Tokenize once and build length-sorted, dynamically padded batches."""
        per, lengths = self._encode(samples)
        order = np.argsort(lengths, kind="stable")
        pad_id = self.tokenizer.pad_token_id or 0
        batches = []
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            L = max(lengths[i] for i in idx)
            batch = {}
            for k, vals in per.items():
                fill = pad_id if k == "input_ids" else 0
                if self.mcq:
                    arr = [[seq + [fill] * (L - len(seq)) for seq in vals[i]] for i in idx]
                else:
                    arr = [vals[i] + [fill] * (L - len(vals[i])) for i in idx]
                batch[k] = torch.tensor(arr, dtype=torch.long)
            batches.append((idx, batch))
        return {"n": len(samples), "batches": batches}

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def logits(self, prepared, mc_dropout: bool = False, mc_p: Optional[float] = None):
        if mc_dropout:
            _enable_mc_dropout(self.model, p=mc_p)
        else:
            self.model.eval()
        out = None
        for idx, batch in prepared["batches"]:
            b = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                lg = self.model(**b).logits.float().cpu().numpy()
            if out is None:
                out = np.zeros((prepared["n"], lg.shape[-1]), dtype=np.float32)
            out[idx] = lg
        self.model.eval()
        return out

    def mc_logits(self, prepared, n_samples: int, mc_p: Optional[float], seed: int):
        """(S, N, K) MC-dropout stack; the dropout masks are seeded by `seed`."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        stack = [self.logits(prepared, mc_dropout=True, mc_p=mc_p) for _ in range(n_samples)]
        return np.stack(stack, axis=0)

    def timed(self, fn, *a, **kw):
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return r, time.perf_counter() - t0
