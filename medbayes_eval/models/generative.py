"""Local generative-model UQ for medical QA (Qwen2.5-7B-Instruct by default).

Implements three API-compatible uncertainty estimators that need only sampled
generations (no internal access):
  - self_consistency: vote distribution over sampled answers; confidence = vote share
  - semantic_entropy:  entropy of the answer-cluster distribution (Kuhn et al. 2023),
                       here over the discrete option clusters
  - verbalized:        the model's stated confidence parsed from its own output

For multiple-choice / closed-label medical QA the answer space is the option set,
so semantic clustering reduces to grouping by the extracted option label. The
predictive distribution is the (smoothed) vote distribution; this plugs into the
exact same record schema and metric suite as the encoder methods.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

import numpy as np
import torch

from ..schema import Sample
from . import uq_methods as uq

_LETTERS = ["A", "B", "C", "D", "E"]
PUBMEDQA_CHOICES = ["yes", "no", "maybe"]


def _build_prompt(sample: Sample, ask_confidence: bool) -> str:
    ctx = f"Context: {sample.context}\n" if sample.context else ""
    if sample.options:
        opts = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(sample.options))
        ask = ("Answer with the single letter of the best option, then on a new line "
               "'Confidence: X%'." if ask_confidence else
               "Answer with the single letter of the best option only.")
        return (f"{ctx}Question: {sample.question}\n{opts}\n{ask}")
    # PubMedQA-style yes/no/maybe
    ask = ("Answer 'yes', 'no', or 'maybe', then on a new line 'Confidence: X%'."
           if ask_confidence else "Answer 'yes', 'no', or 'maybe' only.")
    return f"{ctx}Question: {sample.question}\n{ask}"


def _parse_label(text: str, sample: Sample) -> Optional[int]:
    t = text.strip()
    if sample.options:
        m = re.search(r"\b([A-E])\b", t.upper())
        if m:
            idx = _LETTERS.index(m.group(1))
            return idx if idx < len(sample.options) else None
        return None
    low = t.lower()
    for i, c in enumerate(PUBMEDQA_CHOICES):
        if re.search(rf"\b{c}\b", low):
            return i
    return None


def _parse_confidence(text: str) -> Optional[float]:
    m = re.search(r"confidence[:\s]*([0-9]+(?:\.[0-9]+)?)\s*%", text.lower())
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / 100.0))
    return None


class GenerativeModel:
    """Wraps a local instruct LLM and exposes sampling-based UQ over option labels."""

    def __init__(self, hub_id: str = "Qwen/Qwen2.5-7B-Instruct",
                 device: str = "cuda", dtype=torch.float16, max_new_tokens: int = 24):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(hub_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hub_id, torch_dtype=dtype, device_map=device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def _generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.device)
        do_sample = temperature > 0 and n > 1
        gen_kw = dict(max_new_tokens=self.max_new_tokens, num_return_sequences=n,
                      pad_token_id=self.tok.eos_token_id, do_sample=do_sample)
        if do_sample:
            gen_kw.update(temperature=temperature, top_p=0.95)
        out = self.model.generate(**inputs, **gen_kw)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)

    def predict_sample(self, sample: Sample, method: str, n_samples: int = 10,
                       temperature: float = 0.7):
        """Return (probs, confidence, uncertainty) for one sample under `method`."""
        k = sample.n_classes
        ask_conf = method == "verbalized"
        prompt = _build_prompt(sample, ask_confidence=ask_conf)
        n = 1 if method == "verbalized" else n_samples
        temp = 0.0 if method == "verbalized" else temperature
        gens = self._generate(prompt, n=n, temperature=temp)

        labels = [_parse_label(g, sample) for g in gens]
        valid = [l for l in labels if l is not None]
        counts = Counter(valid)
        probs = np.full(k, 1e-3)
        for lbl, c in counts.items():
            probs[lbl] += c
        probs = probs / probs.sum()

        if method == "verbalized":
            pred = labels[0] if labels and labels[0] is not None else int(np.argmax(probs))
            vconf = _parse_confidence(gens[0])
            conf = vconf if vconf is not None else float(probs.max())
            sharp = np.full(k, (1.0 - conf) / (k - 1))
            sharp[pred] = conf
            probs = sharp
            return probs, float(conf), float(1.0 - conf)

        if method == "semantic_entropy":
            # entropy over the answer-cluster (option) distribution, normalized
            ent = uq.entropy(probs[None, :])[0]
            conf = 1.0 - ent / np.log(k)
            return probs, float(conf), float(ent)

        # self_consistency (default): confidence = top vote share
        conf = float(probs.max())
        return probs, conf, float(1.0 - conf)


    def predict_shared(self, sample: Sample, n_samples: int = 10,
                       temperature: float = 0.7):
        """Sample once and derive self_consistency + semantic_entropy from the shared
        vote distribution; one greedy generation yields verbalized confidence. Avoids
        regenerating per method."""
        k = sample.n_classes
        prompt = _build_prompt(sample, ask_confidence=False)
        gens = self._generate(prompt, n=n_samples, temperature=temperature)
        labels = [_parse_label(g, sample) for g in gens]
        valid = [l for l in labels if l is not None]
        probs = np.full(k, 1e-3)
        for lbl in valid:
            probs[lbl] += 1
        probs = probs / probs.sum()
        ent = float(uq.entropy(probs[None, :])[0])
        out = {
            "self_consistency": (probs, float(probs.max()), float(1.0 - probs.max())),
            "semantic_entropy": (probs, float(1.0 - ent / np.log(k)), ent),
        }
        # verbalized: one greedy generation that also states confidence
        vprompt = _build_prompt(sample, ask_confidence=True)
        vgen = self._generate(vprompt, n=1, temperature=0.0)[0]
        vlabel = _parse_label(vgen, sample)
        vconf = _parse_confidence(vgen)
        pred = vlabel if vlabel is not None else int(np.argmax(probs))
        conf = vconf if vconf is not None else float(probs.max())
        vprobs = np.full(k, (1.0 - conf) / (k - 1))
        vprobs[pred] = conf
        out["verbalized"] = (vprobs, float(conf), float(1.0 - conf))
        return out


def run_generative_dataset(samples: List[Sample], model: GenerativeModel,
                           methods=("self_consistency", "semantic_entropy", "verbalized"),
                           n_samples: int = 10, temperature: float = 0.7,
                           model_name: str = "qwen2_5_7b", seed: int = 0):
    """Produce per-sample records for a generative model across the given methods."""
    from ..buckets.rules import assign_buckets, sample_severity
    from ..schema import make_record, records_to_frame
    recs = []
    for s in samples:
        buckets = assign_buckets(s)
        sev = sample_severity(buckets)
        for method in methods:
            probs, conf, unc = model.predict_sample(s, method, n_samples, temperature)
            recs.append(make_record(s, model=model_name, method=method, seed=seed,
                                    probs=probs, confidence=conf, uncertainty=unc,
                                    buckets=buckets, severity=sev))
    return records_to_frame(recs)
