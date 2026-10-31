"""UQ methods: turn encoder logits into calibrated predictive distributions plus a
scalar confidence and uncertainty, one function per method.

To avoid redundant forward passes, the pipeline computes deterministic logits and
(optionally) a stack of MC-dropout logits ONCE, then derives every method from
those cached tensors. Each function returns three aligned arrays:
    probs  (N, K)  predictive distribution
    conf   (N,)    scalar confidence in [0,1]
    unc    (N,)    scalar uncertainty
"""
from __future__ import annotations

from typing import List
import numpy as np

_EPS = 1e-12


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def normalized_confidence(probs: np.ndarray) -> np.ndarray:
    """C(p) = 1 - H(p)/log K (the paper's confidence-guided shaping score)."""
    k = probs.shape[-1]
    if k <= 1:
        return np.ones(probs.shape[0])
    return 1.0 - entropy(probs) / np.log(k)


# ---------------------------------------------------------------------------
# deterministic-logit methods
# ---------------------------------------------------------------------------
def predict_base(logits: np.ndarray):
    """MSP baseline: single softmax, confidence = max prob."""
    probs = softmax(logits)
    conf = probs.max(axis=-1)
    return probs, conf, 1.0 - conf


def fit_temperature(calib_logits: np.ndarray, calib_labels: np.ndarray,
                    lr: float = 0.01, n_iter: int = 200) -> float:
    """Fit a single temperature by minimizing NLL on a calibration set (LBFGS)."""
    import torch
    logits = torch.tensor(calib_logits, dtype=torch.float32)
    labels = torch.tensor(calib_labels, dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) > 0
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=n_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / torch.exp(log_T), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).item())


def fit_temperature_on_probs(calib_probs: np.ndarray, calib_labels: np.ndarray,
                             **kw) -> float:
    """Fit temperature using a probability distribution (e.g. the MC-dropout BMA) by
    treating its log as logits. Prevents double-softening when calibrating an
    already-averaged predictive distribution."""
    log_probs = np.log(np.clip(calib_probs, _EPS, 1.0))
    return fit_temperature(log_probs, calib_labels, **kw)


def predict_temperature(logits: np.ndarray, T: float):
    probs = softmax(logits / max(T, _EPS))
    conf = probs.max(axis=-1)
    return probs, conf, 1.0 - conf


# ---------------------------------------------------------------------------
# MC-dropout methods (consume a (S, N, K) stack)
# ---------------------------------------------------------------------------
def predict_mc_dropout(mc_logits: np.ndarray):
    """Mean predictive distribution; confidence = max mean prob; unc = pred entropy."""
    probs_s = softmax(mc_logits, axis=-1)          # (S, N, K)
    mean_probs = probs_s.mean(axis=0)              # (N, K)
    conf = mean_probs.max(axis=-1)
    unc = entropy(mean_probs)
    return mean_probs, conf, unc


def power_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Temperature-calibrate a (Bayesian-averaged) distribution by powering: keeps
    confidence on the probability scale while sharpening/softening per T."""
    if abs(T - 1.0) < 1e-6:
        return probs
    p = np.clip(probs, _EPS, 1.0) ** (1.0 / max(T, _EPS))
    return p / p.sum(axis=-1, keepdims=True)


def predict_medbayes_lite(mc_logits: np.ndarray, T: float = 1.0):
    """MedBayes-Lite (full, attention penalty lambda=0 here): MC-dropout Bayesian
    model average + temperature calibration. Confidence is the calibrated max
    probability (probability scale, valid for ECE/coverage); the predictive entropy
    of the BMA is the epistemic uncertainty used for ranking and abstention quality.

    The confidence-guided abstention gate is applied downstream uniformly via the
    selective-prediction thresholds, so it does not redefine the confidence here.
    """
    bma = softmax(mc_logits, axis=-1).mean(axis=0)   # Bayesian model average
    cal = power_temperature(bma, T)
    conf = cal.max(axis=-1)
    unc = entropy(bma)
    return cal, conf, unc


def predict_uncertainty_attention(mc_logits: np.ndarray,
                                  attn_logits: np.ndarray | None = None):
    """Ablation 'uncertainty-attention only': MC-dropout BMA computed from logits
    that were produced with the uncertainty-weighted attention penalty active. If
    attn_logits is None this reduces to MC-dropout (lambda=0), documented as such."""
    src = attn_logits if attn_logits is not None else mc_logits
    bma = softmax(src, axis=-1).mean(axis=0) if src.ndim == 3 else softmax(src)
    conf = bma.max(axis=-1)
    unc = entropy(bma)
    return bma, conf, unc


def predict_abstain_only(logits: np.ndarray):
    """Ablation 'confidence/abstention only': single deterministic pass; abstention
    is the downstream threshold policy (no MC, no attention, no calibration). The
    predictor equals the base MSP predictor by construction."""
    return predict_base(logits)


# ---------------------------------------------------------------------------
# deep ensemble (consume list of (N, K) logits from K separately trained models)
# ---------------------------------------------------------------------------
def predict_deep_ensemble(logits_list: List[np.ndarray]):
    probs = np.mean([softmax(l) for l in logits_list], axis=0)
    conf = probs.max(axis=-1)
    unc = entropy(probs)
    return probs, conf, unc


# ---------------------------------------------------------------------------
# split-conformal calibrated confidence (LAC)
# ---------------------------------------------------------------------------
def predict_conformal(logits: np.ndarray, calib_logits: np.ndarray,
                      calib_labels: np.ndarray):
    """Split-conformal (LAC) confidence: the conformal coverage level at which the
    predicted class would just leave the prediction set. Continuous, calibrated,
    and compatible with risk-coverage analysis.

    conf(x) = fraction of calibration nonconformity scores <= (1 - p_top(x)),
    where nonconformity = 1 - softmax(true class)."""
    calib_probs = softmax(calib_logits)
    s_calib = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_labels]
    s_calib = np.sort(s_calib)
    probs = softmax(logits)
    p_top = probs.max(axis=-1)
    # conformal p-value of the predicted label = P(s_calib >= test nonconformity),
    # where test nonconformity = 1 - p_top. High p_top -> high conformal confidence.
    n = len(s_calib)
    conf = (n - np.searchsorted(s_calib, 1.0 - p_top, side="left")) / n
    conf = np.clip(conf, 0.0, 1.0)
    return probs, conf, 1.0 - conf
