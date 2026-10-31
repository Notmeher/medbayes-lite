"""Central configuration: severity weights, thresholds, bucket taxonomy, and the
dataset/model registry. Values can be overridden by a YAML file (see configs/).

This module is the single source of truth for constants shared across the pipeline,
so buckets, metrics, plotting, and reporting stay consistent.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import os

try:
    import yaml  # optional; YAML override is convenience only
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ---------------------------------------------------------------------------
# Clinical bucket taxonomy (13 categories from the spec) and severity weights.
# Severity weights encode clinical harm potential of a confident error.
# ---------------------------------------------------------------------------
BUCKETS = [
    "medication_drug_interaction",
    "diagnosis_differential",
    "treatment_recommendation",
    "risk_prognosis",
    "lab_biomarker_interpretation",
    "rare_disease",
    "ambiguous_symptoms",
    "contradictory_evidence",
    "missing_evidence",
    "negation_heavy",
    "numeric_vital_lab_heavy",
    "high_severity",
    "low_severity",
]

# Severity weight per bucket. Buckets not listed default to 1.0 (low-risk factual).
SEVERITY_WEIGHTS = {
    "medication_drug_interaction": 3.0,
    "diagnosis_differential": 3.0,
    "treatment_recommendation": 3.0,
    "risk_prognosis": 2.5,
    "rare_disease": 2.5,
    "lab_biomarker_interpretation": 2.0,
    "numeric_vital_lab_heavy": 2.0,
    "ambiguous_symptoms": 2.0,
    "missing_evidence": 2.0,
    "contradictory_evidence": 2.0,
    "negation_heavy": 1.5,
    "high_severity": 3.0,
    "low_severity": 1.0,
}
DEFAULT_SEVERITY = 1.0

# A prediction is "high severity" if its max bucket severity >= this cutoff. Used
# for harmful-overconfident-error and abstention-recall definitions.
HIGH_SEVERITY_CUTOFF = 2.5

# Confidence thresholds for coverage / selective prediction / abstention.
COVERAGE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
# Thresholds at which overconfident / harmful-overconfident errors are defined.
OVERCONF_THRESHOLDS = [0.7, 0.8, 0.9]

ECE_BINS = 15
N_SEEDS = 5
BOOTSTRAP_N = 2000


@dataclass
class DatasetSpec:
    name: str
    loader: str            # key into data.loaders.LOADERS
    n_classes: int
    split: str = "test"
    max_samples: Optional[int] = None
    hf_id: Optional[str] = None
    hf_config: Optional[str] = None


@dataclass
class ModelSpec:
    name: str
    kind: str              # "encoder_seqcls" | "encoder_mcq" | "generative"
    hf_id: str
    # which datasets this checkpoint is valid for (task head must match)
    datasets: tuple = ()
    input_format: str = "qsep"   # how to build input text (see models.encoder)
    dropout: float = 0.3
    note: str = ""


# ---------------------------------------------------------------------------
# Registry of datasets actually available in this environment.
# MIMIC is declared but gated behind credentialed access (adapter stub).
# ---------------------------------------------------------------------------
DATASETS = {
    "pubmedqa": DatasetSpec("pubmedqa", "pubmedqa", n_classes=3, split="test",
                            hf_id="qiaojin/PubMedQA", hf_config="pqa_labeled"),
    "medmcqa": DatasetSpec("medmcqa", "medmcqa", n_classes=4, split="validation",
                           hf_id="openlifescienceai/medmcqa"),
    "medqa": DatasetSpec("medqa", "medqa", n_classes=4, split="test",
                         hf_id="GBaker/MedQA-USMLE-4-options"),
    "mediq": DatasetSpec("mediq", "mediq", n_classes=2, split="test",
                         hf_id="stellalisy/mediq"),
    "mimic": DatasetSpec("mimic", "mimic", n_classes=2, split="test"),  # adapter stub
}

# ---------------------------------------------------------------------------
# Registry of models. Encoder checkpoints are pre-fine-tuned so predictions are
# real (not random-head). BERT-base is fine-tuned locally as a general baseline.
# ---------------------------------------------------------------------------
MODELS = {
    # --- pretrained MedMCQA-fine-tuned 4-way multiple-choice (genuine, strong) ---
    "biobert_medmcqa": ModelSpec(
        "biobert_medmcqa", "encoder_mcq",
        "RUI525/BioBERT-finetune-MedMCQA-wo-context",
        datasets=("medmcqa", "medqa"), note="pretrained BertForMultipleChoice"),
    "pubmedbert_medmcqa": ModelSpec(
        "pubmedbert_medmcqa", "encoder_mcq",
        "RUI525/PubMedBERT-finetune-MedMCQA-wo-context",
        datasets=("medmcqa", "medqa"), note="pretrained BertForMultipleChoice"),
    # --- pretrained PubMedQA-fine-tuned 3-way classifier ---
    "pubmedbert_pubmedqa": ModelSpec(
        "pubmedbert_pubmedqa", "encoder_seqcls",
        "blizrys/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-pubmedqa",
        datasets=("pubmedqa",), note="pretrained 3-way classifier"),
    # --- locally fine-tuned encoder family (identical training routine) ---
    "bert_base": ModelSpec("bert_base", "encoder_finetune", "bert-base-uncased",
                           datasets=("pubmedqa", "medmcqa", "medqa", "mediq")),
    "biobert": ModelSpec("biobert", "encoder_finetune", "dmis-lab/biobert-base-cased-v1.1",
                         datasets=("pubmedqa", "medmcqa", "medqa", "mediq")),
    "clinicalbert": ModelSpec("clinicalbert", "encoder_finetune",
                              "emilyalsentzer/Bio_ClinicalBERT",
                              datasets=("pubmedqa", "medmcqa", "medqa", "mediq")),
    "pubmedbert": ModelSpec("pubmedbert", "encoder_finetune",
                            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                            datasets=("pubmedqa", "medmcqa", "medqa", "mediq")),
    # --- local generative model (self-consistency / semantic-entropy / verbalized) ---
    "qwen2_5_7b": ModelSpec(
        "qwen2_5_7b", "generative", "Qwen/Qwen2.5-7B-Instruct",
        datasets=("pubmedqa", "medmcqa", "medqa", "mediq")),
}

# backbone hub ids for the four locally fine-tuned encoder families
FINETUNE_BACKBONES = {
    "bert_base": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
}

# UQ methods available per model kind.
ENCODER_METHODS = ["base", "msp", "temp_scaling", "mc_dropout", "deep_ensemble",
                   "conformal", "medbayes_lite"]
GENERATIVE_METHODS = ["self_consistency", "semantic_entropy", "verbalized"]


@dataclass
class EvalConfig:
    """Top-level run configuration. Defaults reproduce the validated slice."""
    datasets: tuple = ("pubmedqa", "medmcqa")
    models: tuple = ("biobert_medmcqa", "clinicalbert_medmcqa", "pubmedbert_medmcqa")
    methods: tuple = ("base", "temp_scaling", "mc_dropout", "medbayes_lite")
    seeds: tuple = (0, 1, 2, 3, 4)
    mc_samples: int = 20
    dropout: float = 0.3
    lambda_attn: float = 1.0          # uncertainty-attention penalty strength
    tau: float = 0.7                  # default abstention threshold
    coverage_thresholds: tuple = tuple(COVERAGE_THRESHOLDS)
    overconf_thresholds: tuple = tuple(OVERCONF_THRESHOLDS)
    max_samples: Optional[int] = None
    device: str = "cuda"
    results_dir: str = RESULTS_DIR
    ensemble_size: int = 5

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        if not _HAS_YAML:
            raise RuntimeError("pyyaml not installed; cannot load YAML config")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {}
        for k, v in raw.items():
            if k in known:
                kwargs[k] = tuple(v) if isinstance(v, list) else v
        return cls(**kwargs)


def severity_for_buckets(buckets) -> float:
    """Max severity weight across a sample's assigned buckets."""
    if not buckets:
        return DEFAULT_SEVERITY
    return max(SEVERITY_WEIGHTS.get(b, DEFAULT_SEVERITY) for b in buckets)
