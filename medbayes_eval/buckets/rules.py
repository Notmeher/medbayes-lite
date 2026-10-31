"""Rule-based assignment of evaluation samples to the 13 clinical buckets.

Every :class:`~medbayes_eval.schema.Sample` is mapped to one or more clinical
buckets defined in :data:`medbayes_eval.config.BUCKETS` using regex / keyword
heuristics over its ``display_text()`` (lowercased). The buckets stratify the
benchmark by clinical task type, evidence structure, and harm potential so that
calibration / abstention metrics can be reported per-bucket and severity-weighted.

The lexicons here are intentionally rich (drug-name suffixes, lab analytes, vital
units, negation cues, etc.) but transparent: ``BUCKET_PATTERNS`` exposes the exact
compiled patterns / keyword lists for inspection and unit testing.

Design notes
------------
* Patterns are precompiled once at import time for speed over large corpora.
* ``low_severity`` is a strict fallback: it is assigned only when no other
  ("high-risk") bucket matched, guaranteeing every sample carries >= 1 bucket.
* ``negation_heavy`` and ``numeric_vital_lab_heavy`` are count-thresholded rather
  than presence-based, so a single stray token does not trigger them.
"""
from __future__ import annotations

import json
import re
from typing import Pattern, Union

import pandas as pd

from .. import config
from ..schema import Sample

# A bucket pattern is either a single compiled regex or a list of keyword regexes.
BucketPattern = Union[Pattern[str], list]


def _alt(keywords: list[str]) -> Pattern[str]:
    """Compile an alternation of literal keyword/substring cues.

    Substrings are matched anywhere (so ``"diagnos"`` catches *diagnosis*,
    *diagnose*, *diagnostic*). Special regex characters in the cues are escaped.
    """
    return re.compile("|".join(re.escape(k) for k in keywords))


# ---------------------------------------------------------------------------
# Lexicons. Kept as module constants so tests can reference them directly.
# ---------------------------------------------------------------------------

# Common drug-name morphological suffixes (word endings) by class.
_DRUG_SUFFIXES = [
    "pril",     # ACE inhibitors (lisinopril)
    "olol",     # beta-blockers (metoprolol)
    "statin",   # statins (atorvastatin)
    "mab",      # monoclonal antibodies (rituximab)
    "cillin",   # penicillins (amoxicillin)
    "azole",    # antifungals / PPIs adjacent (fluconazole)
    "pine",     # dihydropyridines (amlodipine)
    "sartan",   # ARBs (losartan)
    "prazole",  # proton-pump inhibitors (omeprazole)
]
# A drug suffix only counts at a word boundary end, e.g. "...statin\b".
_DRUG_SUFFIX_RE = re.compile(r"\b\w*(?:" + "|".join(_DRUG_SUFFIXES) + r")\b")

# dose amounts like "500 mg" / "10mcg" but NOT lab concentrations like "mg/dl"
_DOSE_RE = re.compile(r"\d\s?(?:mg|mcg|µg|units?)\b(?!\s*/)", re.IGNORECASE)
_MEDICATION_KEYWORDS = [
    "dose", "dosage", "administer", "prescrib", "contraindicat",
    "interaction", "drug", "medication", "antibiotic", "anticoagulant", "regimen",
]

_DIAGNOSIS_KEYWORDS = [
    "diagnos", "differential", "most likely", "consistent with", "suggestive of",
    "presentation of", "what is the cause",
]

_TREATMENT_KEYWORDS = [
    "treat", "therapy", "manage", "management", "first-line",
    "next step in management", "recommend", "intervention", "surgery", "regimen",
]

_RISK_KEYWORDS = [
    "prognos", "risk of", "mortality", "survival", "likelihood", "recurrence",
    "complication", "outcome", "life expectancy",
]

_LAB_KEYWORDS = [
    "lab", "serum", "level", "wbc", "hemoglobin", "hematocrit", "creatinine",
    "sodium", "potassium", "glucose", "biomarker", "titer", "culture", "panel",
    "count", "elevated", "mg/dl", "mmol",
]

_RARE_KEYWORDS = [
    "rare", "uncommon", "atypical", "syndrome", "congenital", "hereditary",
    "genetic disorder", "orphan",
]

_AMBIGUOUS_KEYWORDS = [
    "nonspecific", "vague", "unclear", "could be", "may be", "possible",
    "atypical presentation", "intermittent", "unremarkable",
]

_CONTRADICTORY_KEYWORDS = [
    "however", "but", "despite", "although", "whereas", "contradict",
    "inconsistent", "conflicting", "on the other hand", "paradox",
]

_MISSING_KEYWORDS = [
    "not available", "unknown", "unspecified", "no history", "cannot determine",
    "insufficient", "not reported", "none documented", "not mentioned", "missing",
]

_HIGH_SEVERITY_KEYWORDS = [
    "emergenc", "acute", "severe", "critical", "icu", "resuscitat",
    "life-threatening", "unstable", "hemorrhage", "sepsis",
    "cardiac arrest", "septic shock", "code blue",
]
# "stat" (the STAT order) only as a whole word, never inside statin/prostate/etc.
_HIGH_SEVERITY_EXTRA_RE = re.compile(r"\bstat\b", re.IGNORECASE)

# Negation cues counted for the negation_heavy bucket (whole-word).
_NEGATION_RE = re.compile(
    r"\b(no|not|without|denies|denied|absent|negative|never|none|non-)\b"
)
_NEGATION_THRESHOLD = 3

# Numeric tokens and vital/lab unit patterns counted for numeric_vital_lab_heavy.
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_UNIT_RE = re.compile(
    r"mmhg|bpm|mg/dl|mmol|°c|°f|/min|\bkg\b|beats", re.IGNORECASE
)
_NUMERIC_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Public, inspectable pattern table. Count-based buckets carry None here because
# their logic is thresholded (see _NEGATION_RE / _NUMBER_RE), not a simple match.
# ---------------------------------------------------------------------------
BUCKET_PATTERNS: dict[str, BucketPattern] = {
    "medication_drug_interaction": [_DRUG_SUFFIX_RE, _DOSE_RE, _alt(_MEDICATION_KEYWORDS)],
    "diagnosis_differential": _alt(_DIAGNOSIS_KEYWORDS),
    "treatment_recommendation": _alt(_TREATMENT_KEYWORDS),
    "risk_prognosis": _alt(_RISK_KEYWORDS),
    "lab_biomarker_interpretation": _alt(_LAB_KEYWORDS),
    "rare_disease": _alt(_RARE_KEYWORDS),
    "ambiguous_symptoms": _alt(_AMBIGUOUS_KEYWORDS),
    "contradictory_evidence": _alt(_CONTRADICTORY_KEYWORDS),
    "missing_evidence": _alt(_MISSING_KEYWORDS),
    "negation_heavy": _NEGATION_RE,            # thresholded on match count
    "numeric_vital_lab_heavy": _UNIT_RE,       # thresholded on number+unit count
    "high_severity": [_alt(_HIGH_SEVERITY_KEYWORDS), _HIGH_SEVERITY_EXTRA_RE],
    # low_severity is a computed fallback, not a pattern.
}


def _matches(pattern: BucketPattern, text: str) -> bool:
    """True if any of the bucket's compiled pattern(s) fire on ``text``."""
    if isinstance(pattern, list):
        return any(p.search(text) is not None for p in pattern)
    return pattern.search(text) is not None


# ---------------------------------------------------------------------------
# Strict (word-boundary) variant, used only as a sensitivity analysis for the
# substring false positives of the default rules (e.g. "icu" in "ventricular",
# "but" in "distribution", "lab" in "label").
# ---------------------------------------------------------------------------
def _alt_strict(keywords: list[str]) -> Pattern[str]:
    parts = []
    for k in keywords:
        e = re.escape(k)
        # short cues must be whole words; longer stems keep their suffix freedom
        parts.append(rf"\b{e}\b" if len(k) <= 4 else rf"\b{e}")
    return re.compile("|".join(parts))


STRICT_BUCKET_PATTERNS: dict[str, BucketPattern] = {
    "medication_drug_interaction": [_DRUG_SUFFIX_RE, _DOSE_RE,
                                    _alt_strict(_MEDICATION_KEYWORDS)],
    "diagnosis_differential": _alt_strict(_DIAGNOSIS_KEYWORDS),
    "treatment_recommendation": _alt_strict(_TREATMENT_KEYWORDS),
    "risk_prognosis": _alt_strict(_RISK_KEYWORDS),
    "lab_biomarker_interpretation": _alt_strict(_LAB_KEYWORDS),
    "rare_disease": _alt_strict(_RARE_KEYWORDS),
    "ambiguous_symptoms": _alt_strict(_AMBIGUOUS_KEYWORDS),
    "contradictory_evidence": _alt_strict(_CONTRADICTORY_KEYWORDS),
    "missing_evidence": _alt_strict(_MISSING_KEYWORDS),
    "negation_heavy": _NEGATION_RE,
    "numeric_vital_lab_heavy": _UNIT_RE,
    "high_severity": [_alt_strict(_HIGH_SEVERITY_KEYWORDS), _HIGH_SEVERITY_EXTRA_RE],
}


def explain_buckets(sample: Sample, strict: bool = False) -> dict[str, list[str]]:
    """bucket -> distinct matched cue strings (count buckets report their count)."""
    text = sample.display_text().lower()
    table = STRICT_BUCKET_PATTERNS if strict else BUCKET_PATTERNS
    out: dict[str, list[str]] = {}
    for bucket in assign_buckets(sample, strict=strict):
        if bucket == "low_severity":
            out[bucket] = []
        elif bucket == "negation_heavy":
            out[bucket] = [f"{_negation_count(text)} negation cues"]
        elif bucket == "numeric_vital_lab_heavy":
            out[bucket] = [f"{_numeric_unit_count(text)} numeric/unit tokens"]
        else:
            pats = table[bucket] if isinstance(table[bucket], list) else [table[bucket]]
            hits = []
            for p in pats:
                hits += [m.group(0) for m in p.finditer(text)]
            out[bucket] = sorted(set(hits))
    return out


def _negation_count(text: str) -> int:
    """Number of whole-word negation cues in ``text``."""
    return len(_NEGATION_RE.findall(text))


def _numeric_unit_count(text: str) -> int:
    """Combined count of numeric tokens and vital/lab unit patterns."""
    return len(_NUMBER_RE.findall(text)) + len(_UNIT_RE.findall(text))


def assign_buckets(sample: Sample, strict: bool = False) -> list[str]:
    """Assign ``sample`` to one or more clinical buckets.

    The match is performed over the lowercased ``display_text()`` (question +
    context + options). Returns buckets in the canonical order of
    :data:`config.BUCKETS`. Always returns at least one bucket; if no high-risk
    bucket fires, ``low_severity`` is returned as the fallback.
    """
    text = sample.display_text().lower()
    table = STRICT_BUCKET_PATTERNS if strict else BUCKET_PATTERNS
    assigned: list[str] = []

    for bucket in config.BUCKETS:
        if bucket == "low_severity":
            continue  # handled as fallback below
        if bucket == "negation_heavy":
            if _negation_count(text) >= _NEGATION_THRESHOLD:
                assigned.append(bucket)
            continue
        if bucket == "numeric_vital_lab_heavy":
            if _numeric_unit_count(text) >= _NUMERIC_THRESHOLD:
                assigned.append(bucket)
            continue
        if _matches(table[bucket], text):
            assigned.append(bucket)

    if not assigned:
        assigned.append("low_severity")
    return assigned


def sample_severity(buckets: list[str]) -> float:
    """Max severity weight across a sample's assigned buckets (see config)."""
    return config.severity_for_buckets(buckets)


def bucket_dataframe(samples: list[Sample]) -> pd.DataFrame:
    """Build a tidy DataFrame of bucket assignments for a list of samples.

    Columns: ``uid``, ``buckets`` (JSON string), ``severity`` (float).
    """
    rows = []
    for s in samples:
        buckets = assign_buckets(s)
        rows.append(
            {
                "uid": s.uid,
                "buckets": json.dumps(buckets),
                "severity": sample_severity(buckets),
            }
        )
    return pd.DataFrame(rows, columns=["uid", "buckets", "severity"])


if __name__ == "__main__":
    # Hand-written synthetic samples spanning the main bucket families.
    demo_samples = [
        Sample(
            uid="demo-drug",
            dataset="demo",
            question=(
                "A patient on warfarin is prescribed amoxicillin. What is the "
                "appropriate dose adjustment, and is there a drug interaction or "
                "contraindication for this anticoagulant regimen?"
            ),
            label=0,
            n_classes=4,
            options=["Increase dose", "Decrease dose", "No change", "Stop drug"],
        ),
        Sample(
            uid="demo-ddx",
            dataset="demo",
            question=(
                "A 55-year-old presents with chest pain. What is the most likely "
                "diagnosis given a presentation consistent with acute coronary "
                "syndrome?"
            ),
            label=1,
            n_classes=4,
            options=["MI", "GERD", "Costochondritis", "Anxiety"],
        ),
        Sample(
            uid="demo-negation",
            dataset="demo",
            question=(
                "The patient denies fever, has no cough, no chest pain, and is "
                "not short of breath. There is no history of smoking and never "
                "any cardiac symptoms. Exam is unremarkable."
            ),
            label=0,
            n_classes=2,
        ),
        Sample(
            uid="demo-lab",
            dataset="demo",
            question=(
                "Serum sodium is 128 mmol/L, potassium 5.6, creatinine 2.1 mg/dl, "
                "and WBC count is elevated at 14.2. Blood pressure 90/60 mmHg, "
                "heart rate 110 bpm. Interpret the lab panel."
            ),
            label=1,
            n_classes=3,
        ),
        Sample(
            uid="demo-factual",
            dataset="demo",
            question="Which vitamin is primarily synthesized in the skin via sunlight?",
            label=2,
            n_classes=4,
            options=["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin K"],
        ),
    ]

    for sample in demo_samples:
        buckets = assign_buckets(sample)
        sev = sample_severity(buckets)
        print(f"{sample.uid:14s} severity={sev:>4.1f}  buckets={buckets}")

    print("\nbucket_dataframe:")
    print(bucket_dataframe(demo_samples).to_string(index=False))
