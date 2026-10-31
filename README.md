# MedBayes-Lite

**A Clinical Uncertainty Governance Layer for Risk-Aware Medical Decision Support**

MedBayes-Lite is a lightweight, inference-time governance layer for *pretrained*
clinical predictors. Rather than adding parameters or retraining, it converts
Monte Carlo dropout uncertainty into a calibrated, confidence-guided abstention
policy that defers low-confidence predictions for human review:

```
MC dropout  ->  calibration  ->  confidence-guided abstention  ->  human review
```

This repository contains the method and a **reproducible evaluation pipeline**
(`medbayes_eval/`) that assesses MedBayes-Lite not as a generic calibration trick
but as a *clinical uncertainty governance* framework: does it reduce **harmful
overconfident** clinical errors, abstain appropriately on high-risk and ambiguous
cases, improve risk-coverage trade-offs, and stay useful under medical domain
shift? Every number is produced from real models on real medical benchmarks;
nothing is hand-set, and negative results are reported alongside the positive ones.

---

## Key findings

- **Harmful overconfidence is driven toward zero.** On clinical multiple-choice QA,
  the governance layer substantially improves calibration and reduces confident,
  high-severity errors, with the **largest benefit under medical domain shift**,
  where in-distribution temperature scaling alone does not transfer as well.
- **The Clinical Uncertainty Score (CUS)** — a severity-weighted calibration
  summary — is empirically evaluated and tracks harmful overconfidence
  (r ~ 0.88) and adds variance beyond standard ECE (delta-R^2 ~ 0.19).
- **Honest negative results are kept.** The layer does not improve risk *ranking*
  (AURC is statistically unchanged; temperature scaling matches its calibration;
  deep ensembles rank risk better), gives no benefit on already-calibrated models,
  carries a coverage cost, and does not detect local numeric implausibility. The
  previously proposed ZTI is shown to be threshold-confounded and is demoted.

See [`EXPERIMENTAL_SUMMARY.md`](EXPERIMENTAL_SUMMARY.md) for the full results and
[`paper/`](paper/) for the manuscript and reviewer rebuttal.

---

## What the pipeline evaluates

**Datasets** (`medbayes_eval/data/loaders.py`)
- **PubMedQA** (3-way yes/no/maybe evidence QA)
- **MedMCQA** (4-way single-best-answer clinical exam QA)
- **MedQA-USMLE** (4-way; also the OOD target for MedMCQA-trained models)
- **MediQ** (clinical-note framing of MedQA)
- **MIMIC-III** mortality: credentialed-access adapter stub (no PHI in this repo)

**Models** (`medbayes_eval/config.py`)
- Pretrained, genuinely fine-tuned MCQ checkpoints (BioBERT, PubMedBERT on MedMCQA).
- A locally fine-tuned encoder family (BioBERT, Bio_ClinicalBERT, PubMedBERT,
  BERT-base) trained with one identical routine for clean comparison.
- A local open-weight generative model (Qwen2.5-7B-Instruct), sampling-only.

**UQ methods**: base/MSP, temperature scaling, MC-dropout, deep ensemble
(across fine-tuned seeds), split-conformal (LAC), and **MedBayes-Lite**
(MC-dropout BMA + calibration + confidence-guided abstention); for the generative
contrast: self-consistency, semantic entropy, verbalized confidence.

**Clinical risk categories** (`medbayes_eval/buckets/rules.py`): 13 rule-based
categories (medication/interaction, diagnosis, treatment, risk/prognosis, lab,
rare disease, ambiguous, contradictory, missing evidence, negation-heavy,
numeric-heavy, high/low severity) with clinical severity weights for harm-weighted
metrics.

**Metrics** (`medbayes_eval/metrics/core.py`): accuracy, ECE, NLL, Brier,
AUROC/AUPRC, AURC, E-AURC, coverage@{0.5..0.9}, selective accuracy,
overconfident-error, **harmful overconfident error**, abstention precision/recall,
plus the secondary CUS / ZTI summaries and their correlation with clinical-failure
metrics.

---

## Installation

```bash
git clone https://github.com/eliashossain001/medbayes-lite.git
cd medbayes-lite

conda create -n medbayes python=3.12 -y
conda activate medbayes
pip install -r requirements.txt
```

---

## Running the evaluation

```bash
# 1. per-sample predictions for all (dataset x model x method x seed)
python -m medbayes_eval.run run     --config configs/default.yaml

# 2. OOD domain transfer (MedMCQA-trained -> MedQA): in-distribution vs OOD drift
python -m medbayes_eval.run ood     --config configs/default.yaml

# 3. controlled clinical perturbations: confidence / abstention response
python -m medbayes_eval.run perturb --config configs/default.yaml

# 4. metrics tables, CUS/ZTI correlations, stats, plots, LaTeX/CSV tables
python -m medbayes_eval.run analyze --config configs/default.yaml

# or end to end:
python -m medbayes_eval.run all     --config configs/default.yaml

# quick smoke test (pretrained MCQ only, tiny eval):
python -m medbayes_eval.run all     --config configs/quick.yaml --max-eval 200
```

The generative slice loads a 7B model and is run separately:

```bash
python -m medbayes_eval.run_generative --config configs/default.yaml
```

Configuration (datasets, models, methods, seeds, `mc_samples`, `mc_p` inference
dropout rate, confidence thresholds, OOD pairs) lives in `configs/default.yaml`.
Raise `seeds` to 5 and set `max_eval: null` to scale the full sweep up.

---

## Repository layout

```
medbayes-lite/
├── medbayes_eval/        reproducible evaluation pipeline (config-driven OOP)
│   ├── data/             dataset loaders
│   ├── models/           encoders, fine-tuning, generative, UQ methods
│   ├── buckets/          clinical risk-category rules + severity weights
│   ├── metrics/          calibration / selective-prediction / CUS metrics
│   ├── perturb/          controlled clinical perturbations
│   ├── stats/            paired bootstrap significance tests
│   ├── plotting/         risk-coverage, reliability, abstention plots
│   ├── reporting/        booktabs LaTeX + CSV tables
│   ├── pipeline.py       orchestration
│   └── run.py            CLI: run / ood / perturb / analyze / all
├── configs/              default.yaml, quick.yaml
├── scripts/              fine-tuning, sensitivity sweeps, helper scripts
├── results/              raw predictions, metrics, stats, plots, tables
├── paper/                manuscript (main.tex) + reviewer rebuttal
├── EXPERIMENTAL_SUMMARY.md
├── requirements.txt
└── README.md
```

### Outputs (`results/`)

```
raw_predictions/     per-sample: probs, confidence, uncertainty, buckets,
                     severity, correctness
metrics/             metrics_overall.csv, metrics_by_bucket.csv,
                     ood_transfer.csv, perturbation_report.csv,
                     cus_zti_correlation.csv, cus_incremental_value.csv
statistical_tests/   paired bootstrap (base vs MedBayes-Lite)
plots/               risk-coverage, reliability, harmful-error bars,
                     abstention PR, OOD uncertainty, CUS correlation
tables/              main_results / bucket / ablation / ood in .csv and .tex
```

---

## Design notes

- Encoder "confidence" is the calibrated max predictive probability for **all**
  methods, so ECE / coverage are comparable; a method's distinctive signal lives
  in its `uncertainty` column and abstention behavior.
- MedBayes-Lite calibrates the **MC-averaged** distribution (not single-pass
  logits) to avoid double-softening.
- Conformal confidence is the LAC conformal p-value of the predicted label.
- Deep ensemble is emitted only when checkpoints genuinely differ (fine-tuned seeds).
- CUS and ZTI are reported as **secondary** summaries and explicitly evaluated by
  correlation against harmful overconfidence / AURC / selective accuracy. If they
  do not track clinical failure, the report says so.
- The optional uncertainty-weighted attention extension is **inactive (lambda=0)**
  in all reported experiments; no result depends on it.

---

## Citation

If you use this work, please cite the manuscript in [`paper/`](paper/):

```bibtex
@article{hossain_medbayeslite,
  title   = {MedBayes-Lite: A Clinical Uncertainty Governance Layer for
             Risk-Aware Medical Decision Support},
  author  = {Hossain, Elias and others},
  year    = {2026}
}
```

---

## Contact

**Elias Hossain** — PhD Student, University of Central Florida

[![GitHub](https://img.shields.io/badge/GitHub-eliashossain001-blue?logo=github)](https://github.com/eliashossain001)
