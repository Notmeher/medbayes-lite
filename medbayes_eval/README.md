# MedBayes-Lite: Clinical Uncertainty Governance Evaluation

A reproducible evaluation pipeline that assesses **MedBayes-Lite** not as a generic
calibration trick, but as a **clinical uncertainty governance framework**: does it
reduce *harmful* overconfident clinical errors, abstain appropriately on high-risk /
ambiguous cases, improve risk-coverage trade-offs, stay useful under medical domain
shift, and are its summary scores (CUS, ZTI) actually meaningful?

This package (`medbayes_eval/`) replaces the original ad-hoc scripts with a clean,
config-driven, OOP pipeline. Every number is produced from real models on real
medical data; nothing is hand-set.

## What it evaluates

**Datasets** (`medbayes_eval/data/loaders.py`)
- **PubMedQA** (3-way yes/no/maybe, evidence QA)
- **MedMCQA** (4-way single-best-answer clinical exam QA)
- **MedQA-USMLE** (4-way; also the OOD target for MedMCQA-trained models)
- **MediQ** (clinical atomic-facts reformatting of MedQA; clinical-note framing)
- **MIMIC-III** mortality: credentialed-access adapter stub (no PHI in this repo)

**Models** (`medbayes_eval/config.py :: MODELS`)
- Pretrained, genuinely fine-tuned MCQ checkpoints (BioBERT, PubMedBERT on MedMCQA)
  for strong, confident predictions where overconfidence analysis is meaningful.
- A locally fine-tuned encoder family (BioBERT, Bio_ClinicalBERT, PubMedBERT,
  BERT-base) with one identical training routine for clean comparison.
- A local generative model (Qwen2.5-7B-Instruct) with API-compatible UQ.

**UQ methods**: base/MSP, temperature scaling, MC-dropout, deep ensemble (across
fine-tuned seeds), split-conformal, **MedBayes-Lite** (MC-dropout BMA + calibration +
confidence-guided abstention); generative self-consistency, semantic entropy,
verbalized confidence.

**Clinical buckets** (`medbayes_eval/buckets/rules.py`): 13 rule-based categories
(medication/interaction, diagnosis, treatment, risk/prognosis, lab, rare disease,
ambiguous, contradictory, missing evidence, negation-heavy, numeric-heavy, high/low
severity) with clinical severity weights for harm-weighted metrics.

**Metrics** (`medbayes_eval/metrics/core.py`): accuracy, ECE, NLL, Brier, AUROC/AUPRC,
AURC, E-AURC, coverage@{0.5..0.9}, selective accuracy, overconfident-error,
**harmful overconfident error**, abstention precision/recall, plus the secondary
CUS / ZTI summaries and their correlation with the clinical failure metrics.

## Install

```bash
pip install -r requirements.txt   # torch, transformers, datasets, scikit-learn, scipy, matplotlib, seaborn, pyyaml
```

## Run

```bash
# 1. produce per-sample predictions for all (dataset x model x method x seed)
python -m medbayes_eval.run run     --config configs/default.yaml

# 2. OOD domain transfer (MedMCQA-trained -> MedQA): ID vs OOD drift
python -m medbayes_eval.run ood     --config configs/default.yaml

# 3. controlled clinical perturbations: confidence / abstention response
python -m medbayes_eval.run perturb --config configs/default.yaml

# 4. metrics tables, CUS/ZTI correlations, stats, plots, LaTeX/CSV tables
python -m medbayes_eval.run analyze --config configs/default.yaml

# or everything end to end:
python -m medbayes_eval.run all     --config configs/default.yaml
# quick smoke (pretrained MCQ only, tiny eval):
python -m medbayes_eval.run all     --config configs/quick.yaml --max-eval 200
```

The generative slice is run separately (loads a 7B model):
```bash
python -m medbayes_eval.run_generative --config configs/default.yaml   # see module
```

## Outputs (`results/`)

```
results/
  raw_predictions/     per-sample parquet: probs, confidence, uncertainty,
                       buckets, severity, correctness, all flags derivable
  bucket_assignments/  per-sample bucket labels + severity + frequency
  metrics/             metrics_overall.csv, metrics_by_bucket.csv,
                       metrics_per_seed.csv, metrics_seed_aggregated.csv,
                       ood_transfer.csv, perturbation_report.csv,
                       cus_zti_correlation.csv
  statistical_tests/   paired bootstrap (base vs MedBayes-Lite), seeds
  plots/               risk-coverage, reliability, bucket harmful bars,
                       abstention PR, confidence histograms, OOD uncertainty,
                       severity-weighted error, CUS/ZTI correlation
  tables/              main_results / bucket / ablation / ood / correlation
                       in both .csv and booktabs .tex
```

## Configuration

Edit `configs/default.yaml`: datasets, models, methods, seeds, `mc_samples`,
`mc_p` (inference dropout rate), confidence thresholds, and OOD pairs. To scale the
full sweep up, raise `seeds` to 5 and set `max_eval: null`.

## Design notes / honesty

- Encoder "confidence" is the calibrated max predictive probability for **all**
  methods, so ECE / coverage are comparable; a method's distinctive signal lives in
  the `uncertainty` column and its abstention behavior.
- MedBayes-Lite calibrates the **MC-averaged** distribution (not single-pass logits)
  to avoid double-softening.
- Conformal confidence is the LAC conformal p-value of the predicted label.
- Deep ensemble is only emitted when checkpoints genuinely differ (fine-tuned seeds).
- CUS and ZTI are reported as **secondary** summaries and explicitly validated by
  correlation against harmful-overconfidence / AURC / selective accuracy. If they do
  not track clinical failure, the summary says so.
