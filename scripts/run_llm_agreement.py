"""Agreement analysis for the LLM (and any clinician) rating sheets in revision/clinician_kit."""
import os
import sys

os.environ.setdefault("MEDBAYES_RESULTS_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ["MEDBAYES_RESULTS_DIR"])

from medbayes_eval.revision.clinician_kit import analyze_clinician_ratings
from medbayes_eval.revision.perturbation import analyze_perturbation_reviews
from medbayes_eval.revision.predictions import load_store

store, _ = load_store()
print("[agreement] store loaded", flush=True)
sev = analyze_clinician_ratings(store=store)
for k, v in sev.items():
    print(f"\n=== severity: {k} ===\n{v.round(3).to_string(index=False)}", flush=True)
pert = analyze_perturbation_reviews()
for k, v in (pert or {}).items():
    print(f"\n=== perturbation: {k} ===\n{v.round(3).to_string(index=False)}", flush=True)
print("[agreement] done", flush=True)
