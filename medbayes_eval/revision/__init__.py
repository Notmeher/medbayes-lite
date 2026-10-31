"""Revision (reviewer-response) experiments.

GPU stage  : jobs.py writes logit caches (deterministic + seeded MC-dropout stacks).
CPU stage  : analyses.py recomputes every method, gate, and statistic from the caches.
Orchestration for Kaggle T4 x2 lives in multigpu.py; reproduce.py is the CLI.
"""
