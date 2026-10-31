"""Run revision GPU jobs on every visible GPU (Kaggle "GPU T4 x2").

One worker subprocess per GPU (CUDA_VISIBLE_DEVICES=<i>) pulls jobs from a shared
file queue: a job is claimed by atomically creating `<id>.lock`, finished by writing
`<id>.done`, and failures write `<id>.failed` (other jobs keep running). Jobs whose
cache output already exists finish instantly, so re-running a cell resumes.

Notebook:
    from medbayes_eval.revision.multigpu import run_parallel
    run_parallel(jobs, n_gpus=2)
Worker (internal):
    python -m medbayes_eval.revision.multigpu --worker <queue_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from typing import List

from .. import config
from . import paths


def _qpath(qdir, job_id, ext):
    return os.path.join(qdir, f"{job_id}.{ext}")


def _stream(proc, prefix, log_path):
    with open(log_path, "a") as log:
        for line in proc.stdout:
            print(f"{prefix} {line}", end="", flush=True)
            log.write(line)
            log.flush()


def run_parallel(jobs: List[dict], n_gpus: int | None = None, queue_name: str = "main",
                 gpu_groups: List[str] | None = None, env_extra: dict | None = None) -> dict:
    """Execute `jobs` (dicts with id/kind/args/cost) across GPUs; return status.

    `gpu_groups` overrides the default one-worker-per-GPU split, e.g. ["0,1"] runs a
    single worker that sees both GPUs (used for the 7B generative model)."""
    import torch
    n_gpus = n_gpus if n_gpus is not None else torch.cuda.device_count()
    if gpu_groups is None:
        gpu_groups = [str(g) for g in range(n_gpus)]
    qdir = os.path.join(paths.QUEUE, queue_name)
    os.makedirs(qdir, exist_ok=True)
    # stale locks from a crashed session would block their jobs forever
    for f in os.listdir(qdir):
        if f.endswith((".lock", ".failed")):
            os.remove(os.path.join(qdir, f))
    with open(os.path.join(qdir, "jobs.json"), "w") as f:
        json.dump(sorted(jobs, key=lambda j: -j.get("cost", 1)), f, indent=1)

    env = {**os.environ, **(env_extra or {}),
           "PYTHONPATH": config.REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    if not gpu_groups or not torch.cuda.is_available():
        _worker(qdir)  # CPU smoke tests run in-process
    else:
        procs, threads = [], []
        for g, devices in enumerate(gpu_groups):
            p = subprocess.Popen(
                [sys.executable, "-m", "medbayes_eval.revision.multigpu", "--worker", qdir],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env={**env, "CUDA_VISIBLE_DEVICES": devices}, cwd=config.REPO_ROOT)
            tag = f"[gpu{devices.replace(',', '+')}]"
            t = threading.Thread(target=_stream, daemon=True,
                                 args=(p, tag, os.path.join(paths.LOGS, f"{queue_name}_w{g}.log")))
            t.start()
            procs.append(p)
            threads.append(t)
        for p in procs:
            p.wait()
        for t in threads:
            t.join(timeout=5)
    status = {"done": [], "failed": {}, "missing": []}
    for j in jobs:
        if os.path.exists(_qpath(qdir, j["id"], "done")):
            status["done"].append(j["id"])
        elif os.path.exists(_qpath(qdir, j["id"], "failed")):
            status["failed"][j["id"]] = open(_qpath(qdir, j["id"], "failed")).read()[-2000:]
        else:
            status["missing"].append(j["id"])
    print(f"[multigpu:{queue_name}] done={len(status['done'])} "
          f"failed={len(status['failed'])} missing={len(status['missing'])}")
    for jid, tb in status["failed"].items():
        print(f"--- FAILED {jid} ---\n{tb}")
    return status


def _worker(qdir: str):
    from .jobs import run_job
    jobs = json.load(open(os.path.join(qdir, "jobs.json")))
    for job in jobs:
        jid = job["id"]
        if os.path.exists(_qpath(qdir, jid, "done")):
            continue
        try:
            fd = os.open(_qpath(qdir, jid, "lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        print(f"[job] start {jid}", flush=True)
        try:
            out = run_job(job)
            with open(_qpath(qdir, jid, "done"), "w") as f:
                json.dump({"output": out, "seconds": time.time() - t0}, f, default=str)
            print(f"[job] done  {jid} ({time.time() - t0:.0f}s)", flush=True)
        except Exception:
            tb = traceback.format_exc()
            with open(_qpath(qdir, jid, "failed"), "w") as f:
                f.write(tb)
            print(f"[job] FAILED {jid}\n{tb}", flush=True)
        finally:
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    _worker(ap.parse_args().worker)
