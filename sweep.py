"""
Run a relevance sweep: every (task x relevance config x seed) -> train + evaluate,
then aggregate into summary.csv.

    python -m relevance.sweep --config relevance/configs/sweeps/proposal.yaml --dry-run
    python -m relevance.sweep --config relevance/configs/sweeps/proposal.yaml
    python -m relevance.sweep --config ... --only task=dishes_to_sink,relevance=s2
    python -m relevance.sweep --config ... --aggregate-only

Runs that already have eval_summary.json are skipped, so a sweep can be resumed
or split across machines (each job with its own --only filter). --dry-run prints
one command per run, e.g. for a SLURM job array on TACC.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import subprocess
import sys

import numpy as np

METRICS = ["safe_success", "success", "timeout", "zero_damage", "min_health", "mean_health",
           "weighted_health", "num_damaged_objects", "in_scope_damage", "out_of_scope_damage", "length"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="sweep YAML (experiment config + a 'sweep' block)")
    p.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    p.add_argument("--only", default=None, help="filter, e.g. task=dishes_to_sink,relevance=s2,seed=0")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args(argv)


def expand(cfg: dict, only: str | None):
    sweep = cfg["sweep"]
    filters = dict(kv.split("=", 1) for kv in only.split(",")) if only else {}
    for task, rel, seed in itertools.product(sweep["tasks"], sweep["relevance"], sweep["seeds"]):
        run = {"task": task, "relevance": rel, "seed": seed}
        if all(str(run[k]) == v for k, v in filters.items()):
            yield run


def main(argv=None):
    args = parse_args(argv)
    from relevance.common import load_experiment_config, run_dir
    from relevance.weights import load_relevance_config

    cfg = load_experiment_config(args.config)
    py = sys.executable
    runs = list(expand(cfg, args.only))

    if not args.aggregate_only:
        for i, run in enumerate(runs):
            rel_name = load_relevance_config(run["relevance"], task=run["task"]).name
            out = run_dir({**cfg, **run}, rel_name)
            train_cmd = [py, "-m", "relevance.train", "--config", args.config, "--task", run["task"],
                         "--relevance", str(run["relevance"]), "--seed", str(run["seed"])]
            eval_cmd = [py, "-m", "relevance.evaluate", "--run", out]
            tag = f"[{i + 1}/{len(runs)}] {run['task']} / {rel_name} / seed {run['seed']}"
            if args.dry_run:
                print(f"{tag}\n  {' '.join(train_cmd)}\n  {' '.join(eval_cmd)}")
                continue
            if os.path.exists(os.path.join(out, "eval_summary.json")):
                print(f"{tag}: done, skipping")
                continue
            print(f"{tag}: training")
            if not os.path.exists(os.path.join(out, "model.zip")):
                subprocess.run(train_cmd, check=True)
            if not args.skip_eval:
                print(f"{tag}: evaluating")
                subprocess.run(eval_cmd, check=True)
        if args.dry_run:
            return

    aggregate(cfg)


def aggregate(cfg: dict):
    from relevance.common import REPO_ROOT
    out_root = cfg["out_dir"] if os.path.isabs(cfg["out_dir"]) else os.path.join(REPO_ROOT, cfg["out_dir"])
    summaries = []
    for path in glob.glob(os.path.join(out_root, "*", "*", "seed*", "eval_summary.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    if not summaries:
        print(f"[sweep] no eval_summary.json under {out_root}")
        return

    groups = {}
    for s in summaries:
        groups.setdefault((s["task"], s["relevance"]), []).append(s)

    rows = []
    for (task, rel), group in sorted(groups.items()):
        row = {"task": task, "relevance": rel, "penalty_weight": group[0]["penalty_weight"],
               "n_seeds": len(group), "episodes": sum(g["episodes"] for g in group)}
        for m in METRICS:
            vals = np.array([g[m] for g in group], dtype=float)
            row[f"{m}_mean"] = round(float(vals.mean()), 4)
            row[f"{m}_std"] = round(float(vals.std()), 4)
        rows.append(row)

    path = os.path.join(out_root, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'task':22s} {'relevance':16s} {'seeds':>5s} {'safe':>12s} {'success':>12s} {'timeout':>12s}")
    for r in rows:
        print(f"{r['task']:22s} {r['relevance']:16s} {r['n_seeds']:5d} "
              f"{r['safe_success_mean']:6.2f}+-{r['safe_success_std']:<4.2f} "
              f"{r['success_mean']:6.2f}+-{r['success_std']:<4.2f} {r['timeout_mean']:6.2f}+-{r['timeout_std']:<4.2f}")
    print(f"\n[sweep] wrote {path}")


if __name__ == "__main__":
    main()
