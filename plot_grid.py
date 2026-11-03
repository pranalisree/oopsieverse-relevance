"""
Heatmaps of a 2-object relevance grid (one cell per trained policy, averaged over seeds).

    python -m relevance.plot_grid --config relevance/configs/sweeps/grid_close_drawer.yaml

Reads runs/.../<cell>/seed*/eval_summary.json and writes grid_results.png (safe success, success,
timeout, damage) and grid_results.csv into the sweep's out_dir. Cells without results are left blank,
so this can be run while the sweep is still going.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import yaml

PANELS = [  # (metric, title, colormap, is_rate)
    ("safe_success", "Safe success rate", "Greens", True),
    ("success", "Task success rate", "Blues", True),
    ("timeout", "Timeout rate", "Oranges", True),
    ("total_damage", "Damage per episode (all objects)", "Reds", False),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="sweep YAML written by relevance.grid")
    p.add_argument("--out", default=None, help="output folder (default: the sweep's out_dir)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from relevance.common import REPO_ROOT, load_experiment_config
    from relevance.weights import CONFIG_DIR

    cfg = load_experiment_config(args.config)
    task = cfg["sweep"]["tasks"][0]
    out_root = cfg["out_dir"] if os.path.isabs(cfg["out_dir"]) else os.path.join(REPO_ROOT, cfg["out_dir"])
    out = args.out or out_root
    os.makedirs(out, exist_ok=True)

    cells = {}
    labels = None
    for name in cfg["sweep"]["relevance"]:
        with open(os.path.join(CONFIG_DIR, task, f"{name}.yaml")) as f:
            grid = yaml.safe_load(f)["grid"]
        labels = grid["labels"]
        runs = []
        for path in glob.glob(os.path.join(out_root, task, name, "seed*", "eval_summary.json")):
            with open(path) as f:
                s = json.load(f)
            s["total_damage"] = s["in_scope_damage"] + s["out_of_scope_damage"]
            runs.append(s)
        cells[tuple(grid["levels"])] = (name, runs)

    rows_lv = sorted({lv[0] for lv in cells})
    cols_lv = sorted({lv[1] for lv in cells})

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 4.8))
    for ax, (metric, title, cmap, is_rate) in zip(axes, PANELS):
        grid = np.full((len(rows_lv), len(cols_lv)), np.nan)
        for (a, b), (_, runs) in cells.items():
            if runs:
                grid[rows_lv.index(a), cols_lv.index(b)] = np.mean([r[metric] for r in runs])
        vmax = 1.0 if is_rate else max(1.0, np.nanmax(grid) if np.isfinite(grid).any() else 1.0)
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.0, vmax=vmax)
        for i in range(len(rows_lv)):
            for j in range(len(cols_lv)):
                v = grid[i, j]
                text = "–" if np.isnan(v) else (f"{100 * v:.0f}%" if is_rate else f"{v:.1f}")
                dark = not np.isnan(v) and v > 0.6 * vmax
                ax.text(j, i, text, ha="center", va="center", fontsize=11,
                        color="white" if dark else "#1f1f1f")
        ax.set_xticks(range(len(cols_lv)), [f"{v:g}" for v in cols_lv])
        ax.set_yticks(range(len(rows_lv)), [f"{v:g}" for v in rows_lv])
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(labels[1], fontsize=12, weight="bold")
        ax.set_ylabel(labels[0], fontsize=12, weight="bold")
        ax.set_title(title, fontsize=12, pad=48)
    n_seeds = max((len(r) for _, r in cells.values()), default=0)
    fig.suptitle(f"{task}: relevance grid ({labels[0]} x {labels[1]}), up to {n_seeds} seed(s) per cell",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    png = os.path.join(out, "grid_results.png")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out, "grid_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([labels[0], labels[1], "config", "n_seeds"] + [m for m, *_ in PANELS])
        for (a, b), (name, runs) in sorted(cells.items()):
            vals = [round(float(np.mean([r[m] for r in runs])), 4) if runs else "" for m, *_ in PANELS]
            w.writerow([a, b, name, len(runs)] + vals)

    done = sum(1 for _, r in cells.values() if r)
    print(f"[plot_grid] {done}/{len(cells)} cells have results")
    print(f"[plot_grid] wrote {png}\n[plot_grid] wrote {csv_path}")


if __name__ == "__main__":
    main()
