"""Experiment config loading and env factories shared by train / evaluate / sweep."""

from __future__ import annotations

import copy
import os
from functools import partial
from typing import Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "experiment.yaml")


def load_experiment_config(path: Optional[str] = None, **overrides) -> dict:
    """Load an experiment YAML (defaults from ``configs/experiment.yaml``) and apply non-None overrides."""
    with open(DEFAULT_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    if path and os.path.abspath(path) != DEFAULT_CONFIG:
        with open(path, "r") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for key, value in overrides.items():
        if value is None:
            continue
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def run_dir(cfg: dict, relevance_name: str) -> str:
    out = cfg["out_dir"]
    if not os.path.isabs(out):
        out = os.path.join(REPO_ROOT, out)
        # Keep run outputs out of git without touching the repo's .gitignore.
        top = os.path.join(REPO_ROOT, os.path.normpath(cfg["out_dir"]).split(os.sep)[0])
        os.makedirs(top, exist_ok=True)
        ignore = os.path.join(top, ".gitignore")
        if not os.path.exists(ignore):
            with open(ignore, "w") as f:
                f.write("*\n")
    return os.path.join(out, cfg["task"], relevance_name, f"seed{cfg['seed']}")


def _build_gym_env(task, relevance, env_cfg, seed, verbose):
    from relevance.env import RelevanceGymEnv
    return RelevanceGymEnv(task=task, relevance=relevance, seed=seed, verbose=verbose, **env_cfg)


def env_factory(cfg: dict, relevance, rank: int, track: Optional[str] = None, seed_offset: int = 0):
    """Picklable zero-arg constructor (needed by SubprocVecEnv on Windows)."""
    env_cfg = copy.deepcopy(cfg["env"])
    if track is not None:
        env_cfg["track"] = track
    seed = int(cfg["seed"]) * 1000 + seed_offset + rank
    return partial(_build_gym_env, cfg["task"], relevance, env_cfg, seed, rank == 0)
