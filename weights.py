"""
Relevance configs: which objects the damage penalty cares about, and how much.

A relevance YAML looks like::

    task: dishes_to_sink
    penalty_weight: 0.1        # w in r = R_task - w * sum_e rho_e * d_e
    default_relevance: 0.0     # rho for every tracked object not matched below
    relevance:
      robot: 1.0               # any robot
      bowl: 1.0                # exact object name
      ref:sink: 0.75           # fixture the task registered as ``env.sink``
      type:counter: 0.25       # every object whose class is Damageable<Counter>
      contains:cab: 0.5        # every object whose name contains "cab"
    nearby:                    # optional, evaluated at every reset
      - of: [bowl, cup, plate, ref:sink]
        radius: 0.30           # metres, between geom bounding boxes
        relevance: 1.0
      # Size-matched control: count the objects the rule above would add, then
      # give that many objects at least ``min_distance`` away the relevance instead.
      - of: [bowl, cup, plate, ref:sink]
        radius: 0.30
        select: far_size_matched
        min_distance: 1.5
        relevance: 1.0

When several keys match one object the most specific wins:
name > ref: > robot > type: > contains: > nearby > default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml


CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "relevance")

# Lower number = more specific.
_PRIORITY = {"name": 0, "ref": 1, "robot": 2, "type": 3, "contains": 4, "nearby": 5, "default": 6}


@dataclass
class NearbyRule:
    of: List[str]
    radius: float
    relevance: float
    select: str = "near"          # "near" | "far_size_matched"
    min_distance: float = 1.5     # far_size_matched only
    seed: int = 0                 # far_size_matched only: which far objects get picked


@dataclass
class RelevanceConfig:
    name: str
    task: Optional[str] = None
    penalty_weight: float = 0.1
    default_relevance: float = 0.0
    relevance: Dict[str, float] = field(default_factory=dict)
    nearby: List[NearbyRule] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict, name: str = "custom") -> "RelevanceConfig":
        cfg = cls(
            name=raw.get("name", name),
            task=raw.get("task"),
            penalty_weight=float(raw.get("penalty_weight", 0.1)),
            default_relevance=float(raw.get("default_relevance", 0.0)),
            relevance={str(k): float(v) for k, v in (raw.get("relevance") or {}).items()},
            nearby=[
                NearbyRule(of=[str(k) for k in r["of"]], radius=float(r["radius"]),
                           relevance=float(r.get("relevance", 1.0)), select=r.get("select", "near"),
                           min_distance=float(r.get("min_distance", 1.5)), seed=int(r.get("seed", 0)))
                for r in (raw.get("nearby") or [])
            ],
        )
        for rule in cfg.nearby:
            if rule.select not in ("near", "far_size_matched"):
                raise ValueError(f"[{cfg.name}] nearby.select must be 'near' or 'far_size_matched', got '{rule.select}'")
        for key, rho in list(cfg.relevance.items()) + [("default_relevance", cfg.default_relevance)]:
            if not 0.0 <= rho <= 1.0:
                raise ValueError(f"[{cfg.name}] relevance for '{key}' must be in [0, 1], got {rho}")
        return cfg

    @classmethod
    def uniform(cls, rho: float = 1.0, penalty_weight: float = 0.0) -> "RelevanceConfig":
        """Every object gets the same relevance (used for full-scene scoring)."""
        return cls(name=f"uniform_{rho:g}", penalty_weight=penalty_weight, default_relevance=rho)

    # ── Resolution ──────────────────────────────────────────────────────

    def resolve(self, env, objects: Iterable, verbose: bool = False) -> Dict[str, float]:
        """Return ``{object_name: rho}`` for every object in *objects*."""
        objects = list(objects)
        best: Dict[str, tuple] = {o.name: (_PRIORITY["default"], self.default_relevance, "default") for o in objects}
        used_keys = set()

        def offer(obj_name, kind, rho, key):
            if _PRIORITY[kind] < best[obj_name][0]:
                best[obj_name] = (_PRIORITY[kind], rho, key)

        for key, rho in self.relevance.items():
            matched = match_key(env, objects, key)
            if matched:
                used_keys.add(key)
            for obj in matched:
                offer(obj.name, _key_kind(key), rho, key)

        for rule in self.nearby:
            anchors = [o for key in rule.of for o in match_key(env, objects, key)]
            if not anchors:
                print(f"[relevance:{self.name}] nearby rule matched no anchor objects: {rule.of}")
                continue
            near = objects_within(env, anchors, objects, rule.radius)
            if rule.select == "near":
                for obj in near:
                    offer(obj.name, "nearby", rule.relevance, f"nearby({rule.radius}m)")
                continue
            # Only objects the near rule would newly add count towards the size.
            n = sum(1 for o in near if best[o.name][2] == "default")
            anchor_names = {a.name for a in anchors}
            far = [o for o in objects
                   if best[o.name][2] == "default" and o.name not in anchor_names
                   and min(object_distance(env, a, o) for a in anchors) >= rule.min_distance]
            far.sort(key=lambda o: o.name)
            if len(far) < n:
                print(f"[relevance:{self.name}] only {len(far)} objects beyond {rule.min_distance}m, wanted {n}")
            picks = np.random.default_rng(rule.seed).permutation(len(far))[:n]
            for i in picks:
                offer(far[i].name, "nearby", rule.relevance, f"far_size_matched(n={n})")

        unused = set(self.relevance) - used_keys
        if unused:
            print(f"[relevance:{self.name}] WARNING: keys matched no object (typo?): {sorted(unused)}")

        if verbose:
            print(f"[relevance:{self.name}] w={self.penalty_weight}")
            for name, (_, rho, key) in sorted(best.items(), key=lambda kv: (-kv[1][1], kv[0])):
                if rho > 0 or key != "default":
                    print(f"    rho={rho:<5g} {name:45s} <- {key}")
            n_default = sum(1 for _, (_, _, k) in best.items() if k == "default")
            print(f"    ({n_default} other objects at default rho={self.default_relevance:g})")

        return {name: rho for name, (_, rho, _) in best.items()}


def load_relevance_config(path_or_name: str, task: Optional[str] = None) -> RelevanceConfig:
    """Load a relevance YAML by path, or by short name from ``configs/relevance/<task>/``."""
    path = path_or_name
    if not os.path.exists(path):
        if task is None:
            raise FileNotFoundError(f"Relevance config '{path_or_name}' not found (pass a task to look it up by name)")
        path = os.path.join(CONFIG_DIR, task, f"{path_or_name}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No relevance config '{path_or_name}' for task '{task}' (looked in {path})")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = RelevanceConfig.from_dict(raw, name=os.path.splitext(os.path.basename(path))[0])
    if task is not None and cfg.task is not None and cfg.task != task:
        raise ValueError(f"Relevance config '{path}' is for task '{cfg.task}', not '{task}'")
    return cfg


# ── Key matching ────────────────────────────────────────────────────────

def _key_kind(key: str) -> str:
    if key == "robot":
        return "robot"
    for prefix in ("ref", "type", "contains"):
        if key.startswith(prefix + ":"):
            return prefix
    return "name"


def object_type(obj) -> str:
    """``DamageableHingeCabinet`` -> ``hingecabinet``."""
    return type(obj).__name__.replace("Damageable", "").lower()


def match_key(env, objects: List, key: str) -> List:
    kind = _key_kind(key)
    if kind == "name":
        return [o for o in objects if o.name == key]
    if kind == "robot":
        return [o for o in objects if env._is_robot(o)]
    value = key.split(":", 1)[1]
    if kind == "ref":
        ref = getattr(env, value, None)
        ref_name = getattr(ref, "name", None)
        return [o for o in objects if ref_name is not None and o.name == ref_name]
    if kind == "type":
        wanted = value.replace("_", "").lower()
        return [o for o in objects if object_type(o) == wanted]
    return [o for o in objects if value.lower() in o.name.lower()]


# ── Geometry ────────────────────────────────────────────────────────────

def _object_boxes(env, obj) -> tuple:
    """World axis-aligned boxes ``(centres (N,3), half_sizes (N,3))`` of the object's collision geoms."""
    model, data = env.sim.model._model, env.sim.data._data
    body_ids = []
    for link in getattr(obj, "damageable_links", []) or []:
        try:
            body_ids.append(env.sim.model.body_name2id(link))
        except Exception:
            continue
    empty = (np.zeros((0, 3)), np.zeros((0, 3)))
    if not body_ids:
        return empty
    on_body = np.isin(np.asarray(model.geom_bodyid), body_ids)
    collides = (np.asarray(model.geom_contype) | np.asarray(model.geom_conaffinity)) != 0
    gids = np.nonzero(on_body & collides)[0]
    if len(gids) == 0:
        gids = np.nonzero(on_body)[0]
    if len(gids) == 0:
        return empty
    aabb = np.asarray(model.geom_aabb)[gids]            # local [centre(3), half(3)]
    rot = np.asarray(data.geom_xmat)[gids].reshape(-1, 3, 3)
    centres = np.asarray(data.geom_xpos)[gids] + np.einsum("nij,nj->ni", rot, aabb[:, :3])
    halves = np.einsum("nij,nj->ni", np.abs(rot), aabb[:, 3:])
    return centres, halves


def object_distance(env, a, b) -> float:
    """Approximate surface distance between two objects (min gap over pairs of geom bounding boxes)."""
    (ca, ha), (cb, hb) = _object_boxes(env, a), _object_boxes(env, b)
    if len(ca) == 0 or len(cb) == 0:
        return float("inf")
    gap = np.abs(ca[:, None, :] - cb[None, :, :]) - (ha[:, None, :] + hb[None, :, :])
    return float(np.linalg.norm(np.maximum(gap, 0.0), axis=-1).min())


def objects_within(env, anchors: List, objects: List, radius: float) -> List:
    anchor_names = {a.name for a in anchors}
    out = []
    for obj in objects:
        if obj.name in anchor_names:
            continue
        if any(object_distance(env, a, obj) <= radius for a in anchors):
            out.append(obj)
    return out


def damage_penalty(prev_health: Dict[str, float], health: Dict[str, float],
                   rho: Dict[str, float], weight: float) -> tuple:
    """Return ``(penalty, per_object_damage)`` for one step: ``w * sum rho_e * max(0, h_prev - h)``."""
    damage = {n: max(0.0, prev_health.get(n, h) - h) for n, h in health.items()}
    penalty = weight * sum(rho.get(n, 0.0) * d for n, d in damage.items())
    return penalty, damage
