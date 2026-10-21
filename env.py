"""
Environments for relevance-weighted damage-aware RL.

* ``RelevanceEnvMixin`` is mixed in front of any registered Damageable<Task>
  class (no base edits). It tracks damage on every scene object, attaches a
  relevance to each, and speeds up the contact scan.
* ``RelevanceGymEnv`` is a gymnasium env for SB3 with reward
  ``R_task - w * sum_e rho_e * d_e``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from robosuite.controllers import load_composite_controller_config

from damagesim.utils.misc_utils import SAFE_ENV_HEALTH_THRESHOLD
from oopsiebench.envs.registry import EnvironmentRegistry
from relevance.fast_contacts import ContactCache, use_fast_contacts
from relevance.weights import RelevanceConfig, damage_penalty, load_relevance_config

TRACK_MODES = ("all", "relevant", "default")


class RelevanceEnvMixin:
    """
    Mixin placed before a ``Damageable<Task>`` class in the MRO.

    Args:
        relevance_config: ``RelevanceConfig``; defaults to rho=1 for everything.
        track: ``"all"`` tracks every damageable object in the scene,
            ``"relevant"`` only those with rho > 0 (same reward, faster),
            ``"default"`` keeps OopsieBench's per-task allowlist.
        soft_reset: after the first reset, keep the built kitchen (layout,
            fixtures, object models) and only re-sample object placements.
            RoboCasa otherwise rebuilds the whole scene on every reset (~30-100 s).
    """

    def __init__(self, *args, relevance_config: Optional[RelevanceConfig] = None, track: str = "all",
                 verbose_relevance: bool = False, soft_reset: bool = False, **kwargs):
        if track not in TRACK_MODES:
            raise ValueError(f"track must be one of {TRACK_MODES}, got '{track}'")
        self.relevance_config = relevance_config or RelevanceConfig.uniform(1.0)
        self.track_mode = track
        self.relevance: Dict[str, float] = {}
        self._verbose_relevance = verbose_relevance
        self._soft_reset = soft_reset
        self._contact_cache = ContactCache()
        super().__init__(*args, **kwargs)

    def reset(self):
        if self._soft_reset and getattr(self, "sim", None) is not None:
            # Kitchen hard-codes hard_reset=True; robosuite skips _load_model when it is False.
            self.hard_reset = False
            self._resample_object_placements()
        return super().reset()

    def _resample_object_placements(self, attempts: int = 10):
        initializer = getattr(self, "placement_initializer", None)
        if initializer is None:
            return
        for _ in range(attempts):
            try:
                self.object_placements = initializer.sample(placed_objects=self.fxtr_placements)
                return
            except Exception:
                continue
        # Keep the previous placements if sampling keeps failing.

    def initialize_damageable_objects(self):
        if self.track_mode == "default":
            super().initialize_damageable_objects()
        else:
            # The core env tracks every damageable object when task_name == "default".
            task_name = self.task_name
            self.task_name = "default"
            try:
                super().initialize_damageable_objects()
            finally:
                self.task_name = task_name

        objects = self.unique_damageable_objects()
        self.relevance = self.relevance_config.resolve(self, objects, verbose=self._verbose_relevance)
        self._verbose_relevance = False  # print the resolution once, not every reset

        if self.track_mode == "relevant":
            for obj in objects:
                if self.relevance[obj.name] <= 0:
                    obj.set_track_damage(False)
            self.relevance = {n: r for n, r in self.relevance.items() if r > 0}

    def unique_damageable_objects(self) -> list:
        # Fixtures appear in both model.mujoco_objects and self.fixtures as the same instance.
        seen, out = set(), []
        for obj in self.get_damageable_objects():
            if id(obj) not in seen:
                seen.add(id(obj))
                out.append(obj)
        return out

    def _update_all_health(self):
        obj_damage_info = {}
        if self.lock_health:
            return obj_damage_info
        objects = self.unique_damageable_objects()
        self._contact_cache.refresh(self.sim)
        use_fast_contacts(objects, self._contact_cache)
        for obj in objects:
            obj.update_health()
            obj_damage_info[obj.name] = obj.damage_info
        return obj_damage_info

    def health_snapshot(self) -> Dict[str, float]:
        return {obj.name: float(obj.health) for obj in self.unique_damageable_objects()}


def make_env_class(task: str):
    base = EnvironmentRegistry.get(task).damageable_class
    return type(f"Relevance{base.__name__}", (RelevanceEnvMixin, base), {})


def make_raw_env(task: str, relevance_config: Optional[RelevanceConfig] = None, track: str = "all",
                 seed: Optional[int] = None, verbose_relevance: bool = False, soft_reset: bool = False,
                 **env_kwargs):
    """Build a Relevance<Damageable<Task>> env with low-dim defaults (no cameras, no segmentation)."""
    cfg = EnvironmentRegistry.get(task)
    kwargs = dict(
        robots=cfg.robot,
        controller_configs=load_composite_controller_config(robot=cfg.robot),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        render_segmentation=False,
        control_freq=cfg.control_freq,
        ignore_done=True,
        seed=seed,
    )
    kwargs.update(env_kwargs)
    return make_env_class(task)(relevance_config=relevance_config, track=track,
                                verbose_relevance=verbose_relevance, soft_reset=soft_reset, **kwargs)


class RelevanceGymEnv(gym.Env):
    """
    Gymnasium wrapper with the relevance-weighted damage penalty.

    ``terminated`` = task success, ``truncated`` = horizon reached.
    ``info["episode_metrics"]`` is set on the last step of every episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str,
        relevance: Union[str, RelevanceConfig] = "s4",
        horizon: int = 500,
        track: str = "relevant",
        obs_keys: Sequence[str] = ("robot0_proprio", "object-state"),
        action_parts: Optional[Sequence[str]] = ("right", "right_gripper"),
        include_health_obs: bool = False,
        soft_reset: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False,
        env_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.task = task
        self.relevance_config = (relevance if isinstance(relevance, RelevanceConfig)
                                 else load_relevance_config(relevance, task=task))
        self.horizon = int(horizon)
        self.obs_keys = list(obs_keys)
        self.include_health_obs = include_health_obs

        self.env = make_raw_env(task, self.relevance_config, track=track, seed=seed,
                                verbose_relevance=verbose, soft_reset=soft_reset, **(env_kwargs or {}))
        obs, _ = self.env.reset()  # robots are only built on the first reset
        self._init_episode()

        low, high = self.env.action_spec
        self._full_action_dim = low.shape[0]
        if action_parts:
            split = self.env.robots[0].composite_controller._action_split_indexes
            self._action_idx = np.concatenate([np.arange(*split[p]) for p in action_parts])
        else:
            self._action_idx = np.arange(self._full_action_dim)
        self.action_space = gym.spaces.Box(low[self._action_idx].astype(np.float32),
                                           high[self._action_idx].astype(np.float32))
        dim = self._flatten_obs(obs).shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _init_episode(self):
        self._t = 0
        self._prev_health = self.env.health_snapshot()
        self._ep = {"task_return": 0.0, "damage_penalty": 0.0,
                    "in_scope_damage": 0.0, "out_of_scope_damage": 0.0}

    def _flatten_obs(self, obs) -> np.ndarray:
        parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in self.obs_keys]
        if self.include_health_obs:
            health = self._prev_health
            in_scope = [n for n, r in self.env.relevance.items() if r > 0 and n in health]
            if in_scope:
                w = np.array([self.env.relevance[n] for n in in_scope])
                h = np.array([health[n] for n in in_scope])
                parts.append(np.array([np.dot(w, h) / w.sum(), h.min()], dtype=np.float32) / 100.0)
            else:
                parts.append(np.ones(2, dtype=np.float32))
        return np.concatenate(parts)

    # ── Gym API ─────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        obs, _ = self.env.reset()
        self._init_episode()
        return self._flatten_obs(obs), {}

    def step(self, action):
        full = np.zeros(self._full_action_dim)
        full[self._action_idx] = np.clip(action, self.action_space.low, self.action_space.high)
        obs, task_reward, _, _ = self.env.step(full)
        self._t += 1

        health = self.env.health_snapshot()
        rho = self.env.relevance
        penalty, damage = damage_penalty(self._prev_health, health, rho, self.relevance_config.penalty_weight)
        self._prev_health = health

        in_scope = sum(d for n, d in damage.items() if rho.get(n, 0.0) > 0)
        self._ep["task_return"] += float(task_reward)
        self._ep["damage_penalty"] += penalty
        self._ep["in_scope_damage"] += in_scope
        self._ep["out_of_scope_damage"] += sum(damage.values()) - in_scope

        success = bool(self.env._check_success())
        terminated = success
        truncated = (not success) and self._t >= self.horizon

        info = {"task_reward": float(task_reward), "damage_penalty": penalty, "is_success": success}
        if terminated or truncated:
            info["episode_metrics"] = episode_metrics(health, rho, success, truncated, self._t, self._ep)
        return self._flatten_obs(obs), float(task_reward) - penalty, terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def episode_metrics(final_health: Dict[str, float], rho: Dict[str, float], success: bool, timeout: bool,
                    length: int, totals: Dict[str, float]) -> dict:
    """Per-episode metrics from the proposal. Safety is judged on every tracked object."""
    healths = np.array(list(final_health.values())) if final_health else np.array([100.0])
    in_scope = [n for n, r in rho.items() if r > 0 and n in final_health]
    weighted = (sum(rho[n] * final_health[n] for n in in_scope) / sum(rho[n] for n in in_scope)
                if in_scope else 100.0)
    return {
        "success": success,
        "timeout": timeout,
        "safe_success": success and bool(healths.min() >= SAFE_ENV_HEALTH_THRESHOLD),
        "zero_damage": bool(healths.min() >= 100.0 - 1e-6),
        "safe_success_zero_damage": success and bool(healths.min() >= 100.0 - 1e-6),
        "length": length,
        "min_health": float(healths.min()),
        "mean_health": float(healths.mean()),
        "weighted_health": float(weighted),
        "num_damaged_objects": int((healths < 100.0 - 1e-6).sum()),
        **totals,
        "final_health": dict(final_health),
    }
