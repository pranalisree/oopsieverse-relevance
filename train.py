"""
Train one PPO policy for a (task, relevance config, seed).

    python -m relevance.train --task dishes_to_sink --relevance s2 --seed 0
    python -m relevance.train --config relevance/configs/smoke.yaml

Writes to runs/<task>/<relevance>/seed<k>/: model.zip, vecnormalize.pkl,
config.yaml, relevance.yaml, checkpoints/ and tb/ (TensorBoard).
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import yaml


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="experiment YAML merged over configs/experiment.yaml")
    p.add_argument("--task", default=None)
    p.add_argument("--relevance", default=None, help="relevance config name or path")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from relevance.common import env_factory, load_experiment_config, run_dir
    from relevance.weights import load_relevance_config

    cfg = load_experiment_config(
        args.config, task=args.task, relevance=args.relevance, seed=args.seed, out_dir=args.out_dir,
        **{"ppo.total_timesteps": args.total_timesteps, "ppo.n_envs": args.n_envs},
    )
    relevance = load_relevance_config(cfg["relevance"], task=cfg["task"])
    out = run_dir(cfg, relevance.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out, "relevance.yaml"), "w") as f:
        rel = dataclasses.asdict(relevance)
        yaml.safe_dump(rel, f, sort_keys=False)
    print(f"[train] {cfg['task']} / {relevance.name} / seed {cfg['seed']} -> {out}")

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
    from relevance.callbacks import EpisodeMetricsCallback

    ppo = cfg["ppo"]
    n_envs = int(ppo["n_envs"])
    factories = [env_factory(cfg, relevance, rank) for rank in range(n_envs)]
    venv = SubprocVecEnv(factories, start_method="spawn") if n_envs > 1 else DummyVecEnv(factories)
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=ppo["normalize_obs"], norm_reward=ppo["normalize_reward"],
                        gamma=ppo["gamma"])

    callbacks = [EpisodeMetricsCallback()]
    if ppo.get("checkpoint_every"):
        callbacks.append(CheckpointCallback(
            save_freq=max(1, int(ppo["checkpoint_every"]) // n_envs),
            save_path=os.path.join(out, "checkpoints"), name_prefix="model", save_vecnormalize=True,
        ))

    if cfg["wandb"]["enabled"]:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], name=f"{cfg['task']}-{relevance.name}-s{cfg['seed']}",
                   config={**cfg, "relevance_config": rel}, sync_tensorboard=True, dir=out)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=int(ppo["n_steps"]), batch_size=int(ppo["batch_size"]), n_epochs=int(ppo["n_epochs"]),
        learning_rate=float(ppo["learning_rate"]), gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]), clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        policy_kwargs=dict(net_arch=list(ppo["net_arch"]), activation_fn=torch.nn.Tanh),
        tensorboard_log=os.path.join(out, "tb"), seed=int(cfg["seed"]), verbose=1,
        device=ppo.get("device", "auto"),
    )
    model.learn(total_timesteps=int(ppo["total_timesteps"]), callback=CallbackList(callbacks),
                tb_log_name="ppo")
    model.save(os.path.join(out, "model.zip"))
    venv.save(os.path.join(out, "vecnormalize.pkl"))
    venv.close()
    print(f"[train] done -> {out}")
    return out


if __name__ == "__main__":
    main()
