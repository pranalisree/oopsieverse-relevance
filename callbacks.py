"""SB3 callback that logs relevance / damage episode metrics to TensorBoard."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """Averages every scalar in ``info["episode_metrics"]`` under ``episode/`` per rollout."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if not metrics:
                continue
            for key, value in metrics.items():
                if isinstance(value, (bool, int, float)):
                    self.logger.record_mean(f"episode/{key}", float(value))
        return True
