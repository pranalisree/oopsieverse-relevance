# relevance: scaled object relevance for damage-aware RL

This extension package adds a relevance weight to every object for PPO in OopsieVerse/RoboCasa. It changes nothing
in `damagesim/`, `oopsiebench/` or `externals/`; everything is done through subclassing and wrappers.

    r(s_t) = R_task(s_t) − w · Σ_e ρ_e · d_e(t),     ρ_e ∈ [0, 1]

`d_e(t)` is the health object `e` lost on this step, added up over all of its parts (`damage_mode: sum`, the
default; `damage_mode: min` uses only the weakest part), and `w` is `penalty_weight`. The binary scopes S1–S4 from
the proposal are the special case where every ρ is 0 or 1.

**New here?** See [PIPELINE.md](PIPELINE.md) for a step-by-step walkthrough, from picking a task to a results table.

## Setup (once)

This repo is a folder that goes **inside** an OopsieVerse checkout, named `relevance`:

```bash
# 1. Install OopsieVerse with RoboCasa as usual (see its docs/install.md)
git clone https://github.com/UT-Austin-RobIn/oopsieverse.git
cd oopsieverse
python install.py --new_env --robocasa

# 2. Clone this repo into it as the `relevance` folder
git clone https://github.com/pranalisree/oopsieverse-relevance.git relevance

# 3. Extra dependency
conda activate oopsieverse_robocasa
pip install "stable-baselines3>=2.3,<2.7"
```

Run every command below from the `oopsieverse` folder, not from inside `relevance/`.

On Windows, if `conda` isn't on PATH, call the env's Python directly:
`$py = "$HOME\miniconda3\envs\oopsieverse_robocasa\python.exe"`, then `& $py -m relevance.<command> ...`.

## Commands

| What | Command |
|---|---|
| List the objects in a scene, their distances and ρ | `python -m relevance.inspect_scene --task dishes_to_sink --relevance s3` |
| Train one policy | `python -m relevance.train --task dishes_to_sink --relevance s2 --seed 0` |
| Evaluate it (full-scene damage) | `python -m relevance.evaluate --run runs/dishes_to_sink/s2/seed0` |
| No-training baseline | `python -m relevance.evaluate --task dishes_to_sink --relevance s2 --policy random --episodes 5` |
| Quick end-to-end check | `python -m relevance.sweep --config relevance/configs/sweeps/smoke.yaml` |
| Full study (print the commands) | `python -m relevance.sweep --config relevance/configs/sweeps/proposal.yaml --dry-run` |
| One slice of the study (e.g. one TACC job) | `python -m relevance.sweep --config ...proposal.yaml --only task=dishes_to_sink,relevance=s3` |
| Rebuild summary.csv | `python -m relevance.sweep --config ...proposal.yaml --aggregate-only` |
| Unit tests | `python -m pytest relevance/tests -q` |

TensorBoard: `tensorboard --logdir runs`. Episode metrics are logged under `episode/*`.

## Relevance configs: `configs/relevance/<task>/<name>.yaml`

```yaml
task: dishes_to_sink
penalty_weight: 0.1
default_relevance: 0.0          # ρ for every object not matched below
damage_mode: sum                # sum: add health lost by every part (default); min: weakest part only
relevance:
  robot: 1.0                    # any robot
  bowl: 1.0                     # exact object name
  ref:sink: 0.75                # the fixture the task registered as env.sink (works with random layouts)
  type:counter: 0.25            # every object whose class is DamageableCounter
  contains:cab: 0.5             # every object whose name contains "cab"
nearby:                         # re-evaluated at every reset
  - of: [bowl, cup, plate, ref:sink]
    radius: 0.30                # metres between collision-geom bounding boxes
    relevance: 0.5
```

When several keys match an object, the most specific key wins: name > `ref:` > `robot` > `type:` > `contains:` >
`nearby` > default. Keys that match no object print a warning. Run `inspect_scene` to see the real names, types
and `ref:` aliases.

Conditions shipped for `dishes_to_sink` and `counter_to_microwave`:

| Name | Meaning |
|---|---|
| `w0` | task reward only (w = 0) |
| `s1` | robot + target object(s) |
| `s2` | S1 + the manipulated fixture (sink / microwave) |
| `s3` | S2 + every object within 30 cm of an S2 object |
| `s4` | every object in the scene (the OopsieVerse reward) |
| `s3_size_matched` | S2 + as many objects as S3 adds, but chosen ≥ 1.5 m away (the control) |
| `s3_nearby_025/050/075` | S2 at 1.0 + the nearby objects at 0.25 / 0.5 / 0.75 |
| `graded` | target 1.0, manipulated 0.75, nearby 0.5, rest of the scene 0.25 |

To add a task, create `configs/relevance/<task>/` with YAMLs for any task in `oopsiebench/envs/registry.py`.

## Experiment config: `configs/experiment.yaml`

This file holds the defaults for horizon, observations, action parts and PPO hyperparameters. A `--config` file
only overrides the keys it sets. A sweep file is an experiment config plus a `sweep:` block with tasks, relevance
names and seeds. Keep `env` and `ppo` identical across conditions.

- **Observations:** `robot0_proprio` + `object-state`, low-dimensional, with no cameras or segmentation.
- **Actions:** arm + gripper only (7-D). The mobile base and torso are held at zero. Set
  `env.action_parts: null` for the full 12-D action.
- **Episodes:** `terminated` = task success, `truncated` = `env.horizon` reached (this counts as a timeout).

## Metrics (evaluate.py, sweep summary)

Evaluation always tracks **every** object in the scene, whatever scope the policy was trained with.

- `safe_success`: the task succeeded and every object, robot included, ended at health ≥ 95
  (`SAFE_ENV_HEALTH_THRESHOLD`).
- `zero_damage`, `safe_success_zero_damage`: the stricter version, where nothing loses any health.
- `success`, `timeout`, `length`.
- `min_health`, `mean_health`, `weighted_health` (Σρh/Σρ over ρ > 0), `num_damaged_objects`, and final health
  per object (in `eval_episodes.csv`).
- `in_scope_damage` / `out_of_scope_damage`: total damage to objects with ρ > 0 versus ρ = 0.

## Implementation notes

- **`env.py`:** `RelevanceEnvMixin` is placed before the registered `Damageable<Task>` class. It sets
  `task_name="default"` while damage tracking is initialised, so the core env tracks every object. With
  `env.track: relevant` (the training default), objects with ρ = 0 are then switched off: they cannot change the
  reward, so this is faster.
- **`fast_contacts.py`:** builds MuJoCo's contact table once per step instead of once per part. Damage values
  are unchanged (they match the original evaluator exactly), and steps are 10–20× faster with full-scene tracking.
- **Soft reset (`env.soft_reset: true`):** RoboCasa rebuilds the whole kitchen on every reset (30–100 s). With
  soft reset, the env builds the kitchen once and then only re-samples object placements each episode (~1 s). So
  each env process keeps one kitchen layout, while different seeds and ranks get different layouts. This matters
  for `counter_to_microwave`, whose layout is randomized.
- **Compute:** about 130 ms per step per env on a laptop CPU, mostly MuJoCo physics and the robosuite controller.
  2M steps with 8 envs is roughly 9 hours per run, and the 60-run study needs far more than a laptop. Use many
  envs (`ppo.n_envs`) on a TACC node, and split the sweep with `--only` or `--dry-run`.
- **Objects listed twice:** fixtures appear in both `model.mujoco_objects` and `env.fixtures`, so the base env
  updates their health twice per step. This env updates each object once.
- **`penalty_weight` is a placeholder (0.1).** Damage per step can reach 100, while task rewards are around 10.
  Tune `w` on S4 before running the sweep.
- **Sparse reward in `counter_to_microwave`:** it gives +10 only once the cup is in the microwave. PPO from
  scratch will probably need reward shaping (reach / grasp terms) to learn it at all.
