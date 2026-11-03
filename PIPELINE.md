# Pipeline: train a policy on a task with your own relevance weights

The steps below go from picking a task to a results table. The example uses `dishes_to_sink` with a custom
weight file called `my_weights`. To use another task, swap in its name.

---

## Step 0: Open a terminal (do this every time)

All commands run from the **`oopsieverse`** folder, not from inside `relevance/`.

**Windows PowerShell:**
```powershell
cd $HOME\Desktop\oopsieverse
$py = "$HOME\miniconda3\envs\oopsieverse_robocasa\python.exe"
```
Adjust the two paths if your oopsieverse folder or Miniconda install is somewhere else.

**Mac / Linux / TACC:**
```bash
cd ~/oopsieverse
conda activate oopsieverse_robocasa
```
On Mac/Linux, replace `& $py` with `python` in every command below.

---

## Step 1: Pick a task

Any of the 16 RoboCasa tasks registered in `oopsiebench/envs/registry.py` works. The policy learns from the
task's own reward, which varies a lot:

| Reward type | Tasks | RL from scratch |
|---|---|---|
| Gradual (partial credit) | `dishes_to_sink`, `open_drawer`, `close_drawer`, `open_single_door`, `wipe_counter` | Best bet |
| Only at the very end | `counter_to_microwave`, `turn_on_faucet`, `turn_on_microwave`, `turn_on_stove` | Hard |
| None (always 0) | `pick_egg` | Won't learn |

---

## Step 2: See what's in the scene

```powershell
& $py -m relevance.inspect_scene --task dishes_to_sink --anchors bowl cup plate ref:sink
```

Example output:
```
 dist(m)  rho  name                     type          refs
  anchor    0  bowl                     mjcfobject
  anchor    0  sink_island_group        sink          ref:sink
   0.000    0  island_island_group      counter       ref:counter
   0.017    0  dishwasher_island_group  dishwasher
   0.179    0  PandaOmron               wheeledrobot  (robot)
```

- **name** is what you write in the YAML.
- **type** can be used as `type:<type>`, for example `type:counter`.
- **refs** are stable nicknames such as `ref:sink`. Use these for tasks with random kitchen layouts.
- **dist** is the distance in metres from the objects listed after `--anchors`.

---

## Step 3: Write your weights

Create `relevance/configs/relevance/dishes_to_sink/my_weights.yaml`. The file name becomes the condition's name.

```yaml
task: dishes_to_sink
penalty_weight: 0.1          # w: overall strength of the damage penalty
default_relevance: 0.0       # rho for any object not listed below
damage_mode: sum             # sum: add damage over all of an object's parts (default); min: weakest part only
relevance:
  robot: 1.0                 # any robot
  bowl: 1.0                  # exact object name
  cup: 1.0
  plate: 0.5
  ref:sink: 0.75             # the fixture the task calls "sink"
  type:counter: 0.25         # every counter
nearby:                      # optional, recomputed every episode
  - of: [bowl, cup, plate]
    radius: 0.30             # metres
    relevance: 0.5
```

The reward the robot learns from:
```
reward = task_reward - penalty_weight * sum over objects of (relevance * damage this step)
```

**Ways to name objects:**

| Key | Matches |
|---|---|
| `bowl` | the object named exactly `bowl` |
| `ref:sink` | the fixture the task registered as `sink` |
| `robot` | any robot |
| `type:counter` | every object of that type |
| `contains:cab` | every object whose name contains `cab` |

If several keys match one object, the most specific wins:
name > `ref:` > `robot` > `type:` > `contains:` > `nearby` > default.

**Check that it does what you expect:**
```powershell
& $py -m relevance.inspect_scene --task dishes_to_sink --relevance my_weights
```
This prints the relevance each object received and which key gave it. It also warns about keys that matched
nothing, which usually means a typo.

---

## Step 4: Quick test run (about 5–10 min)

```powershell
& $py -m relevance.train --task dishes_to_sink --relevance my_weights --seed 0 --total-timesteps 2000 --n-envs 2
```

If this finishes without errors, the setup is good.

---

## Step 5: Real training run

```powershell
& $py -m relevance.train --task dishes_to_sink --relevance my_weights --seed 0
```

Defaults come from `relevance/configs/experiment.yaml`: 2M steps, 8 parallel envs, and 500-step episodes.

- **Laptop:** about 9 hours per run.
- **TACC:** add `--n-envs 32` or more, depending on the number of CPU cores.

Output goes to `runs/dishes_to_sink/my_weights/seed0/`:

| File | What it is |
|---|---|
| `model.zip` | trained policy |
| `vecnormalize.pkl` | observation scaling (needed for evaluation) |
| `config.yaml`, `relevance.yaml` | exact settings used |
| `checkpoints/` | snapshots every 200k steps |
| `tb/` | TensorBoard training curves |

---

## Step 6: Watch training (optional, in a second terminal)

```powershell
& $py -m tensorboard.main --logdir runs
```

Open the link it prints and look under **episode/**:

| Metric | Question it answers |
|---|---|
| `success` | Is it finishing the task? |
| `safe_success` | Is it finishing without damaging anything? |
| `damage_penalty` | Is it learning to be careful? |
| `timeout` | Is it freezing up / too cautious? |

---

## Step 7: Evaluate

```powershell
& $py -m relevance.evaluate --run runs/dishes_to_sink/my_weights/seed0
```

This runs 30 episodes and scores damage on **every** object in the kitchen, including objects with relevance 0.
That is how you catch a policy that ignored the counter and smashed it.

Output (in the run folder):
- `eval_summary.json`: safe_success, success, timeout, min/mean health, damage in scope vs out of scope.
- `eval_episodes.csv`: one row per episode, including each object's final health.

**Safe success** means the task succeeded **and** every object, robot included, ended with health ≥ 95.

---

## Step 8: Run all seeds and conditions with a sweep

Create `relevance/configs/sweeps/mine.yaml`:
```yaml
out_dir: runs/mine
sweep:
  tasks: [dishes_to_sink]
  relevance: [s1, s4, my_weights]
  seeds: [0, 1, 2]
```

```powershell
# Preview the runs (prints commands, runs nothing)
& $py -m relevance.sweep --config relevance/configs/sweeps/mine.yaml --dry-run

# Train + evaluate everything
& $py -m relevance.sweep --config relevance/configs/sweeps/mine.yaml

# Run only part of it, e.g. one condition per TACC job
& $py -m relevance.sweep --config relevance/configs/sweeps/mine.yaml --only relevance=my_weights

# Rebuild the results table from finished runs
& $py -m relevance.sweep --config relevance/configs/sweeps/mine.yaml --aggregate-only
```

Finished runs are skipped, so you can stop the sweep and restart it.

The results table is written to `runs/mine/summary.csv`. It has one row per (task, condition) with the mean and
standard deviation over seeds of safe_success, success, timeout, health and damage.

---

## Adding a new task

1. Pick a task from `oopsiebench/envs/registry.py`.
2. Make a folder: `relevance/configs/relevance/<task>/`.
3. Run `inspect_scene --task <task>` to get object names.
4. Add YAMLs. These two work for any task without changes:
   - `s4.yaml`: `default_relevance: 1.0` and no `relevance:` entries (everything matters).
   - `w0.yaml`: `penalty_weight: 0.0` (task reward only).
5. Add the task to your sweep file's `tasks:` list.

---

## Built-in conditions (dishes_to_sink and counter_to_microwave)

| Name | Who the robot cares about |
|---|---|
| `w0` | nobody (task reward only) |
| `s1` | robot + target object(s) |
| `s2` | s1 + the sink / microwave |
| `s3` | s2 + everything within 30 cm |
| `s4` | everything in the kitchen |
| `s3_nearby_025` / `050` / `075` | s2 fully, nearby objects at 0.25 / 0.5 / 0.75 |
| `graded` | target 1.0, sink/microwave 0.75, nearby 0.5, everything else 0.25 |
| `s3_size_matched` | same number of objects as s3, but far away (control) |

---

## Cheat sheet

```
pick task
  -> inspect_scene                      (see objects)
  -> write YAML                         (set weights)
  -> inspect_scene --relevance <name>   (check weights)
  -> train --total-timesteps 2000       (quick test)
  -> train                              (real run)
  -> evaluate                           (full-scene safety)
  -> sweep                              (all seeds + conditions -> summary.csv)
```
