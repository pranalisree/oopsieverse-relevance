# Relevance grid on poincare (close_drawer, Drawer × Robot)

This trains one PPO policy per cell of a weight grid and colours the grid by **safe success rate**. It's the
2-object grid with real results in it.

- **Task:** `close_drawer`. The robot pushes an open drawer shut, then moves its hand away. It gives reward every
  step as the drawer closes, which makes it the task most likely to learn overnight.
- **Grid:** Drawer relevance (rows) × Robot relevance (columns), levels 0 / 0.5 / 1, so 9 policies. Every other
  object has relevance 0 during training. **Evaluation scores damage on every object in the kitchen.**
- **Defaults:** 300k steps per policy, 4 envs each, 200-step episodes, 20 eval episodes, 1 seed,
  `penalty_weight` 1.0, damage summed over parts.

---

## 1. Log in and start tmux (so it keeps running after you disconnect)

```bash
ssh fri@poincare.csres.utexas.edu
tmux new -s grid          # later: reattach with  tmux attach -t grid
nproc                     # number of CPU cores: decides how many cells train at once
```

## 2. Install (skip whatever is already there)

```bash
cd ~
git clone https://github.com/UT-Austin-RobIn/oopsieverse.git
cd oopsieverse
python install.py --new_env --robocasa        # creates the oopsieverse_robocasa env (downloads kitchen assets)
git clone https://github.com/pranalisree/oopsieverse-relevance.git relevance
conda activate oopsieverse_robocasa
pip install "stable-baselines3>=2.3,<2.7"
```

If the repo is already there: `cd ~/oopsieverse/relevance && git pull`.

## 3. Check that it works (about 5–10 min)

```bash
cd ~/oopsieverse
conda activate oopsieverse_robocasa
python -m relevance.train --task close_drawer --relevance grid_100_100 --seed 0 \
    --total-timesteps 2000 --n-envs 2 --out-dir runs/check
```

Look for the PPO table with an `fps` line. Each env runs about 5–8 steps per second.

## 4. Run the grid

```bash
bash relevance/scripts/run_grid.sh relevance/configs/sweeps/grid_close_drawer.yaml
```

- It trains `nproc / 5` cells at once. To set this yourself, add a number at the end, e.g. `... grid_close_drawer.yaml 9`
  runs all 9 at once, which needs about 45 cores.
- Detach with `Ctrl-b` then `d`. Log out; it keeps going.
- Progress: `tail -f runs/grid_close_drawer/logs/grid_100_100.log`
- If it stops, run the same command again; finished cells are skipped.

**Time:** one cell is about 300k steps / (4 envs × ~6 steps/s) ≈ **3.5 h**, plus ~15 min of evaluation. With
all 9 running at once (≥ 45 cores) the grid is done in about 4 h; with 4 at a time (~20 cores), about 12 h.

## 5. Get the heatmap

When it finishes it writes:

- `runs/grid_close_drawer/grid_results.png`: heatmaps of safe success, task success, timeout and damage
- `runs/grid_close_drawer/grid_results.csv`: the same numbers as a table
- `runs/grid_close_drawer/summary.csv`: every metric, per cell

To plot partial results while it is still running:
`python -m relevance.plot_grid --config relevance/configs/sweeps/grid_close_drawer.yaml`

Copy the image to your laptop (run this **on your laptop**):

```bash
scp fri@poincare.csres.utexas.edu:~/oopsieverse/runs/grid_close_drawer/grid_results.png .
```

---

## Changing the grid

```bash
# different objects / levels / budget, then run step 4 again
python -m relevance.grid --task close_drawer --objects ref:drawer robot --labels Drawer Robot \
    --levels 0 0.25 0.5 0.75 1 --total-timesteps 500000 --seeds 0 1
```

Use `python -m relevance.inspect_scene --task <task>` to find object keys. Options: `--penalty-weight`,
`--default-relevance` (weight for every other object), `--damage-mode sum|min`, `--horizon`, `--n-envs`,
`--eval-episodes`.

## Reading the result

- **(0, 0)** is the task-reward-only baseline: no damage penalty at all.
- **(1, 1)** cares fully about both.
- A cell with high safe success and low weights supports the hypothesis: the robot stays safe without being told
  to care fully about both objects.
- If (1, 1) has a high **timeout** rate, the penalty made the robot too cautious to finish.
- 300k steps and 1 seed is a first look, not a final result. Treat differences of less than about 15–20
  percentage points as noise until there are more seeds.
