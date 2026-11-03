#!/usr/bin/env bash
# Train + evaluate every cell of a relevance grid in parallel, then plot the heatmaps.
#
#   bash relevance/scripts/run_grid.sh relevance/configs/sweeps/grid_close_drawer.yaml [JOBS]
#
# Run from the oopsieverse folder with the oopsieverse_robocasa env active, inside tmux so it
# survives logging out. JOBS = cells trained at once (default: CPU cores / (n_envs + 1)).
# Finished cells are skipped, so re-running resumes. Logs: <out_dir>/logs/<cell>.log
set -euo pipefail

CFG=${1:?usage: run_grid.sh <sweep yaml> [jobs]}
read -r OUT NENVS CELLS < <(python - "$CFG" <<'EOF'
import sys
from relevance.common import load_experiment_config
c = load_experiment_config(sys.argv[1])
print(c["out_dir"], c["ppo"]["n_envs"], ",".join(c["sweep"]["relevance"]))
EOF
)
JOBS=${2:-$(( $(nproc) / (NENVS + 1) ))}
(( JOBS < 1 )) && JOBS=1
mkdir -p "$OUT/logs"

echo "[run_grid] $(tr ',' '\n' <<<"$CELLS" | wc -l) cells, $JOBS at a time, logs in $OUT/logs"
tr ',' '\n' <<<"$CELLS" | xargs -P "$JOBS" -I{} sh -c \
  "python -m relevance.sweep --config '$CFG' --only relevance={} > '$OUT/logs/{}.log' 2>&1 \
   && echo '[run_grid] done {}' || echo '[run_grid] FAILED {} (see $OUT/logs/{}.log)'"

python -m relevance.sweep --config "$CFG" --aggregate-only
python -m relevance.plot_grid --config "$CFG"
