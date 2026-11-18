#!/usr/bin/env bash
# Train + evaluate every cell of a relevance grid in parallel, then plot the heatmaps.
#
#   bash relevance/scripts/run_grid.sh relevance/configs/sweeps/grid_close_drawer.yaml [JOBS]
#
# Run from the oopsieverse folder with the oopsieverse_robocasa env active, inside tmux so it
# survives logging out. JOBS = most cells trained at once (default: CPU cores / (n_envs + 1)).
# A cell only starts when enough RAM is free for it, so the grid can share the machine with
# other runs without the OOM killer taking them out:
#   needed = n_envs * MEM_PER_PROC_GB + 1 GB (trainer) + MEM_MARGIN_GB   (defaults 2.5 and 2)
# Set PY to choose the python (default: python on PATH).
# Finished cells are skipped, so re-running resumes. Logs: <out_dir>/logs/<cell>.log
set -euo pipefail

CFG=${1:?usage: run_grid.sh <sweep yaml> [jobs]}
PY=${PY:-python}
MEM_PER_PROC_GB=${MEM_PER_PROC_GB:-2.5}
MEM_MARGIN_GB=${MEM_MARGIN_GB:-2}
read -r OUT NENVS CELLS < <("$PY" - "$CFG" <<'EOF'
import sys
from relevance.common import load_experiment_config
c = load_experiment_config(sys.argv[1])
print(c["out_dir"], c["ppo"]["n_envs"], ",".join(c["sweep"]["relevance"]))
EOF
)
JOBS=${2:-$(( $(nproc) / (NENVS + 1) ))}
(( JOBS < 1 )) && JOBS=1
NEED_KB=$(awk -v n="$NENVS" -v p="$MEM_PER_PROC_GB" -v m="$MEM_MARGIN_GB" \
  'BEGIN { printf "%d", (n * p + 1 + m) * 1024 * 1024 }')
mkdir -p "$OUT/logs"

mem_free_kb() { awk '/MemAvailable/ { print $2 }' /proc/meminfo; }
log() { echo "[run_grid $(date '+%H:%M')] $*"; }

run_cell() {
  local cell=$1
  if "$PY" -m relevance.sweep --config "$CFG" --only "relevance=$cell" > "$OUT/logs/$cell.log" 2>&1; then
    log "done $cell"
  else
    log "FAILED $cell (see $OUT/logs/$cell.log)"
  fi
}

IFS=',' read -ra CELL_LIST <<< "$CELLS"
log "${#CELL_LIST[@]} cells, at most $JOBS at a time, each needs $(( NEED_KB / 1024 / 1024 )) GB free RAM to start; logs in $OUT/logs"
for cell in "${CELL_LIST[@]}"; do
  while (( $(jobs -rp | wc -l) >= JOBS )) || (( $(mem_free_kb) < NEED_KB )); do
    sleep 60
  done
  log "start $cell ($(( $(mem_free_kb) / 1024 / 1024 )) GB free)"
  run_cell "$cell" &
  sleep 120   # let the new cell's envs allocate before checking memory again
done
wait

"$PY" -m relevance.sweep --config "$CFG" --aggregate-only
"$PY" -m relevance.plot_grid --config "$CFG"
