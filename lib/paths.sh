# Shared layout roots, sourced by aider/*.sh and harbor/harbor.sh.
#   EVAL_DIR   this repo
#   DATA_DIR   combined results (the nested results repo): TSVs + viz
#   RUNS_DIR   per-run generated output (gitignored); runners use $RUNS_DIR/<runner>
#   LISTS_DIR  exercise/task lists (lists/tmp = one-off lists)
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$EVAL_DIR/results}"
RUNS_DIR="${RUNS_DIR:-$EVAL_DIR/runs}"
LISTS_DIR="${LISTS_DIR:-$EVAL_DIR/lists}"
