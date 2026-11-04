#!/bin/bash
set -euo pipefail

# Aider Eval Sweep — runs ./run.sh against several llama-server configs in
# turn (different models, or the same model with one setting varied, e.g. a
# few -ctv values) and collects results.
#
# Each sweep point means: swap the persistent llama-cpp-turboquant `llama-cpp`
# compose service to this combo's config via /mnt/llm/model-configs/mcfg.py
# (writes docker-compose.override.yml, `docker compose up -d`, blocks until a
# real completion succeeds -- not just /health, and dies with recent logs on
# a crash/timeout instead of hanging), run the eval via ./run.sh (which
# already tracks results into results/aider-results.tsv -- see run.sh's
# track_eval), move to the next combo.
#
# This USES your persistent llama-cpp-turboquant `llama-cpp` compose service
# directly (swapping its config combo-to-combo) rather than a throwaway
# container, so nothing needs stopping/restarting around the sweep. If it's
# already serving a request when a combo wants to swap in, mcfg refuses by
# default (WAIT_IDLE=1 waits for idle instead, FORCE=1 restarts mid-request)
# -- see WAIT_IDLE/FORCE below. The sweep leaves whatever the LAST combo
# resolved to running when it finishes.
#
# Usage:
#   CONFIGS=configs.txt ./sweep.sh [run-cmd] [-- extra run.sh args]
#
#   # or vary one flag across values on top of a shared base config:
#   BASE_FLAGS="-m /models/foo.gguf --alias foo --temp 0.6 --ctx-size 32768" \
#     VARY_FLAG=-ctv VARY_VALUES="q8_0 turbo3 turbo2" ./sweep.sh
#
# configs.txt format: one combo per line, TAB-separated tag + llama-server
# flags (everything after --host/--port, which the sweep adds itself — don't
# include --server, it's a tools.sh dispatch flag that doesn't apply since
# this bypasses tools.sh via --entrypoint /app/llama-server directly). Each
# combo's flags are parsed like a shell command line (quote multi-word
# values, e.g. --alias "Gemma 4"), so write them the same way you would in
# llama-cpp-turboquant/docker-compose.yml's command: block. Every combo MUST
# include --alias, since run.sh's model arg has to match it exactly. Blank
# lines and #-comments are skipped. Example:
#   qwen3.5-q8       -m /models/Qwen3.5-9B.gguf --alias Qwen3.5 -ctk q8_0 -ctv q8_0     --temp 0.6 --top-p 0.95 --top-k 20 --ctx-size 32768
#   qwen3.5-turbo3   -m /models/Qwen3.5-9B.gguf --alias Qwen3.5 -ctk q8_0 -ctv turbo3   --temp 0.6 --top-p 0.95 --top-k 20 --ctx-size 32768
#
# CONFIGS can ALSO be a comma-separated list of models instead of a
# configs.txt path, e.g. CONFIGS="Foo.gguf,Bar.gguf" -- each entry is
# resolved by /mnt/llm/model-configs/mcfg.py in eval mode (rules keyed on
# the GGUF's own header: arch, MTP head, quant...; see models.yaml there).
# An entry can carry per-model overrides/overlays after the filename, e.g.
#   CONFIGS="Foo.gguf +tpl,Foo.gguf -ctv q6_0,Bar.gguf +nomtp"
# Every resolved config is printed before running; `mcfg explain --mode eval
# Foo.gguf` shows which rule set each flag. MCFG_MODE overrides the mode.
#
# run-cmd is any evals/aider/run.sh command (default: quick — 5 python
# exercises, a reasonable tradeoff of signal vs. time-per-combo for a sweep).
# Extra args after `--` are forwarded to run.sh; TRIES/MAX_TOKENS/etc. env
# vars work exactly as they do for a plain ./run.sh invocation. --clean is
# added automatically unless you already passed --clean/-c/--resume
# yourself, so a sweep is always independent combo-to-combo by default (see
# run.sh's auto_clean/guard_reuse — matters for python/full, which don't
# auto-clean on a plain ./run.sh call).
#
# Recipe: gap-fill a model roster + auto seed-2 retry on 100%-failing
# exercises. run.sh's own SKIP_EXISTING/SEED_TRIES (see its fromfile help)
# pass straight through here since they're ordinary env vars and this
# script just execs `MODEL=$ALIAS ./run.sh "$RUN_CMD" ...` per combo:
#   CONFIGS="ModelA.gguf,ModelB.gguf,ModelC.gguf" \
#     SKIP_EXISTING=1 SEED_TRIES=2 \
#     ./sweep.sh fromfile lists/some-exercise-list.txt
# (no trailing -- --clean needed: the exercise-list path is just the next
# positional arg after run-cmd, and --clean is added automatically since
# nothing else here already implies --clean/-c/--resume)
# For each resolved model this runs the exercise list once (dropping
# anything that's EVER passed before, since SKIP_EXISTING=1), then -- for
# whatever's still failing after that -- automatically retries with seed 2
# before moving to the next model, closing out both "never attempted" gaps
# and "seed got unlucky" gaps in one pass instead of a hand-built one-off
# driver script per campaign (see the seed2-retry-sweep session this
# recipe was distilled from: 3 of 4 confirmed-failing exercises across a
# 6-model roster turned out to just be seed variance).
#
# Other seed/skip combos (full semantics in `./run.sh --help`, fromfile):
#   fill every missing seed-2 record, even where seed 1 passed:
#     SEEDS=2 SEED_ALWAYS=2 SKIP_EXISTING=seed ./sweep.sh fromfile <list>
#   fill seed 2 only for exercises that never passed at any seed:
#     SEEDS=2 SKIP_EXISTING=pass,seed ./sweep.sh fromfile <list>
#   skip only what already passed at the seed being run:
#     SKIP_EXISTING=seed-pass ./sweep.sh fromfile <list>
#   scale every exercise's max duration to each model's measured decode speed
#   (max(1, 100/tok-per-s), probed by run.sh after each combo comes up):
#     MAX_DUR_SCALE=auto ./sweep.sh fromfile <list>
#
# Env vars:
#   CONFIGS                path to a configs.txt (see format above). Required
#                          unless BASE_FLAGS + VARY_FLAG + VARY_VALUES are set.
#   BASE_FLAGS/VARY_FLAG/VARY_VALUES   synthesize a configs.txt-equivalent
#                          sweep of one flag; see usage above.
#   LLAMA_REPO_DIR          path to the llama-cpp-turboquant checkout (holds
#                          docker-compose.yml, the compose service mcfg
#                          manages). Default /mnt/llm/llama-cpp-turboquant.
#   MCFG                   path to mcfg.py (default /mnt/llm/model-configs/mcfg.py)
#   MCFG_MODE=eval         mode every combo resolves in (rule facts can match
#                          on --mode; eval is model-configs' benchmark-tuned
#                          profile set).
#   HEALTH_TIMEOUT=900     seconds `mcfg use --wait` gets to bring a combo up
#                          and answer a real completion before giving up
#                          (mcfg's own default; cold loads off slow drives
#                          can take ~450s -- see its --timeout).
#   WAIT_IDLE=0             if 1, and the service is currently mid-request
#                          when a combo wants to swap in, wait for it to go
#                          idle instead of refusing (mcfg's --wait-idle).
#   FORCE=0                 if 1, swap in a combo even if the service is
#                          currently mid-request, interrupting it (mcfg's
#                          --force). Takes priority over WAIT_IDLE if both
#                          are set. Leave both 0 (default) to just refuse and
#                          let you decide by hand -- don't interrupt or race
#                          against something else using the server.
#   MIN_TPS=0              if >0, after a combo comes up (mcfg's --wait
#                          already confirmed a completion succeeds), run a
#                          further smoke completion and read the logged
#                          `tg = N t/s` back; skip the combo (marking it
#                          failed, same as a run.sh failure) if under this
#                          threshold instead of committing to a full eval run.
#                          Catches a bad layer split (e.g. --fit-ctx leaving
#                          CPU-resident layers a manual --n-gpu-layers/
#                          --ctx-size split wouldn't) that answers fine but
#                          decodes at a few tok/s for hours -- see
#                          turboquant-fit-vs-manual-moe-offload notes. 0
#                          (default) disables the check.
#
# Output: every combo's row lands in results/aider-results.tsv via run.sh's
# own tracking, tagged with that combo's actual server settings (model,
# sampling, -ctk/-ctv, etc. — see run.sh's SRV_BLACKLIST for what's
# excluded). This script additionally copies just this sweep's rows to
# runs/aider/sweep-<timestamp>.tsv and prints them at the end.

export CALLER_PWD="${CALLER_PWD:-$PWD}"   # run.sh resolves relative list paths against it
cd "$(dirname "$0")"
source ../lib/paths.sh
AIDER_RUNS="$RUNS_DIR/aider"

# The model server (llama-cpp-turboquant in docker inside WSL) lives on the NAS,
# not on this host: LLAMA_REPO_DIR/MCFG are paths ON the NAS, and everything
# that touches them (mcfg.py, `docker compose`, `docker logs`) runs over ssh.
# This sweep only drives the llama-nas target; Strata isn't swapped by it.
NAS_HOST="${NAS_HOST:-192.168.2.15}"
export TARGET=llama-nas
nas() { ssh -o BatchMode=yes "$NAS_HOST" "$@"; }
LLAMA_REPO_DIR="${LLAMA_REPO_DIR:-/mnt/llm/llama-cpp-turboquant}"
MCFG="${MCFG:-/mnt/llm/model-configs/mcfg.py}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
WAIT_IDLE="${WAIT_IDLE:-0}"
FORCE="${FORCE:-0}"
MIN_TPS="${MIN_TPS:-0}"

# ── Parse combos: CONFIGS file, or BASE_FLAGS x VARY_FLAG x VARY_VALUES ─────

TAGS=()
COMBO_FLAGS=()

if [ -n "${CONFIGS:-}" ] && [ -f "$CONFIGS" ]; then
  while IFS=$'\t' read -r tag flags || [ -n "${tag:-}" ]; do
    [ -z "${tag:-}" ] && continue
    [[ "$tag" == \#* ]] && continue
    TAGS+=("$tag")
    COMBO_FLAGS+=("$flags")
  done < "$CONFIGS"
elif [ -n "${CONFIGS:-}" ] && [[ "$CONFIGS" == *.gguf* ]]; then
  IFS=',' read -ra GGUF_LIST <<< "$CONFIGS"
  for entry in "${GGUF_LIST[@]}"; do
    read -ra entry_words <<< "$entry"
    [ "${#entry_words[@]}" -eq 0 ] && continue
    resolved="$("$MCFG" resolve --mode "${MCFG_MODE:-eval}" --format tsv "${entry_words[@]}")" || {
      echo "ERROR: failed to resolve a config for '$entry' -- see above." >&2
      exit 1
    }
    tag="${resolved%%$'\t'*}"
    flags="${resolved#*$'\t'}"
    echo "Resolved $entry -> --alias $tag: $flags" >&2
    TAGS+=("$tag")
    COMBO_FLAGS+=("$flags")
  done
elif [ -n "${CONFIGS:-}" ]; then
  echo "ERROR: CONFIGS='$CONFIGS' is neither an existing file nor a .gguf list." >&2
  exit 1
elif [ -n "${VARY_FLAG:-}" ] && [ -n "${VARY_VALUES:-}" ]; then
  : "${BASE_FLAGS:?BASE_FLAGS required when VARY_FLAG/VARY_VALUES are set}"
  VARY_VALUES_NORM="${VARY_VALUES//,/ }"
  read -ra VLIST <<< "$VARY_VALUES_NORM"
  for v in "${VLIST[@]}"; do
    TAGS+=("${VARY_FLAG}=${v}")
    COMBO_FLAGS+=("$BASE_FLAGS $VARY_FLAG $v")
  done
else
  echo "ERROR: set CONFIGS=<file>, or BASE_FLAGS + VARY_FLAG + VARY_VALUES." >&2
  echo "See the header comment in $0 for the two formats." >&2
  exit 1
fi

[ "${#TAGS[@]}" -ge 1 ] || { echo "ERROR: no combos found in CONFIGS." >&2; exit 1; }

RUN_CMD="${1:-quick}"
[ $# -ge 1 ] && shift
if [ "${1:-}" = "--" ]; then
  shift
fi
EXTRA_RUN_ARGS=("$@")
HAS_OVERRIDE=0
for a in "${EXTRA_RUN_ARGS[@]}"; do
  case "$a" in --clean|-c|--resume) HAS_OVERRIDE=1 ;; esac
done
[ "$HAS_OVERRIDE" -eq 0 ] && EXTRA_RUN_ARGS+=(--clean)

echo "Sweeping ${#TAGS[@]} combo(s) with run-cmd '$RUN_CMD': ${TAGS[*]}" >&2

# Combo flag strings are parsed like a shell command line (via eval), so
# quoting works exactly like it would on a real command line or in
# docker-compose.yml's command: block -- e.g. --alias "Gemma 4".
extract_alias() {
  local flags_str="$1" flag_arr i
  eval "flag_arr=($flags_str)"
  for ((i = 0; i < ${#flag_arr[@]}; i++)); do
    if [[ "${flag_arr[$i]}" == "--alias" && $((i + 1)) -lt ${#flag_arr[@]} ]]; then
      echo "${flag_arr[$((i + 1))]}"
      return 0
    fi
  done
  return 1
}

# -> prints "MODEL\tREST_FLAGS_STRING" -- pulls the -m/--model value out as
# mcfg's positional MODEL (it needs a real GGUF to derive facts from), and
# returns everything else as override flags. Since mcfg's use/resolve apply
# overrides last ("later contributions override earlier ones"), passing an
# already-fully-formed flags string back this way is safe and idempotent --
# it just reasserts exactly what we already have, whether that came from
# mcfg resolve (CONFIGS=<gguf list>) or a hand-written configs.txt/BASE_FLAGS
# line.
split_model_and_rest() {
  local flags_str="$1" flag_arr i model="" rest=()
  eval "flag_arr=($flags_str)"
  for ((i = 0; i < ${#flag_arr[@]}; i++)); do
    if [[ "${flag_arr[$i]}" == "-m" || "${flag_arr[$i]}" == "--model" ]] && [ -z "$model" ]; then
      model="${flag_arr[$((i + 1))]}"
      i=$((i + 1))
    else
      rest+=("${flag_arr[$i]}")
    fi
  done
  [ -n "$model" ] || return 1
  printf '%s\t%s\n' "$model" "$(printf '%q ' "${rest[@]}")"
}

# Short smoke completion + read the server's own logged `tg = N t/s` back.
# mcfg's --wait already confirmed a completion succeeds; this is purely the
# MIN_TPS speed check on top of that. Returns 1 (skip this combo) if under
# MIN_TPS or no speed line showed up.
speed_gate() {
  local container http_code
  container="$(nas "cd $(printf %q "$LLAMA_REPO_DIR") && docker compose ps -q llama-cpp")"
  http_code="$(curl -s -o /dev/null -w '%{http_code}' "http://$NAS_HOST:8080/completion" -H 'Content-Type: application/json' -d '{
    "prompt": "Write a short paragraph about the history of bridges.",
    "n_predict": 200,
    "temperature": 0.6
  }')"
  if [ "$http_code" != "200" ]; then
    echo "smoke decode speed: unknown tok/s (threshold ${MIN_TPS}, completion request got HTTP ${http_code:-<none>})" >&2
    return 1
  fi
  sleep 2
  local tps
  # Non-speculative builds log a plain "tg = N t/s" summary line. Models running
  # --spec-type draft-mtp (all the 35B-A3B MoE configs) never print that line at
  # all -- their print_timing summary uses "eval time = ... / N tokens (... M
  # tokens per second)" instead. Try the normal format first, fall back to the
  # spec-decode one so those models don't get spuriously skipped as "unknown".
  tps="$(nas docker logs "$container" --since 2m 2>&1 | grep -oE 'tg = *[0-9.]+ t/s' | tail -1 | grep -oE '[0-9.]+')"
  if [ -z "$tps" ]; then
    tps="$(nas docker logs "$container" --since 2m 2>&1 | grep -E 'eval time = ' | grep -v 'prompt eval time' | tail -1 | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+')"
  fi
  echo "smoke decode speed: ${tps:-unknown} tok/s (threshold ${MIN_TPS})" >&2
  [ -n "$tps" ] && awk "BEGIN{exit !($tps >= $MIN_TPS)}"
}

SWEEP_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FAILED=()

BUSY_FLAG=()
[ "$FORCE" = "1" ] && BUSY_FLAG=(--force)
[ "$FORCE" != "1" ] && [ "$WAIT_IDLE" = "1" ] && BUSY_FLAG=(--wait-idle)

for i in "${!TAGS[@]}"; do
  TAG="${TAGS[$i]}"
  FLAGS="${COMBO_FLAGS[$i]}"
  echo "" >&2
  echo "=== [$((i + 1))/${#TAGS[@]}] $TAG ===" >&2

  ALIAS="$(extract_alias "$FLAGS")" || {
    echo "ERROR: combo '$TAG' has no --alias in its flags -- skipping (run.sh needs it to match)." >&2
    FAILED+=("$TAG (no --alias)")
    continue
  }

  split="$(split_model_and_rest "$FLAGS")" || {
    echo "ERROR: combo '$TAG' has no -m/--model in its flags -- skipping." >&2
    FAILED+=("$TAG (no -m)")
    continue
  }
  MODEL="${split%%$'\t'*}"
  REST_STR="${split#*$'\t'}"
  eval "REST_ARR=($REST_STR)"

  echo "Bringing up $ALIAS (mcfg use, timeout ${HEALTH_TIMEOUT}s) -- full command below" >&2
  if ! nas "cd $(printf %q "$LLAMA_REPO_DIR") && $(printf %q "$MCFG") use --mode $(printf %q "${MCFG_MODE:-eval}") --up --wait \
      --timeout $(printf %q "$HEALTH_TIMEOUT") $(printf '%q ' "${BUSY_FLAG[@]}" "$MODEL" "${REST_ARR[@]}")"; then
    echo "ERROR: mcfg use failed for combo '$TAG' -- see its output above." >&2
    FAILED+=("$TAG (mcfg use failed)")
    continue
  fi

  if [ "$MIN_TPS" != "0" ] && ! speed_gate; then
    echo "SKIPPING $TAG -- decode speed below MIN_TPS=$MIN_TPS threshold, not worth a full run" >&2
    FAILED+=("$TAG (below MIN_TPS)")
    continue
  fi

  echo "Running: MODEL=$ALIAS ./run.sh $RUN_CMD ${EXTRA_RUN_ARGS[*]}" >&2
  # Direct to the NAS's published :8080 (not the localhost of the aider
  # container, and bypassing the :3002 prompt-router/logger proxy).
  if ! LLAMA_BASE_URL="http://$NAS_HOST:8080/v1" \
      MODEL="$ALIAS" ./run.sh "$RUN_CMD" "${EXTRA_RUN_ARGS[@]}"; then
    echo "ERROR: ./run.sh failed for combo '$TAG'" >&2
    FAILED+=("$TAG (run.sh failed)")
  fi
done

# ── Summary: just this sweep's rows out of the shared aider-results.tsv ─────

TRACK_TSV="${TRACK_TSV:-$DATA_DIR/aider-results.tsv}"
SWEEP_TSV="$AIDER_RUNS/sweep-$(date +%Y%m%dT%H%M%S).tsv"

if [ -f "$TRACK_TSV" ]; then
  python3 -c "
import csv, sys

tsv_file, since, out_file = sys.argv[1:4]
with open(tsv_file, newline='') as f:
    r = csv.reader(f, delimiter='\t')
    header = next(r)
    rows = [row for row in r if row and row[0] >= since]
with open(out_file, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(header)
    w.writerows(rows)
print(f'{len(rows)} row(s) from this sweep -> {out_file}')
" "$TRACK_TSV" "$SWEEP_START" "$SWEEP_TSV"
  column -t -s $'\t' "$SWEEP_TSV" >&2 || true
else
  echo "No $TRACK_TSV found -- nothing to summarize." >&2
fi

echo "" >&2
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Sweep finished with ${#FAILED[@]} failure(s): ${FAILED[*]}" >&2
else
  echo "Sweep finished, all ${#TAGS[@]} combo(s) ran." >&2
fi
echo "The persistent llama-cpp service is left running whatever the last combo resolved to." >&2

[ "${#FAILED[@]}" -eq 0 ]
