#!/bin/bash
set -euo pipefail

# Aider Eval Sweep — runs ./run.sh against several llama-server configs in
# turn (different models, or the same model with one setting varied, e.g. a
# few -ctv values) and collects results.
#
# Different shape than scripts/perplexity-sweep.sh in llama-cpp-turboquant:
# that script spins up a throwaway `docker run` per request against the
# one-shot llama-perplexity binary. llama-server is instead a persistent
# HTTP service that an aider benchmark run talks to over many requests, so
# each sweep point here means: stop the previous server container, start a
# fresh one with this combo's flags, wait for /health, run the eval via
# ./run.sh (which already tracks results into results/eval-results.tsv —
# see run.sh's track_eval), tear the server down, move to the next combo.
#
# This STOPS your persistent llama-cpp-turboquant `llama-cpp` compose
# service for the sweep's duration (it needs host port 8080) and does NOT
# restart it automatically afterward — bring it back yourself when done:
#   cd /mnt/llm/llama-cpp-turboquant && docker compose up -d llama-cpp
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
#     ./sweep.sh fromfile results/some-exercise-list.txt
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
# Env vars:
#   CONFIGS                path to a configs.txt (see format above). Required
#                          unless BASE_FLAGS + VARY_FLAG + VARY_VALUES are set.
#   BASE_FLAGS/VARY_FLAG/VARY_VALUES   synthesize a configs.txt-equivalent
#                          sweep of one flag; see usage above.
#   IMAGE                  llama-cpp-turboquant server image; default
#                          llama-cpp-turboquant-llama-cpp:latest
#   LLAMA_REPO_DIR          path to the llama-cpp-turboquant checkout; only
#                          used to `docker compose stop llama-cpp` there when
#                          STOP_EXISTING=1. Default /mnt/llm/llama-cpp-turboquant.
#   MCFG                   path to mcfg.py (default /mnt/llm/model-configs/mcfg.py)
#   MCFG_MODE=eval         mode CONFIGS=<gguf list> entries resolve in
#   MODELS_DIR/MODELS2_DIR/MODELS3_DIR   optional host-dir overrides for the
#                          /models, /models2, /models3 mounts; every /models*
#                          mount otherwise comes from docker-compose.yml's
#                          volumes (via `mcfg mounts`).
#   HEALTH_TIMEOUT=1800    seconds to wait for a combo's server to report
#                          healthy before giving up on it (model load can be
#                          slow, especially cold-cache on a big GGUF).
#   MIN_TPS=0              if >0, after a combo's server reports healthy, run
#                          a short smoke completion and read its logged
#                          `tg = N t/s` back; skip the combo (marking it
#                          failed, same as a run.sh failure) if under this
#                          threshold instead of committing to a full eval run.
#                          Catches a bad layer split (e.g. --fit-ctx leaving
#                          CPU-resident layers a manual --n-gpu-layers/
#                          --ctx-size split wouldn't) that loads fine and
#                          passes /health but decodes at a few tok/s for
#                          hours -- see turboquant-fit-vs-manual-moe-offload
#                          notes. 0 (default) disables the check.
#   STOP_EXISTING=0        if 1, automatically stop the persistent llama-cpp
#                          compose service if it's currently running.
#                          Otherwise the sweep refuses to start rather than
#                          silently killing a server something else may be
#                          relying on.
#
# Output: every combo's row lands in results/eval-results.tsv via run.sh's
# own tracking, tagged with that combo's actual server settings (model,
# sampling, -ctk/-ctv, etc. — see run.sh's SRV_BLACKLIST for what's
# excluded). This script additionally copies just this sweep's rows to
# results/sweep-<timestamp>.tsv and prints them at the end.

cd "$(dirname "$0")"

trap 'echo; echo "Interrupted -- stopping sweep container." >&2; docker rm -f aider-sweep-llama >/dev/null 2>&1 || true; exit 130' INT TERM

IMAGE="${IMAGE:-llama-cpp-turboquant-llama-cpp:latest}"
LLAMA_REPO_DIR="${LLAMA_REPO_DIR:-/mnt/llm/llama-cpp-turboquant}"
MCFG="${MCFG:-/mnt/llm/model-configs/mcfg.py}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
STOP_EXISTING="${STOP_EXISTING:-0}"
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

# ── Make sure port 8080 is free for the sweep's own server ─────────────────

EXISTING="$(docker ps --filter 'name=llama-cpp-turboquant-llama-cpp' --filter status=running --format '{{.Names}}' | head -1)"
if [ -n "$EXISTING" ]; then
  if [ "$STOP_EXISTING" = "1" ]; then
    echo "Stopping persistent llama-cpp service ($EXISTING) to free port 8080 for the sweep..." >&2
    (cd "$LLAMA_REPO_DIR" && docker compose stop llama-cpp)
  else
    echo "ERROR: persistent llama-cpp service is running ($EXISTING) and holds port 8080." >&2
    echo "Stop it yourself (cd $LLAMA_REPO_DIR && docker compose stop llama-cpp)," >&2
    echo "or re-run with STOP_EXISTING=1 to have this script stop it for you." >&2
    exit 1
  fi
fi

# Same /models* mounts as the compose service (mcfg reads them from
# docker-compose.yml), so any path mcfg resolves exists in here too.
# MODELS_DIR/MODELS2_DIR/MODELS3_DIR still override their mount if set.
MOUNT_ARGS=()
while IFS=$'\t' read -r host cont; do
  case "$cont" in
    /models) host="${MODELS_DIR:-$host}" ;;
    /models2) host="${MODELS2_DIR:-$host}" ;;
    /models3) host="${MODELS3_DIR:-$host}" ;;
  esac
  MOUNT_ARGS+=(-v "$host:$cont:ro")
done < <("$MCFG" mounts)
[ "${#MOUNT_ARGS[@]}" -gt 0 ] || { echo "ERROR: '$MCFG mounts' returned nothing." >&2; exit 1; }

# Combo flag strings are parsed like a shell command line (via eval), so
# quoting works exactly like it would on a real command line or in
# docker-compose.yml's command: block -- e.g. --alias "Gemma 4".
start_server() {
  local flags_str="$1" flag_arr
  eval "flag_arr=($flags_str)"
  docker rm -f aider-sweep-llama >/dev/null 2>&1 || true
  docker run -d --rm --name aider-sweep-llama --gpus all -p 8080:8080 \
    "${MOUNT_ARGS[@]}" \
    --entrypoint /app/llama-server "$IMAGE" \
    --host 0.0.0.0 --port 8080 "${flag_arr[@]}" >/dev/null
}

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

wait_for_health() {
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while true; do
    if ! docker ps --filter 'name=^aider-sweep-llama$' --filter status=running --format '{{.Names}}' | grep -q .; then
      echo "ERROR: aider-sweep-llama exited before becoming healthy -- check: docker logs aider-sweep-llama" >&2
      return 1
    fi
    if curl -sf "http://localhost:8080/health" >/dev/null 2>&1; then
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "ERROR: timed out after ${HEALTH_TIMEOUT}s waiting for aider-sweep-llama to become healthy" >&2
      return 1
    fi
    sleep 5
  done
}

# Short smoke completion + read the server's own logged `tg = N t/s` back.
# Returns 1 (skip this combo) if under MIN_TPS or no speed line showed up.
# Retries the completion itself up to 3x -- seen a 503 "Loading model" moments
# after /health already reported ok on a big --fit-ctx model, which a bare
# `curl -sf ... >/dev/null` (ignoring the exit code) silently turns into a
# false "unknown speed, skip" instead of a real readiness check.
speed_gate() {
  local attempt http_code
  for attempt in 1 2 3; do
    http_code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8080/completion" -H 'Content-Type: application/json' -d '{
      "prompt": "Write a short paragraph about the history of bridges.",
      "n_predict": 200,
      "temperature": 0.6
    }' 2>/dev/null)"
    [ "$http_code" = "200" ] && break
    echo "smoke completion attempt $attempt got HTTP ${http_code:-<none>} -- retrying in 10s" >&2
    sleep 10
  done
  sleep 2
  local tps
  tps="$(docker logs aider-sweep-llama --since 2m 2>&1 | grep -oE 'tg = *[0-9.]+ t/s' | tail -1 | grep -oE '[0-9.]+')"
  echo "smoke decode speed: ${tps:-unknown} tok/s (threshold ${MIN_TPS}, last http=${http_code:-<none>})" >&2
  [ -n "$tps" ] && awk "BEGIN{exit !($tps >= $MIN_TPS)}"
}

SWEEP_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FAILED=()

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

  echo "Starting server: $FLAGS" >&2
  start_server "$FLAGS"

  echo "Waiting for /health (timeout ${HEALTH_TIMEOUT}s)..." >&2
  if ! wait_for_health; then
    FAILED+=("$TAG (server never became healthy)")
    docker rm -f aider-sweep-llama >/dev/null 2>&1 || true
    continue
  fi

  if [ "$MIN_TPS" != "0" ] && ! speed_gate; then
    echo "SKIPPING $TAG -- decode speed below MIN_TPS=$MIN_TPS threshold, not worth a full run" >&2
    FAILED+=("$TAG (below MIN_TPS)")
    docker rm -f aider-sweep-llama >/dev/null 2>&1 || true
    continue
  fi

  echo "Running: MODEL=$ALIAS ./run.sh $RUN_CMD ${EXTRA_RUN_ARGS[*]}" >&2
  if ! LLAMA_BASE_URL="http://host.docker.internal:8080/v1" LLAMA_CONTAINER="aider-sweep-llama" \
      MODEL="$ALIAS" ./run.sh "$RUN_CMD" "${EXTRA_RUN_ARGS[@]}"; then
    echo "ERROR: ./run.sh failed for combo '$TAG'" >&2
    FAILED+=("$TAG (run.sh failed)")
  fi

  docker rm -f aider-sweep-llama >/dev/null 2>&1 || true
done

# ── Summary: just this sweep's rows out of the shared eval-results.tsv ─────

TRACK_TSV="${TRACK_TSV:-$(pwd)/results/eval-results.tsv}"
SWEEP_TSV="results/sweep-$(date +%Y%m%dT%H%M%S).tsv"

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
echo "Your persistent llama-cpp service was NOT restarted -- bring it back with:" >&2
echo "  cd $LLAMA_REPO_DIR && docker compose up -d llama-cpp" >&2

[ "${#FAILED[@]}" -eq 0 ]
