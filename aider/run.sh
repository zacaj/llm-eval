#!/bin/bash
set -euo pipefail

# Aider Benchmark Runner
# Runs against your llama.cpp server (see docker-compose.yml / LLAMA_BASE_URL)
#
# The model name after openai/ should match your llama.cpp --alias.
# Check what your server reports: curl http://100.71.29.121:3002/v1/models
#
# Edit formats (set via EDIT_FORMAT=<format>, e.g. EDIT_FORMAT=diff ./run.sh quick):
#   whole    - model outputs entire file (safest for weaker models) [default]
#   diff     - model outputs search/replace blocks (faster, needs stronger model)
#   udiff    - unified diff format
# The dedicated `diff` command is just shorthand for EDIT_FORMAT=diff ./run.sh quick.
#
# --reasoning-effort / --thinking-tokens send OpenAI-o-series/Anthropic-style
# fields (extra_body.reasoning_effort / thinking.budget_tokens) that llama.cpp
# likely ignores for a Qwen3-style chat-template reasoning model — so they're
# sent as a best-effort hint but NOT relied on. The actual hard cap is
# `max_tokens` (a standard OpenAI field llama.cpp maps straight to n_predict),
# which bounds total generation (thinking + answer) regardless of whether the
# model cooperates with softer reasoning-effort hints.
#
# REQUEST_TIMEOUT auto-scales from MAX_TOKENS assuming ~70 tok/s decode, plus
# a fixed buffer for prompt prefill. Override either directly if needed.
#
# Note: the benchmark harness (benchmark.py) hardcodes stream=False — there
# is no streaming option.
#
# Usage:
#   ./run.sh <command> [exercise ...] [flags...]
#
# The model is NOT a positional arg -- set MODEL=<alias> (your llama-server's
# --alias) if it isn't "Qwen3.6-35B-A3B" (every model swapped into
# docker-compose.yml's active block gets aliased to this same string
# regardless of actual weights, so this rarely needs overriding -- see
# srv.-m in the TSVs to tell runs apart instead). Bare
# words after <command> are exercise-name substrings (aider's --keywords) --
# e.g. `./run.sh smoke connect` targets the "connect" exercise specifically,
# NOT a model named "connect". Anything starting with '-' (e.g. --clean,
# --resume) is forwarded to benchmark.py as-is; everything before the first
# such flag is treated as exercise names.
#
# smoke/quick/polyglot/diff use a fixed, deterministic default exercise set
# per command (not random) so repeated runs are directly comparable across
# models/configs -- see the `random` command if you actually want a random
# sample instead.
#
# Examples:
#   ./run.sh smoke                          # 1 fixed python exercise, ~2 min
#   ./run.sh quick                          # 5 fixed python exercises, ~10 min
#   ./run.sh quick zipper leap              # just those 2 exercises instead
#   ./run.sh random 5                       # 5 RANDOM python exercises
#   ./run.sh python                         # all python exercises, ~30 min
#   ./run.sh full                           # all 225 exercises, ~2-4 hours
#   MODEL=my-alias ./run.sh quick           # specify model name
#   ./run.sh stats <run-dir>                # regenerate stats from a completed run
#   ./run.sh shell                          # open interactive shell in container
#
# Per-run output goes to runs/aider/<run-name>/; tracked rows to results/aider-results*.tsv

CALLER_PWD="${CALLER_PWD:-$PWD}"   # relative list paths resolve against the caller's cwd
cd "$(dirname "$0")"
source ../lib/paths.sh
AIDER_RUNS="$RUNS_DIR/aider"

# ── Target server ───────────────────────────────────────────────────────────
# This eval host is NOT the machine running the model server any more. TARGET
# picks which remote server a run talks to; each sets the OpenAI base URL the
# aider containers use, the /health URL used for probes, and how server
# settings get recorded into the TSVs:
#   llama-nas  llama.cpp in docker inside WSL on the NAS. Requests go through
#              the :3002 prompt-logger proxy (as before); health/probes hit
#              :8080 directly. srv.* columns come from `docker inspect` over
#              ssh://$NAS_HOST.
#   strata     Strata (Qwen3.8-Flash-Next engine) on the NAS's Windows host,
#              over its own tailscale address. No container to inspect, so
#              srv.* columns come from its /props endpoint instead.
# LLAMA_BASE_URL / LLAMA_SERVER_URL / LLAMA_HEALTH_URL still override.
TARGET="${TARGET:-llama-nas}"
NAS_HOST="${NAS_HOST:-192.168.2.15}"          # WSL ssh on :22 (docker lives here)
STRATA_URL="${STRATA_URL:-http://100.94.194.48:8080}"
LLAMA_DOCKER=()                                # docker CLI prefix for the model server's host
case "$TARGET" in
  llama-nas)
    LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://$NAS_HOST:8080}"
    LLAMA_BASE_URL="${LLAMA_BASE_URL:-http://$NAS_HOST:3002/v1}"
    LLAMA_DOCKER=(docker -H "ssh://$NAS_HOST")
    ;;
  strata)
    LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-$STRATA_URL}"
    LLAMA_BASE_URL="${LLAMA_BASE_URL:-$LLAMA_SERVER_URL/v1}"
    ;;
  *)
    echo "ERROR: TARGET='$TARGET' is not one of llama-nas|strata." >&2
    exit 1
    ;;
esac
LLAMA_HEALTH_URL="${LLAMA_HEALTH_URL:-$LLAMA_SERVER_URL/health}"
export LLAMA_BASE_URL    # read by docker-compose.yml as OPENAI_API_BASE

CMD="${1:-help}"
# Strata serves whatever one model it has loaded under its own alias (any
# name is accepted); default MODEL to that so the TSV `model` column says
# which model actually ran, instead of the llama default below.
_DEFAULT_MODEL="Qwen3.6-35B-A3B"
if [[ "$TARGET" == "strata" && -z "${MODEL:-}" && "$CMD" != "help" ]]; then
  _DEFAULT_MODEL="$(curl -sf --max-time 10 "$LLAMA_SERVER_URL/v1/models" 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)" || _DEFAULT_MODEL=""
  if [[ -z "$_DEFAULT_MODEL" ]]; then
    echo "ERROR: TARGET=strata but couldn't read a model id from $LLAMA_SERVER_URL/v1/models (is it up/exposed?). Set MODEL= explicitly to skip this." >&2
    exit 1
  fi
fi
MODEL="${MODEL:-$_DEFAULT_MODEL}"  # your llama-server's --alias; NOT a positional arg (see header).
                                    # Every model swapped into docker-compose.yml's active block
                                    # gets aliased to this same string regardless of actual weights
                                    # (see llama-cpp-turboquant/docker-compose.yml), so this rarely
                                    # needs overriding in practice -- distinguish runs via srv.-m in
                                    # the TSVs instead.
# A bare alias (the normal case) is wrapped in litellm's openai/ prefix and
# routed to your local llama-server (via OPENAI_API_BASE in docker-compose.yml).
# A MODEL already containing '/' is treated as a full litellm model spec and
# used as-is instead -- e.g. MODEL=openrouter/<vendor>/<slug> routes straight
# to OpenRouter using OPENROUTER_API_KEY (see docker-compose.yml), no local
# server involved at all.
if [[ "$MODEL" == */* ]]; then
  AIDER_MODEL="$MODEL"
else
  AIDER_MODEL="openai/$MODEL"
fi
# Filesystem-safe stand-in for AIDER_MODEL, used everywhere a per-model file
# path is needed (model-settings YAML, fromfile's results dir) so two runs
# against different models -- e.g. a local run and an OpenRouter run at the
# same time -- never share a path and clobber each other's state.
MODEL_SLUG="$(echo "$AIDER_MODEL" | tr '/:' '--')"
REASONING_EFFORT="${REASONING_EFFORT-low}"  # empty string disables
# The effort level is a bundle, not just the --reasoning-effort hint: the
# Qwen3.8 template turns it into a system-prompt sentence, so a model asked for
# more effort thinks longer -- and then needs a bigger thinking cap, token cap
# and wall-clock allowance or it just gets truncated/timed out. 'low' is the
# baseline every earlier result was run at (4k thinking, 16k tokens, 1x
# timeouts); the others scale from it. The numbers are a first guess -- tune
# them after seeing a real run. Each can still be set explicitly:
#   REASONING_BUDGET   thinking-token cap sent PER REQUEST (llama-server's
#                      reasoning_budget_tokens), so no server restart is needed.
#                      Empty = don't send, the server's --reasoning-budget applies.
#                      (-1 can't mean "unlimited" per request -- the server reads
#                      it as "unset"; use a big number.) Ignored for MODEL=x/y
#                      remote specs: OpenRouter has OR_REASONING_TOKENS instead.
#   MAX_TOKENS         total generation cap (thinking + answer), below
#   EFFORT_DUR_MULT    extra multiplier on every max_dur, stacks with MAX_DUR_SCALE
# REQUEST_TIMEOUT already follows MAX_TOKENS, so it scales with no extra knob.
[[ "$REASONING_EFFORT" == "high" ]] && REASONING_EFFORT=xhigh   # the template treats them the same
case "${REASONING_EFFORT:-low}" in
  low)    _EFFORT_BUDGET=4096;  _EFFORT_MAX_TOKENS=16000; _EFFORT_DUR_MULT=1 ;;
  medium) _EFFORT_BUDGET=8192;  _EFFORT_MAX_TOKENS=20000; _EFFORT_DUR_MULT=1.5 ;;
  xhigh)  _EFFORT_BUDGET=16384; _EFFORT_MAX_TOKENS=32000; _EFFORT_DUR_MULT=2.5 ;;
  *)
    echo "ERROR: REASONING_EFFORT='$REASONING_EFFORT' is not one of low|medium|xhigh (high = xhigh)." >&2
    exit 1
    ;;
esac
REASONING_BUDGET="${REASONING_BUDGET-$_EFFORT_BUDGET}"
EFFORT_DUR_MULT="${EFFORT_DUR_MULT:-$_EFFORT_DUR_MULT}"
# Non-low runs get their own results dir / settings file (appended to
# SAMPLING_SLUG below); low keeps the old names so existing dirs still resolve.
EFFORT_SLUG=""
[[ -n "$REASONING_EFFORT" && "$REASONING_EFFORT" != "low" ]] && EFFORT_SLUG="-${REASONING_EFFORT}"
# OpenRouter-only knobs, sent via extra_params.extra_body in the model-settings YAML.
# OR_PROVIDER: comma-separated provider order, e.g. "DeepInfra" or "DeepInfra,Together"
#   (pinned with allow_fallbacks=false so a silent fallback can't change quantization).
# OR_REASONING_TOKENS: exact thinking budget (reasoning.max_tokens); OpenRouter accepts
#   effort OR max_tokens, not both, so set REASONING_EFFORT= (empty) when using this.
OR_PROVIDER="${OR_PROVIDER:-}"
OR_REASONING_TOKENS="${OR_REASONING_TOKENS:-}"
EDIT_FORMAT="${EDIT_FORMAT:-whole}"         # whole|diff|udiff|... forwarded as aider's --edit-format
                                             # (applies to smoke/quick/python/polyglot/full/custom;
                                             # the dedicated `diff` command always forces diff, that's its purpose)

# Fail fast with a clear message instead of a cryptic UnknownEditFormat
# traceback from deep inside the container 30+ seconds into a run. Full list
# from aider/coders/base_coder.py; only whole/diff/diff-fenced/patch/udiff/
# udiff-simple make sense for benchmarking (architect/editor-*/ask/help are
# chat-mode or architect-internal formats, not something you'd pick here).
_VALID_EDIT_FORMATS="help ask diff diff-fenced whole patch udiff udiff-simple architect editor-diff editor-whole editor-diff-fenced context"
case " $_VALID_EDIT_FORMATS " in
  *" $EDIT_FORMAT "*) ;;
  *)
    echo "ERROR: EDIT_FORMAT='$EDIT_FORMAT' is not a real aider edit format." >&2
    echo "Did you mean 'whole' (aider's default, and this script's default)?" >&2
    echo "Valid values: $_VALID_EDIT_FORMATS" >&2
    echo "(for benchmarking, you almost always want: whole, diff, or udiff)" >&2
    exit 1
    ;;
esac
TRIES="${TRIES:-6}"                         # fix-up attempts per exercise (default 2 gave too many false
                                             # negatives; 6 matches the baseline methodology used across
                                             # every model comparison so far -- see shortlist-*.txt's own
                                             # "passed_at_try/6" annotations)
MAX_TOKENS="${MAX_TOKENS:-$_EFFORT_MAX_TOKENS}"  # hard cap on thinking+answer tokens (default 16000 at low; see REASONING_EFFORT)
                                             # (6-8k caps truncated legitimate answers on some
                                             # exercises: needed as low as ~4243/try to as much as
                                             # 8486 total observed. 16k gives real headroom to avoid
                                             # false failures from truncation, not model capability.)

# Sampling, sent per request via extra_params (see write_model_settings).
# Without this, aider's default use_temperature=True sends temperature=0 on
# every request, overriding llama-server's --temp -- so every run before
# these existed was greedy decoding, and "reruns" were byte-identical replays
# (see mtp-off-runaway-generation.md). Under greedy, any numerics change (MTP
# on/off, -ub, KV cache type, a new build) flips near-tied tokens and acts
# like an uncontrolled new seed. A fixed SEED makes that variance explicit:
# same SEED + same settings reproduces a run, a different SEED is a new
# independent sample (run several and compare pass rates, not single
# pass/fails). top-p/top-k/min-p aren't sent, so the server's own flags apply
# (captured as srv.--top-p etc. in the TSVs). TEMPERATURE=0 restores the old
# greedy behavior; SEED then has no effect.
# TEMPERATURE=server sends no temperature at all (aider's use_temperature: false),
# so the server's own sampling defaults apply -- the default for Strata, whose
# recommended sampling lives in its run config (read back via /config, see
# capture_strata_settings). llama keeps 0.6 so existing rows stay comparable.
_DEFAULT_TEMPERATURE=0.6
[[ "$TARGET" == "strata" ]] && _DEFAULT_TEMPERATURE=server
TEMPERATURE="${TEMPERATURE:-$_DEFAULT_TEMPERATURE}"
SEED="${SEED:-1}"
if ! [[ "$TEMPERATURE" =~ ^([0-9]+(\.[0-9]+)?|server)$ ]]; then
  echo "ERROR: TEMPERATURE='$TEMPERATURE' is not a non-negative number or 'server'." >&2
  exit 1
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SEED='$SEED' is not a non-negative integer." >&2
  exit 1
fi
# EXTRA_BODY: a JSON object merged into every request body (aider's
# extra_params.extra_body), for toying with sampler knobs the server accepts
# but run.sh has no dedicated variable for, e.g.
#   EXTRA_BODY='{"top_k":20,"min_p":0.1,"top_p":0.9}'
# Fields the server doesn't know are typically ignored, so check the server's
# captured settings (srv.* columns) if a knob seems to do nothing. It is part
# of the results-dir/settings-file name (a short hash, so runs with different
# bodies never share a dir or satisfy each other's skip checks) and is recorded
# verbatim in the extra_body column of the TSVs.
EXTRA_BODY="${EXTRA_BODY:-}"
EXTRA_SLUG=""
if [[ -n "$EXTRA_BODY" ]]; then
  if ! EXTRA_BODY=$(python3 -I -c '
import json, sys
try:
    d = json.loads(sys.argv[1])
except ValueError as e:
    sys.exit("invalid JSON (%s)" % e)
if not isinstance(d, dict) or not d:
    sys.exit("must be a non-empty JSON object")
bad = {"reasoning_budget_tokens", "provider", "reasoning"} & d.keys()
if bad:
    sys.exit("%s already set by run.sh (REASONING_BUDGET / OR_PROVIDER / OR_REASONING_TOKENS)" % sorted(bad))
print(json.dumps(d, sort_keys=True, separators=(",", ":")))
' "$EXTRA_BODY" 2>&1); then
    echo "ERROR: EXTRA_BODY is not usable: $EXTRA_BODY" >&2
    exit 1
  fi
  EXTRA_SLUG="-x$(printf '%s' "$EXTRA_BODY" | sha1sum | cut -c1-6)"
fi
# Part of every per-run file path (model-settings YAML, fromfile's results
# dir) so runs of the same model at different seeds never share a dir.
SAMPLING_SLUG="temp${TEMPERATURE}-seed${SEED}${EFFORT_SLUG}${EXTRA_SLUG}"
DECODE_TPS="${DECODE_TPS:-25}"              # assumed tok/s, for timeout auto-calc -- kept conservative
                                             # since it needs to cover the slowest configs tested (partial
                                             # CPU-offload "-fit on" models can run ~25-30 tok/s), not just
                                             # full-GPU ones; costs a fast model nothing but timeout headroom
PREFILL_BUFFER="${PREFILL_BUFFER:-120}"     # seconds, headroom for prompt prefill
AUTO_TIMEOUT=$(( (MAX_TOKENS + DECODE_TPS - 1) / DECODE_TPS + PREFILL_BUFFER ))
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-$AUTO_TIMEOUT}"  # seconds, overrides aider's 600s default

# Wall-clock cap PER EXERCISE across all its --tries combined (seconds; 0 =
# disabled, the old unbounded behavior). Guards against aider's own internal
# reflection loop (see patches/patch_benchmark_max_duration.py) burning
# excessive time on a model that keeps tripping the same error without
# converging -- observed: 88 min on one exercise for one slow model, vs.
# every genuine pass/clean-fail across every model tested clearing in under
# 6 min. Default errs above that ceiling, not at it: a model too slow to
# finish inside it isn't a useful comparison point regardless of whether it
# would eventually get there. Once a try pushes cumulative duration past
# this, remaining tries for that exercise are skipped (recorded as
# pass=false, bailed_on_max_duration=true in the TSVs) rather than run at
# full length.
MAX_DUR="${MAX_DUR:-480}"

# Uniform multiplier applied to every max_dur value -- the global default
# above AND any per-exercise override from a fromfile -- so a known-slower
# model family (e.g. a dense 27B running 2-3x slower per-token than the
# 35B-A3B family this project mostly tests) just needs
# `MAX_DUR_SCALE=2.5 ./run.sh ...` once, rather than hand-multiplying every
# line of every shortlist file. 1 = no change (default). Applied at both
# AIDER_BENCHMARK_MAX_DUR export points below via scale_max_dur().
#
# MAX_DUR_SCALE=auto: the MAX_DUR values (here and in the shortlist files) were
# tuned against a ~100-120 tok/s model, so they're too tight for a slower one.
# auto probes the live server's decode speed once per run.sh invocation and
# sets scale = max(1, AUTO_SCALE_BASE_TPS / measured), AUTO_SCALE_BASE_TPS
# default 100 -- never below 1x, so a fast model keeps the stock limits. It
# only corrects for decode speed, not for how many tokens a model chooses to
# generate; the MAX_DUR values themselves already carry that fuzz. If the
# probe fails the run aborts (rather than silently using 1x).
MAX_DUR_SCALE="${MAX_DUR_SCALE:-1}"
AUTO_SCALE_BASE_TPS="${AUTO_SCALE_BASE_TPS:-100}"
scale_max_dur() {
  local s="$MAX_DUR_SCALE"
  [[ "$s" == "auto" ]] && s=1   # not yet resolved (see resolve_auto_max_dur_scale)
  awk -v d="$1" -v s="$s" -v m="$EFFORT_DUR_MULT" 'BEGIN { printf "%.0f", d * s * m }'
}

# measure_decode_tps: in server-capture.sh (shared with harbor/harbor.sh)

resolve_auto_max_dur_scale() {
  [[ "$MAX_DUR_SCALE" == "auto" ]] || return 0
  local tps
  tps="$(measure_decode_tps)"
  if [[ -z "$tps" ]] || ! awk -v t="$tps" 'BEGIN { exit !(t > 0) }'; then
    echo "ERROR: MAX_DUR_SCALE=auto couldn't measure decode speed (see probe errors above). Aborting rather than silently running with 1x limits; set MAX_DUR_SCALE=<number> to override." >&2
    exit 1
  else
    MAX_DUR_SCALE="$(awk -v t="$tps" -v b="$AUTO_SCALE_BASE_TPS" 'BEGIN { s = b / t; if (s < 1) s = 1; printf "%.2f", s }')"
    echo "-- MAX_DUR_SCALE=auto: decode ${tps} tok/s vs ${AUTO_SCALE_BASE_TPS} baseline -> ${MAX_DUR_SCALE}x (min 1x)" >&2
  fi
  export AIDER_BENCHMARK_MAX_DUR="$(scale_max_dur "$MAX_DUR")"
}

# Raises aider's ChatSummary trigger (main_model.max_chat_history_tokens,
# default 1024 -- see patches/patch_benchmark_max_chat_history.py) well above
# what a single --tries-bounded exercise transcript realistically reaches.
# Left at the default, summarize_start() fires a background-threaded extra
# completion call on nearly every turn; against our --parallel 1 llama-server
# that races the next turn's request for the single KV-cache slot and evicts
# its cached prefix, forcing a full prompt reprocess every time (visible as
# "erased invalidated context checkpoint" spam in the server log). 0 restores
# aider's original (slow, here) behavior. Kept comfortably under the
# --fit-ctx 32768 / max_tokens 16000 generation reserve so a genuinely
# oversized transcript still hits a normal context-overflow error instead of
# being silently masked.
MAX_CHAT_HISTORY_TOKENS="${MAX_CHAT_HISTORY_TOKENS:-12000}"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

# Bare `-e AIDER_BENCHMARK_MAX_DUR` (no =value) so docker reads whatever this
# exported var is set to AT THE MOMENT `$DOCKER_RUN ...` actually runs, not
# frozen at this assignment -- lets `fromfile`'s per-exercise-group overrides
# (see the fromfile case below) change it between invocations just by
# reassigning/re-exporting AIDER_BENCHMARK_MAX_DUR before each group's call.
export AIDER_BENCHMARK_MAX_DUR="$(scale_max_dur "$MAX_DUR")"
export AIDER_BENCHMARK_MAX_CHAT_HISTORY_TOKENS="$MAX_CHAT_HISTORY_TOKENS"
DOCKER_RUN="docker compose run --rm -e AIDER_BENCHMARK_MAX_DUR -e AIDER_BENCHMARK_MAX_CHAT_HISTORY_TOKENS benchmark"

mkdir -p "$AIDER_RUNS"

# Snapshot before the docker run starts, so track_per_test/track_aggregate
# can scope their aggregation to files this invocation actually wrote (see
# run_benchmark_and_track below).
RUN_START_TS="$(date +%s)"

# Global (not per-call local): advances across repeated run_benchmark_and_track
# calls within the same script invocation (fromfile's per-group loop calls it
# once per exercise group against the same results dir) so track_per_test's
# mtime-watermark dedup doesn't re-process an earlier group's already-tracked
# exercises as if they were new. Single-call commands (smoke/quick/full/...)
# are unaffected -- same start value, same behavior as before.
TRACK_WATERMARK="$RUN_START_TS"

# ── Result tracking (aider-results.tsv) ──────────────────────────────────────
# After each benchmark run, snapshot the llama-server's current model/sampling
# settings (via `docker inspect`, same trick as bench-replay.sh) alongside
# aider's own eval config and the run's aggregate pass-rate stats, then append
# one row to aider-results.tsv. Columns grow/shrink automatically as flags
# change across runs (dynamic-header merge, same as bench-replay.sh).

TRACK_TSV="${TRACK_TSV:-$DATA_DIR/aider-results.tsv}"
# One row per exercise per run (language, exercise name, pass/fail, tries
# taken, tokens, cost, ...) alongside the same run/model/server settings
# columns as TRACK_TSV, so per-exercise questions ("did model A handle
# zipper better than model B") are a plain filter on one file instead of
# needing to dig through individual .aider.chat.history.md transcripts.
PER_TEST_TSV="${PER_TEST_TSV:-$DATA_DIR/aider-results-per-test.tsv}"
# Server settings capture (detect_llama_container, capture_srv_settings,
# strata_settings_effective, ...): shared with harbor/harbor.sh
source "$EVAL_DIR/lib/server-capture.sh"

# Shared dynamic-header TSV writer, embedded in both track_per_test and
# track_aggregate below (each is its own `python3 -c`, a fresh interpreter,
# so nothing to import from -- duplicated on purpose rather than factored
# into a helper .py file, to keep every track_* function self-contained).
# STATIC columns first, then whatever keys in new_rows aren't in
# static_cols/stat_cols (settings -- grows/shrinks as flags change run to
# run), then any such columns only the *existing* file had (so old rows keep
# their data), then the fixed STAT columns last.
_WRITE_TSV_PY='
def write_tsv(tsv_file, static_cols, stat_cols, new_rows):
    # Whole read-modify-write cycle happens under an exclusive lock on a
    # sibling .lock file (not the TSV itself, so a reader never has to worry
    # about lock semantics) -- without this, two concurrent runs (e.g. a
    # local-model run and an OpenRouter run at the same time) polling on
    # their own timers can each read the same N-row file, both compute N+1,
    # and whichever writes last silently overwrites the other'"'"'s new row.
    import fcntl
    os.makedirs(os.path.dirname(tsv_file) or ".", exist_ok=True)
    lock_path = tsv_file + ".lock"
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            static_set, stat_set = set(static_cols), set(stat_cols)
            existing_rows = []
            existing_header = None
            if os.path.isfile(tsv_file):
                with open(tsv_file, newline="") as f:
                    reader = csv.reader(f, delimiter="\t")
                    existing_header = next(reader, None)
                    for r in reader:
                        existing_rows.append(r)

            old_mid_cols = [c for c in (existing_header or []) if c not in static_set and c not in stat_set]
            seen_mid = set(old_mid_cols)
            new_mid_cols = []
            for nrow in new_rows:
                for k in nrow:
                    if k not in static_set and k not in stat_set and k not in seen_mid:
                        new_mid_cols.append(k)
                        seen_mid.add(k)
            new_header = static_cols + new_mid_cols + old_mid_cols + stat_cols

            if existing_header and existing_header != new_header:
                rebuilt = []
                for old_row in existing_rows:
                    old_dict = dict(zip(existing_header, old_row))
                    rebuilt.append([old_dict.get(c, "") for c in new_header])
                existing_rows = rebuilt

            with open(tsv_file, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t", lineterminator="\n")
                writer.writerow(new_header)
                for r in existing_rows:
                    writer.writerow(r)
                for nrow in new_rows:
                    writer.writerow([nrow.get(c, "") for c in new_header])
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
'

# Append per-test rows for any .aider.results.json with mtime > since_ts
# (strictly greater, so repeated calls never reprocess the same file) to
# PER_TEST_TSV, then print the new watermark (max mtime just processed, or
# the unchanged since_ts if nothing was new) as the ONLY thing on stdout --
# so `watermark="$(track_per_test ...)"` captures it cleanly. Meant to be
# called repeatedly while a run is still in progress (see
# run_benchmark_and_track below) so results survive a mid-run crash instead
# of only landing once at the very end. Safe/idempotent to call any number
# of times, including with no new files (a no-op, unchanged watermark).
track_per_test() {
  local cmd_name="$1" run_dir="$2" edit_format="$3" since_ts="$4" srv_file="$5"
  if [[ ! -d "$run_dir" ]]; then
    echo "$since_ts"
    return
  fi
  python3 -c "
import csv, json, os, sys
from pathlib import Path
from datetime import datetime, timezone

$_WRITE_TSV_PY

per_test_tsv_file, run_dir, cmd_name, model, srv_file, since_ts, px_log = sys.argv[1:8]
since_ts = float(since_ts)
run_dir = Path(run_dir)
aider_settings = {}
for kv in sys.argv[8:]:
    k, _, v = kv.partition('=')
    aider_settings[k] = v

settings = dict(aider_settings)
if os.path.isfile(srv_file):
    with open(srv_file) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                settings[k] = v

results = []
max_mtime = since_ts
for fname in run_dir.glob('*/exercises/practice/*/.aider.results.json'):
    try:
        mtime = fname.stat().st_mtime
        if mtime <= since_ts:
            continue  # already picked up by an earlier poll
        data = json.loads(fname.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    rel = fname.relative_to(run_dir).parts  # (language, 'exercises', 'practice', exercise, '.aider.results.json')
    data['_language'] = rel[0]
    data['_exercise'] = rel[3] if len(rel) > 4 else fname.parent.name
    data['_mtime'] = mtime
    results.append(data)
    max_mtime = max(max_mtime, mtime)

if not results:
    print(since_ts)
    sys.exit(0)

# eval proxy rows (results/aider-eval-proxy.tsv) for one exercise: its X-Eval-Session tag,
# within the exercise's own time window (a rerun into the same dir adds new rows)
px_rows = []
if os.path.isfile(px_log):
    with open(px_log, newline='') as f:
        px_rows = list(csv.DictReader(f, delimiter='\t'))
def px_stats(run_name, lang, ex, end_ts, duration):
    tag = f'{run_name}/{lang}/{ex}'
    lo = end_ts - duration - 120
    mine = []
    for row in px_rows:
        if row.get('session') != tag:
            continue
        try:
            ts = datetime.fromisoformat(row['timestamp']).timestamp()
        except (KeyError, ValueError):
            continue
        if lo <= ts <= end_ts + 60:
            mine.append(row)
    if not mine:
        return {}
    num = lambda row, k: int(row[k]) if (row.get(k) or '').isdigit() else 0
    uniq = lambda k: ','.join(sorted({row[k] for row in mine if row.get(k)}))
    return {'px_requests': str(len(mine)),
            'px_output_tokens': str(sum(num(row, 'output_tokens') for row in mine)),
            'px_reasoning_tokens': str(sum(num(row, 'reasoning_tokens') for row in mine)),
            'px_answer_tokens': str(sum(num(row, 'answer_tokens') for row in mine)),
            'px_budget_cutoffs': str(sum(num(row, 'budget_cutoff') for row in mine)),
            'px_effort': uniq('effort'), 'px_reasoning_budget': uniq('reasoning_budget'),
            # time the model spent on requests (incl. prompt processing): tok/s without test runs
            'px_model_sec': str(round(sum(num(row, 'duration_ms') for row in mine) / 1000))}

# Counts from the exercise's .aider.chat.history.md: aider's own counters mix these up
# (num_user_asks = its auto-answered confirm prompts, num_error_outputs = server AND
# model errors), and the transcript is overwritten when the run dir is reused, so it's
# captured here. Same patterns as results/viz/outcomes.py HISTORY_PATTERNS.
import re
HIST_PATTERNS = {
    'hist_auth_errors':   r'AuthenticationError',
    'hist_server_errors': r'ServiceUnavailableError|Loading model|InternalServerError|Connection error|APIConnectionError',
    'hist_context':       r'exceeds the available context|possibly exhausted context window',
    'hist_output_limit':  r'possibly exceeded output limit',
    'hist_edit_format':   r'did not conform to the edit format',
    'hist_lint':          r'Attempt to fix lint errors\?',
    'hist_wrong_file':    r'Create new file\?|Allow edits to file that has not been added',
}
def hist_stats(lang, ex):
    h = run_dir / lang / 'exercises' / 'practice' / ex / '.aider.chat.history.md'
    try:
        text = h.read_text(errors='replace')
    except OSError:
        return {}
    return {k: str(len(re.findall(p, text))) for k, p in HIST_PATTERNS.items()}

timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
per_test_rows = []
for r in results:
    outcomes = r.get('tests_outcomes', [])
    passed = bool(outcomes) and outcomes[-1]
    t_row = {
        'timestamp': timestamp,
        'run_cmd': cmd_name,
        'language': r.get('_language', ''),
        'exercise': r.get('_exercise', ''),
        'model': model,
    }
    t_row.update(settings)
    t_row['pass'] = 'true' if passed else 'false'
    t_row['tries_taken'] = str(len(outcomes))
    t_row['passed_at_try'] = str(len(outcomes)) if passed else ''
    t_row['cost'] = f'{r.get(\"cost\", 0) or 0:.4f}'
    t_row['duration'] = f'{r.get(\"duration\", 0) or 0:.1f}'
    t_row['error_outputs'] = str(r.get('num_error_outputs', 0) or 0)
    t_row['num_malformed_responses'] = str(r.get('num_malformed_responses', 0) or 0)
    t_row['syntax_errors'] = str(r.get('syntax_errors', 0) or 0)
    t_row['indentation_errors'] = str(r.get('indentation_errors', 0) or 0)
    t_row['user_asks'] = str(r.get('num_user_asks', 0) or 0)
    t_row['lazy_comments'] = str(r.get('lazy_comments', 0) or 0)
    t_row['exhausted_context_windows'] = str(r.get('num_exhausted_context_windows', 0) or 0)
    t_row['bailed_on_max_duration'] = 'true' if r.get('bailed_on_max_duration') else 'false'
    t_row['test_timeouts'] = str(r.get('test_timeouts', 0) or 0)
    t_row['prompt_tokens'] = str(r.get('prompt_tokens', 0) or 0)
    t_row['completion_tokens'] = str(r.get('completion_tokens', 0) or 0)
    t_row['aider_commit'] = r.get('commit_hash', '') or ''
    t_row.update(px_stats(run_dir.name, r.get('_language', ''), r.get('_exercise', ''), r.get('_mtime', 0), r.get('duration', 0) or 0))
    t_row.update(hist_stats(r.get('_language', ''), r.get('_exercise', '')))
    per_test_rows.append(t_row)

TEST_STATIC = ['timestamp', 'run_cmd', 'language', 'exercise', 'model']
TEST_STATS = ['pass', 'passed_at_try', 'tries_taken', 'cost', 'duration',
              'error_outputs', 'num_malformed_responses', 'syntax_errors',
              'indentation_errors', 'user_asks', 'lazy_comments',
              'exhausted_context_windows', 'bailed_on_max_duration', 'test_timeouts',
              'prompt_tokens', 'completion_tokens', 'aider_commit',
              'px_requests', 'px_output_tokens', 'px_reasoning_tokens', 'px_answer_tokens',
              'px_budget_cutoffs', 'px_effort', 'px_reasoning_budget', 'px_model_sec',
              'hist_auth_errors', 'hist_server_errors', 'hist_context', 'hist_output_limit',
              'hist_edit_format', 'hist_lint', 'hist_wrong_file']
write_tsv(per_test_tsv_file, TEST_STATIC, TEST_STATS, per_test_rows)

names = ', '.join(r['exercise'] for r in per_test_rows)
print(f'  +{len(per_test_rows)} per-test row(s): {names}', file=sys.stderr)
print(max_mtime)
" "$PER_TEST_TSV" "$run_dir" "$cmd_name" "$AIDER_MODEL" "$srv_file" "$since_ts" "$EVAL_PROXY_LOG" \
      "edit_format=$edit_format" "tries=$TRIES" "max_tokens=$MAX_TOKENS" \
      "reasoning_effort=$REASONING_EFFORT" "request_timeout=$REQUEST_TIMEOUT" \
      "target=$TARGET" "temperature=$TEMPERATURE" "seed=$SEED" "or_provider=$OR_PROVIDER" "or_reasoning_tokens=$OR_REASONING_TOKENS" "extra_body=$EXTRA_BODY"
}

# Aggregate summary row over the FULL run range [since_ts, now] into
# TRACK_TSV, independent of how many times track_per_test was already called
# during the run -- always recomputed fresh from every matching
# .aider.results.json in range, so it's correct whether the run finished
# cleanly or was interrupted partway through. Call once, at the end,
# regardless of the benchmark run's own exit status (see
# run_benchmark_and_track) -- a crashed/interrupted run still gets a summary
# of whatever it actually completed, instead of no row at all.
track_aggregate() {
  local cmd_name="$1" run_dir="$2" edit_format="$3" since_ts="$4" srv_file="$5"
  if [[ ! -d "$run_dir" ]]; then
    echo "track_aggregate: no results dir $run_dir, skipping" >&2
    return
  fi

  python3 -c "
import csv, json, os, sys
from pathlib import Path
from datetime import datetime, timezone

$_WRITE_TSV_PY

tsv_file, run_dir, cmd_name, model, srv_file, since_ts = sys.argv[1:7]
since_ts = float(since_ts)
run_dir = Path(run_dir)
aider_settings = {}
for kv in sys.argv[7:]:
    k, _, v = kv.partition('=')
    aider_settings[k] = v

settings = dict(aider_settings)
if os.path.isfile(srv_file):
    with open(srv_file) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                settings[k] = v

results = []
for fname in run_dir.glob('*/exercises/practice/*/.aider.results.json'):
    try:
        if fname.stat().st_mtime <= since_ts:
            continue  # leftover from a prior run of this same command
        results.append(json.loads(fname.read_text()))
    except (json.JSONDecodeError, OSError):
        continue

total_tests = len(list(run_dir.glob('*/exercises/practice/*')))
completed = 0
tries = max((len(r.get('tests_outcomes', [])) for r in results), default=0)
passed_by_try = [0] * tries
cost = duration = 0.0
error_outputs = malformed = with_malformed = syntax_errors = test_timeouts = exhausted = 0
user_asks = lazy_comments = indentation_errors = 0
prompt_tokens = completion_tokens = 0
aider_commits = set()

for r in results:
    completed += 1
    outcomes = r.get('tests_outcomes', [])
    if outcomes and outcomes[-1]:
        for i in range(len(outcomes) - 1, tries):
            passed_by_try[i] += 1
    cost += r.get('cost', 0) or 0
    duration += r.get('duration', 0) or 0
    test_timeouts += r.get('test_timeouts', 0) or 0
    error_outputs += r.get('num_error_outputs', 0) or 0
    exhausted += r.get('num_exhausted_context_windows', 0) or 0
    nm = r.get('num_malformed_responses', 0) or 0
    malformed += nm
    if nm:
        with_malformed += 1
    syntax_errors += r.get('syntax_errors', 0) or 0
    indentation_errors += r.get('indentation_errors', 0) or 0
    user_asks += r.get('num_user_asks', 0) or 0
    if r.get('commit_hash'):
        aider_commits.add(r['commit_hash'])
    lazy_comments += r.get('lazy_comments', 0) or 0
    prompt_tokens += r.get('prompt_tokens', 0) or 0
    completion_tokens += r.get('completion_tokens', 0) or 0

timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

if completed == 0:
    print(f'[{timestamp}] track_aggregate: {cmd_name} produced no results under {run_dir}'
          f' -- skipping row, not writing to {tsv_file}')
    sys.exit(0)

row = {
    'timestamp': timestamp,
    'run_cmd': cmd_name,
    'run_dir': str(run_dir),
    'model': model,
}
row.update(settings)

row['total_tests'] = str(total_tests)
row['completed_tests'] = str(completed)
# One column per actual try (pass_rate_1..pass_rate_N, N = tries_observed),
# not just first/last -- TRIES is user-configurable per run.sh invocation, so
# the column count varies run to run; these fall into the same dynamic
# grows/shrinks-across-rows bucket as the srv.* columns below, rather than
# the fixed STATS list, since their names/count are not known in advance.
if completed and tries:
    for i in range(tries):
        row[f'pass_rate_{i + 1}'] = f'{100 * passed_by_try[i] / completed:.1f}'
    row['percent_cases_well_formed'] = f'{100 * (1 - with_malformed / completed):.1f}'
else:
    row['percent_cases_well_formed'] = ''
row['tries_observed'] = str(tries)
row['error_outputs'] = str(error_outputs)
row['num_malformed_responses'] = str(malformed)
row['num_with_malformed_responses'] = str(with_malformed)
row['user_asks'] = str(user_asks)
row['lazy_comments'] = str(lazy_comments)
row['syntax_errors'] = str(syntax_errors)
row['indentation_errors'] = str(indentation_errors)
row['exhausted_context_windows'] = str(exhausted)
row['prompt_tokens'] = str(prompt_tokens)
row['completion_tokens'] = str(completion_tokens)
row['test_timeouts'] = str(test_timeouts)
row['total_cost'] = f'{cost:.4f}'
row['seconds_per_case'] = f'{duration / completed:.1f}' if completed else ''
# aider's own short git commit hash (e.g. 'a1b2c3d', or 'a1b2c3d-dirty' if
# the container's aider checkout had local changes) -- more precise than a
# version string, and free: benchmark.py already stamps it into every
# .aider.results.json (see run_test_real's commit_hash= field), no extra
# docker exec needed. Joined with ',' in the rare case a single run somehow
# spans more than one aider commit, so drift is visible rather than silently
# picking one.
row['aider_commit'] = ','.join(sorted(aider_commits))

STATIC = ['timestamp', 'run_cmd', 'run_dir', 'model']
STATS = ['total_tests', 'completed_tests', 'percent_cases_well_formed',
         'tries_observed', 'error_outputs', 'num_malformed_responses',
         'num_with_malformed_responses', 'user_asks', 'lazy_comments',
         'syntax_errors', 'indentation_errors', 'exhausted_context_windows',
         'prompt_tokens', 'completion_tokens', 'test_timeouts', 'total_cost',
         'seconds_per_case', 'aider_commit']
write_tsv(tsv_file, STATIC, STATS, [row])

# Console summary shows NEW passes per try (a delta), not the cumulative
# pass_rate_N stored in the TSV -- cumulative is the right thing to persist
# (lets later analysis ask: solved within a budget of N retries?), but as a
# printed line it's misleading at a glance: once nothing else passes, it just
# repeats the same percentage for every remaining try (e.g. 16 tries with 1
# holdout exercise prints '80.0' fourteen times in a row, which reads like a
# bug even though it's mathematically correct). Trim the trailing all-zero
# tail so a stuck exercise doesn't pad the line with as many entries as TRIES.
pass_delta_by_try = [passed_by_try[i] - (passed_by_try[i - 1] if i > 0 else 0) for i in range(tries)]
last_nonzero = max((i for i in range(tries) if pass_delta_by_try[i] > 0), default=-1)
if last_nonzero == -1:
    passed_at_try_str = '(none passed)'
else:
    shown = ' '.join(f'{i + 1}:{pass_delta_by_try[i]}' for i in range(last_nonzero + 1))
    passed_at_try_str = shown if last_nonzero == tries - 1 else f'{shown} (0 more through try {tries})'

print(f'Eval row appended: {tsv_file}')
print(f'  [{row[\"timestamp\"]}] {cmd_name} model={model} seed={row[\"seed\"]}: '
      f'completed={row[\"completed_tests\"]}/{row[\"total_tests\"]} '
      f'passed={sum(pass_delta_by_try)}/{completed} new-pass-by-try={passed_at_try_str} '
      f'well_formed={row[\"percent_cases_well_formed\"]}% cost=\${row[\"total_cost\"]} '
      f'sec/case={row[\"seconds_per_case\"]}')
" "$TRACK_TSV" "$run_dir" "$cmd_name" "$AIDER_MODEL" "$srv_file" "$since_ts" \
      "edit_format=$edit_format" "tries=$TRIES" "max_tokens=$MAX_TOKENS" \
      "reasoning_effort=$REASONING_EFFORT" "request_timeout=$REQUEST_TIMEOUT" \
      "target=$TARGET" "temperature=$TEMPERATURE" "seed=$SEED" "or_provider=$OR_PROVIDER" "or_reasoning_tokens=$OR_REASONING_TOKENS" "extra_body=$EXTRA_BODY"
}

# Runs benchmark.py in the background, polling every POLL_INTERVAL seconds
# for newly-completed exercises and appending them to PER_TEST_TSV
# immediately (track_per_test above) -- so a mid-run crash, Ctrl-C, or OOM
# doesn't lose results for exercises that already finished, which happens
# with a single end-of-run tracking pass because `set -e` skips it entirely
# on a non-zero exit. track_aggregate still runs exactly once, after the
# background job exits, over the FULL run range, regardless of its exit
# status. $1=run_dir $2=cmd_name $3=edit_format, remaining args are
# benchmark.py's own args (everything from "/results/<name>" onward).
# Returns benchmark.py's real exit status (propagated via `return`, not
# swallowed) so callers/set -e still see a real failure as a failure.
POLL_INTERVAL="${POLL_INTERVAL:-15}"

# Regenerates results/viz/eval-viz.html from PER_TEST_TSV; called whenever new per-test
# rows land. Best-effort: a viz failure must never fail or stall the benchmark run.
# VIZ_REFRESH=0 disables it.
VIZ_REFRESH="${VIZ_REFRESH:-1}"
refresh_viz() {
  [[ "$VIZ_REFRESH" == "0" ]] && return 0
  local dir
  dir="$(dirname "$PER_TEST_TSV")/viz"
  [[ -f "$dir/generate_viz_data.py" ]] || return 0
  (cd "$dir" && python3 generate_viz_data.py >/dev/null 2>&1) \
    || echo "WARNING: viz refresh failed (run generate_viz_data.py in $dir to see why)" >&2
  return 0
}
_CURRENT_DOCKER_PID=""
_CURRENT_SRV_FILE=""
_on_benchmark_interrupt() {
  echo "" >&2
  echo "Interrupted -- stopping benchmark container (results so far are already saved)..." >&2
  [[ -n "$_CURRENT_DOCKER_PID" ]] && kill "$_CURRENT_DOCKER_PID" 2>/dev/null
  [[ -n "$_CURRENT_DOCKER_PID" ]] && wait "$_CURRENT_DOCKER_PID" 2>/dev/null
  # Killing the local `docker compose run` CLI process (above) does NOT
  # reliably stop the actual container it spawned -- confirmed directly:
  # since run_benchmark_and_track backgrounds it with `&`, the CLI isn't the
  # terminal's real attached foreground process, so it doesn't forward the
  # signal to stop the container the way a normal interactive `docker compose
  # run` would. Observed firsthand: the CLI process exits, but the container
  # is left behind (seen sitting in a "Created", never-running, never-cleaned-
  # up state despite --rm). Force-remove anything still around from this
  # project by name pattern as a hard guarantee Ctrl+C actually stops it,
  # not just the wrapper.
  docker ps -aq --filter "name=aider-benchmark-benchmark-run" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1
  [[ -n "$_CURRENT_SRV_FILE" ]] && rm -f "$_CURRENT_SRV_FILE"
  exit 130
}
trap _on_benchmark_interrupt INT TERM

# Eval proxy (lib/eval_proxy.py, shared with harbor.sh): sits between the aider
# container and LLAMA_BASE_URL and logs one row per request to results/aider-eval-proxy.tsv
# with what the server says it did (Strata's strata_effective: effort, thinking
# budget, reasoning/answer tokens, budget cut-off). The benchmark tags each
# exercise's requests with X-Eval-Session (patches/patch_benchmark_session_header.py),
# so track_per_test can add per-exercise px_* columns. Requests reach the upstream
# unchanged. Started once per run.sh invocation on 127.0.0.1 (the container uses
# host networking); skipped for remote MODEL=x/y specs and with EVAL_PROXY=0.
_RUN_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PROXY_LOG="$DATA_DIR/aider-eval-proxy.tsv"
_EVAL_PROXY_PID=""
start_eval_proxy() {
  [[ -n "$_EVAL_PROXY_PID" || "$MODEL" == */* || "${EVAL_PROXY:-1}" == 0 ]] && return 0
  local out port
  out="$(mktemp)"
  python3 "$EVAL_DIR/lib/eval_proxy.py" --bind 127.0.0.1 --upstream "${LLAMA_BASE_URL%/v1}" \
    --log "$EVAL_PROXY_LOG" > "$out" 2>&1 &
  _EVAL_PROXY_PID=$!
  for _ in $(seq 50); do grep -q listening "$out" && break; sleep 0.1; done
  port="$(awk '/listening/ {print $2}' "$out")"
  if [[ -z "$port" ]]; then
    echo "WARNING: eval proxy failed to start, running without it: $(cat "$out")" >&2
    kill "$_EVAL_PROXY_PID" 2>/dev/null; _EVAL_PROXY_PID=""
  else
    export LLAMA_BASE_URL="http://127.0.0.1:$port/v1"
  fi
  rm -f "$out"
}
trap '[[ -n "$_EVAL_PROXY_PID" ]] && kill "$_EVAL_PROXY_PID" 2>/dev/null' EXIT

run_benchmark_and_track() {
  local run_dir="$1" cmd_name="$2" edit_format="$3"
  shift 3
  start_eval_proxy

  local container srv_file
  srv_file="$(mktemp)"
  _CURRENT_SRV_FILE="$srv_file"
  # MODEL with a '/' bypasses the local llama-server (e.g. openrouter/...), so
  # whatever container happens to be running isn't serving this run -- leave
  # srv_file empty rather than record its unrelated alias/flags as srv.* columns.
  if [[ "$MODEL" != */* ]]; then
    if [[ "$TARGET" == "strata" ]]; then
      strata_settings_effective > "$srv_file"
    else
      container="$(detect_llama_container)"
      llama_settings_effective "$container" > "$srv_file"
    fi
    # The server flag is only the fallback; the per-request budget is what the
    # model actually ran under, so record that in the same column. Keeps one
    # column meaning "thinking cap in effect" across old and new rows.
    if [[ -n "$(local_reasoning_budget)" ]]; then
      sed -i '/^srv\.--reasoning-budget=/d' "$srv_file"
      echo "srv.--reasoning-budget=$(local_reasoning_budget)" >> "$srv_file"
    fi
  fi

  $DOCKER_RUN ./benchmark/benchmark.py "$@" &
  local docker_pid=$!
  _CURRENT_DOCKER_PID="$docker_pid"

  # External stall watchdog: patch_benchmark_try_timeout.py's in-process
  # SIGALRM guard can be silently defeated when litellm's own broad
  # except-and-retry logic catches the raised TryTimedOut exception before it
  # ever escapes coder.run() -- observed firsthand: an uncapped
  # connection-error retry loop ran for 6+ hours on a single exercise despite
  # AIDER_BENCHMARK_MAX_DUR being set correctly (confirmed via the printed
  # "fromfile group: ... max_dur=480s" line), because the interrupt never
  # made it out of litellm's own retry frame. This is a hard, external
  # backstop that can't be defeated the same way: if no new exercise result
  # has landed for stall_ceiling seconds, force-kill the container outright.
  local stall_ceiling=$(( ${AIDER_BENCHMARK_MAX_DUR:-$MAX_DUR} * 4 ))
  [[ $stall_ceiling -lt 1800 ]] && stall_ceiling=1800
  local last_progress_ts
  last_progress_ts=$(date +%s)

  while kill -0 "$docker_pid" 2>/dev/null; do
    sleep "$POLL_INTERVAL"
    local prev_watermark="$TRACK_WATERMARK"
    TRACK_WATERMARK="$(track_per_test "$cmd_name" "$run_dir" "$edit_format" "$TRACK_WATERMARK" "$srv_file")"
    if [[ "$TRACK_WATERMARK" != "$prev_watermark" ]]; then
      last_progress_ts=$(date +%s)
      refresh_viz
    elif (( $(date +%s) - last_progress_ts > stall_ceiling )); then
      echo "" >&2
      echo "WARNING: no exercise progress for ${stall_ceiling}s -- assuming this group is stuck" >&2
      echo "(e.g. an unbounded connection-error retry loop the in-process timeout guard didn't" >&2
      echo "catch). Force-killing this group's container; run.sh will move on to the next group." >&2
      kill "$docker_pid" 2>/dev/null
      docker ps -aq --filter "name=aider-benchmark-benchmark-run" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1
      break
    fi
  done

  local docker_status=0
  wait "$docker_pid" || docker_status=$?
  _CURRENT_DOCKER_PID=""

  # Catch anything that finished between the last poll and process exit.
  TRACK_WATERMARK="$(track_per_test "$cmd_name" "$run_dir" "$edit_format" "$TRACK_WATERMARK" "$srv_file")"
  track_aggregate "$cmd_name" "$run_dir" "$edit_format" "$RUN_START_TS" "$srv_file"
  refresh_viz

  rm -f "$srv_file"
  _CURRENT_SRV_FILE=""
  return "$docker_status"
}

# Generate model-settings YAML matching this run's model name so
# extra_params.timeout actually applies (aider matches by exact name).
# Takes the edit format to record (defaults to $EDIT_FORMAT) so this file
# never disagrees with whatever --edit-format actually gets passed to
# benchmark.py for this invocation.
# Split "$@" into EXERCISE_ARGS (bare words before the first '-'-prefixed
# token -- exercise-name substrings) and PASSTHRU_ARGS (that flag and
# everything after it, forwarded to benchmark.py as-is, e.g. --clean/
# --resume/other benchmark.py flags). Positional-args-then-flags, like any
# normal CLI -- avoids needing to know which flags take a value.
split_exercise_args() {
  EXERCISE_ARGS=()
  PASSTHRU_ARGS=()
  local seen_flag=0 a
  for a in "$@"; do
    if [[ $seen_flag -eq 1 ]]; then
      PASSTHRU_ARGS+=("$a")
    elif [[ "$a" == -* ]]; then
      seen_flag=1
      PASSTHRU_ARGS+=("$a")
    else
      EXERCISE_ARGS+=("$a")
    fi
  done
}

# Join newline-separated stdin into a single comma-separated string (no
# external `paste` dependency).
join_lines_comma() {
  local out="" line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ -z "$out" ]]; then out="$line"; else out="$out,$line"; fi
  done
  echo "$out"
}

# The 6 polyglot-benchmark language dirs (see README: "225 Exercism
# exercises across 6 languages"). Stable repo structure, safe to hardcode.
POLYGLOT_LANGUAGES="cpp go java javascript python rust"

EXERCISE_LIST_CACHE_DIR="$AIDER_RUNS/.exercise-lists"

# Real exercise directory names for a language, queried once from inside the
# container (there's no way to know these ahead of time without asking the
# image what polyglot-benchmark actually contains -- it's not mounted on the
# host) and cached to disk so repeat runs don't pay the container-startup
# cost again. Delete runs/aider/.exercise-lists/ to force a re-list (e.g. after
# rebuilding the image against a newer polyglot-benchmark).
list_exercises() {
  local lang="$1"
  local cache="$EXERCISE_LIST_CACHE_DIR/$lang.txt"
  if [[ ! -s "$cache" ]]; then
    mkdir -p "$EXERCISE_LIST_CACHE_DIR"
    echo "Listing $lang exercises (first time only, cached after this)..." >&2
    $DOCKER_RUN bash -c "find /benchmarks/polyglot-benchmark/$lang/exercises/practice -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort" > "$cache" 2>/dev/null
    if [[ ! -s "$cache" ]]; then
      rm -f "$cache"
      echo "ERROR: failed to list $lang exercises (bad language name, or image not built yet?)" >&2
      exit 1
    fi
  fi
  cat "$cache"
}

# First N exercise names for a language, sorted alphabetically -- the fixed,
# deterministic default set used when no exercise names are given explicitly.
default_exercises() {
  list_exercises "$1" | head -n "$2"
}

# Error out clearly (with the real exercise list) if any named exercise
# doesn't substring-match anything for $lang, rather than silently running 0
# exercises or letting benchmark.py fail 30+ seconds in.
validate_exercises() {
  local lang="$1"
  shift
  local all_exercises name found ex
  all_exercises="$(list_exercises "$lang")"
  for name in "$@"; do
    found=0
    while IFS= read -r ex; do
      [[ "$ex" == *"$name"* ]] && { found=1; break; }
    done <<< "$all_exercises"
    if [[ $found -eq 0 ]]; then
      echo "ERROR: no $lang exercise matches '$name'." >&2
      echo "Available $lang exercises:" >&2
      echo "$all_exercises" | tr '\n' ' ' >&2
      echo "" >&2
      exit 1
    fi
  done
}

# The per-request thinking budget, or nothing: only local runs (MODEL without a
# '/') talk to a llama-server that understands reasoning_budget_tokens.
local_reasoning_budget() {
  [[ "$MODEL" == */* ]] && return 0
  printf '%s' "$REASONING_BUDGET"
}

write_model_settings() {
  local fmt="${1:-$EDIT_FORMAT}"
  # Per-model filename (not a single shared runs/aider/.model-settings.yml) so
  # two runs against different models don't overwrite each other's timeout/
  # max_tokens mid-run -- aider matches entries by exact model name, and a
  # shared file means whichever run wrote last silently wins for every other
  # run's subsequent container launches. Also: "name:" must be AIDER_MODEL,
  # not "openai/$MODEL" -- for a MODEL already containing '/' (e.g.
  # openrouter/<vendor>/<slug>), the old hardcoded openai/ prefix never
  # matched the real model name, so extra_params silently never applied.
  local f="$AIDER_RUNS/.model-settings-${MODEL_SLUG}-${SAMPLING_SLUG}.yml"
  cat > "$f" <<EOF
- name: $AIDER_MODEL
  edit_format: $fmt
  use_repo_map: false
  use_system_prompt: true
  streaming: false
EOF
  # use_temperature: false stops aider sending any temperature (its default is
  # temperature=0, which would override the server's sampling defaults).
  if [[ "$TEMPERATURE" == "server" ]]; then
    echo "  use_temperature: false" >> "$f"
  fi
  {
    echo "  extra_params:"
    echo "    timeout: $REQUEST_TIMEOUT"
    echo "    max_tokens: $MAX_TOKENS"
    [[ "$TEMPERATURE" != "server" ]] && echo "    temperature: $TEMPERATURE"
    echo "    seed: $SEED"
  } >> "$f"
  if [[ -n "$EXTRA_BODY" || -n "$OR_PROVIDER" || -n "$OR_REASONING_TOKENS" || -n "$(local_reasoning_budget)" ]]; then
    echo "    extra_body:" >> "$f"
    if [[ -n "$EXTRA_BODY" ]]; then
      # JSON scalars/arrays/objects are valid YAML flow values, so dump as-is.
      python3 -I -c '
import json, sys
for k, v in json.loads(sys.argv[1]).items():
    print("      %s: %s" % (json.dumps(k), json.dumps(v)))
' "$EXTRA_BODY" >> "$f"
    fi
    if [[ -n "$(local_reasoning_budget)" ]]; then
      echo "      reasoning_budget_tokens: $(local_reasoning_budget)" >> "$f"
    fi
    if [[ -n "$OR_PROVIDER" ]]; then
      local order
      order="[$(echo "$OR_PROVIDER" | sed 's/[^,]\+/"&"/g; s/,/, /g')]"
      echo "      provider: {order: $order, allow_fallbacks: false}" >> "$f"
    fi
    if [[ -n "$OR_REASONING_TOKENS" ]]; then
      echo "      reasoning: {max_tokens: $OR_REASONING_TOKENS}" >> "$f"
    fi
  fi
}

# Common flags shared by every benchmark invocation
common_flags() {
  local flags=(--read-model-settings "/results/.model-settings-${MODEL_SLUG}-${SAMPLING_SLUG}.yml")
  if [[ -n "$REASONING_EFFORT" ]]; then
    flags+=(--reasoning-effort "$REASONING_EFFORT")
  fi
  # KEYWORDS is set per-case-branch below (from named exercises, or a
  # deterministic default set) before common_flags is called.
  if [[ -n "${KEYWORDS:-}" ]]; then
    flags+=(--keywords "$KEYWORDS")
  fi
  echo "${flags[@]}"
}

# Move a stale results dir aside (non-destructive rename), the way aider's
# own benchmark.py --clean claims to. NOT delegated to benchmark.py's --clean:
# its safety check compares dirname's top-level contents (just the language
# subdirs our copytree loop puts there) against the *entire* polyglot-
# benchmark repo's top-level contents (README, .git, docs, ...) -- those sets
# never match, so --clean always hits "ERROR: will not delete dir that does
# not look like original tests" and *returns* rather than raising, which
# Typer treats as a normal (exit 0) completion -- silent no-op, not a crash,
# so `set -e` in this script never catches it either. Doing the move
# ourselves sidesteps that broken check entirely.
do_clean() {
  local run_dir="$1"
  [[ -d "$run_dir" ]] || return 0
  mkdir -p "$AIDER_RUNS/OLD"
  local dest="$AIDER_RUNS/OLD/$(basename "$run_dir")"
  if [[ -e "$dest" ]]; then
    dest="$AIDER_RUNS/OLD/$(date +%Y-%m-%d-%H-%M-%S)-$(basename "$run_dir")"
  fi
  echo "Moving stale $run_dir -> $dest" >&2
  mv "$run_dir" "$dest"
}

# aider's benchmark.py reuses a results dir across invocations by default
# (caches each exercise's .aider.results.json and returns it unmodified on a
# later run instead of re-testing) -- see run_test_real()'s
# `if results_fname.exists(): return res`. That's a deliberate resume-after-
# crash mechanism for long runs, but it means an unqualified rerun against a
# dirty results dir can silently mix a previous model/config's cached answers
# into what looks like a fresh comparison. Refuse to proceed against a dirty
# dir unless the caller explicitly says which behavior they want:
#   --clean/-c   do_clean() the dir, then run.sh does a normal fresh run.
#   --resume     proceed and let benchmark.py reuse/continue whatever's there.
# Neither flag is forwarded to benchmark.py (--clean is handled here instead,
# per do_clean's comment above). Sets GUARD_ARGS to the remaining args.
guard_reuse() {
  local run_dir="$1"
  shift
  local has_clean=0 has_resume=0 a
  for a in "$@"; do
    case "$a" in
      --clean|-c) has_clean=1 ;;
      --resume) has_resume=1 ;;
    esac
  done
  GUARD_ARGS=()
  for a in "$@"; do
    case "$a" in
      --clean|-c|--resume) ;;
      *) GUARD_ARGS+=("$a") ;;
    esac
  done
  if [[ -d "$run_dir" ]] && find "$run_dir" -name ".aider.results.json" -print -quit 2>/dev/null | grep -q .; then
    if [[ $has_clean -eq 0 && $has_resume -eq 0 ]]; then
      echo "ERROR: $run_dir already has results from a previous run." >&2
      echo "Pass --clean to start fresh (moves old results to runs/aider/OLD/<name>)," >&2
      echo "or --resume to continue/reuse the existing results as-is." >&2
      exit 1
    fi
    # Full if-statement, not `[[ ]] && cmd` -- under set -e, a bare `[[ cond ]] &&
    # cmd` as a function's last executed statement propagates COND's own false/1
    # exit status as the function's return value when cond is false (has_clean=0,
    # i.e. --resume was passed instead of --clean), which silently kills the
    # whole script right here with zero output. An if-statement's own exit
    # status never triggers errexit regardless of which branch runs.
    if [[ $has_clean -eq 1 ]]; then
      do_clean "$run_dir"
    fi
  fi
}

# Same idea as guard_reuse, but for the small/fast commands (smoke, quick,
# polyglot, diff): a stale results dir is always a mistake there (a handful
# of exercises, seconds to redo), so just auto-clean instead of blocking.
# --resume overrides (skips cleaning). Sets GUARD_ARGS.
auto_clean() {
  local run_dir="$1"
  shift
  local has_resume=0 a
  for a in "$@"; do
    [[ "$a" == "--resume" ]] && has_resume=1
  done
  GUARD_ARGS=()
  for a in "$@"; do
    case "$a" in
      --clean|-c|--resume) ;;
      *) GUARD_ARGS+=("$a") ;;
    esac
  done
  [[ $has_resume -eq 0 ]] && do_clean "$run_dir"
}

# Parse a shortlist file for `fromfile`: one exercise per line, either
# "<language>\t<exercise>" or a bare "<exercise>" (matched across all
# languages), optionally followed by per-exercise overrides:
#   <language>  <exercise>  [<max_tries>]  [<max_duration>]
# Either override may be "-" or omitted to fall back to this run's global
# TRIES/MAX_DUR (bare "<exercise>"-only lines, with no language, never take
# overrides -- there's no way to place them without positional ambiguity, and
# these lines are meant for quick ad hoc use anyway). Everything from the
# first '#' onward is a comment (so baseline annotations like "# pass  3
# 141.0" can live on the same line -- see compare_to_baseline below, which
# re-parses those same comments).
#
# Prints one TAB-separated "<keyword>\t<tries>\t<max_dur>" line per exercise
# on stdout (tries/max_dur empty string = use this run's default). <keyword>
# is "<lang>/exercises/practice/<exercise>" for language-qualified lines (an
# exact-enough substring match against benchmark.py's own
# "<lang>/exercises/practice/<exercise>" test-dir keys that won't accidentally
# also pull in other languages), or bare "<exercise>" when no language given.
parse_fromfile_entries() {
  local file="$1" line stripped lang ex tries dur
  while IFS= read -r line || [[ -n "$line" ]]; do
    stripped="${line%%#*}"
    stripped="$(echo "$stripped" | xargs)"
    [[ -z "$stripped" ]] && continue
    read -r lang ex tries dur <<< "$stripped"
    # Emit "-" rather than "" for an unset tries/dur: tab is one of bash's
    # special IFS whitespace characters (space/tab/newline), so `read` collapses
    # a run of delimiters and strips empty fields REGARDLESS of what IFS is set
    # to -- an empty field between two tabs (e.g. "kw\t\t1200") silently
    # vanishes and shifts the next real value into the wrong variable, rather
    # than the consumer's `read -r kw entry_tries entry_dur` seeing tries="" as
    # intended. Observed firsthand: a fromfile line overriding only max_dur
    # (tries left at "-" for the run default) silently ran at the DEFAULT
    # max_dur instead, with the override value shifted into $entry_tries
    # instead -- see the matching "-" normalization this now requires below,
    # where these placeholders get read back.
    [[ -z "$tries" ]] && tries="-"
    [[ -z "$dur" ]] && dur="-"
    if [[ -n "$ex" ]]; then
      printf '%s/exercises/practice/%s\t%s\t%s\n' "$lang" "$ex" "$tries" "$dur"
    else
      printf '%s\t-\t-\n' "$lang"
    fi
  done < "$file"
}

# Bonus feature for `fromfile`: if the shortlist file has baseline
# annotations in its comments (# pass|fail  <tries|->  <duration>, the same
# format this project writes into shortlist-*.txt), print a table showing
# where this run's pass/fail diverged from that recorded baseline --
# read straight from the run's own .aider.results.json files rather than
# PER_TEST_TSV, since it's already scoped to exactly this run_dir with no
# risk of picking up rows from an unrelated run of the same exercise name.
compare_to_baseline() {
  local file="$1" run_dir="$2"
  [[ -d "$run_dir" ]] || return 0
  python3 -c "
import json, re, sys
from pathlib import Path

file, run_dir = sys.argv[1:3]

baseline = {}  # (lang, exercise) -> (pass_bool, tries_or_None)
for raw in Path(file).read_text().splitlines():
    line, _, comment = raw.partition('#')
    parts = line.split()
    if not parts:
        continue
    lang, ex = (parts[0], parts[1]) if len(parts) > 1 else (None, parts[0])
    m = re.match(r'\s*(pass|fail)\s+(\S+)', comment)
    if not m:
        continue
    passed = m.group(1) == 'pass'
    tries = None if m.group(2) == '-' else int(m.group(2))
    baseline[(lang, ex)] = (passed, tries)

if not baseline:
    sys.exit(0)

actual = {}  # (lang, exercise) -> (pass_bool, tries)
for fname in Path(run_dir).glob('*/exercises/practice/*/.aider.results.json'):
    lang = fname.relative_to(run_dir).parts[0]
    ex = fname.parent.name
    try:
        data = json.loads(fname.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    outcomes = data.get('tests_outcomes', [])
    actual[(lang, ex)] = (bool(outcomes) and outcomes[-1], len(outcomes))

def lookup(d, lang, ex):
    if (lang, ex) in d:
        return d[(lang, ex)]
    if (None, ex) in d:
        return d[(None, ex)]
    return None

rows = []
for (lang, ex), (base_pass, base_tries) in baseline.items():
    res = lookup(actual, lang, ex)
    if res is None:
        rows.append((lang or '?', ex, 'pass' if base_pass else 'fail', str(base_tries or '-'), 'MISSING', '-', ''))
        continue
    act_pass, act_tries = res
    flag = ''
    if act_pass != base_pass:
        flag = '<<< DIVERGED'
    elif base_pass and act_pass and base_tries and act_tries != base_tries:
        delta = act_tries - base_tries
        # negative = fewer tries than baseline (better), positive = more (worse)
        flag = f'({delta:+d} tries)'
    rows.append((lang or '?', ex, 'pass' if base_pass else 'fail', str(base_tries or '-'),
                 'pass' if act_pass else 'fail', str(act_tries), flag))

if not rows:
    sys.exit(0)

print()
print('=== Comparison vs baseline in', file, '===')
w = max(len(r[1]) for r in rows)
print(f'{\"language\":<12} {\"exercise\":<{w}} {\"baseline\":<8} {\"try\":<4} {\"this run\":<8} {\"try\":<4}')
diverged = 0
for lang, ex, bp, bt, ap, at, flag in sorted(rows, key=lambda r: (r[6] == '', r[1])):
    if flag == '<<< DIVERGED':
        diverged += 1
    print(f'{lang:<12} {ex:<{w}} {bp:<8} {bt:<4} {ap:<8} {at:<4} {flag}')
print()
print(f'{diverged}/{len(rows)} diverged from baseline pass/fail')
" "$file" "$run_dir"
}

# Shared body for the fixed-python-subset commands (smoke/quick/diff): pick
# either the exercises named on the command line, or this preset's fixed
# deterministic default set of size $default_n (see list_exercises/
# default_exercises above -- NOT random, so repeated runs are directly
# comparable across models/configs). $1=name (results subdir + run_cmd),
# $2=default exercise count, $3=edit format to force ("" = use $EDIT_FORMAT),
# remaining args = this command's args after the command name itself.
run_python_subset() {
  local name="$1" default_n="$2" forced_format="$3"
  shift 3
  split_exercise_args "$@"
  auto_clean "$AIDER_RUNS/$name" "${PASSTHRU_ARGS[@]}"
  local num_tests
  if [[ "${#EXERCISE_ARGS[@]}" -gt 0 ]]; then
    validate_exercises python "${EXERCISE_ARGS[@]}"
    KEYWORDS="$(IFS=,; echo "${EXERCISE_ARGS[*]}")"
    num_tests="${#EXERCISE_ARGS[@]}"
  else
    KEYWORDS="$(default_exercises python "$default_n" | join_lines_comma)"
    num_tests="$default_n"
  fi
  local fmt="${forced_format:-$EDIT_FORMAT}"
  write_model_settings "$fmt"
  run_benchmark_and_track "$AIDER_RUNS/$name" "$name" "$fmt" \
    "/results/$name" \
    --model "$AIDER_MODEL" \
    --edit-format "$fmt" \
    --threads "${THREADS:-1}" \
    --tries "$TRIES" \
    --languages python \
    --num-tests "$num_tests" \
    --exercises-dir /benchmarks/polyglot-benchmark \
    --new \
    $(common_flags) \
    "${GUARD_ARGS[@]}"
}

case "$CMD" in
  smoke|quick|diff|python|polyglot|full|fromfile|custom)
    echo "-- reasoning effort '${REASONING_EFFORT:-<unset>}': thinking budget ${REASONING_BUDGET:-<server default>}$([[ "$MODEL" == */* ]] && echo ' (not sent: remote model)'), MAX_TOKENS ${MAX_TOKENS}, request timeout ${REQUEST_TIMEOUT}s, max_dur x${EFFORT_DUR_MULT}" >&2
    resolve_auto_max_dur_scale
    ;;
esac

case "$CMD" in
  build)
    docker compose build
    ;;

  shell)
    $DOCKER_RUN bash
    ;;

  smoke)
    # 1 exercise (fixed default, or the one(s) you named) — verify the setup works
    run_python_subset smoke 1 "" "${@:2}"
    ;;

  quick)
    # 5 python exercises (fixed default set, or the ones you named) — quick feedback loop
    run_python_subset quick 5 "" "${@:2}"
    ;;

  diff)
    # Same as quick but with diff edit format (tests if model can do search/replace)
    run_python_subset diff 5 diff "${@:2}"
    ;;

  random)
    # TRUE random exercise(s) -- unlike smoke/quick/diff, which are fixed and
    # deterministic on purpose so repeated runs are directly comparable
    # across models/configs. Use this for exploratory/coverage testing instead.
    RANDOM_N="${2:-5}"
    split_exercise_args "${@:3}"
    if [[ "${#EXERCISE_ARGS[@]}" -gt 0 ]]; then
      echo "ERROR: 'random' picks exercises randomly -- naming specific ones" >&2
      echo "(${EXERCISE_ARGS[*]}) defeats the point; use 'quick <names>' instead." >&2
      exit 1
    fi
    auto_clean $AIDER_RUNS/random "${PASSTHRU_ARGS[@]}"
    write_model_settings
    run_benchmark_and_track $AIDER_RUNS/random random "$EDIT_FORMAT" \
      /results/random \
      --model "$AIDER_MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads "${THREADS:-1}" \
      --tries "$TRIES" \
      --languages python \
      --num-tests "$RANDOM_N" \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    ;;

  python)
    split_exercise_args "${@:2}"
    guard_reuse $AIDER_RUNS/python "${PASSTHRU_ARGS[@]}"
    if [[ "${#EXERCISE_ARGS[@]}" -gt 0 ]]; then
      validate_exercises python "${EXERCISE_ARGS[@]}"
      KEYWORDS="$(IFS=,; echo "${EXERCISE_ARGS[*]}")"
    fi
    write_model_settings
    # All python exercises (~30-40 depending on model speed), or just the named ones
    run_benchmark_and_track $AIDER_RUNS/python python "$EDIT_FORMAT" \
      /results/python \
      --model "$AIDER_MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads "${THREADS:-1}" \
      --tries "$TRIES" \
      --languages python \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    ;;

  polyglot)
    split_exercise_args "${@:2}"
    auto_clean $AIDER_RUNS/polyglot "${PASSTHRU_ARGS[@]}"
    if [[ "${#EXERCISE_ARGS[@]}" -gt 0 ]]; then
      KEYWORDS="$(IFS=,; echo "${EXERCISE_ARGS[*]}")"
      POLY_NUM_TESTS="${#EXERCISE_ARGS[@]}"
    else
      # Deterministic default: 3 fixed exercises PER LANGUAGE, 18 total.
      POLY_DEFAULTS=""
      for lang in $POLYGLOT_LANGUAGES; do
        lang_default="$(default_exercises "$lang" 3 | join_lines_comma)"
        if [[ -z "$POLY_DEFAULTS" ]]; then POLY_DEFAULTS="$lang_default"; else POLY_DEFAULTS="$POLY_DEFAULTS,$lang_default"; fi
      done
      KEYWORDS="$POLY_DEFAULTS"
      POLY_NUM_TESTS=18
    fi
    write_model_settings
    # 3 exercises per language, 18 total (or just the named ones)
    run_benchmark_and_track $AIDER_RUNS/polyglot polyglot "$EDIT_FORMAT" \
      /results/polyglot \
      --model "$AIDER_MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads "${THREADS:-1}" \
      --tries "$TRIES" \
      --num-tests "$POLY_NUM_TESTS" \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    ;;

  full)
    split_exercise_args "${@:2}"
    guard_reuse $AIDER_RUNS/full "${PASSTHRU_ARGS[@]}"
    if [[ "${#EXERCISE_ARGS[@]}" -gt 0 ]]; then
      KEYWORDS="$(IFS=,; echo "${EXERCISE_ARGS[*]}")"
    fi
    write_model_settings
    # All 225 exercises across all languages, or just the named ones — the real deal
    run_benchmark_and_track $AIDER_RUNS/full full "$EDIT_FORMAT" \
      /results/full \
      --model "$AIDER_MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads "${THREADS:-1}" \
      --tries "$TRIES" \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    ;;

  fromfile)
    # Run just the exercises listed in a shortlist file (one per line, as
    # "<language> <exercise>" or bare "<exercise>", optionally followed by
    # per-exercise "<max_tries> <max_duration>" overrides; '#' starts a
    # comment). See parse_fromfile_entries/compare_to_baseline above and
    # lists/shortlist-15.txt for the expected format. Bare words after the
    # file path are NOT exercise substrings here (the file already says which
    # exercises to run) -- everything after the path is forwarded as flags,
    # same as --clean/--resume elsewhere.
    #
    # Exercises are grouped by their effective (tries, max_dur) pair -- lines
    # with no override fall into the same group as this run's global
    # TRIES/MAX_DUR -- and each group gets its own benchmark.py invocation
    # against the SAME results dir (safe: benchmark.py's --new/dirname
    # handling is a no-op for the absolute paths run.sh always passes, so
    # repeat calls just add to the existing dir; --keywords scopes each call
    # to that group's exercises, which never overlap across groups).
    FROMFILE_PATH="${2:?Usage: ./run.sh fromfile <file> [--clean|--resume]}"
    # relative to the caller's cwd, else a name under lists/ (".txt" optional)
    for _cand in "$FROMFILE_PATH" "$CALLER_PWD/$FROMFILE_PATH" "$LISTS_DIR/$FROMFILE_PATH" "$LISTS_DIR/$FROMFILE_PATH.txt"; do
      [[ "$_cand" == /* && -f "$_cand" ]] && { FROMFILE_PATH="$_cand"; break; }
    done
    [[ -f "$FROMFILE_PATH" ]] || { echo "ERROR: file not found: $FROMFILE_PATH" >&2; exit 1; }
    # Includes the model so two runs against the SAME shortlist file (e.g.
    # comparing several models overnight) never share a results dir. Without
    # this, two concurrent/sequential `fromfile` runs against different
    # models but the same file name silently collide: both track_per_test
    # polling loops scan the same directory and each stamps its OWN model
    # label onto whatever .aider.results.json files it finds, regardless of
    # which run actually produced them -- corrupting both runs' TSV rows with
    # cross-labeled duplicates (observed firsthand: an OpenRouter run and a
    # concurrent local-model run both targeting shortlist-7.txt).
    mapfile -t FROMFILE_ENTRIES < <(parse_fromfile_entries "$FROMFILE_PATH")
    if [[ "${#FROMFILE_ENTRIES[@]}" -eq 0 ]]; then
      echo "ERROR: no exercises found in $FROMFILE_PATH (expected '<language> <exercise>' or '<exercise>' lines)" >&2
      exit 1
    fi

    # Skip + seed controls for `fromfile` (all env vars; see --help):
    #
    # SKIP_EXISTING picks WHICH finished work is skipped, checked against the
    # whole results history (PER_TEST_TSV), not just this run's own dir. A
    # comma list of scopes, any match skips:
    #   pass       passed at ANY seed -- "has this model ever solved this one"
    #   seed-pass  passed at THIS seed only
    #   seed       has a real attempt (pass or fail) at THIS seed -- resumes a
    #              crashed run without redoing what already failed at this seed
    #   resume     = pass,seed (what SKIP_EXISTING=1 has always meant)
    #   0 / none   skip nothing (default)
    # Stall bail-outs (bailed_on_max_duration=true) never count as attempts.
    #
    # Seeds: SEED=N (default 1) + SEED_TRIES=K runs seeds N..N+K-1; SEEDS=
    # ("2", "1 3", "1-3", "1,2") names them explicitly and wins over both.
    # Every seed except the LOWEST in the list is "gated": it only runs
    # exercises that haven't passed at any seed yet (the point of retrying
    # with a new seed). SEED_ALWAYS lifts that gate for the listed seeds
    # ("2", "2 3", or "all") so they run regardless of earlier passes -- e.g.
    # SEEDS=2 SEED_ALWAYS=2 SKIP_EXISTING=seed fills every missing seed-2
    # record; drop SEED_ALWAYS and use SKIP_EXISTING=pass to fill only the
    # exercises that never passed. A lone/lowest seed is never gated, so use
    # SKIP_EXISTING=pass there if you want fail-only.
    SKIP_EXISTING="${SKIP_EXISTING:-0}"
    SKIP_PASS=0; SKIP_SEED_PASS=0; SKIP_SEED_TRIED=0
    _skip_tokens="${SKIP_EXISTING//,/ }"
    for _t in $_skip_tokens; do
      case "$_t" in
        0|none) ;;
        1|resume) SKIP_PASS=1; SKIP_SEED_TRIED=1 ;;
        pass) SKIP_PASS=1 ;;
        seed-pass) SKIP_SEED_PASS=1 ;;
        seed) SKIP_SEED_TRIED=1 ;;
        *) echo "ERROR: SKIP_EXISTING token '$_t' invalid (use 0|none, pass, seed-pass, seed, resume/1; comma-separated)." >&2; exit 1 ;;
      esac
    done
    SEED_TRIES="${SEED_TRIES:-1}"
    if ! [[ "$SEED_TRIES" =~ ^[0-9]+$ ]] || [[ "$SEED_TRIES" -lt 1 ]]; then
      echo "ERROR: SEED_TRIES='$SEED_TRIES' must be a positive integer." >&2
      exit 1
    fi
    BASE_SEED="$SEED"

    SEED_LIST=()
    if [[ -n "${SEEDS:-}" ]]; then
      for _t in ${SEEDS//,/ }; do
        if [[ "$_t" =~ ^[0-9]+$ ]]; then
          SEED_LIST+=("$_t")
        elif [[ "$_t" =~ ^([0-9]+)-([0-9]+)$ && "${BASH_REMATCH[1]}" -le "${BASH_REMATCH[2]}" ]]; then
          for (( _s=BASH_REMATCH[1]; _s<=BASH_REMATCH[2]; _s++ )); do SEED_LIST+=("$_s"); done
        else
          echo "ERROR: SEEDS token '$_t' invalid (use N, N-M, space/comma separated)." >&2
          exit 1
        fi
      done
    else
      for (( _s=BASE_SEED; _s<BASE_SEED+SEED_TRIES; _s++ )); do SEED_LIST+=("$_s"); done
    fi
    SEED_MIN="${SEED_LIST[0]}"
    for _s in "${SEED_LIST[@]}"; do [[ "$_s" -lt "$SEED_MIN" ]] && SEED_MIN="$_s"; done
    SEED_ALWAYS_ALL=0
    declare -A SEED_ALWAYS_SET=()
    _always_tokens="${SEED_ALWAYS:-}"
    for _t in ${_always_tokens//,/ }; do
      if [[ "$_t" == "all" ]]; then
        SEED_ALWAYS_ALL=1
      elif [[ "$_t" =~ ^[0-9]+$ ]]; then
        SEED_ALWAYS_SET[$_t]=1
      else
        echo "ERROR: SEED_ALWAYS token '$_t' invalid (use seed numbers or 'all')." >&2
        exit 1
      fi
    done

    # Sampler identity. Rows are matched on the sampler settings actually in
    # effect, not on the TEMPERATURE label: TEMPERATURE=server sends nothing, so
    # the temperature really used is whatever the server was running (srv.--temp),
    # and top-p/top-k/min-p always come from the server. Without this, rows from
    # a 0.6 server and a 1.0 server both read "server" and wrongly count as the
    # same experiment. Resolved once here from the live server; a row's effective
    # temp is its temperature column, or srv.--temp when that says "server".
    CUR_SRV_TEMP="" CUR_TOP_P="" CUR_TOP_K="" CUR_MIN_P="" SAMPLER_KNOWN=0
    if [[ "$MODEL" != */* ]]; then
      _srv_settings=""
      if [[ "$TARGET" == "strata" ]]; then
        _srv_settings="$(strata_settings_effective)"
      else
        _srv_settings="$(llama_settings_effective "$(detect_llama_container)")"
      fi
      _srv_get() { sed -n "s/^$1=//p" <<< "$_srv_settings" | head -1; }
      CUR_SRV_TEMP="$(_srv_get srv.--temp)"
      CUR_TOP_P="$(_srv_get srv.--top-p)"
      CUR_TOP_K="$(_srv_get srv.--top-k)"
      CUR_MIN_P="$(_srv_get srv.--min-p)"
      [[ -n "$_srv_settings" ]] && SAMPLER_KNOWN=1
    fi
    if [[ "$TEMPERATURE" == "server" ]]; then
      if [[ -z "$CUR_SRV_TEMP" ]]; then
        echo "ERROR: TEMPERATURE=server but the server's temperature could not be read, so existing results can't be matched against it. Is the server up? (Set TEMPERATURE=<float> to bypass.)" >&2
        exit 1
      fi
      CUR_EFF_TEMP="$CUR_SRV_TEMP"
    else
      CUR_EFF_TEMP="$TEMPERATURE"
    fi
    echo "-- sampler in effect: temp=$CUR_EFF_TEMP (TEMPERATURE=$TEMPERATURE) top-p=${CUR_TOP_P:-?} top-k=${CUR_TOP_K:-?} min-p=${CUR_MIN_P:-?}" >&2

    # awk fragment shared by the history lookups below: sets `same` to whether
    # the row's effective sampler equals the current one. Top-p/k/min-p are only
    # compared when the live server could be read (SAMPLER_KNOWN).
    SAMPLER_AWK='
      function norm(x) { return (x == "") ? "" : x + 0 }
      function same_sampler(   et) {
        et = ($t == "server") ? $st : $t
        if (et == "" || norm(et) != norm(ctemp)) return 0
        if (known && (norm($tp) != norm(ctp) || norm($tk) != norm(ctk) || norm($mp) != norm(cmp))) return 0
        return 1
      }'
    SAMPLER_AWK_HDR='if ($i=="srv.--temp") st=i; if ($i=="srv.--top-p") tp=i; if ($i=="srv.--top-k") tk=i; if ($i=="srv.--min-p") mp=i;'

    # already_passed <exercise-keyword>: true if $MODEL has EVER recorded a
    # pass for this exercise at the current $REASONING_EFFORT and $TEMPERATURE in PER_TEST_TSV
    # (an xhigh pass is a different result from a low one; same for temperature), any run_cmd/seed. Keyword is
    # "<lang>/exercises/practice/<exercise>" or bare "<exercise>" (see
    # parse_fromfile_entries above) -- strip to the bare exercise name since
    # that's what PER_TEST_TSV's "exercise" column holds either way. When the
    # keyword carries a language it must match PER_TEST_TSV's "language" column
    # too: exercise names collide across languages (bowling, paasio, ...).
    already_passed() {
      local ex="${1##*/}" lang=""
      [[ "$1" == */exercises/practice/* ]] && lang="${1%%/*}"
      [[ -f "$PER_TEST_TSV" ]] || return 1
      awk -F'\t' -v alias="$MODEL" -v ex="$ex" -v lang="$lang" -v effort="$REASONING_EFFORT" \
        -v ctemp="$CUR_EFF_TEMP" -v ctp="$CUR_TOP_P" -v ctk="$CUR_TOP_K" -v cmp="$CUR_MIN_P" -v known="$SAMPLER_KNOWN" "$SAMPLER_AWK"'
        NR==1 { for (i=1;i<=NF;i++) { if ($i=="srv.--alias") a=i; if ($i=="exercise") e=i; if ($i=="pass") p=i; if ($i=="language") l=i; if ($i=="reasoning_effort") r=i; if ($i=="temperature") t=i; '"$SAMPLER_AWK_HDR"' }; next }
        $a==alias && $e==ex && (lang=="" || $l==lang) && $r==effort && same_sampler() && $p=="true" { found=1 }
        END { exit !found }
      ' "$PER_TEST_TSV"
    }

    # recorded_at_seed <exercise-keyword> <any|pass>: true if $MODEL has a REAL
    # recorded attempt for this exercise at the current $SEED -- "any" = pass
    # or fail, "pass" = passed only. Stall bail-outs (bailed_on_max_duration=
    # true) don't count: those are the server dying/hanging, not the model
    # failing, so they deserve a rerun. Rows from before seeds were recorded
    # (blank seed column) never match.
    recorded_at_seed() {
      local ex="${1##*/}" lang=""
      [[ "$1" == */exercises/practice/* ]] && lang="${1%%/*}"
      [[ -f "$PER_TEST_TSV" ]] || return 1
      awk -F'\t' -v alias="$MODEL" -v ex="$ex" -v lang="$lang" -v seed="$SEED" -v want="$2" -v effort="$REASONING_EFFORT" \
        -v ctemp="$CUR_EFF_TEMP" -v ctp="$CUR_TOP_P" -v ctk="$CUR_TOP_K" -v cmp="$CUR_MIN_P" -v known="$SAMPLER_KNOWN" "$SAMPLER_AWK"'
        NR==1 { for (i=1;i<=NF;i++) { if ($i=="srv.--alias") a=i; if ($i=="exercise") e=i; if ($i=="seed") s=i; if ($i=="pass") p=i; if ($i=="language") l=i; if ($i=="bailed_on_max_duration") b=i; if ($i=="reasoning_effort") r=i; if ($i=="temperature") t=i; '"$SAMPLER_AWK_HDR"' }; next }
        $a==alias && $e==ex && (lang=="" || $l==lang) && $r==effort && same_sampler() && $s==seed && $b!="true" && (want=="any" || $p=="true") { found=1 }
        END { exit !found }
      ' "$PER_TEST_TSV"
    }

    ORIG_TRIES="$TRIES"
    ORIG_MAX_DUR="$AIDER_BENCHMARK_MAX_DUR"
    SEED_IDX=0
    for SEED in "${SEED_LIST[@]}"; do
      SEED_IDX=$(( SEED_IDX + 1 ))
      SAMPLING_SLUG="temp${TEMPERATURE}-seed${SEED}${EFFORT_SLUG}${EXTRA_SLUG}"
      RUN_NAME="fromfile-$(basename "$FROMFILE_PATH" | sed 's/\.[^.]*$//')-${MODEL_SLUG}-${SAMPLING_SLUG}"

      # Filter BEFORE guard_reuse/write_model_settings: a seed with nothing to
      # run must not touch (or, under --clean, archive) its existing run dir.
      SEED_GATED=0
      if [[ "$SEED" != "$SEED_MIN" && "$SEED_ALWAYS_ALL" -eq 0 && -z "${SEED_ALWAYS_SET[$SEED]:-}" ]]; then
        SEED_GATED=1
      fi
      CUR_ENTRIES=()
      N_PASSED=0
      N_SEED_PASSED=0
      N_TRIED=0
      for entry in "${FROMFILE_ENTRIES[@]}"; do
        kw="${entry%%$'\t'*}"
        if [[ "$SEED_GATED" -eq 1 || "$SKIP_PASS" -eq 1 ]] && already_passed "$kw"; then
          N_PASSED=$(( N_PASSED + 1 ))
          continue
        fi
        if [[ "$SKIP_SEED_PASS" -eq 1 ]] && recorded_at_seed "$kw" pass; then
          N_SEED_PASSED=$(( N_SEED_PASSED + 1 ))
          continue
        fi
        if [[ "$SKIP_SEED_TRIED" -eq 1 ]] && recorded_at_seed "$kw" any; then
          N_TRIED=$(( N_TRIED + 1 ))
          continue
        fi
        CUR_ENTRIES+=("$entry")
      done
      echo "-- seed $SEED ($SEED_IDX/${#SEED_LIST[@]}, gated=$SEED_GATED): skipping $N_PASSED passed-at-any-seed, $N_SEED_PASSED passed-at-this-seed, $N_TRIED attempted-at-this-seed (SKIP_EXISTING=$SKIP_EXISTING)" >&2
      if [[ "${#CUR_ENTRIES[@]}" -eq 0 ]]; then
        echo "-- seed $SEED: nothing to attempt" >&2
        continue
      fi
      if [[ "$SEED_GATED" -eq 1 ]]; then
        echo "-- seed $SEED ($SEED_IDX/${#SEED_LIST[@]}): retrying ${#CUR_ENTRIES[@]} not-yet-passed exercise(s)" >&2
      else
        echo "-- seed $SEED ($SEED_IDX/${#SEED_LIST[@]}): attempting ${#CUR_ENTRIES[@]} exercise(s)" >&2
      fi
      guard_reuse "$AIDER_RUNS/$RUN_NAME" "${@:3}"
      write_model_settings

      declare -A FROMFILE_GROUP_KEYWORDS=()
      FROMFILE_GROUP_ORDER=()
      for entry in "${CUR_ENTRIES[@]}"; do
      IFS=$'\t' read -r kw entry_tries entry_dur <<< "$entry"
      [[ "$entry_tries" == "-" ]] && entry_tries=""
      [[ "$entry_dur" == "-" ]] && entry_dur=""
      group_key="${entry_tries:-$TRIES}|${entry_dur:-$MAX_DUR}"
      if [[ -z "${FROMFILE_GROUP_KEYWORDS[$group_key]:-}" ]]; then
        FROMFILE_GROUP_ORDER+=("$group_key")
      fi
      if [[ -n "${FROMFILE_GROUP_KEYWORDS[$group_key]:-}" ]]; then
        FROMFILE_GROUP_KEYWORDS[$group_key]+=",$kw"
      else
        FROMFILE_GROUP_KEYWORDS[$group_key]="$kw"
      fi
      done

      for group_key in "${FROMFILE_GROUP_ORDER[@]}"; do
      TRIES="${group_key%%|*}"
      export AIDER_BENCHMARK_MAX_DUR="$(scale_max_dur "${group_key#*|}")"
      KEYWORDS="${FROMFILE_GROUP_KEYWORDS[$group_key]}"
      echo "-- fromfile group: tries=$TRIES max_dur=${AIDER_BENCHMARK_MAX_DUR}s -- $(echo "$KEYWORDS" | tr ',' '\n' | wc -l) exercise(s)" >&2
      # Fast pre-flight health check: the per-group stall watchdog in
      # run_benchmark_and_track (see its comment above) already catches a
      # server that dies or hangs mid-run, but only after waiting out that
      # GROUP's own multi-thousand-second stall_ceiling. If the server died
      # once, every remaining group in this fromfile independently rediscovers
      # that the hard way -- observed firsthand: 6 groups in a row each burned
      # 35-95 minutes hitting the same litellm connection-error retry loop
      # before their own watchdog gave up, ~4.5 hours wasted against a server
      # that was never coming back on its own. Two quick health checks here
      # (5s apart, to not false-positive on a one-off blip) turn that into a
      # ~15s skip per group instead -- the caller (e.g. a script looping over
      # several models) can then reload the server and retry just the
      # skipped/missing exercises against a healthy one.
      health_ok=0
      for _ in 1 2; do
        if curl -sf --max-time 10 "$LLAMA_HEALTH_URL" 2>/dev/null | grep -Eq '"status": *"ok"'; then
          health_ok=1
          break
        fi
        sleep 5
      done
      if [[ "$health_ok" -eq 0 ]]; then
        echo "WARNING: llama-cpp server not responding at $LLAMA_HEALTH_URL -- skipping this group instead of waiting out its stall timeout (caller should reload the server and retry missing exercises)" >&2
        continue
      fi
      # `if !` (not a bare statement) so a non-zero exit here -- benchmark.py
      # itself failing, or a container getting killed out from under it (e.g.
      # a watchdog reacting to a hang) -- doesn't trip `set -e` and abort the
      # ENTIRE fromfile run. Without this, one bad group silently skips every
      # remaining exercise, which defeats the point of grouping for exactly
      # the unattended/overnight runs this feature is for.
      if ! run_benchmark_and_track "$AIDER_RUNS/$RUN_NAME" "$RUN_NAME" "$EDIT_FORMAT" \
        "/results/$RUN_NAME" \
        --model "$AIDER_MODEL" \
        --edit-format "$EDIT_FORMAT" \
        --threads "${THREADS:-1}" \
        --tries "$TRIES" \
        --exercises-dir /benchmarks/polyglot-benchmark \
        --new \
        $(common_flags) \
        "${GUARD_ARGS[@]}"; then
        echo "WARNING: fromfile group (tries=$TRIES max_dur=${AIDER_BENCHMARK_MAX_DUR}s) exited non-zero -- continuing to next group" >&2
      fi
      done
    done
    TRIES="$ORIG_TRIES"
    SEED="$BASE_SEED"
    export AIDER_BENCHMARK_MAX_DUR="$ORIG_MAX_DUR"

    compare_to_baseline "$FROMFILE_PATH" "$AIDER_RUNS/$RUN_NAME"
    ;;

  stats)
    # Regenerate stats from a previous run
    RUN_DIR="${2:?Usage: ./run.sh stats <run-dir>}"
    $DOCKER_RUN ./benchmark/benchmark.py --stats "/results/$RUN_DIR"
    ;;

  custom)
    split_exercise_args "${@:2}"
    guard_reuse $AIDER_RUNS/custom "${PASSTHRU_ARGS[@]}"
    if [[ "${#EXERCISE_ARGS[@]}" -gt 0 ]]; then
      KEYWORDS="$(IFS=,; echo "${EXERCISE_ARGS[*]}")"
    fi
    # Recover --edit-format from the passed-through flags, if given
    # explicitly; otherwise fall back to $EDIT_FORMAT (default "whole"), same
    # as every other command. Computed before write_model_settings so the
    # YAML's edit_format always matches what's actually requested.
    CUSTOM_EDIT_FORMAT="$EDIT_FORMAT"
    args=("${GUARD_ARGS[@]}")
    for ((i = 0; i < ${#args[@]}; i++)); do
      if [[ "${args[$i]}" == "--edit-format" && $((i + 1)) -lt ${#args[@]} ]]; then
        CUSTOM_EDIT_FORMAT="${args[$((i + 1))]}"
      fi
    done
    write_model_settings "$CUSTOM_EDIT_FORMAT"
    run_benchmark_and_track $AIDER_RUNS/custom custom "$CUSTOM_EDIT_FORMAT" \
      /results/custom \
      --model "$AIDER_MODEL" \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    ;;

  help|*)
    echo "Aider Benchmark Runner"
    echo ""
    echo "Commands:"
    echo "  build                     Build the Docker image (~5-10 min first time)"
    echo "  smoke [exercise ...]      1 FIXED python exercise (~2 min)"
    echo "  quick [exercise ...]      5 FIXED python exercises (~10 min)"
    echo "  diff [exercise ...]       Same as quick but forces diff edit format"
    echo "  random <n> [flags]        <n> RANDOM python exercises (default n=5)"
    echo "  python [exercise ...]     All python exercises, or just the named ones (~30 min)"
    echo "  polyglot [exercise ...]   3 FIXED exercises per language (18 total), or just the named ones"
    echo "  full [exercise ...]       All 225 exercises, or just the named ones (~2-4 hours)"
    echo "  fromfile <file> [flags]   Run just the exercises listed in <file>, one per line:"
    echo "                              '<language> <exercise>'  or bare '<exercise>' (all langs)"
    echo "                            '#' starts a comment -- if it's '# pass|fail <tries|-> <secs>'"
    echo "                            (as written by this project's shortlist-*.txt), the run ends"
    echo "                            with a table of where this model diverged from that baseline."
    echo "                            Results go to runs/aider/fromfile-<basename>/; --clean/--resume"
    echo "                            work the same as full/python. See lists/shortlist-15.txt."
    echo "                            SKIP_EXISTING=<scopes> (comma list) picks what finished work is"
    echo "                            skipped: pass (passed at ANY seed), seed-pass (passed at THIS"
    echo "                            seed), seed (attempted, pass or fail, at THIS seed), resume ="
    echo "                            pass,seed (same as SKIP_EXISTING=1), none/0 (default)."
    echo "                            SEED=N SEED_TRIES=K runs seeds N..N+K-1; SEEDS=\"2\"|\"1 3\"|\"1-3\""
    echo "                            names them explicitly (wins over both). Every seed but the"
    echo "                            lowest is gated: it only runs exercises not yet passed at any"
    echo "                            seed. SEED_ALWAYS=\"2\"|\"2 3\"|all lifts that gate for those seeds."
    echo "                            e.g. fill every missing seed-2 record:"
    echo "                              SEEDS=2 SEED_ALWAYS=2 SKIP_EXISTING=seed"
    echo "                            fill seed 2 only where nothing ever passed:"
    echo "                              SEEDS=2 SKIP_EXISTING=pass,seed"
    echo "                            Also doubles as crash recovery: re-run the same command and"
    echo "                            it skips what's done, redoes what isn't."
    echo "  stats <run-dir>           Regenerate stats from completed run"
    echo "  shell                     Interactive shell in benchmark container"
    echo "  custom [exercise ...] args  Pass arbitrary benchmark.py args"
    echo ""
    echo "smoke/quick/diff/polyglot use a FIXED, deterministic default exercise set"
    echo "(not random) so repeated runs are directly comparable across models/configs;"
    echo "use 'random' if you actually want randomness. [exercise ...] are bare-word"
    echo "substrings of exercise names (aider's --keywords) -- e.g. './run.sh quick"
    echo "zipper leap' runs just those 2 instead of the fixed default 5. Everything"
    echo "from the first '-'-prefixed arg onward is forwarded to benchmark.py as-is."
    echo ""
    echo "Model is NOT a positional arg -- set MODEL=<alias> if it isn't 'Default'"
    echo "(most llama.cpp setups only run one model at a time, so there's rarely a"
    echo "reason to name it every invocation). Check available models/aliases:"
    echo "  curl http://100.71.29.121:3002/v1/models"
    echo ""
    echo "smoke/quick/diff/polyglot/random auto-clean their results dir on every run"
    echo "(old results moved to runs/aider/OLD/<name>) so each invocation is independent --"
    echo "pass --resume to opt out and reuse/continue existing results instead."
    echo ""
    echo "python/full can run for hours, so they don't auto-clean -- if their results"
    echo "dir already has results in it, run.sh refuses to proceed until you pass:"
    echo "  --clean               Move old results to runs/aider/OLD/<name>, start fresh"
    echo "  --resume              Proceed and let benchmark.py reuse/continue them"
    echo "e.g. ./run.sh full --clean"
    echo ""
    echo "Env overrides:"
    echo "  MODEL=<alias>                         (default: Qwen3.6-35B-A3B) your llama-server's --alias"
    echo "  EDIT_FORMAT=whole|diff|udiff|...      (default: whole) aider's --edit-format;"
    echo "                                        applies to smoke/quick/python/polyglot/full/custom"
    echo "                                        (the diff command always forces diff)"
    echo "  TRIES=<n>                             (default: 6) fix-up attempts per exercise"
    echo "  MAX_TOKENS=<n>                        (default: 16000 at low, from REASONING_EFFORT) hard cap, thinking+answer"
    echo "  DECODE_TPS=<n>                         (default: 25) for timeout auto-calc"
    echo "  PREFILL_BUFFER=<seconds>              (default: 120) prompt-eval headroom"
    echo "  REQUEST_TIMEOUT=<seconds>             (default: auto from MAX_TOKENS/DECODE_TPS)"
    echo "  MAX_DUR=<seconds>                     (default: 480/8min, 0 disables) wall-clock cap"
    echo "                                        PER EXERCISE across all its --tries combined --"
    echo "                                        skips remaining tries once exceeded, so a"
    echo "                                        model stuck re-tripping the same error"
    echo "                                        doesn't burn e.g. 90 minutes on one exercise"
    echo "  MAX_DUR_SCALE=<factor>|auto           (default: 1) multiplies MAX_DUR and every"
    echo "                                        per-exercise fromfile override by this factor --"
    echo "                                        e.g. MAX_DUR_SCALE=2.5 for a model known to run"
    echo "                                        2-3x slower per-token than this project's usual"
    echo "                                        comparison set, instead of hand-editing every line."
    echo "                                        auto = max(1, AUTO_SCALE_BASE_TPS(100) / the live"
    echo "                                        server's measured decode tok/s), probed once per run"
    echo "  REASONING_EFFORT=low|medium|xhigh|\"\" (default: low; high = xhigh) a bundle -- sent to the chat"
    echo "                                        template AND scales the rest (first-guess numbers):"
    echo "                                          low    thinking 4096   MAX_TOKENS 16000  max_dur x1"
    echo "                                          medium thinking 8192   MAX_TOKENS 20000  max_dur x1.5"
    echo "                                          xhigh  thinking 16384  MAX_TOKENS 32000  max_dur x2.5"
    echo "                                        REQUEST_TIMEOUT follows MAX_TOKENS. Non-low runs get their own"
    echo "                                        results dir (-xhigh suffix); skip checks only match same effort."
    echo "  REASONING_BUDGET=<n>                  override the thinking-token cap, sent per request (no server"
    echo "                                        restart; recorded in srv.--reasoning-budget). Empty = don't send."
    echo "  VIZ_REFRESH=0                         don't regenerate results/viz/eval-viz.html after each new result"
    echo "  EFFORT_DUR_MULT=<x>                   override the effort's max_dur multiplier (stacks with MAX_DUR_SCALE)"
    echo "  TEMPERATURE=<float>|server            (default: 0.6; server for TARGET=strata) sent per request;"
    echo "                                        0 = greedy; server = send none, the server's own sampling defaults apply"
    echo "  EXTRA_BODY='{\"top_k\":20,\"min_p\":0.1}'   JSON merged into every request body (any sampler knob the server"
    echo "                                        accepts); gets its own results dir (-x<hash>) and an extra_body TSV column"
    echo "  TARGET=llama-nas|strata               (default: llama-nas) which server to eval; NAS_HOST / STRATA_URL override addresses"
    echo "  SEED=<int>                            (default: 1) same seed+settings reproduces a"
    echo "                                        run; change it for another independent sample"
    echo "  SKIP_EXISTING / SEED_TRIES / SEEDS / SEED_ALWAYS    fromfile-only; see 'fromfile' above"
    echo "  LLAMA_BASE_URL=http://host:port/v1    (default: tailscale :3002)"
    echo "  LLAMA_CONTAINER=<name>                (default: auto-detect running llama-server)"
    echo "  TRACK_TSV=<path>                      (default: results/aider-results.tsv)"
    echo "  PER_TEST_TSV=<path>                   (default: results/aider-results-per-test.tsv)"
    echo ""
    echo "Per-run output is saved to runs/aider/; tracked rows to results/"
    echo ""
    echo "Each run (smoke/quick/python/polyglot/full/diff/custom) appends a row to"
    echo "aider-results.tsv: aider's pass-rate stats for that run, plus the model's"
    echo "current behavior-affecting settings (sampling, KV-cache quant, context"
    echo "handling, chat template) read live off the running llama-server via"
    echo "'docker inspect'. Pure perf/infra flags (threads, GPU-layer split,"
    echo "batching, logging, networking) are excluded so they don't pollute"
    echo "behavior comparisons across runs."
    echo ""
    echo "It also appends one row per exercise to aider-results-per-test.tsv (same"
    echo "settings columns, plus language/exercise/pass/passed_at_try/cost/duration/"
    echo "... per exercise) -- e.g. grep zipper results/aider-results-per-test.tsv"
    echo "to see how every model/config you've run has handled that one exercise."
    echo ""
    echo "Compare runs:"
    echo "  ./run.sh shell"
    echo "  # then inside: ./benchmark/benchmark.py --diffs /results/run1 /results/run2"
    ;;
esac
