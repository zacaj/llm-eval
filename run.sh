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
#   ./run.sh <command> [model-alias] [extra-args...]
#
# Examples:
#   ./run.sh smoke                    # 1 python exercise, ~2 min
#   ./run.sh quick                    # 5 python exercises, ~10 min
#   ./run.sh python                   # all python exercises, ~30 min
#   ./run.sh full                     # all 225 exercises, ~2-4 hours
#   ./run.sh quick my-model-alias     # specify model name
#   ./run.sh stats <run-dir>          # regenerate stats from a completed run
#   ./run.sh shell                    # open interactive shell in container
#
# Results go to ./results/<timestamp>--<run-name>/

cd "$(dirname "$0")"

MODEL="${2:-Default}"  # default to your llama.cpp --alias
CMD="${1:-help}"
REASONING_EFFORT="${REASONING_EFFORT-low}"  # empty string disables
EDIT_FORMAT="${EDIT_FORMAT:-whole}"         # whole|diff|udiff|... forwarded as aider's --edit-format
                                             # (applies to smoke/quick/python/polyglot/full/custom;
                                             # the dedicated `diff` command always forces diff, that's its purpose)
TRIES="${TRIES:-4}"                         # fix-up attempts per exercise (default 2 gave too many false negatives)
MAX_TOKENS="${MAX_TOKENS:-16000}"           # hard cap on thinking+answer tokens
                                             # (6-8k caps truncated legitimate answers on some
                                             # exercises: needed as low as ~4243/try to as much as
                                             # 8486 total observed. 16k gives real headroom to avoid
                                             # false failures from truncation, not model capability.)
DECODE_TPS="${DECODE_TPS:-70}"              # assumed tok/s, for timeout auto-calc
PREFILL_BUFFER="${PREFILL_BUFFER:-120}"     # seconds, headroom for prompt prefill
AUTO_TIMEOUT=$(( (MAX_TOKENS + DECODE_TPS - 1) / DECODE_TPS + PREFILL_BUFFER ))
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-$AUTO_TIMEOUT}"  # seconds, overrides aider's 600s default

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

DOCKER_RUN="docker compose run --rm benchmark"

mkdir -p results

# Snapshot before the docker run starts, so track_eval can scope its
# aggregation to files this invocation actually wrote (see track_eval below).
RUN_START_TS="$(date +%s)"

# ── Result tracking (eval-results.tsv) ──────────────────────────────────────
# After each benchmark run, snapshot the llama-server's current model/sampling
# settings (via `docker inspect`, same trick as bench-replay.sh) alongside
# aider's own eval config and the run's aggregate pass-rate stats, then append
# one row to eval-results.tsv. Columns grow/shrink automatically as flags
# change across runs (dynamic-header merge, same as bench-replay.sh).

TRACK_TSV="${TRACK_TSV:-$(pwd)/results/eval-results.tsv}"
LLAMA_CONTAINER="${LLAMA_CONTAINER:-}"

detect_llama_container() {
  if [[ -n "$LLAMA_CONTAINER" ]]; then
    echo "$LLAMA_CONTAINER"
    return
  fi
  local pattern found
  for pattern in "llama-cpp-turboquant-llama-cpp-1" "llama-server" "llama-cpp" "llama"; do
    found="$(docker ps --filter "name=$pattern" --filter "status=running" --format '{{.Names}}' | head -1)"
    if [[ -n "$found" ]]; then
      echo "$found"
      return
    fi
  done
}

# llama-server flags that are purely about *how fast*/*where* inference runs
# (threading, GPU offload split, batching, networking, logging, caching infra,
# speculative-decoding draft config) rather than *how the model behaves*
# (sampling params, context/RoPE handling, KV-cache quant, chat templating).
# Blacklisted so perf-tuning noise doesn't pollute behavior comparisons.
SRV_BLACKLIST="--server --host --port -lv --log-timestamps --log-colors --verbosity \
--threads --threads-batch -t -tb --n-gpu-layers -ngl --n-cpu-moe --tensor-split -ts \
--main-gpu -mg --split-mode -sm --batch-size -b --ubatch-size -ub --parallel -np \
--cont-batching -cb --no-cont-batching --mlock --no-mmap --numa --flash-attn -fa \
--fit --fit-ctx --cache-prompt --checkpoint-every-n-tokens --slot-save-path \
--ctx-checkpoints --metrics --api-key --ssl-key-file --ssl-cert-file --timeout \
--no-webui --draft-max --draft-min --draft-p-min --gpu-layers-draft --model-draft \
--jinja --checkpoint-min-step --cache-ram --cache-idle-slots --kv-unified \
--spec-draft-ngl --spec-draft-n-cpu-moe \
--device --device-draft --spec-draft-type-k --spec-draft-type-v --defrag-thold"



# Extract non-blacklisted llama-server CLI args as key=value lines
# (prefixed srv.<flag>=<value>). Best-effort: silent no-op if no server
# container is running.
capture_srv_settings() {
  local container="$1" inspect_json
  [[ -z "$container" ]] && return
  inspect_json="$(docker inspect "$container" 2>/dev/null)" || return
  [[ -z "$inspect_json" ]] && return
  echo "$inspect_json" | python3 -c "
import json, sys, re

blacklist = set('''$SRV_BLACKLIST'''.split())
data = json.load(sys.stdin)
args = data[0].get('Args', [])

def is_flag(s):
    return s.startswith('-') and not re.match(r'^-\d', s)

i = 0
while i < len(args):
    arg = args[i]
    if is_flag(arg):
        key = arg
        val = ''
        if i + 1 < len(args) and not is_flag(args[i + 1]):
            val = args[i + 1]
            i += 1
        if not val:
            val = 'yes'
        if key not in blacklist:
            print(f'srv.{key}={val}')
        i += 1
    else:
        i += 1
"
}

# Parse a completed run dir's .aider.results.json files into aggregate stats
# (same fields as aider benchmark.py's summarize_results) and append a row to
# eval-results.tsv, merged with this run's aider settings and (if available)
# the llama-server's current model settings.
#
# aider's benchmark.py resolves e.g. "/results/smoke" to itself (--new only
# auto-timestamps a bare name, not a multi-segment path — see
# resolve_dirname() in benchmark.py), so run.sh's fixed per-command dirs
# (results/smoke, results/quick, ...) are reused/accumulated across
# invocations rather than replaced. Scope aggregation to files modified after
# this invocation started so stats aren't polluted by leftover exercises from
# a prior run of the same command.
track_eval() {
  local cmd_name="$1" run_dir="$2" edit_format="$3" since_ts="$4"
  if [[ ! -d "$run_dir" ]]; then
    echo "track_eval: no results dir $run_dir, skipping" >&2
    return
  fi

  local container srv_file
  container="$(detect_llama_container)"
  srv_file="$(mktemp)"
  trap "rm -f '$srv_file'" RETURN
  capture_srv_settings "$container" > "$srv_file"

  python3 -c "
import csv, json, os, sys
from pathlib import Path
from datetime import datetime, timezone

tsv_file, run_dir, cmd_name, model, srv_file, since_ts = sys.argv[1:7]
since_ts = float(since_ts)
run_dir = Path(run_dir)
aider_settings = {}
for kv in sys.argv[7:]:
    k, _, v = kv.partition('=')
    aider_settings[k] = v

results = []
for fname in run_dir.glob('*/exercises/practice/*/.aider.results.json'):
    try:
        if fname.stat().st_mtime < since_ts:
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
    print(f'[{timestamp}] track_eval: {cmd_name} produced no new results under {run_dir}'
          f' (nothing modified since this invocation started) -- skipping row, not writing to {tsv_file}')
    sys.exit(0)

row = {
    'timestamp': timestamp,
    'run_cmd': cmd_name,
    'run_dir': str(run_dir),
    'model': model,
}
row.update(aider_settings)

if os.path.isfile(srv_file):
    with open(srv_file) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                row[k] = v

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

existing_rows = []
existing_header = None
if os.path.isfile(tsv_file):
    with open(tsv_file, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        existing_header = next(reader, None)
        for r in reader:
            existing_rows.append(r)

static_set, stat_set = set(STATIC), set(STATS)
old_mid_cols = [c for c in (existing_header or []) if c not in static_set and c not in stat_set]
new_mid_cols = [k for k in row if k not in static_set and k not in stat_set and k not in old_mid_cols]
new_header = STATIC + new_mid_cols + old_mid_cols + STATS

if existing_header and existing_header != new_header:
    rebuilt = []
    for old_row in existing_rows:
        old_dict = dict(zip(existing_header, old_row))
        rebuilt.append([old_dict.get(c, '') for c in new_header])
    existing_rows = rebuilt

os.makedirs(os.path.dirname(tsv_file) or '.', exist_ok=True)
with open(tsv_file, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')
    writer.writerow(new_header)
    for r in existing_rows:
        writer.writerow(r)
    writer.writerow([row.get(c, '') for c in new_header])

pass_rate_str = ' '.join(
    f'{i + 1}:{row[f\"pass_rate_{i + 1}\"]}' for i in range(tries)
) or '(no tries observed)'

print(f'Eval row appended: {tsv_file}')
print(f'  [{row[\"timestamp\"]}] {cmd_name} model={model}: '
      f'completed={row[\"completed_tests\"]}/{row[\"total_tests\"]} '
      f'pass_rate%={pass_rate_str} '
      f'well_formed={row[\"percent_cases_well_formed\"]}% cost=\${row[\"total_cost\"]} '
      f'sec/case={row[\"seconds_per_case\"]}')
" "$TRACK_TSV" "$run_dir" "$cmd_name" "openai/$MODEL" "$srv_file" "$since_ts" \
      "edit_format=$edit_format" "tries=$TRIES" "max_tokens=$MAX_TOKENS" \
      "reasoning_effort=$REASONING_EFFORT" "request_timeout=$REQUEST_TIMEOUT"
}

# Generate model-settings YAML matching this run's model name so
# extra_params.timeout actually applies (aider matches by exact name).
# Takes the edit format to record (defaults to $EDIT_FORMAT) so this file
# never disagrees with whatever --edit-format actually gets passed to
# benchmark.py for this invocation.
write_model_settings() {
  local fmt="${1:-$EDIT_FORMAT}"
  cat > results/.model-settings.yml <<EOF
- name: openai/$MODEL
  edit_format: $fmt
  use_repo_map: false
  use_system_prompt: true
  streaming: false
  extra_params:
    timeout: $REQUEST_TIMEOUT
    max_tokens: $MAX_TOKENS
EOF
}

# Common flags shared by every benchmark invocation
common_flags() {
  local flags=(--read-model-settings /results/.model-settings.yml)
  if [[ -n "$REASONING_EFFORT" ]]; then
    flags+=(--reasoning-effort "$REASONING_EFFORT")
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
  mkdir -p results/OLD
  local dest="results/OLD/$(basename "$run_dir")"
  if [[ -e "$dest" ]]; then
    dest="results/OLD/$(date +%Y-%m-%d-%H-%M-%S)-$(basename "$run_dir")"
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
      echo "Pass --clean to start fresh (moves old results to results/OLD/<name>)," >&2
      echo "or --resume to continue/reuse the existing results as-is." >&2
      exit 1
    fi
    [[ $has_clean -eq 1 ]] && do_clean "$run_dir"
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

case "$CMD" in
  build)
    docker compose build
    ;;

  shell)
    $DOCKER_RUN bash
    ;;

  smoke)
    auto_clean results/smoke "${@:3}"
    write_model_settings
    # 1 exercise — verify the setup works
    $DOCKER_RUN ./benchmark/benchmark.py /results/smoke \
      --model "openai/$MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads 1 \
      --tries "$TRIES" \
      --languages python \
      --num-tests 1 \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    track_eval smoke results/smoke "$EDIT_FORMAT" "$RUN_START_TS"
    ;;

  quick)
    auto_clean results/quick "${@:3}"
    write_model_settings
    # 5 python exercises — quick feedback loop
    $DOCKER_RUN ./benchmark/benchmark.py /results/quick \
      --model "openai/$MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads 1 \
      --tries "$TRIES" \
      --languages python \
      --num-tests 5 \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    track_eval quick results/quick "$EDIT_FORMAT" "$RUN_START_TS"
    ;;

  python)
    guard_reuse results/python "${@:3}"
    write_model_settings
    # All python exercises (~30-40 depending on model speed)
    $DOCKER_RUN ./benchmark/benchmark.py /results/python \
      --model "openai/$MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads 1 \
      --tries "$TRIES" \
      --languages python \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    track_eval python results/python "$EDIT_FORMAT" "$RUN_START_TS"
    ;;

  polyglot)
    auto_clean results/polyglot "${@:3}"
    write_model_settings
    # Subset across all languages (3 per language = 18 exercises)
    $DOCKER_RUN ./benchmark/benchmark.py /results/polyglot \
      --model "openai/$MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads 1 \
      --tries "$TRIES" \
      --num-tests 3 \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    track_eval polyglot results/polyglot "$EDIT_FORMAT" "$RUN_START_TS"
    ;;

  full)
    guard_reuse results/full "${@:3}"
    write_model_settings
    # All 225 exercises — the real deal
    $DOCKER_RUN ./benchmark/benchmark.py /results/full \
      --model "openai/$MODEL" \
      --edit-format "$EDIT_FORMAT" \
      --threads 1 \
      --tries "$TRIES" \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    track_eval full results/full "$EDIT_FORMAT" "$RUN_START_TS"
    ;;

  diff)
    auto_clean results/diff "${@:3}"
    write_model_settings diff
    # Same as quick but with diff edit format (tests if model can do search/replace)
    $DOCKER_RUN ./benchmark/benchmark.py /results/diff \
      --model "openai/$MODEL" \
      --edit-format diff \
      --threads 1 \
      --tries "$TRIES" \
      --languages python \
      --num-tests 5 \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    track_eval diff results/diff diff "$RUN_START_TS"
    ;;

  stats)
    # Regenerate stats from a previous run
    RUN_DIR="${2:?Usage: ./run.sh stats <run-dir>}"
    $DOCKER_RUN ./benchmark/benchmark.py --stats "/results/$RUN_DIR"
    ;;

  custom)
    # Pass arbitrary args after the model name
    shift  # remove 'custom'
    guard_reuse results/custom "${@:2}"
    # Recover --edit-format from the passed-through args, if the caller gave
    # one explicitly; otherwise fall back to $EDIT_FORMAT (default "whole"),
    # same as every other command. Computed before write_model_settings so
    # the YAML's edit_format always matches what's actually requested.
    CUSTOM_EDIT_FORMAT="$EDIT_FORMAT"
    args=("${GUARD_ARGS[@]}")
    for ((i = 0; i < ${#args[@]}; i++)); do
      if [[ "${args[$i]}" == "--edit-format" && $((i + 1)) -lt ${#args[@]} ]]; then
        CUSTOM_EDIT_FORMAT="${args[$((i + 1))]}"
      fi
    done
    write_model_settings "$CUSTOM_EDIT_FORMAT"
    $DOCKER_RUN ./benchmark/benchmark.py /results/custom \
      --model "openai/$MODEL" \
      --exercises-dir /benchmarks/polyglot-benchmark \
      --new \
      $(common_flags) \
      "${GUARD_ARGS[@]}"
    track_eval custom results/custom "$CUSTOM_EDIT_FORMAT" "$RUN_START_TS"
    ;;

  help|*)
    echo "Aider Benchmark Runner"
    echo ""
    echo "Commands:"
    echo "  build                Build the Docker image (~5-10 min first time)"
    echo "  smoke [model]        1 python exercise (~2 min)"
    echo "  quick [model]        5 python exercises (~10 min)"
    echo "  python [model]       All python exercises (~30 min)"
    echo "  polyglot [model]     3 exercises per language, 18 total (~30 min)"
    echo "  full [model]         All 225 exercises (~2-4 hours)"
    echo "  diff [model]         5 python w/ diff edit format"
    echo "  stats <run-dir>      Regenerate stats from completed run"
    echo "  shell                Interactive shell in benchmark container"
    echo "  custom [model] args  Pass arbitrary benchmark.py args"
    echo ""
    echo "smoke/quick/polyglot/diff auto-clean their results dir on every run (old"
    echo "results moved to results/OLD/<name>) so each invocation is independent --"
    echo "pass --resume to opt out and reuse/continue existing results instead."
    echo ""
    echo "python/full can run for hours, so they don't auto-clean -- if their results"
    echo "dir already has results in it, run.sh refuses to proceed until you pass:"
    echo "  --clean               Move old results to results/OLD/<name>, start fresh"
    echo "  --resume              Proceed and let benchmark.py reuse/continue them"
    echo "e.g. ./run.sh full my-model --clean"
    echo ""
    echo "Model defaults to 'CPU' (your llama.cpp --alias)."
    echo "Check available models: curl http://100.71.29.121:3002/v1/models"
    echo ""
    echo "Env overrides:"
    echo "  EDIT_FORMAT=whole|diff|udiff|...      (default: whole) aider's --edit-format;"
    echo "                                        applies to smoke/quick/python/polyglot/full/custom"
    echo "                                        (the diff command always forces diff)"
    echo "  TRIES=<n>                             (default: 4) fix-up attempts per exercise"
    echo "  MAX_TOKENS=<n>                        (default: 16000) hard cap, thinking+answer"
    echo "  DECODE_TPS=<n>                         (default: 70) for timeout auto-calc"
    echo "  PREFILL_BUFFER=<seconds>              (default: 120) prompt-eval headroom"
    echo "  REQUEST_TIMEOUT=<seconds>             (default: auto from MAX_TOKENS/DECODE_TPS)"
    echo "  REASONING_EFFORT=low|medium|high|\"\"  (default: low, best-effort only)"
    echo "  LLAMA_BASE_URL=http://host:port/v1    (default: tailscale :3002)"
    echo "  LLAMA_CONTAINER=<name>                (default: auto-detect running llama-server)"
    echo "  TRACK_TSV=<path>                      (default: ./results/eval-results.tsv)"
    echo ""
    echo "Results are saved to ./results/"
    echo ""
    echo "Each run (smoke/quick/python/polyglot/full/diff/custom) appends a row to"
    echo "eval-results.tsv: aider's pass-rate stats for that run, plus the model's"
    echo "current behavior-affecting settings (sampling, KV-cache quant, context"
    echo "handling, chat template) read live off the running llama-server via"
    echo "'docker inspect'. Pure perf/infra flags (threads, GPU-layer split,"
    echo "batching, logging, networking) are excluded so they don't pollute"
    echo "behavior comparisons across runs."
    echo ""
    echo "Compare runs:"
    echo "  ./run.sh shell"
    echo "  # then inside: ./benchmark/benchmark.py --diffs /results/run1 /results/run2"
    ;;
esac
