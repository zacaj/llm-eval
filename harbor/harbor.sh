#!/bin/bash
set -euo pipefail

# Harbor runner: agentic evals (Claude Code / terminus-2 / ...) against the
# local model server. Deliberately independent of ../run.sh -- task selection,
# attempts and concurrency are Harbor's own flags, passed straight through.
#
# Usage:
#   ./harbor/harbor.sh gen                          # (re)generate tasks/aider_polyglot from the forked adapter
#   ./harbor/harbor.sh run <agent> [harbor run args...]
#   ./harbor/harbor.sh collect                      # jobs/*/ -> results/harbor-trials.tsv
#   ./harbor/harbor.sh gen-tc <language> <exercise> <calls>   # (re)generate one test-calls task
#   ./harbor/harbor.sh gen-tc --list FILE           # ... every list entry, calls = its 3rd column
#     -> tasks/aider_polyglot_tc/polyglot_<lang>_<exercise>__tc<calls> (run --list also
#        generates missing ones itself)
#
# Test-calls tasks (tasks/aider_polyglot_tc; --list rows with a try count use them
# automatically, TC=1/0 forces all/none -- see TC below): the agent runs as
# non-root and gets a `run-tests` command that runs the hidden tests (still
# unreadable) up to <calls> times; the limit is stated in the instruction and the
# session is killed after the last call. reward.json gains test_calls /
# test_limit / test_limit_hit / first_pass_call.
#
# Examples:
#   ./harbor/harbor.sh run claude-code -i 'polyglot_python_*' -l 5
#   ./harbor/harbor.sh run terminus-2  -i 'polyglot_rust_*' -k 3
#   ./harbor/harbor.sh run oracle      -i 'polyglot_cpp_*' -n 8     # verifier sanity check, no model
#   ./harbor/harbor.sh run claude-code --list smoke-per-language -k 3
#
# --list FILE: exercise list in run.sh's fromfile format ("<language> <exercise>
# [max_tries] [max_dur]" or bare "<exercise>" = every language; "#" comments), so
# the lists/*.txt shortlists work as-is. Each entry becomes an `-i` include
# glob. max_dur (seconds) is honoured: entries are grouped by it and each group
# runs as its own Harbor job (suffix -d<secs>) with a matching
# --agent-timeout-multiplier. max_tries has no Harbor equivalent (ignored, warned).
#
# Env:
#   BASE_URL   model server root, no /v1 (default: NAS prompt-logger proxy)
#   MODEL      model id sent to the server (default: first id from $BASE_URL/v1/models)
#   DATASET    task dir (default: harbor/tasks/aider_polyglot)
#   JOB_NAME   run id that starts every job name of this invocation (default: MMDD-HHMMSS).
#              Job names are <run id>_<tasks>[_d<max_dur>][_k<attempts>]_<agent>[-b<budget>]:
#              tasks = up to 4 short names (go-bowling.tc6), else <list name>.<count>;
#              model and the rest are in the job's config / collect.py columns
#   MAX_DUR    agent wall-clock limit per trial in seconds, for entries without their
#              own max_dur (default: the task's own agent timeout, 1800s for polyglot)
#   MAX_DUR_SCALE  multiplier on every agent time limit (list max_dur, MAX_DUR, or the
#              task default) for slower models, as in run.sh: a number (default 1) or
#              auto = max(1, AUTO_SCALE_BASE_TPS(100) / the server's measured decode tok/s).
#              Job names / max_dur column use the scaled limit; SKIP_EXISTING matches
#              on the unscaled one, so auto-scale drift between jobs doesn't split runs.
#   BUDGET     per-trial output-token budget (default 0 = unlimited), enforced by
#              eval_proxy.py (see EVAL_PROXY_HOST below)
#   SERVER_URL model server itself, not the proxy (default http://$NAS_HOST:8080); its
#              sampling/quality settings are saved to jobs/<job>/server-settings.txt
#              (same capture as run.sh's srv.* columns, via ../lib/server-capture.sh)
#   SKIP_EXISTING  1 = treat -k N as "ensure N graded attempts" per task and only run
#              what's missing, counting matching earlier trials from any job (same
#              agent+version/BUDGET/time limit (or finished within it)/server sampling+model; INVALID and ungraded
#              infra errors don't count) -- rerun the same command to resume
#   EFFORT     Claude Code effort level (CLAUDE_CODE_EFFORT_LEVEL; default low, matching
#              run.sh's REASONING_EFFORT default -- Claude Code's own default is high,
#              which Strata resolves to xhigh). What the server actually resolved is
#              recorded as srv.eff.reasoning_effort / srv.eff.reasoning_budget.
#              Also sets the thinking cap (eval_proxy.py --think-budget): low 6000,
#              medium 12000, high/xhigh 24000; max_tokens = cap + ANSWER_TOKENS.
#              off = reasoning disabled: Claude Code gets CLAUDE_CODE_DISABLE_THINKING=1
#              and the proxy sends thinking {"type":"disabled"} (THINK_BUDGET=-1), so
#              terminus-2 is covered too.
#   THINK_BUDGET   override the thinking cap (0 = leave the request as Claude Code sent it,
#              -1 = thinking disabled)
#   ANSWER_TOKENS  answer/tool-call headroom added to the cap for max_tokens (default 8000)
#   DISALLOWED_TOOLS  Claude Code tools removed for evals (--disallowedTools; default
#              Agent,WebFetch,WebSearch). Agent: subagents swap the prompt back and forth,
#              which throws away the server's prompt cache every time on a parallel=1
#              llama (-> timeouts); web tools let the model fetch the exercise's real
#              tests from GitHub. Recorded in run-settings.txt; set "" to allow all.
#   CC_VERSION Claude Code version to run (default: the one pre-installed in the task
#              images, ARG CC_VERSION in aider_polyglot/task-template/Dockerfile;
#              a different value makes Harbor install it per trial)
#   PROXY_FILTERS  1 (default) = the proxy's prompt filters (tuned for local models)
#              still rewrite Claude Code's system prompt/tools; 0 = stock prompts
#
# Defaults that differ from Harbor's: -n 1 (one model server slot); add -n N to override.
# Anything after <agent> is forwarded to `harbor run`, so later flags win.

cd "$(dirname "$0")"
source ../lib/paths.sh
# jobs/, tasks/, polyglot-benchmark/ are symlinks into runs/harbor/ (gitignored); job and
# trial configs record task paths relative to this dir, so the links keep old jobs readable
mkdir -p "$RUNS_DIR/harbor/jobs" "$RUNS_DIR/harbor/tasks"
# list file: relative to the caller's cwd, else a name under lists/ (".txt" optional)
resolve_list() {
    local c
    for c in "$1" "$OLDPWD/$1" "$LISTS_DIR/$1" "$LISTS_DIR/$1.txt"; do
        [[ "$c" = /* && -f "$c" ]] && { echo "$c"; return; }
    done
    echo "$1"
}

HARBOR=(uvx harbor@0.25.0)
BASE_URL="${BASE_URL:-http://192.168.2.15:3002}"
# Agents talk to eval_proxy.py (one per job, on this host) which forwards to BASE_URL:
# per-trial token budget + per-request log of what the server says it did
# (jobs/<job>/eval-proxy.tsv). EVAL_PROXY_HOST = this host's address as the trial
# containers see it (default: our source IP toward BASE_URL's host).
base_host=$(printf '%s' "$BASE_URL" | sed -E 's|^[a-z]+://([^/:]+).*|\1|')
EVAL_PROXY_HOST="${EVAL_PROXY_HOST:-$(ip route get "$base_host" 2>/dev/null | grep -o 'src [0-9.]*' | cut -d' ' -f2)}"
EVAL_PROXY_PORT="${EVAL_PROXY_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}"
EVAL_URL="http://$EVAL_PROXY_HOST:$EVAL_PROXY_PORT"
# TC: unset = per list row (a row with a try count -> its __tc<count> test-calls task,
# otherwise the classic task; non-list runs are classic); 1 = test-calls for everything;
# 0 = classic for everything. An explicit DATASET overrides all of it.
CLASSIC_DATASET=tasks/aider_polyglot
TC_DATASET=tasks/aider_polyglot_tc
TB2_DATASET=tasks/tb2
TB2_TC_DATASET=tasks/tb2_tc
TB2_SRC="${TB2_SRC:-}"   # terminal-bench-2 checkout, only to generate a missing tb2_tc task
TC="${TC:-}"
DATASET_FORCED="${DATASET:-}"
if [ -z "$DATASET_FORCED" ]; then
    [ "$TC" = 1 ] && DATASET="$TC_DATASET" || DATASET="$CLASSIC_DATASET"
fi
BUDGET="${BUDGET:-0}"
MAX_DUR="${MAX_DUR:-}"
MAX_DUR_SCALE="${MAX_DUR_SCALE:-1}"
EFFORT="${EFFORT:-low}"
case "$EFFORT" in
  low)          _THINK_BUDGET=8000 ;;
  medium)       _THINK_BUDGET=16000 ;;
  high|xhigh)   _THINK_BUDGET=32000 ;;
  off)          _THINK_BUDGET=-1 ;;   # reasoning disabled (see EFFORT in the header)
  *) echo "ERROR: EFFORT='$EFFORT' is not one of low|medium|xhigh|off (high = xhigh)." >&2; exit 1 ;;
esac
THINK_BUDGET="${THINK_BUDGET:-$_THINK_BUDGET}"
ANSWER_TOKENS="${ANSWER_TOKENS:-8000}"
DISALLOWED_TOOLS="${DISALLOWED_TOOLS-Agent,WebFetch,WebSearch}"
# PARALLEL_TOOLS=1 (default): appends a system-prompt line asking Claude Code to batch independent
# tool calls into one turn (fewer model round trips per trial). 0 = Claude Code's stock prompt.
PARALLEL_TOOLS="${PARALLEL_TOOLS:-1}"
PARALLEL_TOOLS_PROMPT="When several tool calls are independent of each other (reading multiple files, running searches, checking several things at once), issue them together in a single turn rather than one per turn. Only sequence calls when a later one needs an earlier one's result."
AUTO_SCALE_BASE_TPS="${AUTO_SCALE_BASE_TPS:-100}"
PROXY_FILTERS="${PROXY_FILTERS:-1}"
NAS_HOST="${NAS_HOST:-192.168.2.15}"
SERVER_URL="${SERVER_URL:-http://$NAS_HOST:3002}"

LANGS=" python go rust javascript java cpp "

# Print one "<dataset>\t<max_dur or ->\t<include glob>" line per entry of a
# fromfile-format list (polyglot rows, or TB2 rows `<task> [calls] [max_dur]` for a
# task dir under tasks/tb2; dataset per the TC rules above), generating any missing
# __tc<count> test-calls task on the way.
list_entries() {
    local file="$1" warned=0 line lang ex tries dur
    [ -f "$file" ] || { echo "list file not found: $file" >&2; exit 1; }
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        read -r lang ex tries dur <<<"$line"
        [ -z "$lang" ] && continue
        if [[ "$LANGS" != *" $lang "* && -d "$TB2_DATASET/$lang" ]]; then
            # Terminal-Bench 2 row: <task> [calls] [max_dur]; calls -> tb2_tc/<task>__tc<calls>
            local task="$lang" calls="$ex" tds="$TB2_DATASET" tglob="$lang" tdir
            dur="$tries"
            [[ "$calls" =~ ^[0-9]+$ ]] || { calls=""; dur="$ex"; }
            [[ "$dur" =~ ^[0-9]+$ ]] || dur=""
            if [ -n "$DATASET_FORCED" ]; then
                tds="$DATASET_FORCED"
            elif [ "$TC" = 1 ] || { [ -z "$TC" ] && [ -n "$calls" ]; }; then
                tds="$TB2_TC_DATASET"
                if [ -n "$calls" ]; then
                    tglob="${task}__tc$calls"
                    if [ ! -d "$tds/$tglob" ]; then
                        [ -n "$TB2_SRC" ] || { echo "missing $tds/$tglob: set TB2_SRC=<terminal-bench-2 checkout> to generate it" >&2; exit 1; }
                        echo "generating missing test-calls task $tglob" >&2
                        python3 tb2/main.py --src "$TB2_SRC" --tasks "$task" --test-calls "$calls" >&2
                    fi
                else
                    tglob="${task}__tc*"
                fi
            fi
            if [ -z "$dur" ] && [ -z "$MAX_DUR" ]; then   # the task's own agent timeout
                tdir=$(cd "$tds" 2>/dev/null && compgen -G "$tglob" | head -1)
                [ -n "$tdir" ] && dur=$(awk '/^\[agent\]/{a=1;next} /^\[/{a=0} a && /^timeout_sec/{print int($3); exit}' "$tds/$tdir/task.toml")
            fi
            printf '%s\t%s\t%s\n' "$tds" "${dur:--}" "$tglob"
            continue
        fi
        if [[ "$LANGS" != *" $lang "* ]]; then
            ex="$lang"; lang="*"; tries=""; dur=""   # bare exercise: all languages, no overrides
        fi
        [[ "$tries" =~ ^[0-9]+$ ]] || tries=""
        local glob="polyglot_${lang}_${ex}" ds="$CLASSIC_DATASET"
        if [ -n "$DATASET_FORCED" ]; then
            ds="$DATASET_FORCED"
        elif [ "$TC" = 1 ] || { [ -z "$TC" ] && [ -n "$tries" ]; }; then
            ds="$TC_DATASET"
            # test-calls variant whose limit is this row's try count (any, if none given)
            if [ -n "$tries" ]; then
                glob+="__tc$tries"
                if [ "$lang" != "*" ] && [ ! -d "$TC_DATASET/$glob" ]; then
                    echo "generating missing test-calls task $glob" >&2
                    ORACLE_SECRET=$(openssl rand -hex 16) python3 aider_polyglot/main.py \
                        --output-dir "$TC_DATASET" --task-ids "${lang}_${ex}" --test-calls "$tries" \
                        2>&1 | grep -E "^(Generated|Error)" >&2
                fi
            else
                glob+="__tc*"
            fi
        elif [ "$TC" = 0 ] && [ -n "$tries" ] && [ $warned -eq 0 ]; then
            echo "note: TC=0, ignoring the try-count column in $file" >&2
            warned=1
        fi
        printf '%s\t%s\t%s\n' "$ds" "${dur:--}" "$glob"
    done <"$file"
}

# The dataset's own per-task agent timeout (polyglot: 1800), which
# --agent-timeout-multiplier scales.
task_agent_timeout() {
    local toml
    toml=$(ls "$DATASET"/*/task.toml | head -1)
    awk '/^\[agent\]/{a=1;next} /^\[/{a=0} a && /^timeout_sec/{print int($3); exit}' "$toml"
}

# srv.* key=value lines for whatever server is behind the proxy: Strata via its
# /props /metrics /config + a 1-token probe of the effective sampler; llama.cpp via
# `docker inspect` of the server container over ssh (non-perf flags only).
# decode tok/s of the model server (server-capture.sh's measure_decode_tps, as run.sh uses)
decode_tps() (
    LLAMA_SERVER_URL="$SERVER_URL" LLAMA_HEALTH_URL="$SERVER_URL/health" MODEL="${model:-x}"
    TARGET=llama-nas
    curl -sf --max-time 10 "$SERVER_URL/health" | grep -q '"service": *"strata"' && TARGET=strata
    source ../lib/server-capture.sh
    measure_decode_tps
)

capture_server() (
    LLAMA_SERVER_URL="$SERVER_URL" LLAMA_DOCKER=(docker -H "ssh://$NAS_HOST")
    MODEL="${model:-x}" TEMPERATURE=server SEED=0 EXTRA_BODY=""
    source ../lib/server-capture.sh
    if curl -sf --max-time 10 "$SERVER_URL/health" | grep -q '"service": *"strata"'; then
        echo "srv.target=strata"
        # the probe has to send a seed; harbor requests never do (Strata picks one per request)
        strata_settings_effective | grep -v '^srv\.eff\.seed_from='
        echo "srv.eff.seed_from=server"
        # thinking settings as a Claude Code request resolves them: through the proxy
        # (filters apply) with the effort harbor.sh passes
        if [ "${agent:-}" = claude-code ]; then
            probe_body="{\"output_config\": {\"effort\": \"$EFFORT\"}, \"thinking\": {\"type\": \"adaptive\"}}"
            # eval_proxy.py rewrites thinking to the explicit THINK_BUDGET; probe what it sends
            [ "$THINK_BUDGET" -gt 0 ] && probe_body="{\"output_config\": {\"effort\": \"$EFFORT\"}, \"thinking\": {\"type\": \"enabled\", \"budget_tokens\": $THINK_BUDGET}, \"reasoning_budget_tokens\": $THINK_BUDGET}"
            [ "$EFFORT" = off ] && probe_body='{"thinking": {"type": "disabled"}}'
            probe_strata_reasoning "$BASE_URL" "$probe_body" \
                "X-Eval-Budget: 1000" $([ "$PROXY_FILTERS" = 0 ] && echo "X-Eval-Filters:0")
        fi
    else
        echo "srv.target=llama"
        capture_srv_settings "$(detect_llama_container)"
    fi
    echo "srv.proxy_filters=$PROXY_FILTERS"
)

model_id() {
    if [ -n "${MODEL:-}" ]; then echo "$MODEL"; return; fi
    curl -sf --max-time 10 "$BASE_URL/v1/models" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
}

cmd="${1:-}"; shift || true
case "$cmd" in
gen)
    # Fresh secret per generation: upstream's default oracle key is public, and
    # the encrypted reference solution sits in the agent's workspace.
    ORACLE_SECRET=$(openssl rand -hex 16) python3 aider_polyglot/main.py "$@"
    ;;
run)
    agent="${1:?usage: harbor.sh run <agent> [harbor args...]}"; shift
    passthru=()
    declare -A groups=()   # "<dataset>|<max_dur or ->" -> newline-separated include globs
    while [ $# -gt 0 ]; do
        if [ "$1" = "--list" ]; then
            # resolve relative to the caller's cwd, not this script's dir
            f=$(resolve_list "$2"); list_file="$f"
            n=0
            while IFS=$'\t' read -r ds dur glob; do
                groups["$ds|$dur"]+="$glob"$'\n'; n=$((n + 1))
            done < <(list_entries "$f")
            [ $n -gt 0 ] || { echo "no entries in $f" >&2; exit 1; }
            shift 2
        else
            passthru+=("$1"); shift
        fi
    done
    [ ${#groups[@]} -gt 0 ] || groups["$DATASET|-"]=""

    # Known test totals (aider_polyglot/expected-totals.tsv -> tasks' tests/expected_total) let the
    # verifier count tests that never ran as failed. For any selected exercise without
    # one, first run the oracle on its classic task (no model needed, so before any
    # SWAP_CMD), let collect.py record the total, and regenerate the selected tasks of
    # that exercise. An exercise whose oracle doesn't pass stays unknown (warned, and
    # retried next run). EXPECTED_PREFILL=0 skips this (nop: only with EXPECTED_PREFILL=1).
    if [ "${EXPECTED_PREFILL:-1}" = 1 ] && [ "$agent" != oracle ] \
        && { [ "$agent" != nop ] || [ "${EXPECTED_PREFILL:-}" = 1 ]; }; then
        pre_globs=()
        for ((i = 0; i < ${#passthru[@]}; i++)); do
            case "${passthru[$i]}" in -i|--include) pre_globs+=("${passthru[$((i + 1))]}") ;; esac
        done
        need=()   # "<dataset>/<task>" whose exercise has no known total
        for key in "${!groups[@]}"; do
            ds="${key%%|*}"
            g=(); while IFS= read -r x; do [ -n "$x" ] && g+=("$x"); done <<<"${groups[$key]}"
            g+=("${pre_globs[@]}"); [ ${#g[@]} -gt 0 ] || g=("*")
            while IFS= read -r t; do
                [[ "$t" =~ ^polyglot_([a-z]+)_(.+)$ ]] || continue
                lang="${BASH_REMATCH[1]}"; ex="${BASH_REMATCH[2]%__tc*}"
                awk -F'\t' -v l="$lang" -v e="$ex" '$1 == l && $2 == e { f = 1 } END { exit !f }' \
                    aider_polyglot/expected-totals.tsv 2>/dev/null || need+=("$ds/$t")
            done < <(cd "$ds" && for x in "${g[@]}"; do compgen -G "$x" || true; done | sort -u)
        done
        if [ ${#need[@]} -gt 0 ]; then
            exs=$(printf '%s\n' "${need[@]}" | sed -E 's#.*/polyglot_([a-z]+)_(.+)$#\1_\2#; s/__tc[0-9]+$//' | sort -u)
            echo "EXPECTED_PREFILL: no known test total for $(echo $exs | wc -w) exercise(s); running the oracle first: $(echo $exs)" >&2
            oi=(); for e in $exs; do oi+=(-i "polyglot_$e"); done
            ( unset SWAP_CMD; CHECK="auto: expected test totals (EXPECTED_PREFILL)" TC=0 DATASET="" \
                SKIP_EXISTING=0 MAX_DUR="" MAX_DUR_SCALE=1 "$PWD/harbor.sh" run oracle "${oi[@]}" ) \
                || echo "EXPECTED_PREFILL: oracle run failed; continuing without totals" >&2
            # that run's collect.py updated aider_polyglot/expected-totals.tsv; regenerate what we're about to run
            ORACLE_SECRET=$(openssl rand -hex 16) python3 aider_polyglot/main.py \
                --task-ids "$(echo $exs | tr ' ' ',')" >/dev/null
            for p in "${need[@]}"; do
                [[ "$p" =~ /polyglot_([a-z]+)_(.+)__tc([0-9]+)$ ]] || continue
                ORACLE_SECRET=$(openssl rand -hex 16) python3 aider_polyglot/main.py \
                    --output-dir "$TC_DATASET" --task-ids "${BASH_REMATCH[1]}_${BASH_REMATCH[2]}" \
                    --test-calls "${BASH_REMATCH[3]}" >/dev/null
            done
            for e in $exs; do
                awk -F'\t' -v k="$e" '$1 "_" $2 == k { f = 1 } END { exit !f }' aider_polyglot/expected-totals.tsv \
                    || echo "EXPECTED_PREFILL: WARNING: oracle didn't pass for $e; its test total stays unknown" >&2
            done
        fi
    fi

    base_timeout=$(task_agent_timeout)
    [ "$agent" = claude-code ] && CC_VERSION=${CC_VERSION:-$(sed -n 's/^ARG CC_VERSION=//p' aider_polyglot/task-template/Dockerfile 2>/dev/null)}
    # SWAP_CMD='ssh ... llm strata foo': command that loads the model to evaluate. With
    # SKIP_EXISTING=1 it only runs if some task still needs attempts. The server's settings
    # (needed to match earlier trials) are unknown until it's loaded, so the pre-check uses
    # the newest earlier job that recorded the same SWAP_CMD (jobs/*/swap-cmd.txt) and its
    # server-settings / decode speed. No such job -> assume work is needed and swap. The
    # real run re-checks against the live server, so a wrong guess only costs a swap.
    if [ -n "${SWAP_CMD:-}" ] && ! [[ "$agent" =~ ^(oracle|nop)$ ]]; then
        if [ "${SKIP_EXISTING:-0}" = 1 ]; then
            prev=""
            while IFS= read -r d; do   # newest first
                [ -f "$d/swap-cmd.txt" ] && [ -f "$d/server-settings.txt" ] \
                    && [ "$(cat "$d/swap-cmd.txt")" = "$SWAP_CMD" ] && { prev="$d"; break; }
            done < <(ls -dt jobs/*/ 2>/dev/null | sed 's#/$##')
            if [ -n "$prev" ]; then
                pk=1; pglobs=()
                for ((i = 0; i < ${#passthru[@]}; i++)); do
                    case "${passthru[$i]}" in
                    -k|--n-attempts) pk="${passthru[$((i + 1))]}" ;;
                    -i|--include) pglobs+=("${passthru[$((i + 1))]}") ;;
                    esac
                done
                pscale=1
                if [ "$MAX_DUR_SCALE" = auto ]; then
                    ptps=$(sed -n 's/^decode_tps=//p' "$prev/run-settings.txt" 2>/dev/null)
                    pscale=$(awk -v t="${ptps:-0}" -v b="$AUTO_SCALE_BASE_TPS" 'BEGIN { s = (t > 0) ? b / t : 1; if (s < 1) s = 1; printf "%.2f", s }')
                else pscale="$MAX_DUR_SCALE"; fi
                todo=0
                for key in "${!groups[@]}"; do
                    pds="${key%%|*}"; pdur="${key#*|}"; [ "$pdur" = "-" ] && pdur="$MAX_DUR"; pbase="${pdur:-$base_timeout}"
                    if [ "$pscale" != 1 ]; then pdur=$(awk -v d="${pdur:-$base_timeout}" -v s="$pscale" 'BEGIN { printf "%.0f", d * s }'); fi
                    pg=(); while IFS= read -r g; do [ -n "$g" ] && pg+=("$g"); done <<<"${groups[$key]}"
                    pg+=("${pglobs[@]}"); [ ${#pg[@]} -gt 0 ] || pg=("*")
                    while IFS=$'\t' read -r rem _; do todo=$((todo + rem)); done < <(
                        python3 skip_existing.py --jobs jobs --dataset "$pds" --agent "$agent" ${CC_VERSION:+--agent-version "$CC_VERSION"} \
                            --budget "$BUDGET" --think-budget "$THINK_BUDGET" --max-dur "${pdur:-$base_timeout}" --max-dur-base "$pbase" --k "$pk" --server "$prev/server-settings.txt" "${pg[@]}")
                done
                if [ "$todo" -eq 0 ]; then
                    echo "SWAP_CMD: nothing left to run (per $prev); not swapping models" >&2
                    exit 0
                fi
                echo "SWAP_CMD: $todo attempt(s) still needed; swapping" >&2
            else
                echo "SWAP_CMD: no earlier job recorded for this command; swapping" >&2
            fi
        fi
        eval "$SWAP_CMD" || { echo "SWAP_CMD failed" >&2; exit 1; }
    fi

    agent_spec="$agent"
    hdrs="X-Eval-Budget: $BUDGET"
    [ "$PROXY_FILTERS" = 0 ] && hdrs+=$'\nX-Eval-Filters: 0'
    agent_args=()
    case "$agent" in
    oracle|nop)
        label="$agent"
        ;;
    claude-code)
        model=$(model_id); label="$model"
        # Server ignores the Anthropic model name, but Harbor still wants provider/model.
        export ANTHROPIC_BASE_URL="$EVAL_URL" ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOCAL:-local}"
        # pinned agent version: matches the image's pre-installed build (task-template/Dockerfile
        # CC_VERSION) so Harbor skips its per-trial install; also part of SKIP_EXISTING's match
        CC_VERSION=${CC_VERSION:-$(sed -n 's/^ARG CC_VERSION=//p' aider_polyglot/task-template/Dockerfile 2>/dev/null)}
        parallel_args=()
        [ "$PARALLEL_TOOLS" = 1 ] && parallel_args=(--ak "append_system_prompt=$PARALLEL_TOOLS_PROMPT")
        # EFFORT=off: Claude Code's effort level must still be a real level, and thinking is switched off
        claude_effort=$EFFORT; think_off_args=()
        if [ "$EFFORT" = off ]; then claude_effort=low; think_off_args=(--ae CLAUDE_CODE_DISABLE_THINKING=1); fi
        agent_args=(-m "anthropic/$model" ${CC_VERSION:+--ak "version=$CC_VERSION"}
            ${DISALLOWED_TOOLS:+--ak "disallowed_tools=$DISALLOWED_TOOLS"}
            "${parallel_args[@]}"
            # no telemetry/autoupdate/background title calls hitting the model server
            --ae CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
            --ae "CLAUDE_CODE_EFFORT_LEVEL=$claude_effort"
            "${think_off_args[@]}"
            # newline-separated; proxy keys the budget on Claude Code's per-session metadata.user_id
            --ae "ANTHROPIC_CUSTOM_HEADERS=$hdrs")
        ;;
    terminus-2)
        model=$(model_id); label="$model"
        export OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL:-local}"
        # budget rejection -> graded agent exit instead of an ungraded trial (harbor_ext.py)
        agent_spec="harbor_ext:BudgetedTerminus2"
        export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
        hdr_json=$(printf '%s' "$hdrs" | python3 -c 'import json,sys; print(json.dumps(dict(l.split(": ",1) for l in sys.stdin.read().splitlines())))')
        agent_args=(-m "openai/$model" --ak "api_base=$EVAL_URL/v1"
            # per-trial session uuid -> X-Eval-Session (the proxy's budget key)
            --ak 'session_id_headers=["X-Eval-Session"]'
            --ak "llm_call_kwargs={\"extra_headers\": $hdr_json}")
        ;;
    *)
        echo "no wiring for agent '$agent' yet (add a case in harbor.sh)" >&2; exit 1
        ;;
    esac
    case "$agent" in claude-code) tail_part=cc ;; terminus-2) tail_part=t2 ;; *) tail_part="$agent" ;; esac
    [ "$BUDGET" -gt 0 ] && tail_part+="-b$(( BUDGET >= 1000 ? BUDGET / 1000 : BUDGET ))$([ "$BUDGET" -ge 1000 ] && echo k)"
    run_id="${JOB_NAME:-$(date +%m%d-%H%M%S)}"
    list_name=""; [ -n "${list_file:-}" ] && { list_name=$(basename "$list_file"); list_name="${list_name%.*}"; }
    # job name: run id, then what's in it (short task names), then the rest; see JOB_NAME above
    job_name() {   # job_name <suffix> <task or glob>...
        local suffix="$1"; shift
        local tasks i
        if [ $# -eq 0 ]; then   # no list: name it after any -i globs passed straight to harbor
            set -- "${cli_globs[@]}"
        fi
        if [ $# -eq 0 ]; then tasks="${list_name:-all}"
        elif [ $# -le 4 ]; then
            tasks=$(printf '%s\n' "$@" | sed -E 's/^polyglot_//; s/__tc([0-9]+)$/.tc\1/; s/\*/all/g; s/_/-/' | paste -sd+)
        else tasks="${list_name:-tasks}.$#"; printf '%s\n' "$@" | grep -q '__tc' && tasks+=".tc"
        fi
        printf '%s_%s%s_%s' "$run_id" "${tasks//[^A-Za-z0-9.+_-]/-}" "$suffix" "$tail_part"
    }

    srv_before=""
    [[ "$agent" =~ ^(oracle|nop)$ ]] || srv_before=$(capture_server 2>/dev/null || true)
    # llama's /v1/models only reports its fixed --alias (e.g. Qwen3.6-35B-A3B) whatever
    # GGUF is loaded, so name the model after the GGUF (-m) instead -- that's what Harbor
    # records (UI, collect.py's model column). eval_proxy puts the alias back on the wire.
    server_model="${model:-}"
    gguf=$(printf '%s\n' "$srv_before" | sed -n 's/^srv\.-m=//p' | head -1)
    if [ -z "${MODEL:-}" ] && [ -n "$gguf" ] && printf '%s\n' "$srv_before" | grep -q '^srv.target=llama'; then
        model=$(basename "$gguf" .gguf | sed -E 's/-[0-9]{5}-of-[0-9]{5}$//')
        agent_args[1]="${agent_args[1]%%/*}/$model"
        label="$model"
        echo "model: $model (llama alias $server_model)" >&2
    fi

    # -i globs are resolved here (the job lists task paths, which Harbor's -i doesn't
    # filter). SKIP_EXISTING=1: -k N means "make sure N graded attempts exist"
    # (skip_existing.py has the matching rules); otherwise -k goes to Harbor as is.
    want_k=1; rest=(); cli_globs=()
    i=0
    while [ $i -lt ${#passthru[@]} ]; do
        case "${passthru[$i]}" in
        -k|--n-attempts) want_k="${passthru[$((i + 1))]}"
            [ "${SKIP_EXISTING:-0}" = 1 ] || rest+=("${passthru[@]:$i:2}"); i=$((i + 2)) ;;
        -i|--include) cli_globs+=("${passthru[$((i + 1))]}"); i=$((i + 2)) ;;
        *) rest+=("${passthru[$i]}"); i=$((i + 1)) ;;
        esac
    done
    passthru=("${rest[@]}")
    if [ "${SKIP_EXISTING:-0}" = 1 ]; then
        [ -n "$srv_before" ] || { echo "SKIP_EXISTING needs a model agent (server settings to match)" >&2; exit 1; }
        srv_file=$(mktemp); printf '%s\n' "$srv_before" > "$srv_file"
    fi

    # one Harbor job: dataset, job name, then extra harbor args
    trap 'kill ${proxy_pid:-} ${watch_pid:-} ${progress_pid:-} 2>/dev/null || true' EXIT
    run_job() {   # run_job <job name> <harbor args (task source, ...)>
        local job="$1"; shift
        # run-level settings that aren't server settings -> collect.py columns
        mkdir -p "jobs/$job"
        [ -z "${SWAP_CMD:-}" ] || printf '%s\n' "$SWAP_CMD" > "jobs/$job/swap-cmd.txt"
        # CHECK=<reason>: a harness test, not a model result (collect.py -> harbor-trials-checks.tsv)
        [ -z "${CHECK:-}" ] || printf '%s\n' "$CHECK" > "jobs/$job/CHECK"
        printf 'max_dur_scale=%s\nmax_dur_base=%s\ndecode_tps=%s\n' \
            "$MAX_DUR_SCALE" "${base_dur:-}" "${decode_tps_val:-}" > "jobs/$job/run-settings.txt"
        printf 'think_budget=%s\nanswer_tokens=%s\n' "$THINK_BUDGET" "$ANSWER_TOKENS" >> "jobs/$job/run-settings.txt"
        [ "$agent" = claude-code ] && printf 'disallowed_tools=%s\nparallel_tools=%s\n' "$DISALLOWED_TOOLS" "$PARALLEL_TOOLS" >> "jobs/$job/run-settings.txt"
        proxy_pid=""   # global: the EXIT trap below stops it if we're interrupted
        if ! [[ "$agent" =~ ^(oracle|nop)$ ]]; then
            python3 ../lib/eval_proxy.py --port "$EVAL_PROXY_PORT" --upstream "$BASE_URL" \
                ${server_model:+--model "$server_model"} \
                --think-budget "$THINK_BUDGET" --answer-tokens "$ANSWER_TOKENS" \
                --log "jobs/$job/eval-proxy.tsv" > "jobs/$job/eval-proxy.out" 2>&1 &
            proxy_pid=$!
            local i
            for i in $(seq 50); do grep -q listening "jobs/$job/eval-proxy.out" 2>/dev/null && break; sleep 0.1; done
            grep -q listening "jobs/$job/eval-proxy.out" || { echo "eval_proxy failed to start:" >&2; cat "jobs/$job/eval-proxy.out" >&2; kill $proxy_pid 2>/dev/null; rc=1; return; }
        fi
        # rebuild results/harbor-trials.tsv as each trial finishes: Harbor has no per-trial
        # hook, but a trial's result.json only appears once it's done, so watch for those
        # (set +e: this script's -e/pipefail would end the loop on the first empty glob)
        (
            set +eo pipefail
            seen=0
            while sleep 2; do
                n=$(compgen -G "jobs/$job/*/result.json" | wc -l)
                if [ "$n" != "$seen" ]; then seen=$n; python3 collect.py jobs >/dev/null 2>&1; fi
            done
        ) &
        watch_pid=$!
        # PROGRESS_INTERVAL=<secs> (default 180, 0 = off): grade each running trial's current
        # /app in a side container every that many seconds -> <trial>/progress.tsv
        # (progress_sampler.py; the agent sees nothing). PROGRESS_TIMEOUT caps each test run.
        progress_pid=""
        if [ "${PROGRESS_INTERVAL:-180}" != 0 ] && ! [[ "$agent" =~ ^(oracle|nop)$ ]]; then
            python3 progress_sampler.py "jobs/$job" --interval "${PROGRESS_INTERVAL:-180}" \
                --timeout "${PROGRESS_TIMEOUT:-90}" > "jobs/$job/progress-sampler.log" 2>&1 &
            progress_pid=$!
        fi
        # written up front (Harbor creates jobs/$job immediately) so the live collect above
        # already sees srv.* -- without it, mid-run trials have no srv.-m and the viz shows
        # them under the bare agent model alias as a separate model
        if [ -n "$srv_before" ]; then
            ( for _ in $(seq 60); do [ -d "jobs/$job" ] && break; sleep 1; done
              [ -d "jobs/$job" ] && printf '%s\n' "$srv_before" > "jobs/$job/server-settings.txt" ) &
        fi
        "${HARBOR[@]}" run -y -a "$agent_spec" "${agent_args[@]}" \
            -n 1 -o jobs --job-name "$job" "$@" "${passthru[@]}" || rc=$?
        kill "$watch_pid" 2>/dev/null || true
        [ -z "$progress_pid" ] || kill "$progress_pid" 2>/dev/null || true
        [ -z "$proxy_pid" ] || kill "$proxy_pid" 2>/dev/null || true
        if [ -n "$srv_before" ] && [ -d "jobs/$job" ]; then
            [ -f "jobs/$job/server-settings.txt" ] || printf '%s\n' "$srv_before" > "jobs/$job/server-settings.txt"
            srv_after=$(capture_server 2>/dev/null || true)
            if [ "$srv_after" != "$srv_before" ]; then
                printf '%s\n' "$srv_after" > "jobs/$job/server-settings-after.txt"
                echo "WARNING: server settings changed during $job (see server-settings-after.txt)" >&2
            fi
        fi
    }

    if [ "$MAX_DUR_SCALE" = auto ]; then
        if [[ "$agent" =~ ^(oracle|nop)$ ]]; then MAX_DUR_SCALE=1
        else
            tps=$(decode_tps); decode_tps_val="$tps"
            awk -v t="$tps" 'BEGIN { exit !(t > 0) }' || {
                echo "ERROR: MAX_DUR_SCALE=auto couldn't measure decode speed; set MAX_DUR_SCALE=<number>" >&2; exit 1; }
            MAX_DUR_SCALE=$(awk -v t="$tps" -v b="$AUTO_SCALE_BASE_TPS" 'BEGIN { s = b / t; if (s < 1) s = 1; printf "%.2f", s }')
            echo "MAX_DUR_SCALE=auto: decode $tps tok/s vs $AUTO_SCALE_BASE_TPS baseline -> ${MAX_DUR_SCALE}x (min 1x)" >&2
        fi
    fi

    # Every (dataset, time limit) group goes into ONE Harbor job. Harbor applies a single
    # timeout multiplier per job, so a task whose limit differs from its task.toml runs
    # from a copy with the limit written in: tasks/_dur/<secs>/<same task name> (same
    # name, so collect.py / SKIP_EXISTING see the same task; both read the limit from
    # the task.toml a trial actually ran). The job gets a generated config listing each
    # task path once per attempt to run, so per-task attempt counts (SKIP_EXISTING) fit too.
    rc=0
    task_paths=()   # one entry per trial to run
    names=()        # distinct task names, for the job name
    base_durs=()
    variant() {     # variant <task dir> <secs> -> path of a copy whose agent timeout is <secs>
        local src="$1" secs="$2" dst
        if [ "$(awk '/^\[agent\]/{a=1;next} /^\[/{a=0} a && /^timeout_sec/{print int($3); exit}' "$src/task.toml")" = "$secs" ]; then
            echo "$PWD/$src"; return
        fi
        dst="tasks/_dur/$secs/$(basename "$src")"
        mkdir -p "$dst"
        rsync -a --delete "$src/" "$dst/"
        awk -v t="$secs" '/^\[/{a=($0=="[agent]")} a && /^timeout_sec/{$0="timeout_sec = " t ".0"} {print}' \
            "$src/task.toml" > "$dst/task.toml"
        # instructions that state the limit (tb2_tc) get the copy's limit
        sed -i -E "s/^Your time limit is [0-9]+ minutes;/Your time limit is $(( (secs + 30) / 60 )) minutes;/" \
            "$dst/instruction.md"
        echo "$PWD/$dst"
    }
    for key in "${!groups[@]}"; do
        ds="${key%%|*}"; dur="${key#*|}"; [ "$dur" = "-" ] && dur="$MAX_DUR"
        # scale every limit (an unset one = the task default) for slower models
        base_dur="${dur:-$base_timeout}"; base_durs+=("$base_dur")
        if [ "$MAX_DUR_SCALE" != 1 ]; then
            dur=$(awk -v d="${dur:-$base_timeout}" -v s="$MAX_DUR_SCALE" 'BEGIN { printf "%.0f", d * s }')
        fi
        globs=()
        while IFS= read -r glob; do [ -n "$glob" ] && globs+=("$glob"); done <<<"${groups[$key]}"
        globs+=("${cli_globs[@]}")
        [ ${#globs[@]} -gt 0 ] || globs=("*")

        if [ "${SKIP_EXISTING:-0}" != 1 ]; then
            while IFS= read -r t; do
                task_paths+=("$(variant "$ds/$t" "${dur:-$base_timeout}")"); names+=("$t")
            done < <(cd "$ds" && for g in "${globs[@]}"; do compgen -G "$g" || true; done | sort -u)
            continue
        fi
        skipped=0
        while IFS=$'\t' read -r remaining task have_n have_jobs; do
            if [ "$remaining" -eq 0 ]; then
                skipped=$((skipped + 1))
                echo "SKIP_EXISTING: skip $task: $have_n matching graded attempt(s) >= -k $want_k (in ${have_jobs})" >&2
                continue
            fi
            [ "$have_n" -gt 0 ] && echo "SKIP_EXISTING: $task has $have_n of $want_k (in ${have_jobs}); running $remaining more" >&2
            p=$(variant "$ds/$task" "${dur:-$base_timeout}"); names+=("$task")
            for ((i = 0; i < remaining; i++)); do task_paths+=("$p"); done
        done < <(python3 skip_existing.py --jobs jobs --dataset "$ds" --agent "$agent" ${CC_VERSION:+--agent-version "$CC_VERSION"} \
                    --budget "$BUDGET" --think-budget "$THINK_BUDGET" --max-dur "${dur:-$base_timeout}" --max-dur-base "$base_dur" --k "$want_k" --server "$srv_file" "${globs[@]}")
        [ $skipped -gt 0 ] && echo "SKIP_EXISTING: skipped $skipped task(s) in $ds (listed above)" >&2
    done
    if [ ${#task_paths[@]} -eq 0 ]; then
        echo "nothing to run" >&2
    else
        # one base limit for the whole run -> run-settings.txt; mixed = blank (collect.py derives it)
        base_dur=$(printf '%s\n' "${base_durs[@]}" | sort -u); [[ "$base_dur" == *$'\n'* ]] && base_dur=""
        job=$(job_name "" $(printf '%s\n' "${names[@]}" | sort -u))
        mkdir -p "jobs/$job"
        printf '%s\n' "${task_paths[@]}" | python3 -c 'import json,sys; print(json.dumps({"tasks": [{"path": l.strip()} for l in sys.stdin if l.strip()]}, indent=1))' \
            > "jobs/$job/harbor-tasks.json"
        echo "job $job: ${#task_paths[@]} trial(s) of $(printf '%s\n' "${names[@]}" | sort -u | wc -l) task(s)" >&2
        run_job "$job" -c "jobs/$job/harbor-tasks.json"
    fi
    # keep results/harbor-trials.tsv current (it's a full rebuild from jobs/, so cheap to redo)
    python3 collect.py jobs >/dev/null || echo "WARNING: collect.py failed; run ./harbor/harbor.sh collect" >&2
    exit $rc
    ;;
gen-tc)
    if [ "${1:-}" = "--list" ]; then
        f=$(resolve_list "${2:?gen-tc --list FILE}")
        sel=(--test-calls-file "$f")
    else
        sel=(--task-ids "${1:?language}_${2:?exercise}" --test-calls "${3:?calls}")
    fi
    ORACLE_SECRET=$(openssl rand -hex 16) python3 aider_polyglot/main.py \
        --output-dir "$TC_DATASET" "${sel[@]}"
    ;;
collect)
    python3 collect.py jobs "$@"
    ;;
*)
    sed -n '4,25p' "$0"; exit 1
    ;;
esac
