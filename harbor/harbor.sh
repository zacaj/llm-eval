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
#   ./harbor/harbor.sh run claude-code --list harbor/lists/smoke-per-language.txt -k 3
#
# --list FILE: exercise list in run.sh's fromfile format ("<language> <exercise>
# [max_tries] [max_dur]" or bare "<exercise>" = every language; "#" comments), so
# the results/*.txt shortlists work as-is. Each entry becomes an `-i` include
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
#              Job names / max_dur column / SKIP_EXISTING use the scaled limit.
#   BUDGET     per-trial output-token budget enforced by the prompt-logger proxy's eval
#              mode (default 0 = unlimited). Model requests always carry X-Eval-Budget,
#              which also keeps them off the proxy's routes (one model per eval) -- see
#              the "Eval mode" section of the proxy's docstring.
#   SERVER_URL model server itself, not the proxy (default http://$NAS_HOST:8080); its
#              sampling/quality settings are saved to jobs/<job>/server-settings.txt
#              (same capture as run.sh's srv.* columns, via ../server-capture.sh)
#   SKIP_EXISTING  1 = treat -k N as "ensure N graded attempts" per task and only run
#              what's missing, counting matching earlier trials from any job (same
#              agent+version/BUDGET/time limit (or finished within it)/server sampling+model; INVALID and ungraded
#              infra errors don't count) -- rerun the same command to resume
#   CC_VERSION Claude Code version to run (default: the one pre-installed in the task
#              images, ARG CC_VERSION in adapters/aider_polyglot/task-template/Dockerfile;
#              a different value makes Harbor install it per trial)
#   PROXY_FILTERS  1 (default) = the proxy's prompt filters (tuned for local models)
#              still rewrite Claude Code's system prompt/tools; 0 = stock prompts
#
# Defaults that differ from Harbor's: -n 1 (one model server slot); add -n N to override.
# Anything after <agent> is forwarded to `harbor run`, so later flags win.

cd "$(dirname "$0")"

HARBOR=(uvx harbor@0.25.0)
BASE_URL="${BASE_URL:-http://192.168.2.15:3002}"
# TC: unset = per list row (a row with a try count -> its __tc<count> test-calls task,
# otherwise the classic task; non-list runs are classic); 1 = test-calls for everything;
# 0 = classic for everything. An explicit DATASET overrides all of it.
CLASSIC_DATASET=tasks/aider_polyglot
TC_DATASET=tasks/aider_polyglot_tc
TC="${TC:-}"
DATASET_FORCED="${DATASET:-}"
if [ -z "$DATASET_FORCED" ]; then
    [ "$TC" = 1 ] && DATASET="$TC_DATASET" || DATASET="$CLASSIC_DATASET"
fi
BUDGET="${BUDGET:-0}"
MAX_DUR="${MAX_DUR:-}"
MAX_DUR_SCALE="${MAX_DUR_SCALE:-1}"
AUTO_SCALE_BASE_TPS="${AUTO_SCALE_BASE_TPS:-100}"
PROXY_FILTERS="${PROXY_FILTERS:-1}"
NAS_HOST="${NAS_HOST:-192.168.2.15}"
SERVER_URL="${SERVER_URL:-http://$NAS_HOST:8080}"

LANGS=" python go rust javascript java cpp "

# Print one "<dataset>\t<max_dur or ->\t<include glob>" line per entry of a
# fromfile-format list (dataset per the TC rules above), generating any missing
# __tc<count> test-calls task on the way.
list_entries() {
    local file="$1" warned=0 line lang ex tries dur
    [ -f "$file" ] || { echo "list file not found: $file" >&2; exit 1; }
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        read -r lang ex tries dur <<<"$line"
        [ -z "$lang" ] && continue
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
                    ORACLE_SECRET=$(openssl rand -hex 16) python3 adapters/aider_polyglot/main.py \
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
    source ../server-capture.sh
    measure_decode_tps
)

capture_server() (
    LLAMA_SERVER_URL="$SERVER_URL" LLAMA_DOCKER=(docker -H "ssh://$NAS_HOST")
    MODEL="${model:-x}" TEMPERATURE=server SEED=0 EXTRA_BODY=""
    source ../server-capture.sh
    if curl -sf --max-time 10 "$SERVER_URL/health" | grep -q '"service": *"strata"'; then
        echo "srv.target=strata"
        # the probe has to send a seed; harbor requests never do (Strata picks one per request)
        strata_settings_effective | grep -v '^srv\.eff\.seed_from='
        echo "srv.eff.seed_from=server"
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
    ORACLE_SECRET=$(openssl rand -hex 16) python3 adapters/aider_polyglot/main.py "$@"
    ;;
run)
    agent="${1:?usage: harbor.sh run <agent> [harbor args...]}"; shift
    passthru=()
    declare -A groups=()   # "<dataset>|<max_dur or ->" -> newline-separated include globs
    while [ $# -gt 0 ]; do
        if [ "$1" = "--list" ]; then
            # resolve relative to the caller's cwd, not this script's dir
            f="$2"; [[ "$f" = /* ]] || f="$OLDPWD/$f"; list_file="$f"
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
        export ANTHROPIC_BASE_URL="$BASE_URL" ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOCAL:-local}"
        # pinned agent version: matches the image's pre-installed build (task-template/Dockerfile
        # CC_VERSION) so Harbor skips its per-trial install; also part of SKIP_EXISTING's match
        CC_VERSION=${CC_VERSION:-$(sed -n 's/^ARG CC_VERSION=//p' adapters/aider_polyglot/task-template/Dockerfile 2>/dev/null)}
        agent_args=(-m "anthropic/$model" ${CC_VERSION:+--ak "version=$CC_VERSION"}
            # no telemetry/autoupdate/background title calls hitting the model server
            --ae CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
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
        agent_args=(-m "openai/$model" --ak "api_base=$BASE_URL/v1"
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
            for ((i = 0; i < ${#passthru[@]}; i++)); do
                [[ "${passthru[$i]}" =~ ^(-i|--include)$ ]] && set -- "$@" "${passthru[$((i + 1))]}"
            done
        fi
        if [ $# -eq 0 ]; then tasks="${list_name:-all}"
        elif [ $# -le 4 ]; then
            tasks=$(printf '%s\n' "$@" | sed -E 's/^polyglot_//; s/__tc([0-9]+)$/.tc\1/; s/\*/all/g; s/_/-/' | paste -sd+)
        else tasks="${list_name:-tasks}.$#"; [ "$ds" = "$TC_DATASET" ] && tasks+=".tc"
        fi
        printf '%s_%s%s_%s' "$run_id" "${tasks//[^A-Za-z0-9.+_-]/-}" "$suffix" "$tail_part"
    }
    base_timeout=$(task_agent_timeout)

    srv_before=""
    [[ "$agent" =~ ^(oracle|nop)$ ]] || srv_before=$(capture_server 2>/dev/null || true)

    # SKIP_EXISTING=1: -k N means "make sure N graded attempts exist" (skip_existing.py
    # has the matching rules); -i globs given directly are folded into the selection
    want_k=1; rest=(); cli_globs=()
    if [ "${SKIP_EXISTING:-0}" = 1 ]; then
        [ -n "$srv_before" ] || { echo "SKIP_EXISTING needs a model agent (server settings to match)" >&2; exit 1; }
        i=0
        while [ $i -lt ${#passthru[@]} ]; do
            case "${passthru[$i]}" in
            -k|--n-attempts) want_k="${passthru[$((i + 1))]}"; i=$((i + 2)) ;;
            -i|--include) cli_globs+=("${passthru[$((i + 1))]}"); i=$((i + 2)) ;;
            *) rest+=("${passthru[$i]}"); i=$((i + 1)) ;;
            esac
        done
        passthru=("${rest[@]}")
        srv_file=$(mktemp); printf '%s\n' "$srv_before" > "$srv_file"
    fi

    # one Harbor job: dataset, job name, then extra harbor args
    run_job() {
        local ds="$1" job="$2"; shift 2
        # run-level settings that aren't server settings -> collect.py columns
        mkdir -p "jobs/$job"
        printf 'max_dur_scale=%s\nmax_dur_base=%s\ndecode_tps=%s\n' \
            "$MAX_DUR_SCALE" "$base_dur" "${decode_tps_val:-}" > "jobs/$job/run-settings.txt"
        "${HARBOR[@]}" run -y -p "$ds" -a "$agent_spec" "${agent_args[@]}" \
            -n 1 -o jobs --job-name "$job" "$@" "${passthru[@]}" || rc=$?
        if [ -n "$srv_before" ] && [ -d "jobs/$job" ]; then
            printf '%s\n' "$srv_before" > "jobs/$job/server-settings.txt"
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

    rc=0
    for key in "${!groups[@]}"; do
        ds="${key%%|*}"; dur="${key#*|}"; [ "$dur" = "-" ] && dur="$MAX_DUR"
        # scale every limit (an unset one = the task default) for slower models
        base_dur="${dur:-$base_timeout}"
        if [ "$MAX_DUR_SCALE" != 1 ]; then
            dur=$(awk -v d="${dur:-$base_timeout}" -v s="$MAX_DUR_SCALE" 'BEGIN { printf "%.0f", d * s }')
        fi
        sfx=""; timeout_args=()
        if [ -n "$dur" ]; then
            timeout_args=(--agent-timeout-multiplier "$(awk -v d="$dur" -v t="$base_timeout" 'BEGIN { print d / t }')")
            [ ${#groups[@]} -gt 1 ] && sfx+="_d$dur"
        fi
        globs=()
        while IFS= read -r glob; do [ -n "$glob" ] && globs+=("$glob"); done <<<"${groups[$key]}"

        if [ "${SKIP_EXISTING:-0}" != 1 ]; then
            inc=(); for g in "${globs[@]}"; do inc+=(-i "$g"); done
            run_job "$ds" "$(job_name "$sfx" "${globs[@]}")" "${timeout_args[@]}" "${inc[@]}"
            continue
        fi

        globs+=("${cli_globs[@]}")
        [ ${#globs[@]} -gt 0 ] || globs=("*")
        declare -A need=()   # remaining attempts -> "-i task" args
        skipped=0
        while IFS=$'\t' read -r remaining task have_n have_jobs; do
            if [ "$remaining" -eq 0 ]; then
                skipped=$((skipped + 1))
                echo "SKIP_EXISTING: skip $task: $have_n matching graded attempt(s) >= -k $want_k (in ${have_jobs})" >&2
                continue
            fi
            [ "$have_n" -gt 0 ] && echo "SKIP_EXISTING: $task has $have_n of $want_k (in ${have_jobs}); running $remaining more" >&2
            need[$remaining]+="$task"$'\n'
        done < <(python3 skip_existing.py --jobs jobs --dataset "$ds" --agent "$agent" ${CC_VERSION:+--agent-version "$CC_VERSION"} \
                    --budget "$BUDGET" --max-dur "$dur" --k "$want_k" --server "$srv_file" "${globs[@]}")
        [ $skipped -gt 0 ] && echo "SKIP_EXISTING: skipped $skipped task(s) in $ds (listed above)" >&2
        for remaining in "${!need[@]}"; do
            inc=(); names=()
            while IFS= read -r t; do [ -n "$t" ] && { inc+=(-i "$t"); names+=("$t"); }; done <<<"${need[$remaining]}"
            sub=$(job_name "$sfx$([ ${#need[@]} -gt 1 ] && echo "_k$remaining")" "${names[@]}")
            echo "SKIP_EXISTING: running $(( ${#inc[@]} / 2 )) task(s) x $remaining attempt(s) -> $sub" >&2
            run_job "$ds" "$sub" "${timeout_args[@]}" "${inc[@]}" -k "$remaining"
        done
        unset need
    done
    # keep results/harbor-trials.tsv current (it's a full rebuild from jobs/, so cheap to redo)
    python3 collect.py jobs >/dev/null || echo "WARNING: collect.py failed; run ./harbor/harbor.sh collect" >&2
    exit $rc
    ;;
gen-tc)
    if [ "${1:-}" = "--list" ]; then
        f="${2:?gen-tc --list FILE}"; [[ "$f" = /* ]] || f="$OLDPWD/$f"
        sel=(--test-calls-file "$f")
    else
        sel=(--task-ids "${1:?language}_${2:?exercise}" --test-calls "${3:?calls}")
    fi
    ORACLE_SECRET=$(openssl rand -hex 16) python3 adapters/aider_polyglot/main.py \
        --output-dir "$TC_DATASET" "${sel[@]}"
    ;;
collect)
    python3 collect.py jobs "$@"
    ;;
*)
    sed -n '4,25p' "$0"; exit 1
    ;;
esac
