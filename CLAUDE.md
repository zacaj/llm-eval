# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A harness for running the [aider polyglot benchmark](https://aider.chat/docs/leaderboards/) (225 Exercism exercises: Python, Go, Rust, JS, Java, C++) against **local LLMs served by llama.cpp** (or Strata / OpenRouter), and tracking the results across many models/configs. There is no application code to build or unit-test — the work is bash orchestration, a Docker image, and a few Python patches/scripts.

## Commands

All via `./run.sh <command> [exercise-substrings...] [flags]` (see the big header comment in [run.sh](run.sh) for the full list):

```bash
./run.sh build                  # build the Docker image (aider + exercises + patches)
./run.sh smoke                  # 1 fixed python exercise
./run.sh quick                  # 5 fixed python exercises
./run.sh quick zipper leap      # bare words = aider --keywords substrings, not a model name
./run.sh random 5               # random sample (the others are deterministic on purpose)
./run.sh python | polyglot | full
./run.sh fromfile results/shortlist-7.txt [--clean|--resume]   # run just the listed exercises
./run.sh stats <run-dir>        # regenerate stats for a finished run
./run.sh shell                  # shell inside the container
```

Everything is configured by env vars, not positional args: `MODEL` (llama `--alias`; a value containing `/` is used as a full litellm spec, e.g. `openrouter/<vendor>/<slug>`), `TARGET=llama-nas|strata`, `EDIT_FORMAT` (whole|diff|udiff…), `TRIES` (default 6), `MAX_TOKENS`, `MAX_DUR`, `REASONING_EFFORT=low|medium|xhigh`, `TEMPERATURE` (`server` = send none), `SEED`. Flags starting with `-` are forwarded to aider's `benchmark.py`.

[sweep.sh](sweep.sh) runs `run.sh` over several model/server configs in turn, swapping the persistent `llama-cpp` compose service via `/mnt/llm/model-configs/mcfg.py` (outside this repo). `CONFIGS` is a configs.txt path or a comma list of GGUFs; `SKIP_EXISTING=1 SEED_TRIES=2` gap-fill and auto-retry 100%-failing exercises. Its header comment documents the format.

## Architecture

- **Eval host ≠ model host.** The aider container runs here; the model server is remote (the NAS, or Strata on its Windows host). `TARGET` in run.sh selects base URL, health URL, and how server settings (`srv.*` columns) are captured — `docker inspect` over `ssh://$NAS_HOST` for llama-nas, the `/props` endpoint for Strata. llama-nas requests go through a `:3002` prompt-logger proxy; health probes hit `:8080` directly.
- **Docker image** ([Dockerfile](Dockerfile)) clones upstream aider and applies [patches/](patches/) to `benchmark/benchmark.py` at build time. Each patch fixes a real failure mode (unbounded test output, no per-exercise wall clock, orphaned grandchild test processes, a hung `coder.run()` retrying forever, chat-history growth) and its docstring explains why. If you change harness behaviour, add/edit a patch and rebuild rather than editing the container.
- **Per-run model settings.** run.sh writes an aider model-settings YAML per `MODEL_SLUG`/`SAMPLING_SLUG` (extra_params: seed, temperature, reasoning budget) so concurrent runs of different models/seeds don't clobber each other. `REASONING_EFFORT` is a *bundle* (thinking budget + max tokens + duration multiplier), not just a hint.
- **Result tracking.** While a run executes, `track_per_test` polls `.aider.results.json` files and `track_aggregate` summarises; both append to `results/eval-results-per-test.tsv` / `results/eval-results.tsv` (flock-guarded via the `.lock` files). `refresh_viz` then runs `results/generate_viz_data.py` to rebuild `results/eval-viz.html` from `results/eval-viz-template.html` — edit the template, not the generated HTML.
- **Shortlists.** `results/*.txt` are exercise lists for `fromfile` (`[language] exercise [max_tries max_dur]`, `#` comments). Exercises are grouped by effective (tries, max_dur) and run as separate `benchmark.py` invocations into the same results dir. `shortlist-*.txt` carry baseline pass-try annotations used by `compare_to_baseline`.
- **`extra-exercises/`** holds `full-*` variants (java word-search, go paasio) where helper files normally hidden from the model are moved into the chat; mounted read-only through docker-compose.yml.

## Harbor (agentic evals) — `harbor/`

A separate, independent runner for [Harbor](https://github.com/laude-institute/harbor) (agents like `claude-code` / `terminus-2` in sandbox containers). It deliberately shares nothing with run.sh's selection/seed/skip machinery — use Harbor's own `-i <glob>` / `-l` / `-k` / `-n`.

```bash
./harbor/harbor.sh gen                                         # regenerate harbor/tasks/aider_polyglot (random oracle secret)
./harbor/harbor.sh run claude-code -i 'polyglot_python_*' -l 5 # BASE_URL / MODEL env; -n 1 default
./harbor/harbor.sh run oracle -i 'polyglot_rust_*'             # verifier sanity check, no model
./harbor/harbor.sh run claude-code --list harbor/lists/smoke-per-language.txt -k 3   # run.sh-format lists work too
BUDGET=20000 MAX_DUR=900 ./harbor/harbor.sh run terminus-2 -i polyglot_python_bowling   # per-trial output-token cap + agent wall clock
./harbor/harbor.sh collect                                     # harbor/jobs/* -> results/harbor-trials.tsv
```

- `harbor/adapters/aider_polyglot/` is a **fork** of Harbor's upstream adapter; every change is marked `(fork)`. Upstream only ran the first test for Rust/JS/C++ (ignored/xtest/RUN_ALL_TESTS), C++ effectively scored "compiles", C++ build failures produced no reward, Java missed `@Disabled` variants, and the oracle payload sat in the agent's workspace with a public key.
- Verifier writes `reward.json`: `reward` (binary, upstream-comparable) plus `tests_passed/total/frac/parsed`. Go panics / C++ crashes undercount `tests_total`; collect.py backfills the denominator from oracle trials (`tests_frac_adj`).
- harbor.sh snapshots the model server's output-affecting settings (before + after the run) into `jobs/<job>/server-settings.txt`, using the same functions as run.sh's srv.* columns (now in [server-capture.sh](server-capture.sh), sourced by both); collect.py adds them as srv.* columns.
- Job names: `<run id MMDD-HHMMSS or JOB_NAME>_<tasks>[_d<secs>][_k<n>]_<cc|t2|agent>[-b<budget>]`; `<tasks>` lists up to 4 short names (`go-bowling.tc6`), else `<list name>.<count>[.tc]`. Model and settings live in the job config / collect.py columns, not the name.
- Claude Code is pinned: the task images pre-install `ARG CC_VERSION` (task-template/Dockerfile), and harbor.sh passes it as `--ak version=` so Harbor skips its slow per-trial install (apt nodejs over a flaky mirror hit the 360s setup timeout). Bump it there and regenerate the tasks.
- `SKIP_EXISTING=1` makes `-k N` mean "ensure N graded attempts per task": [harbor/skip_existing.py](harbor/skip_existing.py) counts matching earlier trials across all jobs (same agent+version/BUDGET/output-affecting server settings, and same time limit -- or, for trials that ended on their own, an agent runtime within the requested limit; INVALID and ungraded infra errors excluded) and harbor.sh runs only the remainder, so rerunning a command resumes it. (`uvx harbor@0.25.0 job resume -p harbor/jobs/<job>` resumes one interrupted job in place.)
- **Invalidating trials**: put an `INVALID` file (first line = reason) in `harbor/jobs/<job>/<trial>/`; collect.py copies the reason into the `invalid` column. Harbor's own result.json is left untouched, so `uvx harbor view` still shows them as normal. Filter `invalid` rows out of comparisons.
- collect.py counts tool calls, malformed/unparsed tool calls, final-message-is-tool-markup (ends Claude Code episodes early), edits, build/test runs from the ATIF `agent/trajectory.json`.
- `BUDGET` is enforced by the **prompt-logger proxy's eval mode** (`X-Eval-Budget` header; Strata's copy at `/mnt/NAS-root/mnt/c/programs/Strata/prompt-logger/proxy.py`, accounting in its `logs/eval-budget.tsv`). Eval-mode requests skip the proxy's routes (which would divert title/compact calls to another backend) but keep its filters (tuned for local models) unless `PROXY_FILTERS=0`.
- **Test-calls tasks** (`harbor/tasks/aider_polyglot_tc/polyglot_<lang>_<ex>__tc<N>`, made by `harbor.sh gen-tc`): agent runs non-root and gets a `run-tests` command for the still-hidden tests, N calls max (stated in the instruction; the agent's processes are killed after the last call). `--list` rows with a try count (3rd column) run as `__tc<count>` automatically (missing ones are generated); `TC=1`/`TC=0` forces all/none. Mixed lists run as separate jobs.
- List `max_dur` → one Harbor job per distinct duration (`_d<secs>` in the name) with `--agent-timeout-multiplier`; `max_tries` has no equivalent. `MAX_DUR_SCALE` (number or `auto`, as in run.sh; the decode probe `measure_decode_tps` now lives in server-capture.sh) multiplies every limit, and the scaled value is what job names, the `max_dur` column and SKIP_EXISTING see. `-k` attempts get independent samples: no seed is sent and Strata picks a random one per request.
- Harbor builds one image per trial (deleted after); parallel builds of several toolchains need tens of GB free on `/var`.

## Gotchas

- Default sampling was historically **greedy** (aider sends temperature 0), making "reruns" byte-identical replays; now `TEMPERATURE=0.6` + fixed `SEED`. Compare pass *rates* across several seeds, not single pass/fails. See [mtp-off-runaway-generation.md](mtp-off-runaway-generation.md) and [moe-cache-off-required.md](moe-cache-off-required.md) for server-config findings that affect results.
- `results/` is gitignored; [todo.md](todo.md) is a running log of queued/finished eval runs and findings.
- Reusing a results dir: `guard_reuse` / `auto_clean` decide whether `--clean` is needed — `smoke/quick` auto-clean, `python/full` don't.
