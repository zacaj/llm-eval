# aider-eval

Harness for benchmarking **local LLMs** (llama.cpp, Strata, or OpenRouter) on coding tasks and tracking results across many models/configs. Two independent runners feed one set of result TSVs and one viz:

| Runner | What it runs | Entry point |
|--------|--------------|-------------|
| **aider** | [aider polyglot benchmark](https://aider.chat/docs/leaderboards/) — 225 Exercism exercises (Python, Go, Rust, JS, Java, C++), edit-format loop with test-error retries | [aider/run.sh](aider/run.sh) |
| **Harbor** | Agentic evals ([Harbor](https://github.com/laude-institute/harbor): `claude-code`, `terminus-2`) on the same exercises plus Terminal-Bench 2, in sandbox containers | [harbor/harbor.sh](harbor/harbor.sh) |

The eval host (here) is separate from the model host (the NAS, or Strata on its Windows host). There is no application code to build — it's bash orchestration, Docker images, and Python patches/scripts. See [CLAUDE.md](CLAUDE.md) for the detailed architecture.

## Layout

- `aider/` — run.sh, sweep.sh, Dockerfile, `patches/` (applied to upstream aider's benchmark.py at build), `extra-exercises/`
- `harbor/` — harbor.sh, collect.py, issues.py, skip_existing.py, progress_sampler.py; `aider_polyglot/` (adapter fork) and `tb2/` (task generator)
- `lib/` — shared: `paths.sh`, `server-capture.sh`, `eval_proxy.py`
- `lists/` — reusable exercise/task lists (`lists/tmp/` = gitignored one-offs)
- `runs/` — per-run output (gitignored)
- `results/` — **separate git repo**: combined TSVs, `viz/` (generator + committed `eval-viz.html`), `notes/`
- `docs/` — findings and plans
- `llm-sweep.sh` — scratchpad of model-swap + Harbor run invocations

## aider runner

```bash
./aider/run.sh build                 # build the Docker image
./aider/run.sh smoke                 # 1 fixed python exercise
./aider/run.sh quick [words...]      # 5 fixed python exercises (or keyword matches)
./aider/run.sh python | polyglot | full
./aider/run.sh fromfile lists/shortlist-7.txt [--clean|--resume]
./aider/run.sh stats <run-dir>
./aider/run.sh shell
```

Configured by env vars: `MODEL` (llama `--alias`, or a `/`-containing litellm spec), `TARGET=llama-nas|strata`, `EDIT_FORMAT`, `TRIES` (default 6), `MAX_TOKENS`, `MAX_DUR`, `REASONING_EFFORT`, `TEMPERATURE`, `SEED`. Flags starting with `-` go to aider's `benchmark.py`. The header comment of run.sh is the full reference.

`aider/sweep.sh` runs run.sh over several model/server configs in turn (`CONFIGS`, `SKIP_EXISTING=1`, `SEED_TRIES=2`).

## Harbor runner

```bash
./harbor/harbor.sh gen                                        # generate tasks
./harbor/harbor.sh run claude-code -i 'polyglot_python_*' -l 5
./harbor/harbor.sh run claude-code --list lists/qfn-diff.txt -k 3
./harbor/harbor.sh run oracle -i 'polyglot_rust_*'            # verifier sanity check
./harbor/harbor.sh collect                                    # jobs -> results/harbor-trials.tsv
```

Env: `EFFORT`, `BUDGET` (output-token cap, enforced by the eval proxy), `MAX_DUR`, `TEMPERATURE`, `SKIP_EXISTING=1` (run only the missing attempts), `SWAP_CMD` (swap the model server first), `TC` (test-call tasks). Notable features: eval proxy with per-request logging, progress sampling, issue classification, token-budget list rows.

## Results & viz

Runs append to `results/*.tsv` (aider: `aider-results*.tsv`; Harbor: `harbor-trials*.tsv`, `harbor-progress.tsv`, `aider-eval-proxy.tsv`). `results/viz/generate_viz_data.py` rebuilds `results/viz/eval-viz.html` from `eval-viz-template.html` (edit the template, not the output). Harness failures and `INVALID`-marked trials are excluded from stats.

Compare pass **rates** across several seeds rather than single pass/fails — sampling is no longer greedy by default.

## Docs

- [docs/viz-harbor-plan.md](docs/viz-harbor-plan.md) — Harbor in the viz
- [docs/issue-taxonomy.md](docs/issue-taxonomy.md) — tool-call failure classification
- [docs/mtp-off-runaway-generation.md](docs/mtp-off-runaway-generation.md), [docs/moe-cache-off-required.md](docs/moe-cache-off-required.md) — server-config findings
