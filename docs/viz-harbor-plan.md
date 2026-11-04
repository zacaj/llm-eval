# Plan: Harbor results in the eval viz

Status: plan only, nothing implemented (2026-10-28).

Goal: get Harbor trials into `results/eval-viz.html` alongside the aider runs. Harbor and aider
methodologies differ, so their scores must not be pooled raw. Matched tasks should still be
comparable, and model-level analysis should be able to use both. Future Harbor suites won't be
aider-based. The same work should also add much richer error/glitch analysis, per cell, per
model and per task.

## Grounding (data as of 2026-10-28)

- Aider per-test TSV: 2422 rows, 536 fails. Fail flags: error_outputs 185, user_asks 249,
  exhausted_context 124, bailed (timeout) 66, malformed 36, test_timeouts 21, lazy 3.
  **214 fails have no flag at all**: they used every try and still failed.
- Harbor trials TSV: 180 rows (nearly all claude-code), plus 74 check rows. Exceptions:
  AgentTimeout 30, NonZeroExit 21, Cancelled 8. test_limit_hit 20, final_no_action 124 (known
  false positive on clean finishes), thinking_cutoffs 56, px_budget_cutoffs 19, edit_errors 43.
- The current viz hardcodes two special cases (timeout dot, context square). Aider-specific
  logic is spread across both `generate_viz_data.py` and the template.

## 1. One dataset, filters instead of views

- Runner, suite, agent and config are **filter dimensions** over one pool of attempts. There is
  no separate page or view per suite.
- **Task variant** is the unit for task-level sections (heatmap rows, usefulness, per-test):
  `aider:go/bowling`, `harbor:go/bowling` and `harbor-tc:go/bowling` are separate variants,
  because their difficulty really differs. Each one carries a `twin` key (lang/exercise) for
  grouping and side-by-side display. Non-aider suites have no twin.
- **Model-level sections pool across runners** by scoring each attempt *relative to other
  models on the same task variant*, then averaging across variants. This is the same idea as
  the existing tries-efficiency score. Shrink variants that only a few models ran toward the
  prior (`shrink_toward_prior`). Show the runner mix of each model's n (e.g. "62 aider / 18
  harbor").
- Caveat: tc vs non-tc Harbor probably differ more than the suite name suggests. tc gives
  hidden-test feedback, which is closer to aider's tries. Filtering across them is fine, but
  their variants should stay separate.

### Common attempt record (built in Python by a per-runner adapter)

| field | aider | harbor |
|---|---|---|
| runner / suite | aider / polyglot | harbor:<agent> / aider_polyglot, aider_polyglot_tc, ... |
| task variant, twin | lang/exercise | Harbor task id; twin = lang/exercise when aider-derived |
| model | `short_label(srv.-m)` | `short_label(srv.-m)`, **not** the `model` alias column |
| config | edit_format, tries, effort, sampler | agent+version, budget, effort, tc limit, max_dur |
| pass | `pass` | `reward` |
| partial | (maybe, see §3) | `tests_frac_adj` |
| steps_used / steps_allowed | tries_taken / tries | test_calls / test_limit |
| tokens, turns | completion_tokens, tries | px_output_tokens, agent_steps |
| end_reason, flags | §5 | §5 |
| independence | (model, variant, seed): latest run per seed wins | every trial is independent (no seed is sent) |

The template should only know about the common fields. Runner-specific knowledge lives in the
adapters.

## 2. Limits are censoring, not failure

An attempt that hit a tries, tc, token or time limit **didn't pass within budget X**. It is
right-censored at X. It is not evidence that the model can't solve the task.

- Each model gets **pass-within-budget curves** (Kaplan–Meier style): P(passed by k output
  tokens / by n turns / by n tries-or-calls). Capability is where the curve plateaus, speed is
  how fast it rises, and consistency is the width of the band.
- **Budget axes are tokens or turns, not wall-clock time.** Server speed varies even within
  one run, so time measures the server as much as the model. A trial killed by the time limit
  is still censored, at the *tokens/turns it had used by then*. This also makes most of the
  slowdown corrections (`slow_lost_sec`, `agent_sec_adj`) unnecessary for scoring. Time stays
  available as a secondary axis only.
- **Raising a task's N (or any limit) later is safe.** Old limit-hit attempts are censored at
  the old N and count fully up to it. Beyond that point the curve has fewer samples and the band
  widens automatically. Nothing is invalidated, and the larger margin of error is visible.
- Headline score = the curve's value at a reference budget, using only attempts whose limit was
  ≥ that budget.
- Unlimited-tc runs fit in with no special-casing (censored only by the duration/token limit).
- tc N: record the calls actually used (`steps_used`). Max N is metadata and is not compared
  across values. Whether calls used is a real signal or noise is still to be shown.

## 3. Partial credit and progress curves

`run-tests-inner` already logs every call (`n, timestamp, exit, passed, total`) to
`/var/lib/run-tests/log.tsv` inside the container. But `count_tests.sh` keeps only `test_calls`
and `first_pass_call`, and the file is lost with the container. **Fixing that is phase 0.**

- Progress series per tc attempt: (budget used, tests passed/total) at each call. The budget
  axis should be **cumulative output tokens or turns**. Get it by joining each call's timestamp
  to the per-request rows in `jobs/<job>/eval-proxy.tsv` (cumulative tokens up to that
  request) or to the trajectory step index. Logging the request count in run-tests itself
  would be more direct.
- Metrics:
  - tokens/turns to reach X%;
  - area under the progress curve (fast climbers score well even below 100%);
  - **last-mile ratio**: budget spent on the final ~10% vs. reaching it (e.g. 9/10 tests in a
    few turns, then many more turns on the last one);
  - **regressions**: passed count dropping between calls, a stability signal.
- Non-tc Harbor has no hidden-test feedback during the run. Options: parse the agent's own test
  runs from `trajectory.json` (visible/self-written tests only), or replay workspace snapshots
  against the hidden tests offline (expensive, but gives true curves).
- Aider: I think `.aider.results.json` only has pass/fail per try. Per-try pass counts might be
  recoverable by parsing test output in the chat history. Unverified, and may differ by
  language.
- Noise check needed: run repeated attempts on the same tasks before building charts around
  these curves.

## 4. Four qualities: capability, consistency, speed, stability

Each section declares which of the four it shows, so they don't silently mix:

| section | capability | consistency | speed (tokens/turns) | stability |
|---|---|---|---|---|
| pass-within-budget curves | plateau | band | slope | — |
| capability score at reference budget | ✓ | CI | — | — |
| progress curve / last-mile | partial | — | ✓ | regressions |
| speed / tokens | — | — | ✓ | — |
| end-reason / glitch profile | — | — | — | ✓ |
| heatmap | ✓ | hatching | color mode | markers |

## 5. Outcome taxonomy: one end reason + secondary flags

Implemented and audited in [results/outcomes.py](results/outcomes.py) (phase 1, 2026-10-28):
`classify_aider(row, history)` / `classify_harbor(row)` -> `(end_reason, flags)`.
`python3 results/outcomes.py` prints the tables; `--sample <reason|flag>` lists attempts.

**End reason** (exactly one per attempt), in precedence order per runner:

| end reason | group | aider | harbor |
|---|---|---|---|
| `harness` | excluded | 0 tries *and* 0 tokens (server never answered); AuthenticationError in the history | INVALID; no reward (never graded); CancelledError, AgentSetupTimeoutError |
| `pass` | pass | `pass` | reward 1 |
| `budget` | censored | — | budget_hit |
| `exhausted` | censored | every other fail (aider never stops before its last try) | test_limit_hit (the exit-137 kill after the last call is by design) |
| `timeout` | censored | bailed_on_max_duration (incl. 0-try attempts that produced tokens) | AgentTimeoutError |
| `cut_off` | censored | — | final_turn_cutoff (thinking cut, then it stopped) |
| `glitch` | fail | — | markup_final, or any other agent exception |
| `done_wrong` | fail | — | ended on its own, tests fail |

Counts at audit time: aider 1886 pass / 443 exhausted / 63 timeout / 30 harness. Harbor 108 pass /
14 exhausted / 23 timeout / 1 budget / 2 cut_off / 9 done_wrong / 2 glitch / 30 harness.

`context` moved from end reason to flag: only 18 of 124 context-exhausted aider fails ended
early (all by timeout), and 27 passes had one too. `stopped_early` became `done_wrong`:
`final_no_action` is set on all 101 clean passes. It just means "the agent ended its own
episode", so on a fail it means a confident wrong finish. Checked: the trials end with
"implementation complete" summaries.

**Flags** (passes can have them): context, output_limit, reasoning_cut, malformed, edit_errors,
lint, wrong_file, build_failed, test_timeout, lazy, markup_final, server_error, slow_server,
proxy_rejected (labels in `FLAGS`).

Audit findings:
- **aider `user_asks` is not model questions.** It counts aider's auto-answered confirm prompts:
  "fix lint errors?" (-> `lint`), "create new file?" / "allow edits to unadded file?"
  (-> `wrong_file`), "open URL?" (comes with errors). The viz's "Model asked a clarifying
  question" label is wrong.
- **aider `error_outputs` mixes server and model problems.** Server: AuthenticationError,
  "Loading model", connection errors. Model: edit-format failures, output-limit / context
  warnings. The flags come from `.aider.chat.history.md` patterns instead (`HISTORY_PATTERNS`).
- **Infra failures scored as model fails:** run `random` (ox-alpha via OpenRouter,
  2026-08-22) is 25 rows where every try was an AuthenticationError. Now `harness`. Also 3
  `quick` rows (0 tokens in ~514s).
- **Run dirs are reused**, so only the latest TSV row per (run, language, exercise) has logs on
  disk. 1781 of 2422 attempts still do. Matching older rows to the current logs gave 26 wrong
  matches; latest-only gives 257/258 exact `tests_outcomes` matches for `exhausted`. Going
  forward run.sh `track_per_test` writes them as `hist_*` count columns (auth_errors,
  server_errors, context, output_limit, edit_format, lint, wrong_file), backfilled on 1690
  latest rows. Blank = logs gone, unknown. outcomes.py reads only these columns.
- The 25 `random` auth-failure rows moved to `aider-results-per-test-invalid.tsv` (with
  `invalid_reason`; that file's header now matches the main TSV). Their aggregate row in
  `aider-results.tsv` is left as is, since the viz doesn't read that file.
- **Harbor `tests_parsed=0` fails are real build failures** (Rust borrow/type errors, C++ API
  renames, a Java constructor mismatch), not harness errors -> `build_failed` flag.
- Not yet checked: whether Harbor timeouts were still making progress (needs the phase-0
  progress data). One aider timeout (Darwin cpp/clock, 2026-08-25) ran 23114s with 4.7k
  tokens, probably the old hung-`coder.run()` bug; left as timeout.

### Display

- Heatmap: a "color fails by end reason" mode, extending the "Color passes by" dropdown. One
  generic corner marker for a flag chosen from a dropdown replaces the timeout dot and context
  square. The tooltip lists each attempt's end reason and flags.
- Per model: stacked end-reason bars, normalized per attempt, plus a flag-rate table.
- Per task: flag rates and end-reason mix in the usefulness table. This absorbs "timeout rate".
- Model × flag matrix with a concentration score, to tell whose problem a flag is:
  - concentrated in one model → model problem;
  - concentrated in one task across models → task/harness problem (feeds the usefulness ranking);
  - spread evenly → maybe server/engine.

## 6. Aider vs Harbor (matched twins only)

- Per model: scatter of aider vs Harbor score on the *intersection* of twins, with n shown.
- Per task: delta across models. A big gap suggests a methodology effect (test visibility),
  not model ability.
- Optional split-cell heatmap mode (left = aider, right = Harbor). Always labeled as different
  methodologies and never pooled raw.

## Phasing

0. **Done 2026-10-28:** run-tests log saved as `verifier/run-tests-log.tsv` (ms start/end);
   collect.py joins calls + progress_sampler.py samples to proxy requests ->
   `results/harbor-progress.tsv` (requests/out_tokens before each point) and
   `first_pass_requests/out_tokens`. All tasks regenerated. Only jobs from after this have it.
1. **Done 2026-10-28:** taxonomy + audit -> `results/outcomes.py`, §5 above.
2. **Done 2026-10-28:** `generate_viz_data.py` reads attempt records from
   `load_aider_attempts()` (adapter) and keys tasks as `(variant, language, exercise)`.
   The pure refactor produced byte-identical JSON. Intended changes were then applied on
   top: harness attempts dropped (5; `summary.excluded_harness`), timeout marker = end
   reason, chart4 built from outcomes.py flags (the old "clarifying question" label is
   gone), and new keys `end_reason`/`flags`/`variant` in `detail`, `cost_spread`,
   `matrix`, `usefulness`. The capability score was already relative per task (shrink
   to the task's pool rate + per-model bias), so per-variant tasks give the relative
   pooling without a new formula. Template untouched; page renders the same heatmap,
   no console errors.
3. **Done 2026-10-28:** harbor-trials.tsv comes in as variants `harbor` / `harbor-tc`
   (any tc N pooled; seed = trial id). Every task label carries a runner code: `·A`,
   `·H`, `·HT`. Model keys come from one labeling pass over both runners on the gguf
   plus shared settings: effort, temp/top_p/top_k/min_p, and agent (only differs between
   Harbor runs). The token budget is ignored because it's rarely hit. So Harbor trials
   join the aider model they match: 7 of 8 Harbor models pool, while `Qwen3.6-35B-A3B`
   (no server settings, ambiguous alias) stays separate. The default selection is the
   20 most-run exercises, every variant (32 tasks). "Runner: <variant>" list entries act
   as the runner filter. chart1 shows each model's runner mix. Aider's unrun-seed
   fill-in applies to aider tasks only. tok/s uses the eval proxy's model time
   (`px_model_sec`) when measured, else wall time; run.sh now records `px_model_sec` for
   aider too.
4. **Done 2026-10-28:** "Pass within budget" card (after the capability score). Axes:
   output tokens / tokens relative to the task's typical pass / tries-or-test-calls.
   Per (model, task) Kaplan-Meier: pass = event; a `censored` end reason leaves the
   at-risk set at the budget it had used; a `fail` stays at risk forever. The model
   curve is the mean of its task curves, each held flat past its own data. **Pooled KM
   was tried first and is wrong here**: attempts are only exchangeable within a task, and
   pooled across tasks every curve's tail ran to ~100%. Harbor passes use
   `first_pass_out_tokens` when known, else total output tokens. The top 6 by capability
   are colored (dataviz reference palette as `--cat-1..8`; a model keeps its color while
   it stays highlighted), the others are gray. It has a crosshair tooltip and a table
   view (passed by the end, budget to reach 50%, stopped-by-limit / failed-on-its-own
   shares). Not done: confidence bands; turns axis (aider has no request count yet).
5. **Done 2026-10-28:** Heatmap: "Color fails by" (one color / stopped by a limit vs
   failed on its own, by majority of the cell's fails; `--fail-limit` token), and one
   generic corner dot whose meaning is picked in "Mark cells with" (any end reason or
   flag; default the time limit). This replaces the timeout dot + context square.
   Tooltip rows show each run's end reason and flags. New "How attempts ended" card:
   rows = models or tasks, an outcome bar (passed / stopped by a limit / failed on its
   own), and per-flag rates shaded by ratio to the selection's overall rate (≥2× and 2+
   attempts = bold). This is the concentration view: shading in a model row is that
   model's habit, in a task row across models it points at the task/harness. Early
   reads: java word-search ·A 43% wrong_file; rust forth ·HT 89% edit_errors; Strata
   models 18-23% reasoning_cut. The bottom chart4 ("What failed runs had in common") is
   now mostly redundant with it.
   Follow-ups the same day: (a) Harbor `edit_errors` were false positives. collect.py's
   output-text fallback read Rust's `Error::Variant` in successful edit results as
   failures; edits now count only the tool's own error flag (46 -> 10 trials).
   `test_fail_runs`/`build_fail_runs` still use the text fallback. (b) Thinking cut-offs
   are shown per thinking block (`think_cuts / think_blocks`), not per trial: the
   per-trial flag hits ~50% of Strata trials at any effort. Per block it's Swift xhigh
   ~10-11%, Swift low ~7%, Gyro-M low ~5%, all under the proxy's 4000 thinking cap.
   Correction: llama-served models do run under a thinking cap (the NAS proxy's default
   4000, now backfilled), but llama ends a capped block mid-sentence with no marker, so
   they read as never cut. collect.py now picks the best source per trial
   (`thinking_cutoff_source`: proxy tokens-vs-budget > Strata marker > length estimate).
   Per block, llama Qwen3.8-27B GSQ-RCO is ~11.5% (as high as Swift xhigh); Darwin
   ~2.5%; Qwen3.6 Opus-Distilled ~3.7%.
   The thinking cap is now part of model identity: Harbor's `srv.eff.reasoning_budget`,
   and aider's `srv.--reasoning-budget` (or, for older rows where it's blank, its effort
   bundle's cap: 4096 / 8192 / 16384). Caps are rounded to thousands ('4k'), so aider's
   4096 pools with Harbor's 4000. This split Swift xhigh into aider [xhigh, think=16k]
   and Harbor [xhigh, think=4k]: Harbor xhigh ran under the proxy's 4k default, which
   explains its cut-off rate. A label shows it only when effort doesn't
   already tell the runs apart: `[xhigh]` / `[low]`, but `[low, think=4000]` /
   `[low, think=6000]`.
6. tc progress curves, then the noise check.
7. Twin comparison, then the first non-aider suite.

## Open decisions

- Reference budget(s) for the headline score: per runner, or a token budget shared across
  runners?
- tc vs non-tc variants of the same exercise: kept separate (current plan) or merged under some
  filters?
- Whether the aider per-try partial counts are worth parsing (depends on the phase-6 noise
  check).
