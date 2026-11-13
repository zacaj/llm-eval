# Agent issue taxonomy (Harbor trajectories)

Status: **implemented.** Detectors in [harbor/issues.py](../harbor/issues.py) -> iss_* columns in
harbor-trials.tsv and `results/harness-alerts.tsv` (resolved alerts: `harbor/issue-known.tsv`).
The viz uses them (outcomes.py flags `malformed` / `hallucinated` / `attention` / `mis_wired` in
stability, `overreach` / `env_gap` / `prompt_leak` outside it; stats `iss_malf` / `iss_hall` /
`iss_attn` / `overreach` / `compile_fail` / `masked`). `tool_errors`, `bash_*`, `edit_errors` are
retired; `malformed_calls` stays (it also covers terminus-2 parse errors). Weights are
uncalibrated; precision is from spot-checks (~10 samples per detector), not hand labels.

## Grid

Every finding gets one **category** (what went wrong) and one **layer** (where).

### Categories

| category | meaning | examples |
|---|---|---|
| **malformed** | couldn't produce valid output at that layer — includes typos | broken tool-call syntax / raw markup, bash quoting (`ythonecho`), `/temp` for `/tmp`, edit `old_string` with mangled whitespace |
| **hallucinated** | well-formed, references something that never existed | made-up path, tool name, module, CLI flag |
| **attention** | the model's picture of the state has drifted from reality — see subtypes below | stale `old_string`, re-running `xxd` after being told it's missing, loops |
| **mis-wired** | every piece real and valid, combination doesn't do what was meant | `pkill -f python3` killing its own shell; pipeline hides the output it needed; interactive tools (`nano`, `less`) in a non-interactive shell |
| **overreach** | did what it meant, but shouldn't have | reading `/opt/hidden-tests`, `sudo`, writing to system site-packages, fetching the exercise source online |
| **env gap** | reasonable expectation the harness doesn't meet — **not the model's fault** | `xxd`, `file`, `unzip` missing; `pip install` into read-only system dirs |
| **not an issue** | non-zero exit, nothing went wrong | expected test/build failures, negative probes (`grep` no match), trailing probe setting the exit code after the useful output |

### Attention subtypes

| subtype | meaning | detector | confidence |
|---|---|---|---|
| **stale state** | acts on an earlier version of something | `old_string` matches an earlier version of the file; old `/tmp/run-tests.XXXX` path | reliable |
| **forgot feedback** | repeats something it was already told fails | same normalized target failed earlier in the trial (`xxd`, same missing path) | reliable |
| **looped** | repeats its own output/actions | existing `looped` / `loop_stuck` detectors (moved here from glitch) | existing |
| **lost instruction** | breaks a stated task constraint | partly: `run-tests` past its call limit, budget ignored; generally **placeholder** | heuristic / placeholder |

### Edit failure split
A failed edit's `old_string` is compared against the file as the trial knows it:
- matches an **earlier** version → attention / stale state
- near-match (difflib ratio ≥ 0.9) in the **current** version → malformed (typo, whitespace)
- no close match anywhere → hallucinated (invented code)

Typo vs hallucinated is a detection split, not a behaviour split: a reference within a small edit
distance of something that exists or appeared earlier in the trial counts as malformed, otherwise
hallucinated.

### Layers

`tool-call` → `edit` → `shell` → `program` (builds/tests/scripts the model wrote).

Rollups are views, not stored stats: "malformed anything" = the malformed row; "bash problems" =
the shell column.

### Modifiers (apply to any cell)

- **repeat** — same normalized target (command name, path, tool name, `old_string` hash) already
  failed earlier in the trial. Feeds *attention / forgot feedback*; a repeated *env gap* is
  promoted to it (it was told). Not for deliberate workaround attempts (e.g. several different
  ways of writing to site-packages).
- **recovered** — the next call on the same target succeeded. Separates a slip from a spiral;
  supersedes `bash_streaks`.

## Detectors

| cell | detector | confidence | rough count |
|---|---|---|---|
| tool-call × malformed | `InputValidationError`; tool markup in args (`MARKUP_RE`); terminus-2 parse complaints | reliable | (= today's `malformed_calls`) |
| tool-call × malformed (end) | final message is raw tool markup | reliable | (= `markup_final`, kept as an end reason) |
| tool-call × hallucinated | tool name not in the agent's tool list | reliable | not measured |
| edit × malformed / hallucinated / attention | edit tool error flag, split per "Edit failure split" above | heuristic (3-way split not yet prototyped) | (= `edit_errors`) |
| shell × malformed | bash `syntax error` / `unexpected EOF` / bad substitution; command-not-found for a near-miss of a real command | reliable / heuristic | ~26 + few |
| shell × hallucinated | `No such file` / `cannot access` on a path never seen in the trial and not a near-miss; `ModuleNotFoundError` in ad-hoc scripts | heuristic | ~56 paths, ~28 imports |
| shell × attention | `/tmp/run-tests.\w+` paths; repeated command-not-found / missing path | reliable | ~14 + repeats |
| shell × mis-wired | `pkill`/`killall` with exit 124/143/144; interactive editors/pagers | reliable (narrow) | ~19 |
| shell × mis-wired (pipeline) | pipeline hides the output wanted | **placeholder** | — |
| shell × overreach | command touches `hidden-tests`, `/opt/run-tests`, `/var/lib/run-tests`, `sudo`; writes to system dirs | reliable | ~172 peek, ~32 sudo |
| shell × env gap | command-not-found for a list of common tools (`xxd`, `file`, `unzip`, …); `pip`/`npm` permission into system dirs | reliable | ~106 + ~44 |
| shell × not an issue | test/build commands; negative probes; trailing `ls`/`grep` setting exit 1 with clean output | heuristic | ~955 + 66 |
| shell, exit 0 × masked | `\|\| echo "X"` with X in output; `echo rc=$?` printing non-zero — then classified as above | reliable trigger | ~79 + ~262 |
| program × malformed | compile error in a throwaway probe vs in the solution | **placeholder** | — |
| overreach intent (cheating vs didn't know) | — | **placeholder** | — |
| shell × hallucinated (CLI flags) | `unrecognized option` is detectable; separating from typos isn't | **placeholder** | — |

### Noise to exclude
- gdb's `Error disabling address space randomization: Operation not permitted` (~105) — a warning, gdb works.
- `run-tests` hitting `cp: cannot open …/target/…lock: Permission denied` (12) — looks like a harness bug; investigate.
- `python_ro` / `node_ro` (40, jobs 10-26..10-30 only): personal CLAUDE.md leaking into the system prompt. Flag those trials (`prompt_leak`), don't score the calls.

## Dev-loop rates (not issues)

Some build/test failure is inevitable, but a high *rate* is a bad sign:
- `compile_fail_rate` = compile failures ÷ build attempts, including throwaway probe compiles
  (`rustc /tmp/x.rs`, `g++ …`) that `build_runs` doesn't count today.
- `test_fail_rate` likewise; noisier.

## Harness to-do from env gaps
Add `xxd`, `file`, `unzip`, `python` (alias), `python3` (TB2 images) to the task images. Not `nano` / `less`
(would hang a non-interactive shell).

## Harness alerts (unknowns)

collect.py tracks things that hit an issue trigger but no known rule, so new harness gaps and
classifier blind spots get noticed instead of silently scored:
- command-not-found names not in the env-gap / interactive / typo lists
- permission denied not matching a known pattern (peek, system write, gdb ASLR, run-tests copy)
- any `harness` or `leak` hit
- unclassified non-zero shell calls, grouped by signature (first word + first error line),
  shown once a signature appears in ≥ N trials

Output: `results/harness-alerts.tsv` (signature, category guess, count, first job, example trial)
plus a short summary at the end of each `collect` run. A reviewed `harbor/issue-known.tsv`
(pattern → verdict: env-gap / ignore / mistake) turns a resolved alert into a rule, so only
new things alert. A viz card can come later.

## Migration

| current | becomes |
|---|---|
| `malformed_calls` | tool-call × malformed |
| `markup_final` | kept as end reason; also tool-call × malformed |
| `edit_errors` | edit × {malformed, hallucinated, attention/stale} |
| `tool_errors` | split across cells by tool/layer; dropped |
| `bash_nonzero`, `bash_nonzero_per_step` | raw behaviour only, no weight (or dropped) |
| `bash_streaks`, `bash_streaks3` | replaced by repeat / recovered |
| `looped`, `loop_stuck` | attention / looped (`loop_stuck` stays an end reason) |
| `build_runs`, `build_fail_runs` | feed `compile_fail_rate` |

Stability weights move from per-column to per-category (malformed, attention high; hallucinated
medium; mis-wired medium; env gap 0; repeat bumps). Overreach is its own flag, outside stability.
