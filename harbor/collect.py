#!/usr/bin/env python3
"""Flatten Harbor job dirs into one TSV row per trial.

A trial dir containing an INVALID file (reason inside) gets that reason in the
`invalid` column -- filter those rows out of comparisons.

    python3 collect.py jobs [--out ../results/harbor-trials.tsv]

Rebuilt from scratch on every run (no appending), so it's always consistent
with what's under jobs/.

Sources per trial:
  result.json             reward + test counts (from the forked verifier's
                          reward.json), tokens, timings, exception
  agent/trajectory.json   ATIF steps -> tool calls, tool errors, malformed
                          calls, build/test command runs, edits

tests_total_oracle: a Go panic or C++ crash aborts the test binary, so the
observed total undercounts. Any oracle trial of the same task gives the true
total; tests_frac_adj uses max(observed, oracle) as the denominator.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Commands the agent ran, classified by intent. Matched against the shell
# command string of Bash-like tool calls (claude-code `command`, terminus-2
# `keystrokes`).
TEST_RE = re.compile(
    r"\b(pytest|unittest|go test|cargo test|npm (run )?test|npx jest|jest\b|gradlew? test|ctest)"
)
BUILD_RE = re.compile(
    r"\b(cargo (build|check)|go (build|vet)|javac|gradlew? (build|compile\w*)|cmake|make\b"
    r"|g\+\+|gcc\b|clang\+*|node --check|py_compile|python3? -c ['\"]import)"
)
# A tool call whose arguments carry raw tool-call markup instead of parsed
# fields, or a final message that is an unparsed tool call (ends the episode).
MARKUP_RE = re.compile(r"<tool_call>|<function[=>]|<parameter[=>]|INVOKE_MARKER|<invoke\b")
EXIT_RE = re.compile(r"\[exit_code\] (-?\d+)")
FAIL_OUTPUT_RE = re.compile(
    r"\bFAILED\b|\bFAIL\b|Traceback \(most recent|error(\[E\d+\])?:|Error:|BUILD FAILED|"
    r"panic:|Segmentation fault|SyntaxError"
)
EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "str_replace_editor", "write_file", "edit_file"}
# terminus-2 has no edit tool: it writes files with heredocs/redirects. Source
# files only, so scratch output like `> /tmp/out.txt` isn't an edit.
SHELL_WRITE_RE = re.compile(
    r"(cat|tee)\b[^\n|]*?>{1,2}\s*['\"]?(?!/tmp/)[\w./-]+\.(py|go|rs|js|mjs|java|cpp|cc|h|hpp)\b"
    r"|\b(sed -i|perl -pi)\b"
)
# Harness-side "your reply didn't parse" feedback (terminus-2 JSON/XML parser);
# the step has no tool calls, the complaint is in its observation.
PARSE_ERROR_RE = re.compile(r"No valid JSON object found|Please fix these issues|No valid XML")

COLUMNS = [
    "job", "started_at", "agent", "agent_version", "model", "task", "language", "exercise",
    "trial", "invalid", "check", "max_dur", "max_dur_scale", "max_dur_base", "disallowed_tools", "budget", "budget_hit", "reward", "tests_passed", "tests_total", "tests_total_oracle", "tests_frac",
    "tests_frac_adj", "tests_parsed",
    "test_limit", "test_calls", "test_limit_hit", "first_pass_call",
    "prog_samples", "prog_first_pass_sec", "prog_best_frac", "prog_last_frac",
    "first_pass_requests", "first_pass_out_tokens", "exception",
    "agent_sec", "agent_sec_adj", "slow_lost_sec", "verifier_sec", "total_sec",
    "in_tokens", "cached_tokens", "out_tokens",
    "agent_steps", "tool_calls", "tool_errors", "malformed_calls", "markup_final",
    "thinking_blocks", "thinking_cutoffs", "thinking_cutoff_source", "final_turn_cutoff", "final_no_action",
    "px_requests", "px_rejected", "px_output_tokens", "px_reasoning_tokens", "px_answer_tokens",
    "px_budget_cutoffs", "px_effort", "px_reasoning_budget", "px_model_sec", "px_slow_requests",
    "px_baseline_tok_s",
    "edits", "edit_errors", "test_runs", "test_fail_runs", "build_runs", "build_fail_runs",
    "tools",
]


def secs(timing: dict | None) -> float | None:
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return None
    a = datetime.fromisoformat(timing["started_at"].replace("Z", "+00:00"))
    b = datetime.fromisoformat(timing["finished_at"].replace("Z", "+00:00"))
    return round((b - a).total_seconds(), 1)


def text_of(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def command_of(args: dict) -> str:
    for k in ("command", "keystrokes", "cmd"):
        if isinstance(args.get(k), str):
            return args[k]
    return ""


def result_failed(res: dict) -> bool:
    extra = res.get("extra") or {}
    if extra.get("tool_result_is_error") or extra.get("is_error"):
        return True
    content = text_of(res.get("content"))
    m = EXIT_RE.search(content)
    if m:
        return int(m.group(1)) != 0
    # no exit status (e.g. terminus-2's tmux output): fall back to output text
    return bool(FAIL_OUTPUT_RE.search(content))


def trajectory_stats(path: Path) -> dict:
    s = dict.fromkeys(
        ["agent_steps", "tool_calls", "tool_errors", "malformed_calls", "markup_final",
         "edits", "edit_errors", "test_runs", "test_fail_runs", "build_runs", "build_fail_runs"], 0)
    s["tools"] = ""
    if not path.exists():
        return {k: "" for k in s}
    traj = json.loads(path.read_text())
    tools: dict[str, int] = {}
    last_agent_msg = ""
    for step in traj.get("steps", []):
        if step.get("source") != "agent":
            continue
        s["agent_steps"] += 1
        last_agent_msg = text_of(step.get("message"))
        results = {r.get("source_call_id"): r for r in ((step.get("observation") or {}).get("results") or [])}
        # results without call ids (some agents) pair up positionally
        unkeyed = [r for k, r in results.items() if k is None]
        if not step.get("tool_calls") and any(PARSE_ERROR_RE.search(text_of(r.get("content"))) for r in results.values()):
            s["malformed_calls"] += 1
        for i, call in enumerate(step.get("tool_calls") or []):
            name = call.get("function_name") or ""
            args = call.get("arguments") or {}
            res = results.get(call.get("tool_call_id")) or (unkeyed[i] if i < len(unkeyed) else {})
            s["tool_calls"] += 1
            tools[name] = tools.get(name, 0) + 1
            failed = bool(res) and result_failed(res)
            content = text_of(res.get("content")) if res else ""
            if (res.get("extra") or {}).get("tool_result_is_error"):
                s["tool_errors"] += 1
            if MARKUP_RE.search(text_of(args)) or "InputValidationError" in content:
                s["malformed_calls"] += 1
            if name in EDIT_TOOLS:
                s["edits"] += 1
                # only the tool's own error flag: a successful edit's result quotes the
                # edited code, and the output-text fallback in result_failed() read
                # Rust's `Error::Variant` as a failure (rust forth: ~all edits "failed")
                extra = (res.get("extra") or {}) if res else {}
                s["edit_errors"] += bool(extra.get("tool_result_is_error") or extra.get("is_error"))
            cmd = command_of(args)
            if cmd:
                if name not in EDIT_TOOLS and SHELL_WRITE_RE.search(cmd):
                    s["edits"] += 1
                if TEST_RE.search(cmd):
                    s["test_runs"] += 1
                    s["test_fail_runs"] += failed
                elif BUILD_RE.search(cmd):
                    s["build_runs"] += 1
                    s["build_fail_runs"] += failed
    s["markup_final"] = int(bool(MARKUP_RE.search(last_agent_msg)))
    s["tools"] = ",".join(f"{k}:{v}" for k, v in sorted(tools.items(), key=lambda kv: -kv[1]))
    return s


# Strata closes a thinking block that hits its reasoning budget with this exact line
# (verified: only ever the last line of a thinking block, at ~the budget's length)
CHECK_AGENTS = {"oracle", "nop"}

THINK_CUTOFF = "I have thought about this long enough; time to give my answer."
# llama's reasoning cap ends a thinking block with no marker, mid-sentence. Estimated as
# a block at least this long (Strata calibration, 2026-10-28: 73 of 107 marker-cut blocks
# reach 12k chars, 15 of 1353 uncut ones do) that doesn't end a sentence.
THINK_CUT_EST_CHARS = 12000
SENTENCE_END = tuple('.!?`)"*')


def looks_cut(text: str) -> bool:
    t = text.rstrip()
    return len(t) >= THINK_CUT_EST_CHARS and not t.endswith(SENTENCE_END)


def thinking_stats(agent_dir: Path) -> dict:
    """From Claude Code's stream-json log (agent/claude-code.txt): thinking blocks,
    how many were cut off by the reasoning budget, and whether the final turn was
    cut off and/or ended with no tool call (Claude Code treats that as "done", so a
    model that stops after announcing its next step ends the episode)."""
    keys = ["thinking_blocks", "thinking_cutoffs", "final_turn_cutoff", "final_no_action",
            "_cut_marker", "_cut_est", "_final_cut_est"]
    log = agent_dir / "claude-code.txt"
    if not log.exists():
        return dict.fromkeys(keys, "")
    turns: dict[str, list] = {}   # message id -> content blocks (streamed one per event)
    for line in log.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict) or e.get("type") != "assistant":
            continue
        m = e.get("message") or {}
        turns.setdefault(m.get("id") or str(len(turns)), []).extend(m.get("content") or [])
    s = dict.fromkeys(keys, 0)
    cut = lambda b: b.get("type") == "thinking" and b.get("thinking", "").rstrip().endswith(THINK_CUTOFF)
    est = lambda b: b.get("type") == "thinking" and not cut(b) and looks_cut(b.get("thinking", ""))
    for blocks in turns.values():
        for b in blocks:
            s["thinking_blocks"] += b.get("type") == "thinking"
            s["_cut_marker"] += cut(b)
            s["_cut_est"] += est(b)
    # thinking_cutoffs / final_turn_cutoff are settled in main() (cutoff_source): marker
    # counts here, the proxy's count when it has one
    s["thinking_cutoffs"] = s["_cut_marker"]
    if turns:
        last = list(turns.values())[-1]
        s["final_turn_cutoff"] = int(any(cut(b) for b in last))
        s["_final_cut_est"] = int(any(est(b) for b in last))
        s["final_no_action"] = int(not any(b.get("type") == "tool_use" for b in last))
    return s


def proxy_rows(job_dir: Path) -> list[dict]:
    """jobs/<job>/eval-proxy.tsv (harbor/eval_proxy.py): one row per model request."""
    f = job_dir / "eval-proxy.tsv"
    if not f.exists():
        return []
    with f.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# A request is "unreasonably slow" when it produced >= SLOW_MIN_TOKENS but ran under
# SLOW_FRACTION of its job's baseline speed (median tok/s of that job's requests with
# >= SLOW_MIN_TOKENS output -- same server/model, robust to one slow spell). Small
# requests are skipped: their time is mostly prompt processing, not generation.
SLOW_MIN_TOKENS = 300
SLOW_FRACTION = 0.3


def job_baseline_tok_s(rows: list[dict]) -> float | None:
    v = sorted(float(r["tok_s"]) for r in rows
               if r.get("tok_s") and (r.get("output_tokens") or "0").isdigit()
               and int(r["output_tokens"]) >= SLOW_MIN_TOKENS)
    return v[len(v) // 2] if len(v) >= 5 else None


num = lambda r, k: int(r[k]) if (r.get(k) or "").lstrip("-").isdigit() else 0


def trial_proxy_rows(tdir: Path, rows: list[dict]) -> list[dict]:
    """This trial's rows of its job's eval-proxy.tsv, by session: Claude Code's session
    uuid (= its session file name under agent/sessions), terminus-2's X-Eval-Session
    (starts with the trial name)."""
    sessions = {p.stem for p in (tdir / "agent" / "sessions").rglob("*.jsonl")}
    return [r for r in rows if r["session"] in sessions or r["session"].startswith(tdir.name)]


def epoch(ts: str) -> float | None:
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (TypeError, ValueError):
        return None


PROGRESS_COLUMNS = [
    "job", "trial", "agent", "model", "task", "language", "exercise", "source", "n", "ts",
    "elapsed_sec", "requests_before", "out_tokens_before", "passed", "total", "frac", "reward",
    "status", "test_sec",
]


def progress_points(tdir: Path, d: dict, proxy: list[dict], denom: int) -> list[dict]:
    """Mid-run test results of one trial, one row each, on a model-budget axis as well as
    time: requests_before / out_tokens_before = the trial's completed proxy requests (and
    their output tokens) up to the moment the graded code was taken. Proxy rows are
    logged when a request finishes, so a request counts only if it ended by then.
      source=call    the agent's run-tests calls (verifier/run-tests-log.tsv, test-calls
                     tasks; graded at the call's start)
      source=sample  progress_sampler.py's side grading (progress.tsv; graded at the
                     snapshot, ts - sample_sec)
    Fractions use the same denominator as tests_frac_adj. Without proxy rows (jobs from
    before eval_proxy.py) the budget columns are blank."""
    done = sorted(((epoch(r["timestamp"]), num(r, "output_tokens")) for r in proxy
                   if r.get("note") != "rejected" and epoch(r["timestamp"]) is not None))
    t0 = epoch((d.get("agent_execution") or {}).get("started_at") or d.get("started_at") or "")

    def budget_at(t: float | None) -> dict:
        if t is None or not proxy:
            return {"requests_before": "", "out_tokens_before": ""}
        before = [tok for ts, tok in done if ts <= t]
        return {"requests_before": len(before), "out_tokens_before": sum(before)}

    def point(source, n, t, passed, total, reward, status, test_sec) -> dict:
        p, tt = (int(passed), int(total or 0)) if str(passed).isdigit() else (None, None)
        return {
            "source": source, "n": n,
            "ts": datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds") if t else "",
            "elapsed_sec": round(t - t0) if t and t0 else "",
            **budget_at(t),
            "passed": "" if p is None else p, "total": "" if p is None else tt,
            "frac": "" if p is None else round(p / max(denom, tt, 1), 4),
            "reward": reward, "status": status, "test_sec": test_sec,
        }

    pts = []
    log = tdir / "verifier" / "run-tests-log.tsv"
    if log.exists():
        # n, end ts, exit code, passed, total, start ts (log lines from before the start
        # column was added never reached /logs/verifier, so it is always there)
        for line in log.read_text().splitlines():
            f = line.split("\t")
            if len(f) < 6:
                continue
            start, end = epoch(f[5]), epoch(f[1])
            pts.append(point("call", int(f[0]), start, f[3], f[4], int(f[2] == "0"), "ok",
                             round(end - start, 1) if start and end else ""))
    prog = tdir / "progress.tsv"
    if prog.exists():
        for i, r in enumerate(csv.DictReader(prog.open(), delimiter="\t"), 1):
            end, sec = epoch(r["ts"]), float(r.get("sample_sec") or 0)
            ok = r["status"] == "ok"
            pts.append(point("sample", i, end - sec if end else None, r["passed"] if ok else "",
                             r["total"], r["reward"] if ok else "", r["status"], sec))
    return pts


def proxy_stats(tdir: Path, rows: list[dict]) -> dict:
    """Per-trial totals from the eval proxy log. Rows are keyed by session: Claude Code's
    session uuid (= its session file name under agent/sessions), terminus-2's
    X-Eval-Session (starts with the trial name)."""
    keys = ["px_requests", "px_rejected", "px_output_tokens", "px_reasoning_tokens", "px_answer_tokens",
            "px_budget_cutoffs", "px_effort", "px_reasoning_budget", "px_model_sec", "px_slow_requests",
            "px_baseline_tok_s", "slow_lost_sec"]
    if not rows:
        return dict.fromkeys(keys, "")
    mine = trial_proxy_rows(tdir, rows)
    done = [r for r in mine if r.get("note") != "rejected"]
    uniq = lambda k: ",".join(sorted({r[k] for r in done if r.get(k)}))
    return {
        "px_requests": len(done),
        "px_rejected": len(mine) - len(done),
        "px_output_tokens": sum(num(r, "output_tokens") for r in done),
        "px_reasoning_tokens": sum(num(r, "reasoning_tokens") for r in done),
        "px_answer_tokens": sum(num(r, "answer_tokens") for r in done),
        "px_budget_cutoffs": sum(num(r, "budget_cutoff") for r in done),
        "px_effort": uniq("effort"),
        "px_reasoning_budget": uniq("reasoning_budget"),
        "px_model_sec": round(sum(num(r, "duration_ms") for r in done) / 1000),
        **slow_stats(done, job_baseline_tok_s(rows)),
    }


def slow_stats(done: list[dict], base: float | None) -> dict:
    """Server-side slowdowns (GPU contention, throttling...): requests far below the
    job's normal speed, and the wall time they cost beyond what their output would
    have taken at baseline speed."""
    if not base:
        return {"px_slow_requests": "", "px_baseline_tok_s": "", "slow_lost_sec": ""}
    n, lost = 0, 0.0
    for r in done:
        out = int(r["output_tokens"]) if (r.get("output_tokens") or "").isdigit() else 0
        if out < SLOW_MIN_TOKENS or not r.get("tok_s") or not (r.get("duration_ms") or "").isdigit():
            continue
        if float(r["tok_s"]) < SLOW_FRACTION * base:
            n += 1
            lost += max(0.0, int(r["duration_ms"]) / 1000 - out / base)
    return {"px_slow_requests": n, "px_baseline_tok_s": round(base, 1), "slow_lost_sec": round(lost)}


def task_agent_timeout(tdir: Path) -> float:
    """[agent] timeout_sec from the trial's task.toml (polyglot: 1800)."""
    try:
        cfg = json.loads((tdir / "config.json").read_text())
        toml = Path(cfg["task"]["path"]) / "task.toml"
        sect = toml.read_text().split("[agent]", 1)[1].split("\n[", 1)[0]
        return float(re.search(r"timeout_sec\s*=\s*([0-9.]+)", sect).group(1))
    except (OSError, KeyError, IndexError, AttributeError, ValueError):
        return 1800.0


EXPECTED_TOTALS = Path(__file__).parent / "expected-totals.tsv"


def write_expected_totals(oracle_total: dict[str, int], trials) -> None:
    """Merge fully-passing oracle totals into expected-totals.tsv (lang, exercise, total),
    which the adapter bakes into each task at gen time so the verifier and run-tests
    count tests that never ran (crash/abort/timeout) as failed. Only reward=1 oracle
    trials: every test ran. Existing rows are kept, so pruning old jobs loses nothing."""
    rows: dict[tuple[str, str], int] = {}
    if EXPECTED_TOTALS.exists():
        for line in EXPECTED_TOTALS.read_text().splitlines()[1:]:
            lang, ex, n = line.split("\t")
            rows[(lang, ex)] = int(n)
    for _, d in trials:
        rw = (d.get("verifier_result") or {}).get("rewards") or {}
        if d["agent_info"]["name"] != "oracle" or rw.get("reward") != 1 or not rw.get("tests_total"):
            continue
        m = re.match(r"polyglot_([a-z]+)_(.+)", exercise_key_of(d["task_name"]))
        if m:
            k = (m.group(1), m.group(2))
            rows[k] = max(rows.get(k, 0), rw["tests_total"])
    EXPECTED_TOTALS.write_text("language\texercise\ttests_total\n"
                               + "".join(f"{l}\t{e}\t{n}\n" for (l, e), n in sorted(rows.items())))


COUNT_TESTS = Path(__file__).parent / "adapters/aider_polyglot/task-template/count_tests.sh"


def rescore(tdir: Path, lang: str, rw: dict, expected: int | None) -> dict | None:
    """Recount a trial's saved verifier output with the current count_tests.sh and the
    exercise's known total, so trials graded by an older counter (e.g. an aborted rust
    test binary scored 1/1 instead of 39/50) match new ones. reward is left as graded.
    Java's counter reads the JUnit XML, which isn't kept, so java keeps its stored counts."""
    log = tdir / "verifier" / "test-output.txt"
    if lang in ("", "java") or not log.exists() or "reward" not in rw:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "count_tests.sh"
        shutil.copy2(COUNT_TESTS, script)
        if expected:
            (Path(tmp) / "expected_total").write_text(f"{expected}\n")
        r = subprocess.run(["bash", str(script), lang, str(log), "0" if rw["reward"] == 1 else "1"],
                           capture_output=True, text=True)
    try:
        new = json.loads(r.stdout)
    except ValueError:
        return None
    return {k: new[k] for k in ("tests_passed", "tests_total", "tests_frac", "tests_parsed")}


def progress_cols(tdir: Path, denom: int) -> dict:
    """Summary of progress_sampler.py's mid-run grading (<trial>/progress.tsv): samples
    taken (timeouts/errors included), seconds into the trial of the first sample that
    passed every test, and the best / last passed fraction (timeouts/errors skipped).
    Fractions use the same denominator as tests_frac_adj."""
    f = tdir / "progress.tsv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(f.open(), delimiter="\t"))
    ok = [r for r in rows if r["status"] == "ok" and r["passed"] != ""]
    fr = [int(r["passed"]) / max(denom, int(r["total"] or 0), 1) for r in ok]
    first = next((r["elapsed_sec"] for r in ok if r["reward"] == "1"), "")
    return {"prog_samples": len(rows), "prog_first_pass_sec": first,
            "prog_best_frac": round(max(fr), 4) if fr else "",
            "prog_last_frac": round(fr[-1], 4) if fr else ""}


def exercise_key_of(name: str) -> str:
    m = re.search(r"polyglot_[a-z]+_.+?(?=__tc\d+$|$)", name.rsplit("/", 1)[-1].split("__", 1)[-1])
    return m.group(0) if m else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "harbor-trials.tsv")
    a = ap.parse_args()

    trials = []
    for result_path in sorted(a.jobs_dir.glob("*/*/result.json")):
        d = json.loads(result_path.read_text())
        if "task_name" not in d:
            continue
        trials.append((result_path.parent, d))

    # X-Eval-Budget as passed by harbor.sh (claude-code env / terminus-2 kwargs), and the
    # agent time limit: the task's own timeout x --agent-timeout-multiplier (Harbor
    # leaves defaults out of config.json, hence the fallbacks)
    job_budget: dict[str, int] = {}
    px_cache: dict[str, list[dict]] = {}   # job -> eval-proxy.tsv rows
    job_mult: dict[str, float] = {}
    for cfg in a.jobs_dir.glob("*/config.json"):
        c = json.loads(cfg.read_text())
        m = re.search(r"X-Eval-Budget\W+(\d+)", json.dumps(c.get("agents")))
        job_budget[cfg.parent.name] = int(m.group(1)) if m else 0
        cj = json.dumps(c)
        mm = (re.search(r'"agent_timeout_multiplier": ([0-9.]+)', cj)
              or re.search(r'"timeout_multiplier": ([0-9.]+)', cj))
        job_mult[cfg.parent.name] = float(mm.group(1)) if mm else 1.0

    # server-settings.txt (written by harbor.sh): srv.* key=value lines -> extra columns
    job_srv: dict[str, dict[str, str]] = {}
    for f in a.jobs_dir.glob("*/server-settings.txt"):
        kv = dict(l.split("=", 1) for l in f.read_text().splitlines() if "=" in l)
        if (f.parent / "server-settings-after.txt").exists():
            kv["srv.changed_during_run"] = "yes"
        job_srv[f.parent.name] = kv

    # keyed by "polyglot_<lang>_<exercise>": task names differ by dataset (classic
    # "aider-ai/aider-polyglot__polyglot_rust_forth", test-calls "polyglot_rust_forth__tc4")
    # but share the hidden tests, so an oracle run of either gives the denominator
    exercise_key = exercise_key_of

    oracle_total: dict[str, int] = {}
    for _, d in trials:
        rw = (d.get("verifier_result") or {}).get("rewards") or {}
        if d["agent_info"]["name"] == "oracle" and rw.get("tests_parsed"):
            k = exercise_key(d["task_name"])
            oracle_total[k] = max(oracle_total.get(k, 0), rw.get("tests_total", 0))
    write_expected_totals(oracle_total, trials)
    for line in EXPECTED_TOTALS.read_text().splitlines()[1:]:   # survives pruned oracle jobs
        lang, ex, n = line.split("\t")
        k = f"polyglot_{lang}_{ex}"
        oracle_total[k] = max(oracle_total.get(k, 0), int(n))

    rows = []
    for tdir, d in trials:
        rw = (d.get("verifier_result") or {}).get("rewards") or {}
        ar = d.get("agent_result") or {}
        mi = d["agent_info"].get("model_info") or {}
        task = d["task_name"]
        m = re.search(r"polyglot_([a-z]+)_(.+?)(?:__tc\d+)?$", task)
        o_total = oracle_total.get(exercise_key(task))
        rescored = rescore(tdir, m.group(1) if m else "", rw, o_total)
        if rescored:
            rw = {**rw, **rescored}
        passed, total = rw.get("tests_passed"), rw.get("tests_total")
        denom = max(total or 0, o_total or 0)
        budget = job_budget.get(tdir.parent.name, 0)
        exc = d.get("exception_info") or {}
        out_tok = ar.get("n_output_tokens") or 0
        row = {
            "job": tdir.parent.name,
            "max_dur": round(task_agent_timeout(tdir) * job_mult.get(tdir.parent.name, 1.0)),
            "budget": budget or "",
            "budget_hit": int(budget > 0 and ("budget exhausted" in (exc.get("exception_message") or "")
                                              or out_tok >= 0.98 * budget)) if budget else "",
            "started_at": d.get("started_at", ""),
            "agent": d["agent_info"]["name"],
            "agent_version": d["agent_info"].get("version", ""),
            "model": mi.get("name", ""),
            "task": task.split("__", 1)[-1],  # drop "<dataset>__"; tc tasks keep their "__tc<N>"
            "language": m.group(1) if m else "",
            "exercise": m.group(2) if m else "",
            "trial": tdir.name,
            # <trial>/INVALID (first line = reason): trial ran under a known harness bug;
            # keep the row for the record but exclude it from comparisons
            "invalid": ((tdir / "INVALID").read_text().splitlines() or ["yes"])[0][:200] if (tdir / "INVALID").exists() else "",
            "reward": rw.get("reward", ""),
            "tests_passed": passed if passed is not None else "",
            "tests_total": total if total is not None else "",
            "tests_total_oracle": o_total or "",
            "tests_frac": rw.get("tests_frac", ""),
            "tests_frac_adj": round(passed / denom, 4) if passed is not None and denom else (0 if passed is not None else ""),
            "tests_parsed": rw.get("tests_parsed", ""),
            # test-calls tasks only (run-tests log, read by the verifier)
            **{k: rw.get(k, "") for k in ("test_limit", "test_calls", "test_limit_hit", "first_pass_call")},
            **progress_cols(tdir, denom),
            "exception": (d.get("exception_info") or {}).get("exception_type", ""),
            "agent_sec": secs(d.get("agent_execution")),
            "verifier_sec": secs(d.get("verifier")),
            "total_sec": secs({"started_at": d.get("started_at"), "finished_at": d.get("finished_at")}),
            "in_tokens": ar.get("n_input_tokens", ""),
            "cached_tokens": ar.get("n_cache_tokens", ""),
            "out_tokens": ar.get("n_output_tokens", ""),
        }
        row.update(trajectory_stats(tdir / "agent" / "trajectory.json"))
        row.update(thinking_stats(tdir / "agent"))
        if tdir.parent.name not in px_cache:
            px_cache[tdir.parent.name] = proxy_rows(tdir.parent)
        row.update(proxy_stats(tdir, px_cache[tdir.parent.name]))
        # thinking cut-offs, best source first: the proxy (Strata reports reasoning tokens
        # and the budget per request: tokens >= budget), Strata's cut-off marker line, else
        # a length estimate (llama: no reasoning token count and no marker)
        px_reasoned = any((r.get("reasoning_tokens") or "").isdigit() and (r.get("reasoning_budget") or "").isdigit()
                          for r in trial_proxy_rows(tdir, px_cache[tdir.parent.name]))
        if row.get("thinking_blocks") in ("", None):
            row["thinking_cutoff_source"] = ""
        elif px_reasoned:
            row["thinking_cutoffs"] = row["px_budget_cutoffs"]
            row["thinking_cutoff_source"] = "proxy"
        elif row["_cut_marker"] or job_srv.get(tdir.parent.name, {}).get("srv.target") == "strata":
            row["thinking_cutoff_source"] = "marker"
        else:
            row["thinking_cutoffs"] = row["_cut_est"]
            row["final_turn_cutoff"] = row["_final_cut_est"]
            row["thinking_cutoff_source"] = "length-est"
        # run-tests calls + side samples on a requests/tokens axis -> harbor-progress.tsv;
        # first_pass_* = the budget used when the code first passed every test
        pts = progress_points(tdir, d, trial_proxy_rows(tdir, px_cache[tdir.parent.name]), denom)
        first = min((p for p in pts if p["reward"] in (1, "1") and p["out_tokens_before"] != ""),
                    key=lambda p: p["out_tokens_before"], default=None)
        row["first_pass_requests"] = first["requests_before"] if first else ""
        row["first_pass_out_tokens"] = first["out_tokens_before"] if first else ""
        row["_progress"] = pts
        # server slowdowns: a timed-out trial that lost time to them is invalid (the limit
        # was eaten by the server, not the model); a finished one gets its agent time
        # corrected instead
        lost = row.get("slow_lost_sec") or 0
        if lost and row["exception"] == "AgentTimeoutError" and not row["invalid"]:
            reason = (f"auto: agent timeout after {row['px_slow_requests']} server-slowed request(s) "
                      f"cost ~{lost}s (under {SLOW_FRACTION:.0%} of the job's {row['px_baseline_tok_s']} tok/s)")
            (tdir / "INVALID").write_text(reason + "\n")
            row["invalid"] = reason
        try:
            row["agent_sec_adj"] = round(float(row["agent_sec"]) - lost, 1) if lost else row["agent_sec"]
        except (TypeError, ValueError):
            row["agent_sec_adj"] = ""
        row.update(job_srv.get(tdir.parent.name, {}))
        # llama only reports its --alias as the model id; the loaded GGUF is the real model
        srv = job_srv.get(tdir.parent.name, {})
        if srv.get("srv.target") == "llama" and srv.get("srv.-m"):
            row["model"] = re.sub(r"-\d{5}-of-\d{5}$", "", Path(srv["srv.-m"]).name.removesuffix(".gguf"))
        # run-settings.txt (harbor.sh): MAX_DUR_SCALE etc.; jobs from before it existed ran unscaled
        rs = tdir.parent / "run-settings.txt"
        kv = dict(l.split("=", 1) for l in rs.read_text().splitlines() if "=" in l) if rs.exists() else {}
        row["max_dur_scale"] = kv.get("max_dur_scale", "1")
        # blank when one job mixed several list limits -> undo the scale
        row["max_dur_base"] = kv.get("max_dur_base") or round(row["max_dur"] / float(row["max_dur_scale"] or 1))
        row["disallowed_tools"] = kv.get("disallowed_tools", "")
        # jobs/<job>/CHECK (first line = reason): a harness test run with a real agent
        # (pilot, budget/proxy test, ...) -> checks TSV like oracle/nop
        ck = tdir.parent / "CHECK"
        row["check"] = ((ck.read_text().splitlines() or ["check"])[0] if ck.exists()
                        else ("agent: " + row["agent"] if row["agent"] in CHECK_AGENTS else ""))
        rows.append(row)

    # harness checks (oracle = reference solution, nop = do nothing) aren't model results:
    # they go to a sibling file so the main TSV is model trials only (oracle trials still
    # feed tests_total_oracle above)
    checks_out = a.out.with_name(a.out.stem + "-checks" + a.out.suffix)
    srv_cols = sorted({k for r in rows for k in r if k.startswith("srv.")})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for out, sel in ((a.out, [r for r in rows if not r["check"]]),
                     (checks_out, [r for r in rows if r["check"]])):
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS + srv_cols, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(sel)
        print(f"{len(sel)} trials -> {out}")
    # one row per mid-run test result (model trials only), see progress_points
    prog_out = a.out.with_name("harbor-progress" + a.out.suffix)
    with prog_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROGRESS_COLUMNS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        n = 0
        for r in rows:
            if r["check"]:
                continue
            for p in sorted(r["_progress"], key=lambda p: p["ts"]):
                w.writerow({**{k: r[k] for k in PROGRESS_COLUMNS[:7]}, **p})
                n += 1
    print(f"{n} progress points -> {prog_out}")


if __name__ == "__main__":
    main()
