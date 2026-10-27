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
from datetime import datetime
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
    "trial", "invalid", "max_dur", "max_dur_scale", "max_dur_base", "budget", "budget_hit", "reward", "tests_passed", "tests_total", "tests_total_oracle", "tests_frac",
    "tests_frac_adj", "tests_parsed",
    "test_limit", "test_calls", "test_limit_hit", "first_pass_call", "exception",
    "agent_sec", "verifier_sec", "total_sec",
    "in_tokens", "cached_tokens", "out_tokens",
    "agent_steps", "tool_calls", "tool_errors", "malformed_calls", "markup_final",
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
                s["edit_errors"] += failed
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


def task_agent_timeout(tdir: Path) -> float:
    """[agent] timeout_sec from the trial's task.toml (polyglot: 1800)."""
    try:
        cfg = json.loads((tdir / "config.json").read_text())
        toml = Path(cfg["task"]["path"]) / "task.toml"
        sect = toml.read_text().split("[agent]", 1)[1].split("\n[", 1)[0]
        return float(re.search(r"timeout_sec\s*=\s*([0-9.]+)", sect).group(1))
    except (OSError, KeyError, IndexError, AttributeError, ValueError):
        return 1800.0


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

    oracle_total: dict[str, int] = {}
    for _, d in trials:
        rw = (d.get("verifier_result") or {}).get("rewards") or {}
        if d["agent_info"]["name"] == "oracle" and rw.get("tests_parsed"):
            oracle_total[d["task_name"]] = max(oracle_total.get(d["task_name"], 0), rw.get("tests_total", 0))

    rows = []
    for tdir, d in trials:
        rw = (d.get("verifier_result") or {}).get("rewards") or {}
        ar = d.get("agent_result") or {}
        mi = d["agent_info"].get("model_info") or {}
        task = d["task_name"]
        m = re.search(r"polyglot_([a-z]+)_(.+?)(?:__tc\d+)?$", task)
        passed, total = rw.get("tests_passed"), rw.get("tests_total")
        o_total = oracle_total.get(task)
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
            "exception": (d.get("exception_info") or {}).get("exception_type", ""),
            "agent_sec": secs(d.get("agent_execution")),
            "verifier_sec": secs(d.get("verifier")),
            "total_sec": secs({"started_at": d.get("started_at"), "finished_at": d.get("finished_at")}),
            "in_tokens": ar.get("n_input_tokens", ""),
            "cached_tokens": ar.get("n_cache_tokens", ""),
            "out_tokens": ar.get("n_output_tokens", ""),
        }
        row.update(trajectory_stats(tdir / "agent" / "trajectory.json"))
        row.update(job_srv.get(tdir.parent.name, {}))
        # run-settings.txt (harbor.sh): MAX_DUR_SCALE etc.; jobs from before it existed ran unscaled
        rs = tdir.parent / "run-settings.txt"
        kv = dict(l.split("=", 1) for l in rs.read_text().splitlines() if "=" in l) if rs.exists() else {}
        row["max_dur_scale"] = kv.get("max_dur_scale", "1")
        row["max_dur_base"] = kv.get("max_dur_base") or row["max_dur"]
        rows.append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        srv_cols = sorted({k for r in rows for k in r if k.startswith("srv.")})
        w = csv.DictWriter(f, fieldnames=COLUMNS + srv_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trials -> {a.out}")


if __name__ == "__main__":
    main()
