#!/usr/bin/env python3
"""SKIP_EXISTING for harbor.sh: how many more attempts each task still needs.

    python3 skip_existing.py --jobs jobs --dataset DIR --agent A --budget N --max-dur S \
        --k K --server server-settings.txt  GLOB [GLOB ...]

Prints "<remaining>\\t<task dir name>\\t<have>\\t<jobs>" for every task matching the
globs (remaining may be 0; jobs = comma list of the jobs holding the counted trials). An existing trial counts toward K when it is the same task, agent (+ --agent-version if given),
eval budget and a compatible agent time limit (time_limit_matches), ran against an equivalent server (SERVER_KEYS
below, compared with the job's server-settings.txt), is not marked INVALID, and
was graded (has a verifier result) -- agent timeouts / budget stops / test-call
limit stops are real outcomes and count; ungraded infra errors don't.

--timeout-mode backstop (a token-budget list row: the time limit is only a safety net)
changes the time rule: any trial that ended on its own counts whatever its limit was,
but one cut off by a time limit never does -- with no time pressure it would have kept
going, even if it hadn't reached the budget yet. A trial that hit a backstop limit
never counts in either mode (harness failure). Tasks whose instruction states the limit
(tb2_tc) also need the same stated budget (instruction_matches).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
from datetime import datetime
from pathlib import Path

from job_settings import LIMIT_LINE_RE, dur_scale, read_kv, task_budgets, think_budget, timeout_mode, variant_instruction

# Output-affecting server settings that must match (others, e.g. speculative
# decoding knobs, change speed not outputs). Missing on both sides = match.
SERVER_KEYS = re.compile(
    r"^srv\.(target|-m|--temp|--top-p|--top-k|--min-p|eff\..*|kv_cache|-ctk|-ctv|"
    r"--reasoning-budget|logit_bias|proxy_filters|--chat-template-file|"
    r"--lora|--lora-scaled|--control-vector|--control-vector-scaled)$"
)
# the thinking cap is compared on its own (job_settings.think_budget, which collect.py /
# the viz use too): newer jobs record it in run-settings.txt, not the server capture
SERVER_SKIP = {"srv.eff.reasoning_budget"}


def server_sig(text: str) -> dict[str, str]:
    kv = dict(l.split("=", 1) for l in text.splitlines() if "=" in l)
    # srv.eff.thinking was added to the capture later; older jobs lack it. Thinking is only
    # off when the effort is off (EFFORT=off), so a known effort level implies thinking on.
    if "srv.eff.thinking" not in kv and "srv.eff.reasoning_effort" in kv:
        kv["srv.eff.thinking"] = "False" if kv["srv.eff.reasoning_effort"] in ("off", "none", "") else "True"
    # *_source keys say where a value came from (e.g. "backfilled"), not what ran
    return {k: v for k, v in kv.items() if SERVER_KEYS.match(k) and not k.endswith("_source")
            and k not in SERVER_SKIP}


def task_timeout(task_dir: Path) -> float:
    try:
        sect = (task_dir / "task.toml").read_text().split("[agent]", 1)[1].split("\n[", 1)[0]
        return float(re.search(r"timeout_sec\s*=\s*([0-9.]+)", sect).group(1))
    except (OSError, IndexError, AttributeError, ValueError):
        return 1800.0


def agent_seconds(r: dict) -> float | None:
    ae = r.get("agent_execution") or {}
    try:
        f = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))
        return (f(ae["finished_at"]) - f(ae["started_at"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def time_limit_matches(r: dict, limit: int, want_limit: int, base: int, want_base: int | None) -> bool:
    """Same configured (unscaled) limit = same run: MAX_DUR_SCALE=auto only adjusts it for
    the server's measured speed, which drifts a little between jobs. Otherwise, a trial
    that hit its time limit was graded on a snapshot taken at that limit, so it doesn't
    match. One that ended on its own (done, budget or test-call stop, ...) after T seconds
    would have played out identically under any limit >= T, whatever it ran with."""
    if limit == want_limit or (want_base is not None and base == want_base):
        return True
    if ((r.get("exception_info") or {}).get("exception_type")) == "AgentTimeoutError":
        return False
    t = agent_seconds(r)
    return t is not None and t <= want_limit


def budget_matches(r: dict, budget: int, want: int, max_req: int) -> bool:
    """Same eval token budget = same run (0 = unlimited). A trial whose budget never bit
    (no "budget exhausted" stop) used fewer tokens than its own cap; if that cap was <=
    the wanted one, the wanted cap would never have bit either, so it played out
    identically. A trial under a bigger (or no) cap matches a smaller wanted one only if
    it stayed far enough under it that the wanted cap would never have clamped a request:
    total output + one full request (max_req = thinking cap + answer tokens) <= wanted.
    A trial that hit its cap only matches the same cap."""
    inf = float("inf")
    b, w = budget or inf, want or inf
    if b == w:
        return True
    exhausted = "budget exhausted" in ((r.get("exception_info") or {}).get("exception_message") or "")
    if b < w:
        return not exhausted
    used = (r.get("agent_result") or {}).get("n_output_tokens")
    return not exhausted and isinstance(used, int) and used + max_req <= w


def states_budget(trial_task: Path) -> bool:
    try:
        return "Your output budget is " in (trial_task / "instruction.md").read_text()
    except OSError:
        return False


def instruction_matches(trial_task: Path, want_text: str | None) -> bool:
    """Tasks whose instruction states the agent's limit (tb2_tc) only count toward a run
    that tells the agent the same thing. Unknown (task copy gone) = assume it matches."""
    if want_text is None:
        return True
    try:
        return (trial_task / "instruction.md").read_text() == want_text
    except OSError:
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--agent-version", default="")
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--think-budget", type=int, default=0)
    ap.add_argument("--temperature", default="", help="forced temperature (harbor.sh TEMPERATURE); '' = agent's own")
    ap.add_argument("--max-dur", default="")
    ap.add_argument("--max-dur-base", default="", help="the limit before the auto dur scale")
    ap.add_argument("--timeout-mode", choices=["fail", "backstop"], default="fail",
                    help="fail = the time limit ends the attempt (legacy rows); backstop = token-budget "
                         "row, the limit is only a safety net")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("globs", nargs="+")
    a = ap.parse_args()

    want_srv_all = dict(l.split("=", 1) for l in a.server.read_text().splitlines() if "=" in l)
    want_srv = server_sig(a.server.read_text())
    want_think = str(a.think_budget) if a.think_budget else want_srv_all.get("srv.eff.reasoning_budget", "")
    want_base = round(float(a.max_dur_base)) if a.max_dur_base else None
    tasks = sorted(p.name for p in a.dataset.iterdir()
                   if p.is_dir() and any(fnmatch.fnmatch(p.name, g) for g in a.globs))
    # Backstop runs: what their task copies will tell the agent (the token budget, on tasks
    # whose instruction states a limit; None = states none). Fail-mode runs keep the old
    # rule (stated minutes follow the auto scale, which drifts between jobs, so they're not
    # compared), except that a backstop trial -- told a token budget -- never counts there.
    want_instr: dict[str, str | None] = dict.fromkeys(tasks)
    if a.timeout_mode == "backstop":
        for t in tasks:
            try:
                txt = (a.dataset / t / "instruction.md").read_text()
            except OSError:
                continue
            if LIMIT_LINE_RE.search(txt):
                want_instr[t] = variant_instruction(txt, "backstop", 0, a.budget)

    have: dict[str, int] = {t: 0 for t in tasks}
    where: dict[str, dict[str, int]] = {t: {} for t in tasks}
    for job in a.jobs.iterdir():
        cfg_path, srv_path = job / "config.json", job / "server-settings.txt"
        if not cfg_path.exists() or not srv_path.exists():
            continue
        if server_sig(srv_path.read_text()) != want_srv:
            continue
        cfg = json.loads(cfg_path.read_text())
        agents = cfg.get("agents") or [{}]
        cj = json.dumps(agents)
        m = re.search(r"X-Eval-Budget\W+(\d+)", cj)
        job_budget = int(m.group(1)) if m else 0
        per_task = task_budgets(job)   # eval-settings.json: per-task budgets win over the header
        if think_budget(job) != want_think:
            continue
        scale = dur_scale(job)
        rs = read_kv(job / "run-settings.txt")
        if rs.get("temperature", "") != a.temperature:
            continue
        try:
            max_req = int(rs.get("think_budget") or 0) + int(rs.get("answer_tokens") or 0)
        except ValueError:
            max_req = 0
        max_req = max_req if max_req > 0 else 16000   # jobs from before these were recorded
        mult = cfg.get("agent_timeout_multiplier") or cfg.get("timeout_multiplier") or 1.0
        for res in job.glob("*/result.json"):
            tdir = res.parent
            if (tdir / "INVALID").exists():
                continue
            r = json.loads(res.read_text())
            info = r.get("agent_info") or {}
            if info.get("name") != a.agent or not r.get("verifier_result"):
                continue
            if a.agent_version and info.get("version") != a.agent_version:
                continue
            task_path = Path(json.loads((tdir / "config.json").read_text())["task"]["path"])
            task = task_path.name
            if task not in have:
                continue
            # the task.toml the trial ran: harbor.sh's tasks/_dur/<secs>/ copies carry
            # their own limit (fall back to the dataset's if that copy is gone)
            src = task_path if (task_path / "task.toml").exists() else a.dataset / task
            limit = round(task_timeout(src) * mult)
            want_limit = round(float(a.max_dur)) if a.max_dur else round(task_timeout(a.dataset / task))
            if not budget_matches(r, per_task.get(task, job_budget), a.budget, max_req):
                continue
            if not instruction_matches(src, want_instr[task]):
                continue
            timed_out = ((r.get("exception_info") or {}).get("exception_type")) == "AgentTimeoutError"
            trial_mode = timeout_mode(task_path)
            if trial_mode == "backstop" and timed_out:
                continue   # hit the safety net: a harness failure (collect.py marks it INVALID too)
            if a.timeout_mode == "fail" and trial_mode == "backstop" and states_budget(src):
                continue   # that agent was told a token budget, not a time limit
            if a.timeout_mode == "backstop":
                # no time pressure in the wanted run: a trial cut off by a time limit (one that
                # ended the attempt) would have kept going, so only trials that ended on their
                # own count, whatever limit they had
                ok = not timed_out
            else:
                ok = time_limit_matches(r, limit, want_limit, round(limit / scale), want_base)
            if ok:
                have[task] += 1
                where[task][job.name] = where[task].get(job.name, 0) + 1

    for t in tasks:
        jobs = ",".join(f"{j}" + (f" x{n}" if n > 1 else "") for j, n in sorted(where[t].items()))
        print(f"{max(0, a.k - have[t])}\t{t}\t{have[t]}\t{jobs}")


if __name__ == "__main__":
    main()
