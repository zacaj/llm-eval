"""Per-job run settings shared by skip_existing.py (which trials count as the same run)
and collect.py (-> the viz's model grouping), so the two agree on what is a different config."""
from __future__ import annotations

import re
import sys
from pathlib import Path

# tb2_tc instructions state the agent's limit (tb2/main.py); polyglot ones don't
LIMIT_LINE_RE = re.compile(r"^Your time limit is [0-9]+ minutes;|^Your output budget is [0-9]+k tokens;", re.M)


def read_kv(path: Path) -> dict[str, str]:
    return dict(l.split("=", 1) for l in path.read_text().splitlines() if "=" in l) if path.exists() else {}


def think_budget(job: Path, srv: dict[str, str] | None = None) -> str:
    """The thinking cap a job ran with: harbor.sh's THINK_BUDGET (run-settings.txt), else what
    the server capture recorded (Strata's resolved cap; backfilled 4000 for old llama jobs)."""
    tb = read_kv(job / "run-settings.txt").get("think_budget", "")
    if tb and tb != "0":
        return tb
    if srv is None:
        srv = read_kv(job / "server-settings.txt")
    return srv.get("srv.eff.reasoning_budget", "")


def task_budgets(job: Path) -> dict[str, int]:
    """Per-task output-token budgets the eval proxy enforced (harbor.sh's
    jobs/<job>/eval-settings.json; task dir name -> budget). Empty for jobs from before it:
    their budget is the job-wide X-Eval-Budget header in config.json."""
    import json
    try:
        d = json.loads((job / "eval-settings.json").read_text())
    except (OSError, ValueError):
        return {}
    return {t["name"]: int(t.get("budget") or 0) for t in d.get("tasks", []) if t.get("name")}


def timeout_mode(task_path: str | Path) -> str:
    """How a trial's agent time limit counted, from the task copy it ran (harbor.sh's
    variant()): "backstop" = tasks/_backstop/<secs>/ (a token-budget row: the limit is only a
    stuck-trial safety net, and hitting it is a harness failure); "fail" = anything else
    (tasks/_dur/<secs>/ or the dataset's own task: the limit ends the attempt and the work
    is graded as it is -- every trial from before token-budget rows existed)."""
    return "backstop" if "/_backstop/" in str(task_path) else "fail"


def variant_instruction(text: str, mode: str, secs: int, budget: int) -> str:
    """The instruction a harbor.sh task copy gets: a stated time limit says the copy's limit
    ("fail" mode), or -- token-budget rows, where the time limit is only a backstop -- the
    output budget instead. Texts without the line (polyglot) come back unchanged.
    Shared with skip_existing.py, so trials only count toward a run that told the agent
    the same thing."""
    if mode == "backstop" and budget > 0:
        new = f"Your output budget is {round(budget / 1000)}k tokens;"
    else:
        new = f"Your time limit is {(secs + 30) // 60} minutes;"
    return LIMIT_LINE_RE.sub(new, text)


def dur_scale(job: Path) -> float:
    """MAX_DUR_SCALE the job ran at (1 before run-settings.txt existed)."""
    try:
        return float(read_kv(job / "run-settings.txt").get("max_dur_scale") or 1)
    except ValueError:
        return 1.0


if __name__ == "__main__":
    # harbor.sh variant(): job_settings.py instruction <file> <fail|backstop> <secs> <budget>
    # rewrites <file> in place
    if len(sys.argv) == 6 and sys.argv[1] == "instruction":
        p = Path(sys.argv[2])
        p.write_text(variant_instruction(p.read_text(), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])))
    else:
        sys.exit("usage: job_settings.py instruction <file> <fail|backstop> <secs> <budget>")
