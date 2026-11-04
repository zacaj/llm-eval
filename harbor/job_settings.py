"""Per-job run settings shared by skip_existing.py (which trials count as the same run)
and collect.py (-> the viz's model grouping), so the two agree on what is a different config."""
from __future__ import annotations

from pathlib import Path


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


def dur_scale(job: Path) -> float:
    """MAX_DUR_SCALE the job ran at (1 before run-settings.txt existed)."""
    try:
        return float(read_kv(job / "run-settings.txt").get("max_dur_scale") or 1)
    except ValueError:
        return 1.0
