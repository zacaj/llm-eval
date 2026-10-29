#!/usr/bin/env python3
"""Background progress sampler for one Harbor job (started by harbor.sh run_job).

Every --interval seconds, for each running trial of the job, it snapshots the trial
container's /app (minus build output) and runs the task's own verifier script
(tests/test.sh) on that snapshot in a *separate* throwaway container from the same
image, with the hidden tests mounted read-only. The agent's container is only read
(docker exec tar), so the agent never sees the tests, the test processes or any
change to /app -- this works for classic tasks (root agent) and test-calls tasks alike,
and doesn't use up run-tests calls.

One row per sample -> jobs/<job>/<trial>/progress.tsv:
  ts  elapsed_sec  passed  total  reward  sample_sec  status
elapsed_sec = since the trial container started (includes agent setup). status: ok,
timeout (killed after --timeout; counts as unknown, not 0), error.

Samples cost CPU next to the agent (capped with --cpus), so a slow build/test suite
can slow the agent down a little; --interval 0 in harbor.sh (PROGRESS_INTERVAL=0)
turns this off.
"""
import argparse
import json
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def project_name(trial: str) -> str:
    # Harbor: compose project = sanitize(f"{trial_name}__env")
    n = f"{trial}__env".lower()
    if not re.match(r"^[a-z0-9]", n):
        n = "0" + n
    return re.sub(r"[^a-z0-9_-]", "-", n)


def running_containers() -> dict[str, tuple[str, str, float]]:
    """compose project -> (container id, image id, started-at epoch)"""
    out = subprocess.run(["docker", "ps", "-q", "--filter", "label=com.docker.compose.project"],
                         capture_output=True, text=True).stdout.split()
    if not out:
        return {}
    info = json.loads(subprocess.run(["docker", "inspect", *out], capture_output=True, text=True).stdout or "[]")
    res = {}
    for c in info:
        proj = (c.get("Config", {}).get("Labels") or {}).get("com.docker.compose.project")
        if not proj or (c.get("Config", {}).get("Labels") or {}).get("com.docker.compose.service") != "main":
            continue
        started = c["State"].get("StartedAt", "")[:26].rstrip("Z")
        try:
            t0 = datetime.fromisoformat(started).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            t0 = time.time()
        res[proj] = (c["Id"], c["Image"], t0)
    return res


def sample(trial_dir: Path, cid: str, image: str, t0: float, a) -> None:
    cfg = json.loads((trial_dir / "config.json").read_text())
    tests = Path(cfg["task"]["path"]) / "tests"
    start = time.time()
    status, rw = "error", {}
    with tempfile.TemporaryDirectory(prefix="harbor-progress-") as tmp:
        snap, out = Path(tmp) / "app", Path(tmp) / "out"
        snap.mkdir(); out.mkdir()
        try:
            tar = subprocess.Popen(["docker", "exec", cid, "tar", "-C", "/app", "--exclude=./target",
                                    "--exclude=./build", "--exclude=./.gradle", "-cf", "-", "."],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            subprocess.run(["tar", "-C", str(snap), "-xf", "-"], stdin=tar.stdout, check=True,
                           stderr=subprocess.DEVNULL, timeout=60)
            tar.wait(timeout=10)
            name = f"progress-{trial_dir.name.lower()}-{int(start)}"
            try:
                subprocess.run(["docker", "run", "--rm", "--name", name, f"--cpus={a.cpus}",
                                "-e", f"TEST_TIMEOUT={a.timeout}",
                                "-v", f"{snap}:/app", "-v", f"{tests}:/tests:ro", "-v", f"{out}:/logs/verifier",
                                "--entrypoint", "bash", image, "/tests/test.sh"],
                               capture_output=True, timeout=a.timeout + 60)
                rw = json.loads((out / "reward.json").read_text())
                timed_out = "TESTS TIMED OUT" in (out / "test-output.txt").read_text(errors="replace")
                status = "timeout" if timed_out else "ok"
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                status = "timeout"
        except (OSError, ValueError, subprocess.SubprocessError, KeyError):
            pass
        finally:
            # files the container wrote as root: clean them via a container so the
            # TemporaryDirectory cleanup doesn't fail on permissions
            subprocess.run(["docker", "run", "--rm", "-v", f"{tmp}:/x", "--entrypoint", "sh", image,
                            "-c", "rm -rf /x/app /x/out"], capture_output=True, timeout=120)
    end = time.time()
    row = [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), f"{end - t0:.0f}",
           str(rw.get("tests_passed", "")), str(rw.get("tests_total", "")), str(rw.get("reward", "")),
           f"{end - start:.0f}", status]
    f = trial_dir / "progress.tsv"
    new = not f.exists()
    with f.open("a") as fh:
        if new:
            fh.write("ts\telapsed_sec\tpassed\ttotal\treward\tsample_sec\tstatus\n")
        fh.write("\t".join(row) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("job_dir", type=Path)
    p.add_argument("--interval", type=int, default=180, help="seconds between samples of a trial")
    p.add_argument("--timeout", type=int, default=90, help="test run cap (TEST_TIMEOUT) per sample")
    p.add_argument("--cpus", default="2")
    a = p.parse_args()
    last: dict[str, float] = {}       # trial -> time its last sample started
    busy: dict[str, threading.Thread] = {}
    while True:
        conts = running_containers()
        for td in a.job_dir.iterdir():
            if not td.is_dir() or (td / "result.json").exists() or not (td / "config.json").exists():
                continue
            c = conts.get(project_name(td.name))
            if not c or (td.name in busy and busy[td.name].is_alive()):
                continue
            cid, image, t0 = c
            if time.time() - last.get(td.name, t0) < a.interval:
                continue
            last[td.name] = time.time()
            busy[td.name] = threading.Thread(target=sample, args=(td, cid, image, t0, a), daemon=True)
            busy[td.name].start()
        time.sleep(10)


if __name__ == "__main__":
    main()
