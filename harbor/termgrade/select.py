#!/usr/bin/env python3
"""Shortlist TermGrade tasks from the publisher's own graded trials.

Needs, in --src (runs/termgrade-environments): task_source.jsonl + tasks.jsonl from
ai-and/termgrade-environments and data/{tests,trials}.parquet from
ai-and/termgrade-trajectories. Run with pyarrow: `uv run --with pyarrow python -I select.py`.

Their numbers are Terminus-2 (JSON + tmux keystrokes), no test feedback, 12h cap, so they
overstate difficulty for Claude Code (first pilot, 2026-11-18: 3 of 4 picks with qwen3.6-27B
pass@1 0.12-0.62 were first-call passes for our mid models). Hence the defaults: tasks the
local-class models (qwen3.6-27B, gemma-4-31B) mostly failed that a strong model solves.

Filters out tests that likely contradict the instructions: a test that strong models
(kimi/glm/deepseek) fail in most of their near-miss trials (all but <= 2 tests passed), a
test nobody ever passed, or any skipped test.
"""
import argparse
import collections
import json
import re
import statistics as st
from pathlib import Path

import pyarrow.parquet as pq

STRONG = {"kimi", "glm", "deepseek"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(__file__).resolve().parents[2] / "runs" / "termgrade-environments")
    ap.add_argument("--local-max", type=float, default=0.25, help="max qwen and gemma pass@1")
    ap.add_argument("--strong-min", type=float, default=0.75, help="min best strong-model pass@1")
    ap.add_argument("--min-tests", type=int, default=10)
    ap.add_argument("--exclude", default="", help="comma list of tasks already tried")
    a = ap.parse_args()
    idx = {json.loads(l)["task"]: json.loads(l) for l in open(a.src / "tasks.jsonl")}
    src = {json.loads(l)["task"]: json.loads(l) for l in open(a.src / "task_source.jsonl")}
    tr = {r["trial"]: r for r in pq.read_table(a.src / "trials.parquet").to_pylist()}
    by = collections.defaultdict(list)
    for r in pq.read_table(a.src / "tests.parquet").to_pylist():
        if tr.get(r["trial"], {}).get("counts_toward_k"):
            by[r["task"]].append(r)
    excl = set(filter(None, a.exclude.split(",")))
    out = []
    for t, rs in by.items():
        i = idx[t]
        if t in excl or i["split"] != "unseen" or not i["oracle_verified"]:
            continue
        names = {c["name"] for r in rs for c in r["cases"] or []}
        if len(names) < a.min_tests:
            continue
        ever, fail_near, strong_n = collections.Counter(), collections.Counter(), 0
        skipped = False
        for r in rs:
            cs = r["cases"] or []
            skipped |= any(c["status"] == "skipped" for c in cs)
            for c in cs:
                ever[c["name"]] += c["status"] == "passed"
            if r["model"] in STRONG:
                strong_n += 1
                p = sum(c["status"] == "passed" for c in cs)
                if cs and len(cs) - 2 <= p < len(cs):
                    for c in cs:
                        if c["status"] != "passed":
                            fail_near[c["name"]] += 1
        if skipped or any(ever[n] == 0 for n in names):
            continue
        if strong_n and max(fail_near.values(), default=0) / strong_n > 0.4:
            continue
        rew, frac = collections.defaultdict(list), collections.defaultdict(list)
        for r in rs:
            rew[r["model"]].append(r["reward"])
            cs = r["cases"] or []
            if cs:
                frac[r["model"]].append(sum(c["status"] == "passed" for c in cs) / len(cs))
        m = {k: st.mean(v) for k, v in rew.items()}
        mf = {k: st.mean(v) for k, v in frac.items()}
        if "qwen" not in m or "gemma" not in m:
            continue
        if max(m["qwen"], m["gemma"]) > a.local_max:
            continue
        best_strong = max((m[k] for k in STRONG if k in m), default=0)
        if best_strong < a.strong_min:
            continue
        ins = src[t]["instruction"]
        kind = "debug" if re.search(r"\b(broken|bug|buggy|fix (the|it|this))\b", ins[:800], re.I) else "build"
        loc_frac = st.mean([mf.get("qwen", 0), mf.get("gemma", 0)])
        out.append((kind, t, len(names), m["qwen"], m["gemma"], m.get("gemma_think", -1),
                    m.get("kimi", -1), m.get("glm", -1), m.get("deepseek", -1), loc_frac,
                    ins[:70].replace("\n", " ")))
    out.sort(key=lambda o: (o[0] != "debug", -o[9]))
    print(f"{len(out)} tasks; kind task tests qwen gemma gem_think kimi glm deepseek local_test_frac")
    for o in out:
        print(*[f"{x:.2f}" if isinstance(x, float) else x for x in o])


if __name__ == "__main__":
    main()
