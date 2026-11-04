#!/usr/bin/env python3
"""Patch aider's benchmark/benchmark.py to add a wall-clock cap per exercise.

Observed need: a slow (partial-CPU-offload) model spent 88 minutes across all
6 --tries on a single exercise, repeatedly tripping the same lint error via
aider's own internal reflection loop (up to 3 extra fix-up rounds per try,
see aider/coders/base_coder.py's max_reflections) without ever converging on
a working fix. run_test_real() already tracks cumulative `dur` across tries
but never acts on it -- there's no existing mechanism to bail out of a
clearly-not-converging exercise early.

Adds an opt-in wall-clock cap (AIDER_BENCHMARK_MAX_DUR env var, seconds; unset
or 0 disables it, preserving existing behavior). Checked once per try, right
after that try's coder.run() call completes and before running the (also
time-consuming) unit test subprocess -- so a stuck/slow try still finishes
naturally rather than being killed mid-generation, and the next try just
never starts once the budget is blown. Surfaces whether this fired via a new
`bailed_on_max_duration` field in run_test_real's results dict, so it's
distinguishable in aider-results-per-test.tsv from a exercise that genuinely
used and exhausted all --tries.

Usage: patch_benchmark_max_duration.py <path-to-benchmark.py>
Idempotent: safe to run twice (skips if already patched).
"""
import sys

OLD_INIT = '''    test_outcomes = []
    for i in range(tries):
        start = time.time()'''

NEW_INIT = '''    test_outcomes = []
    # See patches/patch_benchmark_max_duration.py for why this exists.
    max_dur = float(os.environ.get("AIDER_BENCHMARK_MAX_DUR", "0") or 0)
    bailed_on_max_dur = False
    for i in range(tries):
        start = time.time()'''

OLD_CHECK = '''        dur += time.time() - start

        if not no_aider:'''

NEW_CHECK = '''        dur += time.time() - start

        if max_dur and dur > max_dur:
            bailed_on_max_dur = True
            break

        if not no_aider:'''

OLD_RESULTS = '''    results = dict(
        testdir=str(testdir),
        testcase=testdir.name,
        model=main_model.name,
        edit_format=edit_format,
        tests_outcomes=test_outcomes,
        cost=coder.total_cost,
        duration=dur,
        test_timeouts=timeouts,'''

NEW_RESULTS = '''    results = dict(
        testdir=str(testdir),
        testcase=testdir.name,
        model=main_model.name,
        edit_format=edit_format,
        tests_outcomes=test_outcomes,
        cost=coder.total_cost,
        duration=dur,
        bailed_on_max_duration=bailed_on_max_dur,
        test_timeouts=timeouts,'''

PATCHES = [
    (OLD_INIT, NEW_INIT),
    (OLD_CHECK, NEW_CHECK),
    (OLD_RESULTS, NEW_RESULTS),
]


def main():
    if len(sys.argv) != 2:
        print("usage: patch_benchmark_max_duration.py <path-to-benchmark.py>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    if "bailed_on_max_dur" in content:
        print(f"{path}: already patched, skipping")
        return

    for old, new in PATCHES:
        if old not in content:
            print(f"ERROR: {path} does not contain an expected block -- "
                  "upstream source has likely changed, patch needs updating:\n"
                  f"{old[:80]}...", file=sys.stderr)
            sys.exit(1)
        content = content.replace(old, new)

    with open(path, "w") as f:
        f.write(content)
    print(f"{path}: patched run_test_real() with an AIDER_BENCHMARK_MAX_DUR wall-clock cap")


if __name__ == "__main__":
    main()
