#!/usr/bin/env python3
"""Patch aider's benchmark/benchmark.py to strip test-runner timings from test output.

Upstream cleanup_test_output() already means to do this ("remove timing
info, to avoid randomizing the response to GPT"), but its one regex only
matches pytest/cargo's "in 0.05s". Every other runner's timings leak through
into the chat history and the next try's prompt:

  jest    "✕ puzzle with three letters (2 ms)", "Time:        0.512 s, estimated 1 s"
  go      "--- FAIL: TestPushAndPop (0.00s)", "FAIL	linkedlist	0.003s"
  gradle  "BUILD FAILED in 1s", "BUILD FAILED in 799ms"

Those vary run to run, so try 2+ get a slightly different prompt every time.
With a fixed seed (see SEED/TEMPERATURE in run.sh) any prompt difference acts
like a new seed, so reruns of the same config diverge after try 1 for no
reason the model controls. Observed: two otherwise identical Darwin-35B runs
on alphametics differed only in "(2 ms)" vs "(3 ms)" in the try-2 prompt.

Hooks onto the upstream regex line so it applies before the truncation added
by patch_benchmark_truncate_output.py (stable text -> stable cut point), and
works whether or not that patch has been applied.

Usage: patch_benchmark_strip_timings.py <path-to-benchmark.py>
Idempotent: safe to run twice (skips if already patched).
"""
import sys

ANCHOR = '''    res = re.sub(r"\\bin \\d+\\.\\d+s\\b", "", output)
'''

ADDED = '''    # See patches/patch_benchmark_strip_timings.py for why these exist.
    res = re.sub(r" \\(\\d+(?:\\.\\d+)? m?s\\)$", "", res, flags=re.M)  # jest per-test
    res = re.sub(r"^Time:\\s+[\\d.]+ m?s(?:, estimated [\\d.]+ m?s)?\\n?", "", res, flags=re.M)  # jest summary
    res = re.sub(r"^(\\s*--- (?:FAIL|PASS|SKIP): \\S+) \\(\\d+\\.\\d+s\\)", r"\\1", res, flags=re.M)  # go per-test
    res = re.sub(r"^((?:ok|FAIL)\\s+\\S+)\\s+\\d+\\.\\d+s$", r"\\1", res, flags=re.M)  # go per-package
    res = re.sub(r"^(BUILD (?:FAILED|SUCCESSFUL)) in (?:\\d+(?:ms|h|m|s) ?)+$", r"\\1", res, flags=re.M)  # gradle
'''


def main():
    if len(sys.argv) != 2:
        print("usage: patch_benchmark_strip_timings.py <path-to-benchmark.py>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    if "patch_benchmark_strip_timings.py" in content:
        print(f"{path}: already patched, skipping")
        return

    if content.count(ANCHOR) != 1:
        print(f"ERROR: {path} does not contain exactly one expected cleanup_test_output() timing regex -- "
              "upstream source has likely changed, patch needs updating", file=sys.stderr)
        sys.exit(1)

    content = content.replace(ANCHOR, ANCHOR + ADDED)
    with open(path, "w") as f:
        f.write(content)
    print(f"{path}: patched cleanup_test_output() to strip jest/go/gradle timings")


if __name__ == "__main__":
    main()
