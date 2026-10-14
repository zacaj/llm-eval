#!/usr/bin/env python3
"""Patch aider's benchmark/benchmark.py to cap test-output length.

Upstream gap (confirmed against Aider-AI/aider, no fix as of the commit this
image vendors): cleanup_test_output() only strips timing strings, never caps
length, and the full result gets appended verbatim to the aider chat history
on every retry (see run_test_real -> history_fname.write). A single bug that
triggers a verbose diagnostic path -- GCC's operator== overload-candidate
dump on an STL container type mismatch, or a Go test that logs one line per
failed loop iteration with no cap -- can produce hundreds of KB from one
one-line mistake. That gets baked into every subsequent turn and compounds
across --tries, eventually exceeding the model's context window outright
(observed: 165k -> 327k -> 489k -> 651k -> 813k tokens across 5 retries on
one exercise). At that point the exercise fails structurally, not on
reasoning merit -- and it happens identically regardless of model quality,
since every model hits the same oversized request.

Truncates to head + tail (the actual error is usually at the top for GCC,
and the final pass/fail status is always at the bottom; the noise is in the
middle) rather than a flat cutoff, so the model still sees the real error
and the final build/test status either side of the elision.

Usage: patch_benchmark_truncate_output.py <path-to-benchmark.py>
Idempotent: safe to run twice (skips if already patched).
"""
import sys

OLD = '''def cleanup_test_output(output, testdir):
    # remove timing info, to avoid randomizing the response to GPT
    res = re.sub(r"\\bin \\d+\\.\\d+s\\b", "", output)
    res = res.replace(str(testdir), str(testdir.name))
    return res'''

NEW = '''def truncate_test_output(output, max_chars=8000, head_chars=5000, tail_chars=2000):
    # See patches/patch_benchmark_truncate_output.py for why this exists.
    if len(output) <= max_chars:
        return output
    dropped = len(output) - head_chars - tail_chars
    head = output[:head_chars]
    tail = output[-tail_chars:]
    return (
        f"{head}\\n\\n"
        f"... [{dropped} characters elided -- output was {len(output)} characters"
        f" total; truncated to avoid exceeding the model's context window] ...\\n\\n"
        f"{tail}"
    )


def cleanup_test_output(output, testdir):
    # remove timing info, to avoid randomizing the response to GPT
    res = re.sub(r"\\bin \\d+\\.\\d+s\\b", "", output)
    res = res.replace(str(testdir), str(testdir.name))
    res = truncate_test_output(res)
    return res'''


def main():
    if len(sys.argv) != 2:
        print("usage: patch_benchmark_truncate_output.py <path-to-benchmark.py>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    if "def truncate_test_output(" in content:
        print(f"{path}: already patched, skipping")
        return

    if OLD not in content:
        print(f"ERROR: {path} does not contain the expected cleanup_test_output() body -- "
              "upstream source has likely changed, patch needs updating", file=sys.stderr)
        sys.exit(1)

    content = content.replace(OLD, NEW)
    with open(path, "w") as f:
        f.write(content)
    print(f"{path}: patched cleanup_test_output() to cap test-output length")


if __name__ == "__main__":
    main()
