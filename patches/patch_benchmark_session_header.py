#!/usr/bin/env python3
"""Patch aider's benchmark/benchmark.py to tag each exercise's model requests.

Adds an `X-Eval-Session: <run dir>/<language>/<exercise>` header (via the
model's extra_params.extra_headers, which litellm sends on every completion)
so the eval proxy run.sh puts in front of the model server (harbor/eval_proxy.py)
can attribute each request it logs -- with the server's own report of effort,
thinking budget, reasoning/answer tokens and budget cut-offs -- to an exercise.
track_per_test then sums those into the px_* columns of eval-results-per-test.tsv.

Aider requests carry no session id of their own, and with --threads > 1
exercises interleave, so timing alone can't attribute them.

Usage: patch_benchmark_session_header.py <path-to-benchmark.py>
Idempotent: safe to run twice (skips if already patched).
"""
import sys

ANCHOR = '''    edit_format = edit_format or main_model.edit_format
'''

ADDED = '''    # See patches/patch_benchmark_session_header.py for why this exists.
    if not main_model.extra_params:
        main_model.extra_params = {}
    _session = f"{testdir.parts[-5]}/{testdir.parts[-4]}/{testdir.name}"
    main_model.extra_params["extra_headers"] = {
        **(main_model.extra_params.get("extra_headers") or {}),
        "X-Eval-Session": _session,
    }
'''

MARKER = "patch_benchmark_session_header.py"


def main():
    path = sys.argv[1]
    src = open(path).read()
    if MARKER in src:
        print(f"{path}: already patched (session header)")
        return
    if src.count(ANCHOR) != 1:
        sys.exit(f"{path}: anchor not found exactly once; benchmark.py changed upstream?")
    src = src.replace(ANCHOR, ADDED + ANCHOR)
    open(path, "w").write(src)
    print(f"{path}: patched run_test_real() with an X-Eval-Session header per exercise")


if __name__ == "__main__":
    main()
