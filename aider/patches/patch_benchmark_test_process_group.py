#!/usr/bin/env python3
"""Patch aider's benchmark/benchmark.py so the per-exercise test-runner
timeout actually kills the test process it times out.

Observed need: `run_unit_tests()` runs the test command with
`subprocess.run(command, timeout=timeout, ...)`. On timeout, `subprocess.run`
only SIGKILLs the *direct* child (e.g. the `go` binary, `cargo`, `pytest`,
`./gradlew`). Runners like `go test` and `cargo test` compile and exec a
*separate* test binary as a grandchild process; killing the direct child
leaves that grandchild running, now orphaned and reparented, completely
unaffected by the "timeout." A model-generated solution with a genuine
infinite/exponential-blowup loop (observed firsthand: a `two-bucket` solution
left a compiled `twobucket.test` binary pegging a CPU core for well over an
hour after benchmark.py had already logged the try as timed out and moved on)
then runs forever until someone notices and kills it by hand -- on a
multi-run overnight box, enough of these stack up to threaten the whole
machine's load average.

Fix: run the test command in its own process group (`start_new_session=True`,
i.e. setsid) and, on timeout, SIGKILL the whole group via `os.killpg` instead
of just the direct child. Preserves the existing `subprocess.TimeoutExpired`
contract the caller already handles.

Usage: patch_benchmark_test_process_group.py <path-to-benchmark.py>
Idempotent: safe to run twice (skips if already patched).
"""
import sys

OLD_IMPORT = "import re\nimport shutil\nimport subprocess"
NEW_IMPORT = "import re\nimport shutil\nimport signal\nimport subprocess"

OLD_RUN = '''    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        cwd=testdir,
        encoding="utf-8",
        errors="replace",
    )

    success = result.returncode == 0
    res = result.stdout'''

NEW_RUN = '''    # See patches/patch_benchmark_test_process_group.py for why this isn't a
    # plain subprocess.run(timeout=...) call.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=testdir,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        test_stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        raise

    success = proc.returncode == 0
    res = test_stdout'''

PATCHES = [
    (OLD_IMPORT, NEW_IMPORT),
    (OLD_RUN, NEW_RUN),
]


def main():
    if len(sys.argv) != 2:
        print("usage: patch_benchmark_test_process_group.py <path-to-benchmark.py>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    if "start_new_session=True" in content:
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
    print(f"{path}: patched run_unit_tests() to kill the whole test process group on timeout")


if __name__ == "__main__":
    main()
