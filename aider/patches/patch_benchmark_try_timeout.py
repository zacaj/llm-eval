#!/usr/bin/env python3
"""Patch aider's benchmark/benchmark.py so AIDER_BENCHMARK_MAX_DUR (see
patch_benchmark_max_duration.py) can actually preempt a hung try, not just
refuse to start the next one.

Observed need: a `coder.run()` call hit a real "Connection error / servers are
down or overloaded" from litellm partway through an exercise, then retried
with exponential backoff that kept doubling well past aider's own advertised
60s cap (`aider.models.RETRY_TIMEOUT`) -- up to 8192s-and-climbing waits,
observed firsthand keeping one exercise "in progress" for 10+ hours before
being noticed and killed by hand. AIDER_BENCHMARK_MAX_DUR was set to 3600 for
that run and did nothing, because run_test_real()'s existing check
(`if max_dur and dur > max_dur`) only runs *after* coder.run() returns --
whatever is actually happening inside that call (a stuck retry loop in
aider/litellm, a slow model, anything) is invisible to it until it's over.

Fix: wrap the coder.run() call in a hard, preemptive SIGALRM-based timeout
sized to the *remaining* max_dur budget (max_dur - dur so far), so it composes
with the existing cumulative-across-tries accounting instead of adding a
second independent cap. If the alarm fires mid-call, coder.run() is
interrupted right there via a raised exception, the try is marked bailed and
the tries loop stops, same end state the existing check already produces for
the between-tries case. signal.alarm/setitimer only work in the main thread,
so this only activates there (benchmark.py's parallel-exercise mode via `lox`
runs exercises in worker threads); single-threaded runs -- the common case
for these benchmarks -- get the full protection, others silently keep the old
cooperative-only behavior rather than crashing.

Usage: patch_benchmark_try_timeout.py <path-to-benchmark.py>
Idempotent: safe to run twice (skips if already patched).
"""
import sys

OLD_IMPORT = "import sys\nimport time\nimport traceback"
NEW_IMPORT = "import sys\nimport threading\nimport time\nimport traceback"

OLD_ANCHOR = 'BENCHMARK_DNAME = Path(os.environ.get("AIDER_BENCHMARK_DIR", "tmp.benchmarks"))'

NEW_ANCHOR = '''BENCHMARK_DNAME = Path(os.environ.get("AIDER_BENCHMARK_DIR", "tmp.benchmarks"))


# See patches/patch_benchmark_try_timeout.py for why this exists.
class TryTimedOut(Exception):
    pass


def _raise_try_timed_out(signum, frame):
    raise TryTimedOut()'''

OLD_RUN = '''        else:
            response = coder.run(with_message=instructions, preproc=False)

        dur += time.time() - start

        if max_dur and dur > max_dur:
            bailed_on_max_dur = True
            break'''

NEW_RUN = '''        else:
            # See patches/patch_benchmark_try_timeout.py for why this isn't a
            # bare coder.run() call.
            remaining = (max_dur - dur) if max_dur else 0
            if remaining > 0 and threading.current_thread() is threading.main_thread():
                old_handler = signal.signal(signal.SIGALRM, _raise_try_timed_out)
                signal.setitimer(signal.ITIMER_REAL, remaining)
                try:
                    response = coder.run(with_message=instructions, preproc=False)
                except TryTimedOut:
                    response = ""
                    bailed_on_max_dur = True
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                response = coder.run(with_message=instructions, preproc=False)

        dur += time.time() - start

        if bailed_on_max_dur or (max_dur and dur > max_dur):
            bailed_on_max_dur = True
            break'''

PATCHES = [
    (OLD_IMPORT, NEW_IMPORT),
    (OLD_ANCHOR, NEW_ANCHOR),
    (OLD_RUN, NEW_RUN),
]


def main():
    if len(sys.argv) != 2:
        print("usage: patch_benchmark_try_timeout.py <path-to-benchmark.py>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    if "TryTimedOut" in content:
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
    print(f"{path}: patched run_test_real() to hard-timeout a hung coder.run() call")


if __name__ == "__main__":
    main()
