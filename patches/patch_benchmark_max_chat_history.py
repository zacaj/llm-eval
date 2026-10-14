#!/usr/bin/env python3
"""Patch aider's benchmark/benchmark.py to raise/disable chat-history summarization.

Observed need: aider's ChatSummary feature (aider/history.py) defaults
main_model.max_chat_history_tokens to 1024 (or up to 8192 if the model's
context window is known -- ours isn't, since these are local openai/<alias>
models litellm has no metadata for). Once a run's accumulated history crosses
that threshold, Coder.summarize_start() fires a *background-threaded* extra
completion call (a totally different system prompt, the whole prior
transcript pasted as its user content) on essentially every turn.

Against a real multi-tenant API that's harmless. Against our --parallel 1
llama-server it's not: that thread's request and the next real turn's request
race for the single KV-cache slot, each evicting the other's cached prefix.
Traced via server-context.cpp's "erased invalidated context checkpoint" log
lines back to exactly this -- see data/logs/prompts/ for captured examples of
the interleaved summarizer requests. Net effect: most turns fully reprocess
their prompt from scratch instead of incrementally extending the cache,
which is a large chunk of why these runs are slow.

Adds an opt-in override (AIDER_BENCHMARK_MAX_CHAT_HISTORY_TOKENS env var,
tokens; unset or 0 preserves aider's existing default/behavior). Set it above
your exercises' realistic transcript size (e.g. close to --fit-ctx) to
suppress spurious summarization during these short, bounded --tries runs.

Usage: patch_benchmark_max_chat_history.py <path-to-benchmark.py>
Idempotent: safe to run twice (skips if already patched).
"""
import sys

OLD = '''    dump(main_model.max_chat_history_tokens)

    if num_ctx:'''

NEW = '''    dump(main_model.max_chat_history_tokens)

    # See patches/patch_benchmark_max_chat_history.py for why this exists.
    _max_chat_history_tokens = int(
        os.environ.get("AIDER_BENCHMARK_MAX_CHAT_HISTORY_TOKENS", "0") or 0
    )
    if _max_chat_history_tokens:
        main_model.max_chat_history_tokens = _max_chat_history_tokens

    if num_ctx:'''


def main():
    if len(sys.argv) != 2:
        print("usage: patch_benchmark_max_chat_history.py <path-to-benchmark.py>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    if "_max_chat_history_tokens" in content:
        print(f"{path}: already patched, skipping")
        return

    if OLD not in content:
        print(f"ERROR: {path} does not contain the expected block -- "
              "upstream source has likely changed, patch needs updating:\n"
              f"{OLD[:80]}...", file=sys.stderr)
        sys.exit(1)

    content = content.replace(OLD, NEW)

    with open(path, "w") as f:
        f.write(content)
    print(f"{path}: patched main() with an AIDER_BENCHMARK_MAX_CHAT_HISTORY_TOKENS override")


if __name__ == "__main__":
    main()
