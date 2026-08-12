# Aider Coding Benchmark

Evaluates local LLMs via the [aider polyglot benchmark](https://aider.chat/docs/leaderboards/) — 225 Exercism exercises across 6 languages (Python, Go, Rust, JS, Java, C++).

## Setup

```bash
# Build the Docker image (first time only, ~5-10 min)
./run.sh build

# Make sure llama.cpp is running on port 3002
# The model alias (--alias in llama.cpp) is used as the model name

# Smoke test (1 exercise, ~2 min)
./run.sh smoke

# Quick eval (5 python exercises, ~10 min)
./run.sh quick

# Specify a different model alias
./run.sh quick my-model-alias
```

## Commands

| Command | Exercises | Est. Time | Description |
|---------|-----------|-----------|-------------|
| `smoke` | 1 python | ~2 min | Verify setup works |
| `quick` | 5 python | ~10 min | Fast iteration |
| `python` | all python | ~30 min | Python-only eval |
| `polyglot` | 18 (3/lang) | ~30 min | Cross-language sample |
| `full` | 225 | ~2-4 hrs | Full benchmark |
| `diff` | 5 python | ~10 min | Tests diff edit format |

## What It Tests

Each exercise: model gets instructions + starter code, edits the file via aider's edit format, tests run, if tests fail model gets error output and tries again (2 attempts by default).

This is an **agent loop**, not raw completion — the model must understand edit formats, parse test errors, and self-correct.

## Results

Results saved to `./results/<timestamp>--<run-name>/`. Each exercise gets:
- `.aider.results.json` — pass/fail, timing, tokens, cost
- `.aider.chat.history.md` — full conversation log

### Metrics collected per exercise:
- **pass/fail** on attempt 1 and attempt 2
- **tokens sent/received**
- **cost** (if using a paid API)
- **time per exercise**
- **edit format errors** (malformed responses)
- **syntax/lint errors** in generated code
- **test output** on failure

### Aggregate stats:
- `pass_rate_1` — % passing on first attempt
- `pass_rate_2` — % passing after retry with error feedback
- Total cost, time, error breakdowns

## Comparing Runs

```bash
./run.sh shell
# Inside container:
./benchmark/benchmark.py --diffs /results/run1 /results/run2
```

## Leaderboard Comparison

Published results at https://aider.chat/docs/leaderboards/ include:
- GPT-4o, GPT-4.5, o1/o3/o4
- Claude Opus/Sonnet/Haiku (3.5, 4.x)
- Gemini 2.5 Pro/Flash
- DeepSeek V3/R1
- Llama 3.1/3.3 70B/405B
- Qwen 2.5 Coder 32B
- Mistral Large

Your local results are directly comparable to these.
