# Runaway generation on MoE 35B IQ3 builds: correlates with MTP off

Status: probably NOT a llama.cpp bug (see correction below); not fully closed. Found 2026-09-26 while running the 27B-vs-35B aider comparison
(`27b-diff-list.txt`). Everything below is from aider benchmark runs on one build, so treat it as
a lead, not a diagnosis.

## CORRECTION (added 2026-09-27)

All runs below were **greedy** (aider sends `temperature: 0`, overriding `--temp`), not temp 0.6 as
stated in the flag lists and caveats. Verified: Darwin IQ3 alphametics try 1 was byte-identical
(~13.6k tokens) across 3 separate no-MTP runs, and the two MTP runs matched each other but diverged
from no-MTP on the reply's first word (MTP verifies in a multi-token batch, different kernels, a
near-tie argmax flips). So "MTP on vs off" behaves like a different random seed on a greedy replay,
not a systematic decode bug: each config is effectively n=1 per prompt. See memory note
`aider-evals-run-greedy-temp0`. The "MTP-off loops" pattern above is therefore weak evidence; a real
test needs several seeds per config (run.sh now supports `TEMPERATURE` / `SEED`).

## Symptom

On some prompts the model never finishes its reply. It writes repetitive code until it hits the
request's `max_tokens` (16000), aider prints `Output tokens: ~13,6xx of 0 -- possibly exceeded
output limit!`, records the try as an error output, and the next try does the same. The whole
exercise then fails with every try used.

- Aider's counter for this is `num_exhausted_context_windows` (also `num_error_outputs`). The name
  is misleading: it is the reply hitting the output cap, not the context filling. Worst
  prompt+reply seen was ~22.5k tokens against 32k and 95k contexts.
- `completion_tokens` is recorded as 0 for these runs (aider logs usage after a clean finish).
- Example log: alphametics history is 11,421 lines; one line (`return null;`) repeats ~270 times.
- Not a duration timeout: `bailed_on_max_duration` false, `test_timeouts` 0 on all of these.

## Model and reproduction

- Model: `Darwin-35B-A3B-Opus-unsloth-UD-IQ3_XXS-MTP-zacaj.gguf` (MoE, 12.7 GiB on GPU, has MTP head)
  at `/mnt/2508/Backup 2/` (`/models2/` in the container). GPU: 16 GB, ~15 GB free.
- Exercise that reproduces it most reliably: javascript `alphametics`
  (aider polyglot, whole edit format, reasoning low, temp 0.6 / top-p 0.95 / top-k 20 / min-p 0.01,
  5 tries, 1000s cap before MAX_DUR_SCALE=2.5).
- Run: `cd /mnt/llm/evals/aider && MODEL=<alias> MAX_DUR_SCALE=2.5 ./run.sh fromfile results/alpha-only.txt --clean`
  with `alpha-only.txt` containing `javascript alphametics 5 1000`.
- Server common flags: `--jinja --checkpoint-min-step 4000 --ctx-checkpoints 32 --cache-prompt -b 2048 -ub 512
  --parallel 1 --flash-attn on -ctk q8_0 -ctv q8_0 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.01`.
- MTP flags when on: `--spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-ngl 99
  --spec-draft-n-cpu-moe 0 --spec-type draft-mtp --spec-draft-n-max 2`.

## Evidence (Darwin IQ3, alphametics)

All configs have the whole model on the GPU (fit log for the auto-fit case: `offloaded 42/42 layers
to GPU`, `MoE cache disabled`; its only change was context 262144 -> 94976).

| Config | Context | MTP | Chat template | Result |
|---|---|---|---|---|
| auto-fit (`--fit-ctx 32768`), first run | 94,976 | off | none | FAIL, 5/5 tries runaway, 1447s |
| auto-fit, rerun | 94,976 | off | none | FAIL, 5/5 tries runaway, 1497s |
| `-ngl 99 --n-cpu-moe 0 --ctx-size 32768` | 32,768 | **off** | none | FAIL, 5/5 tries runaway, 456s |
| same, MTP on | 32,768 | on | file | pass on try 3, 51s |
| same, MTP on | 32,768 | on | none | pass on try 3, 51s |
| `--ctx-size 94976`, MTP on | 94,976 | on | none | pass on try 3, 64s |

Same pattern on the full 8-exercise list (`27b-diff-list.txt`): auto-fit/no-MTP 2/8 in 2479s,
MTP on 5/8 in 592-692s. Result dirs: `results/fromfile-27b-diff-list-openai-Darwin-35B-A3B-Opus-unsloth-UD-IQ3_XXS-MTP-zacaj{,-vram-mtp,-vram-mtp-notpl,-moe-rerun}`,
`results/fromfile-alpha-only-...{-vram-nomtp,-vram-mtp-95k}`. Per-exercise rows are in
`eval-results-per-test.tsv` (model name carries the suffix).

## Ruled out

- Chat template: identical results with and without `/models/chat_template.jinja` (MTP on).
- Context size: 32k and 95k behave the same for a given MTP setting.
- GPU placement: all layers and experts on GPU in every config.
- Duration caps and test timeouts (see above).

## Caveats (why this is not yet a diagnosis)

- Small n: one or two runs per config, one exercise for the clean A/B.
- MTP on is not immune. With MTP on, Darwin IQ3 word-search hit the output cap 3 times (with the
  template) and Ornith-1.0-35B IQ3 had alphametics 4 and sgf-parsing 4 runaway tries (`-vram-mtp`
  runs). So MTP-on lowers the rate here, it does not remove it.
- Without MTP, other exercises did not always loop: scale-generator passed on the auto-fit rerun.
- MTP verification should be lossless in distribution, so a difference is unexpected. It could be a
  genuine llama.cpp/fork bug in the non-MTP path, a sampling/state difference between paths, or
  chance that only looks systematic. Not yet distinguished.
- 27B dense builds were run with MTP (except plan-DASLab-imatrix, no MTP), so they show no such
  effect; no evidence yet either way for dense models without MTP.

## Next steps (not done)

1. Reproduce directly against the server: same alphametics prompt, temp 0.6, MTP on vs off, many
   samples, count how often the reply reaches 16000 tokens. Then try temp 0 / fixed seed.
2. Check whether the looping reply repeats a token pattern the moment it starts (log the first
   divergent tokens) and whether `--spec-type draft-mtp` changes sampler/RNG state.
3. Try the same prompt on Ornith and a Qwen3.6-35B build with MTP off/on to see if it is
   model-specific.
4. Consider rerunning the 35B tables with MTP on. Affected results (35B, MTP off): Darwin Q4_K_XL,
   Claude-4.7 Q4_K_XL, UD-IQ4_XS Base, and the first-pass Darwin IQ3 and Ornith rows.
5. If it holds up, capture the exact llama.cpp commit of the running image and file it upstream.

Related mislabel to keep in mind: `ctx_exh` / `num_exhausted_context_windows` in the results JSON
means "hit the output cap", not "ran out of context".
