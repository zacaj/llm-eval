# MoE cache must be disabled on this machine for MoE models that fit dense

Status: resolved, working config confirmed. Found 2026-09-27 while testing the new
`Qwen3.6-35B-A3B-Q8_0-unsoth.gguf` build.

## Symptom

A 35B MoE Q8 model (35.2 GiB file, 16 GB VRAM card, 31 GiB system RAM) loaded fine under
`--fit-ctx 32768` (auto), which picked the fork's "MoE cache" placement:
`MoE cache fit selected main-device dense placement with 8015 MiB projected cache capacity
for 32640 MiB of routed expert weights (up to 24.6% coverage)`. Decode was **0.06-0.2 tok/s**
-- effectively unusable, worse than a 27B dense build that doesn't fit in VRAM at all.
`free -h` showed no unusual memory pressure or eviction; the slowness was per-token, not a
one-time stall.

## Root cause

The MoE cache is a small, fixed-size, *rotating* GPU-side cache (sized from free VRAM) that
holds a fraction of the routed expert weight at a time (here, 24.6%). Cache misses trigger a
host-side `memcpy()` straight out of the mmap'd GGUF file (`ggml-cuda/moe-cache.cu`,
`job.source = host_base + expert*expert_size`) into a staging buffer, then an async
copy to the GPU slot -- i.e. **every** expert, hit or miss, ultimately computes on GPU; there
is no CPU-compute fallback. Because MoE routing picks a near-uniform-random subset of experts
per token, a cache covering only 24.6% of the weight sees a very high miss rate, and every
miss that isn't in RAM page cache pays a real disk seek. The *entire* CPU-side working set
(34.6 GiB here) needs to be page-cache-resident to avoid this, since which slice is "hot"
changes every token -- there's no static partition. With only ~26-28 GiB free RAM, a
double-digit-percent miss rate against page cache was enough to make nearly every token a
disk read.

This is a bad tradeoff specifically when the *total* VRAM+RAM budget could hold the model
dense (no cache needed at all) -- the fork's `--fit-ctx` auto-fit chose the cache path anyway
instead of finding the classic dense-split option it had already computed and discarded
internally (visible in its own log: `n_layer=41, n_part=27, mem=13910 MiB`, a full-GPU-layers
candidate that was never selected).

Moving the file to a different disk (`/mnt/2508` -> `/home`) appeared to fix it (0.20 -> 2.38
tok/s) but this was very likely a page-cache-warming artifact from the `cp` itself, not a real
disk-speed difference -- confirmed by re-testing the *same* `/mnt/2508` copy cold (page cache
verified near-empty via `free -h` beforehand) with the cache disabled instead (see below) and
getting a *better* result than the `/home` number, on the "slower" disk.

## Fix

Add `--moe-cache off` to the server flags. This makes `--fit-ctx`'s auto-fit fall back to its
classic dense/CPU-split search instead of the cache-aware path. Two more flags matter once you
do this:

- **`--no-mmap`** (deprecated alias for `--load-mode <mode>`, but still works): when
  `--moe-cache off` leaves any layer's experts CPU-resident (via the same per-layer
  `tensor_buft_overrides` mechanism as `--n-cpu-moe`), llama.cpp warns `tensor overrides to
  CPU are used with mmap enabled -- consider using --no-mmap for better performance`. With
  mmap, that CPU tensor is lazily page-faulted on first touch during inference (one slow
  cold-prefill stall, ~0.3 tok/s for the first request). `--no-mmap` forces a bulk sequential
  read of everything (including CPU-pinned tensors) during the load phase instead, so
  inference never pays a page-fault cost.
- **MTP** (`--spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-ngl 99
  --spec-draft-n-cpu-moe 0 --spec-type draft-mtp --spec-draft-n-max 2`): with MTP off, this
  model's `blk.40.nextn.*` tensors are dead weight (`model has unused tensor ... -- ignoring`)
  and the fit search leaves 1/42 layers CPU-resident (41/42 offloaded). With MTP on, the nextn
  layer is needed, and fit's dense-split search found room for **42/42 layers fully on GPU**
  (11.85 GiB) -- zero CPU-resident layers, so no cold-page stall at all, plus the speculative
  decoding speedup itself.

## Confirmed numbers (`Qwen3.6-35B-A3B-Q8_0-unsoth.gguf`, `/mnt/2508/Downloads 2`, cold page cache)

| Config | Layers on GPU | Steady decode |
|---|---|---|
| `--fit-ctx 32768` (auto, MoE-cache mode) | 41/42 (cache, 24.6% coverage) | 0.06-0.2 tok/s |
| `--fit-ctx 32768 --moe-cache off` (no MTP, mmap) | 41/42 (1 CPU layer) | ~3 tok/s (after a one-time ~68s cold-page stall) |
| `--fit-ctx 32768 --moe-cache off --no-mmap` + MTP flags | **42/42** (fully dense) | **~12.6 tok/s** (eval 14.19 tok/s), MTP draft accept 63% |

Working full command (server):
```
--server --host 0.0.0.0 --port 8080 --jinja -lv 4 --log-timestamps --log-colors on
--cache-prompt --no-mmap
--checkpoint-min-step 4000 --ctx-checkpoints 32
-m <path>/Qwen3.6-35B-A3B-Q8_0-unsoth.gguf --alias "Qwen3.6-35B-A3B-Q8_0-unsloth"
--fit-ctx 32768 --moe-cache off
--spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-ngl 99
--spec-draft-n-cpu-moe 0 --spec-type draft-mtp --spec-draft-n-max 2
--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.01
-b 2048 -ub 2048 --parallel 1 --flash-attn on -ctk q8_0 -ctv q8_0
```
(`-ub` raised from 512 to 2048 after the ubatch sweep below -- ~3x faster prefill, no downside.)

## Aider harness timeout settings for this speed

Measured ~12-14 tok/s decode. `run.sh` defaults assume `DECODE_TPS=25`. Use:
```
DECODE_TPS=10 MAX_DUR_SCALE=2.5 MODEL=<alias> ./run.sh fromfile results/27b-diff-list.txt
```
(`DECODE_TPS=10` keeps ~30% margin below measured eval speed for `REQUEST_TIMEOUT`;
`MAX_DUR_SCALE=2.5` = `25/10`, matching the same factor used for other slow builds this
session, to scale up `27b-diff-list.txt`'s per-exercise `max_dur` values.)

## `spec-draft-n-max` tuning (tested 2026-09-27)

Real eval traffic showed draft acceptance climbing to 80-86% (well above the single-prompt
smoke test's 63%), with mean accepted length 2.6-2.73 against `n_max=2`'s ceiling -- looked
like a candidate for raising `n_max`. Tested `n_max=3` head-to-head on the same exercise
(`go simple-linked-list`, same try/outcome both times):

| | `n_max=2` | `n_max=3` |
|---|---|---|
| seconds_per_case | 688.6 | 696.3 |
| draft acceptance | 80-86% | 70-74% |
| mean accepted len | 2.6-2.73 | 3.1-3.23 |

Net: a wash to slightly worse. Longer drafts mean len rose but acceptance rate dropped enough
to eat the gain (more verification cost per round for a lower hit rate). **Keep `n_max=2` for
this model.**

## `-ub` (ubatch) tuning (tested 2026-09-27)

The 27b-diff-list run above bailed on 3/8 exercises hitting the (scaled) duration cap, with
word-search's failure driven by prompt reprocessing (28.8k prompt tokens across 9 tries) more
than generation speed. Swept `-ub` (with `-b 2048` fixed) against a fresh ~16k-token prompt
each time (unique nonce prefix, to defeat `--cache-prompt`), reloading the server cold for each
value:

| `-ub` | prefill speed |
|---|---|
| 256 | 88 tok/s |
| 512 (prior config) | 160 tok/s |
| 1024 | 289 tok/s |
| **2048 (= `-b`, the ceiling)** | **~500-514 tok/s** |

Prefill scales almost linearly with `-ub` for this model -- a **~3x** win going from 512 to
2048. VRAM headroom at `-ub 2048`: 1171 MiB free (`nvidia-smi`), comfortably clear of the
91 MiB collapse point from [[turboquant-fit-vs-manual-moe-offload]]'s cliff-not-slope finding.
Decode speed unaffected (300 tokens in 19.65s = 15.3 tok/s, draft acceptance ~58% on this
single sample, consistent with the earlier 12.6-15 tok/s range). **Adopt `-ub 2048` (matching
`-b`) as the default for this model** -- no downside found, only measured at this one context
length/prompt shape so re-check headroom if `--ctx-size`/`-b` change.

## Takeaway for future models on this machine

**Always pass `--moe-cache off` for MoE models here**, at least until fit's cache-vs-dense
choice is fixed/tuned upstream -- the auto-fit's cache path has been pathological every time
it's engaged so far on this hardware (16 GB VRAM, 31 GiB RAM). If a model is too big to go
fully dense even with cache off, that's a legitimate case for the cache -- but check the
`n_part=N, mem=M` candidates fit already logs internally before assuming cache is required.
