# Model-server settings capture, shared by run.sh and harbor/harbor.sh (sourced).
# Moved verbatim out of run.sh. Callers set before use:
#   LLAMA_SERVER_URL  server root (:8080, not the proxy)
#   LLAMA_DOCKER      docker CLI prefix for the server host (llama), e.g. (docker -H ssh://host)
#   MODEL TEMPERATURE SEED EXTRA_BODY   what a real request would send (probe_strata_effective)
#   TARGET LLAMA_HEALTH_URL             strata|llama-nas and llama's /health URL (measure_decode_tps)

LLAMA_CONTAINER="${LLAMA_CONTAINER:-}"

detect_llama_container() {
  if [[ -n "$LLAMA_CONTAINER" ]]; then
    echo "$LLAMA_CONTAINER"
    return
  fi
  local pattern found
  for pattern in "llama-cpp-turboquant-llama-cpp-1" "llama-server" "llama-cpp" "llama"; do
    found="$("${LLAMA_DOCKER[@]}" ps --filter "name=$pattern" --filter "status=running" --format '{{.Names}}' | head -1)"
    if [[ -n "$found" ]]; then
      echo "$found"
      return
    fi
  done
}

# llama-server flags that are purely about *how fast*/*where* inference runs
# (threading, GPU offload split, batching, networking, logging, caching infra,
# speculative-decoding draft config) rather than *how the model behaves*
# (sampling params, context/RoPE handling, KV-cache quant, chat templating).
# Blacklisted so perf-tuning noise doesn't pollute behavior comparisons.
SRV_BLACKLIST="--server --host --port -lv --log-timestamps --log-colors --verbosity \
--threads --threads-batch -t -tb --n-gpu-layers -ngl --n-cpu-moe --tensor-split -ts \
--main-gpu -mg --split-mode -sm --batch-size -b --ubatch-size -ub --parallel -np \
--cont-batching -cb --no-cont-batching --mlock --no-mmap --numa --flash-attn -fa \
--fit --fit-ctx --cache-prompt --checkpoint-every-n-tokens --slot-save-path \
--ctx-checkpoints --metrics --api-key --ssl-key-file --ssl-cert-file --timeout \
--no-webui --draft-max --draft-min --draft-p-min --gpu-layers-draft --model-draft \
--jinja --checkpoint-min-step --cache-ram --cache-idle-slots --kv-unified \
--spec-draft-ngl --spec-draft-n-cpu-moe \
--device --device-draft --spec-draft-type-k --spec-draft-type-v --defrag-thold"



# Strata has no container to `docker inspect`, so its settings come from three
# endpoints: /props (model identity), /metrics .engine (KV-cache type etc.) and
# /config (the run config's sampling defaults). Keys reuse llama's flag names
# where they mean the same thing so rows stay comparable/filterable. Only
# behavior-affecting engine fields are kept (perf-only ones -- expert cache,
# arena, workers, PCIe, VRAM -- are dropped, same as SRV_BLACKLIST does for
# llama). Sampling: an absent config value means the server falls back to the
# engine default, which is greedy (serve/server.py), so that is recorded as
# --temp=0 rather than left blank. Best-effort: every fetch is a silent no-op
# if the server doesn't answer.
capture_strata_settings() {
  local props metrics cfg
  props="$(curl -sf --max-time 10 "$LLAMA_SERVER_URL/props" 2>/dev/null)" || props="{}"
  metrics="$(curl -sf --max-time 10 "$LLAMA_SERVER_URL/metrics" 2>/dev/null)" || metrics="{}"
  cfg="$(curl -sf --max-time 10 "$LLAMA_SERVER_URL/config" 2>/dev/null)" || cfg="{}"
  python3 -c '
import json, sys
def load(s):
    try: return json.loads(s)
    except Exception: return {}
props, metrics, cfg = (load(a) for a in sys.argv[1:4])
out = {}
path = props.get("model_path", "")
if path: out["srv.-m"] = path.replace("\\", "/").rsplit("/", 1)[-1]
if props.get("model_alias"): out["srv.--alias"] = props["model_alias"]
if props.get("build_info"): out["srv.build_info"] = props["build_info"]
eng = metrics.get("engine", {})
for src, dst in (("context", "srv.--ctx-size"), ("kv", "srv.kv_cache"),
                 ("logit_bias", "srv.logit_bias"), ("spec_min_p", "srv.spec_min_p")):
    if eng.get(src) is not None: out[dst] = eng[src]
vals = {k["key"]: k.get("value") for k in cfg.get("keys", [])}
for src, dst in (("sampling.temperature", "srv.--temp"), ("sampling.top_p", "srv.--top-p"),
                 ("sampling.top_k", "srv.--top-k"), ("sampling.min_p", "srv.--min-p")):
    if vals.get(src) is not None: out[dst] = vals[src]
if "srv.--temp" not in out and cfg: out["srv.--temp"] = 0   # engine default is greedy
for k, v in out.items(): print(f"{k}={v}")
' "$props" "$metrics" "$cfg" 2>/dev/null || true
}

# EXTRA_BODY sampler overrides beat the server defaults, so the recorded
# srv.--top-p/--top-k/--min-p (and the skip-existing comparison, which reads the
# same values) must reflect what was sent, not /config's default. Without this a
# min_p=0.1 run records min-p=0.05 and SKIP_EXISTING treats it as a duplicate of
# the baseline. Filters srv.* "key=value" lines on stdin; no-op without EXTRA_BODY.
apply_extra_body_overrides() {
  if [[ -z "$EXTRA_BODY" ]]; then cat; return; fi
  python3 -I -c '
import json, sys
want = json.loads(sys.argv[1])
m = {"top_p": "srv.--top-p", "top_k": "srv.--top-k", "min_p": "srv.--min-p"}
over = {m[k]: v for k, v in want.items() if k in m}
for line in sys.stdin:
    k = line.split("=", 1)[0]
    if k not in over: sys.stdout.write(line)
for k, v in over.items(): print(f"{k}={v}")
' "$EXTRA_BODY"
}

# Strata only: one 1-token request carrying the same sampling fields a real
# eval request would (TEMPERATURE unless "server", SEED, EXTRA_BODY), then read
# back strata_effective.engine (falling back to .sampling on older servers) -- the
# values the sampler actually uses (defaults filled in, overrides applied, out-of-range
# values clamped). Emits srv.eff.{temp,top_p,top_k,min_p,seed_from}
# lines so the TSVs record what was *in effect*, not just what was sent. The
# response also echoes every request field (typos included), so only those four
# resolved keys are trusted. If EXTRA_BODY asked for one of them and the server
# reports a different value, warn loudly: that knob was ignored or clamped.
# Best-effort: silent no-op (no lines) if the server doesn't answer or doesn't
# report strata_effective.
probe_strata_effective() {
  local resp
  resp="$(python3 -I -c '
import json, sys, urllib.request
base, model, temp, seed, extra = sys.argv[1:6]
body = {"model": model, "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 1, "seed": int(seed),
        "chat_template_kwargs": {"enable_thinking": False}}
if temp != "server": body["temperature"] = float(temp)
if extra: body.update(json.loads(extra))
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req, timeout=120))))
' "$LLAMA_SERVER_URL" "$MODEL" "$TEMPERATURE" "$SEED" "$EXTRA_BODY" 2>/dev/null)" || return 0
  python3 -I -c '
import json, sys
d = json.loads(sys.argv[1])
eff = d.get("strata_effective") or {}
# engine = what the sampler really uses (clamped/validated, unknown keys dropped);
# sampling is only the request echo + defaults. Older servers have no engine section.
samp = eff.get("engine") or eff.get("sampling") or {}
sf = (eff.get("engine") or {}).get("seed_from")
if sf: print("srv.eff.seed_from=" + str(sf))
want = json.loads(sys.argv[2]) if sys.argv[2] else {}
for src, dst in (("temperature", "temp"), ("top_p", "top_p"), ("top_k", "top_k"), ("min_p", "min_p")):
    if samp.get(src) is not None: print(f"srv.eff.{dst}={samp[src]}")
    if src in want and samp.get(src) != want[src]:
        print(f"WARNING: EXTRA_BODY {src}={want[src]} but server reports {samp.get(src)} -- knob ignored or clamped", file=sys.stderr)
' "$resp" "$EXTRA_BODY" || true
}

# Strata: /config defaults + the startup probe's resolved values. The probe's
# strata_effective is authoritative for top-p/top-k/min-p (it reflects request
# overrides on top of the defaults), so those replace the /config-derived
# srv.--top-p/--top-k/--min-p; the srv.eff.* lines are kept too. If the probe
# gave nothing (server down / no strata_effective), fall back to applying
# EXTRA_BODY's sampler fields directly. Used for both the recorded columns and
# the SKIP_EXISTING comparison, so the two always agree.
strata_settings_effective() {
  local base eff
  base="$(capture_strata_settings)"
  eff="$(probe_strata_effective)"
  if [[ -z "$eff" ]]; then
    printf '%s\n' "$base" | apply_extra_body_overrides
    return
  fi
  { printf '%s\n' "$base" | grep -v -E '^srv\.--(top-p|top-k|min-p)='
    sed -n -E 's/^srv\.eff\.top_p=/srv.--top-p=/p; s/^srv\.eff\.top_k=/srv.--top-k=/p; s/^srv\.eff\.min_p=/srv.--min-p=/p' <<< "$eff"
    printf '%s\n' "$eff"
  } | grep -v '^$'
}

# Extract non-blacklisted llama-server CLI args as key=value lines
# (prefixed srv.<flag>=<value>). Best-effort: silent no-op if no server
# container is running.
capture_srv_settings() {
  local container="$1" inspect_json
  [[ -z "$container" ]] && return
  inspect_json="$("${LLAMA_DOCKER[@]}" inspect "$container" 2>/dev/null)" || return
  [[ -z "$inspect_json" ]] && return
  echo "$inspect_json" | python3 -c "
import json, sys, re

blacklist = set('''$SRV_BLACKLIST'''.split())
data = json.load(sys.stdin)
args = data[0].get('Args', [])

def is_flag(s):
    return s.startswith('-') and not re.match(r'^-\d', s)

i = 0
while i < len(args):
    arg = args[i]
    if is_flag(arg):
        key = arg
        val = ''
        if i + 1 < len(args) and not is_flag(args[i + 1]):
            val = args[i + 1]
            i += 1
        if not val:
            val = 'yes'
        if key not in blacklist:
            print(f'srv.{key}={val}')
        i += 1
    else:
        i += 1
"
}

# Decode tok/s of the live server: two 200-token completions (the first pays
# one-time warmup), reading llama-server's own timings.predicted_per_second
# from the response -- unlike scraping docker logs this works for spec-decode
# builds too. Prints nothing if the server can't answer.
measure_decode_tps() {
  local i out resp tps=""
  # Strata: /metrics already holds decode speed from real requests (totals
  # since start, else the live mean of a long in-progress generation) -- free,
  # no probe load. Only trusted once it covers enough output: right after a
  # restart the totals are a few tiny requests (1-token capture probes) whose
  # per-request overhead reads as ~10 tok/s; then fall through to a real probe.
  if [[ "$TARGET" == "strata" ]]; then
    tps="$(curl -sf --max-time 10 "$LLAMA_SERVER_URL/metrics" 2>/dev/null | python3 -c '
import sys, json
d = json.load(sys.stdin)
t = d.get("totals", {})
live = d.get("live") or {}
if t.get("decode_ms") and (t.get("output_tokens") or 0) >= 2000:
    print(t["output_tokens"] / (t["decode_ms"] / 1000))
elif live.get("tok_s_mean") and (live.get("generated") or 0) >= 500:
    print(live["tok_s_mean"])
' 2>/dev/null)" || tps=""
    if [[ -n "$tps" ]]; then echo "$tps"; return 0; fi
    echo "-- /metrics has no decode stats yet (fresh server?); probing with a real completion" >&2
  fi
  # Probe: two 600-token completions (the first pays one-time warmup), reading
  # the server's own timings.predicted_per_second. Strata has no /completion,
  # so it gets the OpenAI chat endpoint (which also returns timings).
  local url body
  if [[ "$TARGET" == "strata" ]]; then
    url="$LLAMA_SERVER_URL/v1/chat/completions"
    body='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Write a detailed multi-section essay about the history of bridges."}],"max_tokens":600,"temperature":0.6,"stream":false}'
  else
    url="${LLAMA_HEALTH_URL%/health}/completion"
    body='{"prompt":"Write a detailed multi-section essay about the history of bridges.","n_predict":600,"temperature":0.6}'
  fi
  for i in 1 2; do
    resp="$(curl -sS --max-time 180 "$url" -H 'Content-Type: application/json' -d "$body" 2>&1)" || { echo "-- decode probe $i/2 against $url: curl failed: ${resp:0:300}" >&2; continue; }
    out="$(python3 -c 'import sys,json; print(json.load(sys.stdin)["timings"]["predicted_per_second"])' <<<"$resp" 2>/dev/null)" \
      || { echo "-- decode probe $i/2 against $url: no timings.predicted_per_second in response: ${resp:0:300}" >&2; out=""; }
    [[ "$out" =~ ^[0-9.]+$ ]] && tps="$out"
  done
  echo "$tps"
}

# Strata only: what thinking settings a real request ends up with, read from the
# server's own strata_effective (after any proxy filters and engine mapping -- e.g.
# effort "high" resolves to "xhigh"). Send it through the same URL/headers as the
# real requests. Args: base URL, Anthropic-dialect JSON fields to merge (what the
# caller sends, e.g. {"output_config":{"effort":"low"},"thinking":{"type":"adaptive"}}),
# then extra "Header: value" lines. Emits srv.eff.reasoning_{effort,budget,budget_source}.
# max_tokens 1 keeps it cheap; the budget is read from reasoning_budget_requested
# (before Strata shrinks it to fit max_tokens). Silent no-op if not reported.
probe_strata_reasoning() {
  local base="$1" fields="$2"; shift 2
  python3 -I -c '
import json, sys, urllib.request
base, fields = sys.argv[1], json.loads(sys.argv[2] or "{}")
body = {"model": "x", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
body.update(fields)
hdrs = {"Content-Type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "x"}
for h in sys.argv[3:]:
    k, _, v = h.partition(":"); hdrs[k.strip()] = v.strip()
req = urllib.request.Request(base + "/v1/messages", json.dumps(body).encode(), hdrs)
eff = json.load(urllib.request.urlopen(req, timeout=300)).get("strata_effective") or {}
for src, dst in (("reasoning_effort", "reasoning_effort"),
                 ("reasoning_budget_requested", "reasoning_budget"),
                 ("reasoning_budget_source", "reasoning_budget_source")):
    if eff.get(src) is not None: print(f"srv.eff.{dst}={eff[src]}")
' "$base" "$fields" "$@" 2>/dev/null || true
}
