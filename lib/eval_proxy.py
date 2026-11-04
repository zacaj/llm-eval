#!/usr/bin/env python3
"""Eval-side proxy between the Harbor agents and the model server's own proxy.

    python3 eval_proxy.py --port 0 --upstream http://nas:3002 --log jobs/<job>/eval-proxy.tsv

harbor.sh starts one per job (prints "listening <port>" on stdout once ready) and
points the agents at it. It keeps everything eval-specific out of the NAS-side
prompt-logger proxy, which stays about the engine (filters, reasoning cap, routes):

  * Per-trial output-token budget. The agent sends "X-Eval-Budget: N"; requests are
    keyed by session (X-Eval-Session header -- terminus-2 -- else Claude Code's
    metadata.user_id). max_tokens is clamped to what's left, and once fewer than
    --min-tokens remain the request is rejected with "eval token budget exhausted"
    (Claude Code exits on it; harbor_ext.BudgetedTerminus2 maps it to a graded stop).
    Upstream gets "X-Eval-Budget: 0": still eval mode there (no routes) but unlimited.
  * One TSV row per request with what the server says it actually did, from
    strata_effective (the final message_delta when streaming): effort, thinking
    budget, reasoning/answer tokens, stop reason. collect.py joins the rows to trials.

Responses are streamed through as they arrive (never buffered); only a small tail of
each SSE line is parsed. stdlib only.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

BUDGET_MSG = "eval token budget exhausted"
INFERENCE_PATHS = {"/v1/messages", "/v1/chat/completions", "/v1/completions", "/v1/responses"}
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer",
       "upgrade", "host", "content-length", "accept-encoding"}
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
COLUMNS = ["timestamp", "session", "path", "budget", "max_tokens_sent", "output_tokens", "total_output",
           "stop_reason", "effort", "reasoning_budget", "reasoning_tokens", "answer_tokens",
           "budget_cutoff", "duration_ms", "tok_s", "note"]

args = None
spent: dict[str, int] = {}
lock = threading.Lock()


def log_row(**kw):
    with lock, open(args.log, "a", encoding="utf-8") as f:
        f.write("\t".join(str(kw.get(c, "")) for c in COLUMNS) + "\n")


def session_key(headers, data) -> str:
    if headers.get("X-Eval-Session"):
        return headers["X-Eval-Session"]
    uid = str(((data or {}).get("metadata") or {}).get("user_id") or "")
    # Claude Code's user_id embeds the session uuid (= its session file name)
    m = UUID_RE.findall(uid)
    return m[-1] if m else uid


class Stats:
    """Pulls usage / strata_effective out of a (streamed or whole) response."""

    def __init__(self):
        self.out = 0
        self.stop = ""
        self.eff: dict = {}
        self.buf = b""

    def feed_json(self, d):
        if not isinstance(d, dict):
            return
        for u in (d.get("usage"), (d.get("message") or {}).get("usage") if isinstance(d.get("message"), dict) else None):
            if isinstance(u, dict):
                self.out = max(self.out, u.get("output_tokens") or u.get("completion_tokens") or 0)
        delta = d.get("delta") if isinstance(d.get("delta"), dict) else {}
        self.stop = delta.get("stop_reason") or d.get("stop_reason") or self.stop
        for ch in d.get("choices") or []:
            if isinstance(ch, dict) and ch.get("finish_reason"):
                self.stop = ch["finish_reason"]
        if isinstance(d.get("strata_effective"), dict):
            self.eff = d["strata_effective"]

    def feed_sse(self, chunk: bytes):
        self.buf += chunk
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            line = line.strip()
            # only the events that can carry usage / strata_effective, not every token delta
            if line.startswith(b"data:") and (b"usage" in line or b"strata_effective" in line
                                               or b"finish_reason\": \"" in line or b"stop_reason" in line):
                try:
                    self.feed_json(json.loads(line[5:]))
                except ValueError:
                    pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.forward(None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.forward(self.rfile.read(n) if n else b"")

    def send_json(self, code, payload):
        out = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(out)

    def forward(self, body):
        path = self.path.split("?", 1)[0]
        data, key, budget, sent_max = None, "", 0, ""
        inference = body and path in INFERENCE_PATHS
        if inference:
            try:
                data = json.loads(body)
            except ValueError:
                data = None
            key = session_key(self.headers, data)
            if args.think_budget > 0 and path == "/v1/messages" and isinstance(data, dict):
                # Claude Code sends thinking "adaptive", which Strata caps at its default 4000;
                # an explicit budget is honored, so set it here (EFFORT picks the size in harbor.sh)
                data["thinking"] = {"type": "enabled", "budget_tokens": args.think_budget}
                # the Strata prompt-logger's cap_reasoning_budget filter only treats the flat
                # field as explicit; without it the filter's default (8000) overrides the above
                data["reasoning_budget_tokens"] = args.think_budget
                data["max_tokens"] = args.think_budget + args.answer_tokens
                body = json.dumps(data).encode()
            elif args.think_budget < 0 and path == "/v1/messages" and isinstance(data, dict):
                # EFFORT=off (harbor.sh passes -1): reasoning disabled, max_tokens left as sent
                data["thinking"] = {"type": "disabled"}
                body = json.dumps(data).encode()
            try:
                budget = int(self.headers.get("X-Eval-Budget") or 0)
            except ValueError:
                budget = 0
            if budget > 0 and key and isinstance(data, dict):
                with lock:
                    remaining = budget - spent.get(key, 0)
                if remaining < min(args.min_tokens, budget):
                    log_row(timestamp=now(), session=key, path=path, budget=budget,
                            total_output=budget - remaining, note="rejected")
                    msg = {"type": "error", "error": {"type": "invalid_request_error", "message": BUDGET_MSG}} \
                        if path == "/v1/messages" else \
                        {"error": {"message": BUDGET_MSG, "type": "invalid_request_error", "code": "budget_exhausted"}}
                    return self.send_json(400, msg)
                field = "max_completion_tokens" if "max_completion_tokens" in data else "max_tokens"
                cur = data.get(field)
                data[field] = min(cur, remaining) if isinstance(cur, int) else remaining
                body = json.dumps(data).encode()
            if isinstance(data, dict) and args.model and data.get("model") != args.model:
                # the agent is configured with a descriptive name (e.g. the GGUF); the
                # server wants its own id
                data["model"] = args.model
                body = json.dumps(data).encode()
            if isinstance(data, dict):
                sent_max = data.get("max_completion_tokens") or data.get("max_tokens") or ""

        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if inference and "X-Eval-Budget" in self.headers:
            # upstream: still eval mode (no routes) but no second budget; callers that
            # didn't ask for eval mode (run.sh's aider) reach it unchanged
            hdrs["X-Eval-Budget"] = "0"
        up = urlsplit(args.upstream)
        conn = (http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection)(
            up.hostname, up.port or (443 if up.scheme == "https" else 80), timeout=args.timeout)
        stats = Stats()
        t0 = time.monotonic()
        try:
            conn.request(self.command, (up.path.rstrip("/") + self.path), body=body, headers=hdrs)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            sse = "text/event-stream" in (resp.getheader("Content-Type") or "")
            whole = b""
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                if inference:
                    if sse:
                        stats.feed_sse(chunk)
                    elif len(whole) < 8 << 20:
                        whole += chunk
            self.wfile.write(b"0\r\n\r\n")
            if inference and not sse:
                try:
                    stats.feed_json(json.loads(whole))
                except ValueError:
                    pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as e:
            if inference:
                log_row(timestamp=now(), session=key, path=path, budget=budget, note=f"upstream error: {e}")
            try:
                self.send_json(502, {"error": {"message": f"eval proxy: upstream error: {e}"}})
            except OSError:
                pass
            return
        finally:
            conn.close()
        if inference:
            with lock:
                spent[key] = total = spent.get(key, 0) + stats.out
            e = stats.eff
            rb = e.get("reasoning_budget_requested")
            rt = e.get("reasoning_tokens")
            dur = time.monotonic() - t0
            log_row(timestamp=now(), session=key, path=path, budget=budget, max_tokens_sent=sent_max,
                    output_tokens=stats.out, total_output=total, stop_reason=stats.stop,
                    effort=e.get("reasoning_effort", ""), reasoning_budget=rb if rb is not None else "",
                    reasoning_tokens=rt if rt is not None else "",
                    answer_tokens=e.get("answer_tokens", ""),
                    # reasoning_tokens includes Strata's injected cut-off line, so >= budget = cut off
                    budget_cutoff=int(rt >= rb) if isinstance(rt, int) and isinstance(rb, int) and rb > 0 else "",
                    # wall time incl. prompt processing; tok_s = output / wall time, so a
                    # server-side slowdown (GPU contention etc.) shows up per request
                    duration_ms=round(dur * 1000), tok_s=f"{stats.out / dur:.1f}" if dur > 0 and stats.out else "")


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--min-tokens", type=int, default=256)
    ap.add_argument("--model", default="", help="rewrite each request's model field to this server id")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--think-budget", type=int, default=0,
                    help="set thinking to enabled/budget_tokens=N on /v1/messages (0 = leave the request alone, "
                         "-1 = thinking disabled)")
    ap.add_argument("--answer-tokens", type=int, default=8000,
                    help="with --think-budget: max_tokens = think budget + this (answer/tool-call headroom)")
    args = ap.parse_args()
    header = "\t".join(COLUMNS)
    try:
        with open(args.log, encoding="utf-8") as f:
            old = f.readline().rstrip("\n")
    except OSError:
        old = header
    if old and old != header:
        # columns changed: set the old log aside rather than append misaligned rows
        os.replace(args.log, f"{args.log}.{datetime.now():%Y%m%d-%H%M%S}.old")
    with open(args.log, "a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write(header + "\n")
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    srv.daemon_threads = True
    print(f"listening {srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
