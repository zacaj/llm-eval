FROM buildpack-deps:jammy

# System deps + Python 3.11 + Java 21
RUN apt-get update && apt-get install -y \
    software-properties-common \
    cmake \
    libboost-all-dev \
    libtbb-dev \
    ca-certificates-java \
    openjdk-21-jdk \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Go 1.21
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then GOARCH="amd64"; \
    elif [ "$ARCH" = "aarch64" ]; then GOARCH="arm64"; \
    else false; fi && \
    curl -L "https://golang.org/dl/go1.21.5.linux-$GOARCH.tar.gz" -o go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && rm go.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Rust — install outside /root (mode 700) so the container can run as a
# non-root uid at runtime (needed to keep bind-mounted ./results writable
# by the host user instead of root).
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
ADD https://sh.rustup.rs /tmp/rustup.sh
RUN chmod +x /tmp/rustup.sh && /tmp/rustup.sh -y && rm /tmp/rustup.sh \
    && chmod -R a+rX /opt/rustup /opt/cargo
ENV PATH="/opt/cargo/bin:${PATH}"

# Node.js 20 + JS exercise deps (must be pre-installed for jest tests)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /npm-install && cd /npm-install && npm init -y \
    && npm install \
        jest \
        @babel/core@7.25.2 \
        @exercism/babel-preset-javascript@0.2.1 \
        @exercism/eslint-config-javascript@0.6.0 \
        @types/jest@29.5.12 \
        @types/node@20.12.12 \
        babel-jest@29.6.4 \
        core-js@3.37.1 \
        eslint@8.49.0

# Clone aider + exercises
RUN git clone --depth 1 https://github.com/Aider-AI/aider.git /aider
WORKDIR /aider

# Local patches to vendored aider source -- see patches/ for rationale on each.
# Not fixed upstream as of the commit above (checked commit history on
# benchmark/benchmark.py); this caps test/build output length before it's
# appended to the aider chat history, since some compiler/test-framework
# diagnostics (GCC operator== overload dumps, Go per-iteration test logging)
# are unbounded and can otherwise blow past the model's context window from
# a single one-line bug.
COPY patches/patch_benchmark_truncate_output.py /tmp/patch_benchmark_truncate_output.py
RUN python3 /tmp/patch_benchmark_truncate_output.py /aider/benchmark/benchmark.py \
    && rm /tmp/patch_benchmark_truncate_output.py

# Opt-in wall-clock cap per exercise (AIDER_BENCHMARK_MAX_DUR env var, seconds;
# unset/0 = old unbounded behavior). Needed because aider's own internal
# reflection loop (up to 3 extra lint/test fix-up rounds per --tries attempt,
# see aider/coders/base_coder.py's max_reflections) can burn enormous wall-clock
# time on a model that keeps tripping the same error without converging --
# observed: 88 minutes / 6 tries on one exercise for one slow model.
COPY patches/patch_benchmark_max_duration.py /tmp/patch_benchmark_max_duration.py
RUN python3 /tmp/patch_benchmark_max_duration.py /aider/benchmark/benchmark.py \
    && rm /tmp/patch_benchmark_max_duration.py

# run_unit_tests()'s subprocess.run(timeout=...) only kills the direct child
# (e.g. `go`), not grandchildren it execs (e.g. the compiled `go test` test
# binary) -- so a model solution with a genuine infinite/exponential-blowup
# loop leaves an orphaned process pegging a CPU core forever after the
# "timeout" fires. Observed firsthand: a two-bucket solution's test binary
# ran for over an hour post-timeout and had to be killed by hand. Runs the
# test command in its own process group and SIGKILLs the whole group instead.
COPY patches/patch_benchmark_test_process_group.py /tmp/patch_benchmark_test_process_group.py
RUN python3 /tmp/patch_benchmark_test_process_group.py /aider/benchmark/benchmark.py \
    && rm /tmp/patch_benchmark_test_process_group.py

# AIDER_BENCHMARK_MAX_DUR (see patch_benchmark_max_duration.py) only checks the
# wall clock *between* tries, so it can't do anything about a single try that
# itself hangs -- observed firsthand: a coder.run() call hit a real litellm
# connection error, retried with exponential backoff that kept doubling well
# past aider's own advertised 60s cap (up to 8192s-and-climbing), keeping one
# exercise "in progress" for 10+ hours before it was noticed and killed by
# hand. Wraps coder.run() in a hard SIGALRM-based timeout sized to the
# remaining max_dur budget so a hung try gets interrupted directly instead of
# only blocking the *next* try from starting.
COPY patches/patch_benchmark_try_timeout.py /tmp/patch_benchmark_try_timeout.py
RUN python3 /tmp/patch_benchmark_try_timeout.py /aider/benchmark/benchmark.py \
    && rm /tmp/patch_benchmark_try_timeout.py

# aider's ChatSummary defaults max_chat_history_tokens to 1024 (or up to 8192
# if the model's context window is known -- ours isn't, for local openai/
# aliases). Once history crosses that, summarize_start() fires a background-
# threaded extra completion call on nearly every turn -- fine against a real
# multi-tenant API, but against our --parallel 1 llama-server it races the
# next turn's request for the single KV-cache slot and evicts its cached
# prefix, forcing a full prompt reprocess. Traced via llama-server's "erased
# invalidated context checkpoint" log spam. AIDER_BENCHMARK_MAX_CHAT_HISTORY_TOKENS
# env var (unset/0 = old behavior) raises/disables the threshold.
COPY patches/patch_benchmark_max_chat_history.py /tmp/patch_benchmark_max_chat_history.py
RUN python3 /tmp/patch_benchmark_max_chat_history.py /aider/benchmark/benchmark.py \
    && rm /tmp/patch_benchmark_max_chat_history.py

# Git config in a world-readable file (GIT_CONFIG_GLOBAL, not ~/.gitconfig)
# since the container runs as an arbitrary non-root uid with no $HOME.
ENV GIT_CONFIG_GLOBAL=/etc/gitconfig-aider
RUN git config -f "$GIT_CONFIG_GLOBAL" --add safe.directory /aider \
    && chmod a+r "$GIT_CONFIG_GLOBAL"

RUN git clone --depth 1 https://github.com/Aider-AI/polyglot-benchmark /benchmarks/polyglot-benchmark

# Install aider
RUN pip3 install --no-cache-dir --upgrade pip uv
RUN uv pip install --system --no-cache-dir -e /aider[dev]

# HOME=/tmp: world-writable, covers matplotlib/npm/pip caches for any uid
ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp/mplconfig

RUN mkdir -p /results && chmod a+rwx /results

# Strip jest/go/gradle test timings from test output (upstream only strips
# pytest/cargo's "in 0.05s"), so a fixed SEED reproduces a whole run, not
# just try 1 -- see patches/patch_benchmark_strip_timings.py. Kept last
# (benchmark.py is an editable install, so patching after `uv pip install`
# is fine) so adding it didn't invalidate the polyglot-benchmark clone
# layer above and silently pull a newer exercise set.
COPY patches/patch_benchmark_strip_timings.py /tmp/patch_benchmark_strip_timings.py
RUN python3 /tmp/patch_benchmark_strip_timings.py /aider/benchmark/benchmark.py \
    && rm /tmp/patch_benchmark_strip_timings.py

# X-Eval-Session header per exercise, so run.sh's eval proxy (harbor/eval_proxy.py)
# can attribute each logged request to an exercise -- see
# patches/patch_benchmark_session_header.py. Last layer so adding it reuses the cache.
COPY patches/patch_benchmark_session_header.py /tmp/patch_benchmark_session_header.py
RUN python3 /tmp/patch_benchmark_session_header.py /aider/benchmark/benchmark.py \
    && rm /tmp/patch_benchmark_session_header.py
