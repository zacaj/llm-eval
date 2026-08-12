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
