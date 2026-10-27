FROM buildpack-deps:jammy

# Install common requirements
RUN apt-get update && apt-get install -y \
    tmux \
    asciinema \
    curl \
    && rm -rf /var/lib/apt/lists/*

# (fork) Pre-install a pinned Claude Code (native build) so Harbor's per-trial agent
# install is a no-op: it skips installing when `claude --version` already matches the
# --ak version harbor.sh passes (default read from this line). Avoids a per-trial
# apt nodejs/npm download (mirror stalls hit the 360s setup timeout) and keeps the
# agent version fixed across runs. /usr/local/bin so the non-root `agent` user has it.
ARG CC_VERSION=2.1.311
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- $CC_VERSION \
    && cp -L /root/.local/bin/claude /usr/local/bin/claude && rm -rf /root/.local/share/claude /root/.local/bin/claude \
    && claude --version | grep -q "^$CC_VERSION"

# Language-specific setup
{LANGUAGE_SETUP}

# Set language-specific environment variables
{LANGUAGE_ENV}

# Verify language tools work before copying files
{LANGUAGE_VERIFY}

# Copy workspace into the image
WORKDIR /app
COPY workspace/ ./

# Language-specific post-setup (e.g., make gradlew executable)
{LANGUAGE_POST_SETUP}

# Default environment variables
ENV TEST_DIR=/tests
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
