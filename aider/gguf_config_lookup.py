#!/usr/bin/env python3
"""Find a known-good llama-server config for a .gguf file by scraping every
`command: >` block (active AND commented-out) out of llama-cpp-turboquant's
docker-compose.yml -- that file is effectively a running log of every config
ever run per model, not a clean single-source-of-truth, so this is
best-effort text scraping, not a real YAML/shell parser.

Usage: gguf_config_lookup.py <docker-compose.yml> <gguf-filename-or-path>
Prints "<alias>\\t<flags>" on stdout and exits 0 on exactly one distinct
match. Exits 1 with candidates/reason on stderr otherwise (not found, or
more than one distinct block -- deliberately refuses to guess).

Used by sweep.sh's CONFIGS=<comma-separated .gguf list> mode; see its header
comment for the user-facing behavior.
"""
import re
import sys

COMMAND_RE = re.compile(r'^\s*#?\s*command:\s*>\s*$')
ALIAS_RE = re.compile(r'--alias\s+"([^"]+)"|--alias\s+(\S+)')
M_RE = re.compile(r'-m\s+"?([^\s"]+\.gguf)"?')
DROP_FLAGS_RE = [
    re.compile(r'(?<!\S)--server\b'),
    re.compile(r'(?<!\S)--host\s+\S+'),
    re.compile(r'(?<!\S)--port\s+\S+'),
]


def extract_blocks(lines):
    """Yield (flags_str, alias) for every `command: >` block that has both
    -m and --alias. Only lines that are, after stripping exactly one leading
    '#' and whitespace, real flags (start with '-') are kept -- this file is
    full of interleaved free-text notes (benchmark numbers, todo refs) and
    doubly-commented disabled sub-options ("#     #  --foo") inside these
    blocks, neither of which are part of the actual command line."""
    i = 0
    while i < len(lines):
        if COMMAND_RE.match(lines[i]):
            flag_parts = []
            j = i + 1
            while j < len(lines) and lines[j].strip():
                content = re.sub(r'^#\s*', '', lines[j].strip())
                if content.startswith('-'):
                    flag_parts.append(content)
                j += 1
            flags = ' '.join(flag_parts)
            m_alias = ALIAS_RE.search(flags)
            m_model = M_RE.search(flags)
            if m_alias and m_model:
                alias = m_alias.group(1) or m_alias.group(2)
                cleaned = flags
                for pat in DROP_FLAGS_RE:
                    cleaned = pat.sub('', cleaned)
                yield ' '.join(cleaned.split()), alias
            i = j
        else:
            i += 1


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <docker-compose.yml> <gguf-filename-or-path>", file=sys.stderr)
        return 1
    compose_path, target = sys.argv[1], sys.argv[2]
    target_base = target.rsplit('/', 1)[-1]

    lines = open(compose_path, encoding='utf-8').read().splitlines()
    matches = {}  # flags -> alias, deduplicated by exact flag string
    for flags, alias in extract_blocks(lines):
        m = M_RE.search(flags)
        if m and m.group(1).rsplit('/', 1)[-1] == target_base:
            matches[flags] = alias

    if not matches:
        print(f"NOT FOUND: no command block in {compose_path} references {target_base}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(f"AMBIGUOUS: {len(matches)} distinct configs found for {target_base} -- refusing to guess:", file=sys.stderr)
        for flags, alias in matches.items():
            print(f"  --alias {alias}: {flags[:160]}{'...' if len(flags) > 160 else ''}", file=sys.stderr)
        print("Use a real configs.txt (CONFIGS=<path>) to pick one explicitly.", file=sys.stderr)
        return 1

    flags, alias = next(iter(matches.items()))
    print(f"{alias}\t{flags}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
