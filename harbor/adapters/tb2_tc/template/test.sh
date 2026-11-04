#!/bin/bash
# (tb2_tc) Replaces the task's upstream tests/test.sh. Same pytest file, but run with
# the image's pre-built /opt/tb-pytest venv (root-owned, offline; upstream installs uv +
# pytest from the network on every run) and -v so count_tests.sh can count passes.
# Also run by run-tests-inner with OUT / TEST_DIR pointed at its scratch dirs.
OUT="${OUT:-/logs/verifier}"
TEST_DIR="${TEST_DIR:-/tests}"
mkdir -p "$OUT"
cd /app || exit 1

timeout 600 /opt/tb-pytest/bin/python -m pytest -p no:cacheprovider -rA -v \
    "$TEST_DIR/test_outputs.py" > "$OUT/test-output.txt" 2>&1
exit_code=$?
cat "$OUT/test-output.txt"

[ $exit_code -eq 0 ] && echo 1 > "$OUT/reward.txt" || echo 0 > "$OUT/reward.txt"
# partial credit + test-call history (count_tests.sh: the polyglot fork's, unchanged)
bash "$TEST_DIR/count_tests.sh" python "$OUT/test-output.txt" "$exit_code" \
    > "$OUT/reward.json.tmp" && mv "$OUT/reward.json.tmp" "$OUT/reward.json" \
    || rm -f "$OUT/reward.json.tmp"
cat "$OUT/reward.json" 2>/dev/null || true
if [ -f /var/lib/run-tests/log.tsv ] && [ "$OUT" = /logs/verifier ]; then
    cp /var/lib/run-tests/log.tsv "$OUT/run-tests-log.tsv" || true
fi
exit 0
