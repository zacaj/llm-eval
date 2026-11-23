#!/bin/bash
# (tb2_tc) Replaces the task's upstream tests/test.sh. Same test files, run offline with
# what the image build pre-installed (upstream installs pytest from the network on every
# run) and -v so count_tests.sh can count passes. main.py fills in the @...@ parts from
# the upstream test.sh: the pytest command (a root-owned venv for uvx tasks, the image's
# own python for pip tasks), its `cp /tests/...` setup steps, and whether it also gates
# on the repo's own test suite (counted here as one extra test, repo_test_suite).
# Also run by run-tests-inner with OUT / TEST_DIR pointed at its scratch dirs.
OUT="${OUT:-/logs/verifier}"
TEST_DIR="${TEST_DIR:-/tests}"
mkdir -p "$OUT"
cd @CWD@ || exit 1
@PRE@
repo_rc=0
if [ @REPO_SUITE@ = 1 ]; then
    timeout 600 @PYTEST@ -p no:cacheprovider -q > "$OUT/repo-suite-output.txt" 2>&1
    repo_rc=$?
    tail -5 "$OUT/repo-suite-output.txt"
fi

timeout 600 @PYTEST@ -p no:cacheprovider -rA -v \
    "$TEST_DIR/test_outputs.py" > "$OUT/test-output.txt" 2>&1
exit_code=$?
cat "$OUT/test-output.txt"
if [ @REPO_SUITE@ = 1 ]; then
    # the suite counts as one more test: drop pytest's "N passed in Xs" summary so
    # count_tests.sh falls back to counting the -v "::name PASSED" lines, and add one
    sed -i -E '/^=+ .* in [0-9.]+s/d' "$OUT/test-output.txt"
    echo "test_outputs.py::repo_test_suite $([ $repo_rc -eq 0 ] && echo PASSED || echo FAILED)" \
        >> "$OUT/test-output.txt"
    [ $repo_rc -eq 0 ] || exit_code=1
fi

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
