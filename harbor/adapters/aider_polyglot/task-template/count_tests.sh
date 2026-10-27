#!/bin/bash
# (fork) Count passed/total individual tests from a polyglot verifier run and
# print a Harbor reward.json object on stdout:
#   {"reward": 0|1, "tests_passed": P, "tests_total": T, "tests_frac": P/T, "tests_parsed": 0|1}
# reward is the binary exit-code result (unchanged from upstream semantics);
# tests_parsed=0 means no runner summary was found (usually a compile/collect
# failure), in which case P=T=0 and the true total must come from elsewhere
# (e.g. an oracle run of the same task).
#
# Plain bash+awk on purpose: the task images don't all ship python.
#
# usage: count_tests.sh <language> <test-output.txt> <exit_code>
set -uo pipefail

lang="$1"
log="$2"
exit_code="$3"
passed=0
total=0
parsed=0

case "$lang" in
python)
    # pytest's final "==== 2 failed, 11 passed in 0.12s ====" line
    read -r passed total parsed < <(awk '
        /^=+ .* in [0-9.]+s/ { line = $0 }
        END {
            if (line == "") { print 0, 0, 0; exit }
            p = f = e = 0
            n = split(line, w, /[ ,]+/)
            for (i = 2; i <= n; i++) {
                if (w[i] == "passed") p = w[i-1]
                else if (w[i] == "failed") f = w[i-1]
                else if (w[i] == "error" || w[i] == "errors") e = w[i-1]
            }
            print p, p + f + e, 1
        }' "$log")
    ;;
go)
    # go test -v: "--- PASS: TestX/sub (0.00s)". Count leaves only, so a parent
    # test with subtests isn't counted alongside its children.
    read -r passed total parsed < <(awk '
        match($0, /--- (PASS|FAIL): [^ ]+/) {
            s = substr($0, RSTART + 4, RLENGTH - 4)
            split(s, a, ": ")
            st[a[2]] = a[1]
        }
        END {
            p = t = 0
            for (n in st) {
                leaf = 1
                for (m in st) if (index(m, n "/") == 1) { leaf = 0; break }
                if (!leaf) continue
                t++
                if (st[n] == "PASS") p++
            }
            print p, t, (t > 0 ? 1 : 0)
        }' "$log")
    ;;
rust)
    # one "test result: ok. 12 passed; 0 failed; 0 ignored; ..." per test binary
    read -r passed total parsed < <(awk '
        /^test result: / {
            found = 1
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^passed;?$/) p += $(i-1)
                else if ($i ~ /^failed;?$/) f += $(i-1)
            }
        }
        END { print p + 0, p + f, found + 0 }' "$log")
    ;;
javascript)
    # jest: "Tests:       2 failed, 11 passed, 13 total"
    read -r passed total parsed < <(awk '
        /^Tests:/ {
            p = 0; t = 0; found = 1
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^passed,?$/) p = $(i-1)
                else if ($i ~ /^total$/) t = $(i-1)
            }
        }
        END { print p + 0, t + 0, found + 0 }' "$log")
    ;;
cpp)
    # Catch2 v2: "All tests passed (35 assertions in 12 test cases)" or
    #            "test cases: 12 | 10 passed | 2 failed" / "test cases: 3 | 3 failed"
    read -r passed total parsed < <(awk '
        /^All tests passed/ && match($0, /in [0-9]+ test case/) {
            split(substr($0, RSTART, RLENGTH), a, " "); p = t = a[2]; found = 1
        }
        /^test cases:/ {
            gsub(/\|/, " "); t = $3; f = 0
            for (i = 4; i <= NF; i++) if ($i == "failed") f = $(i-1)
            p = t - f; found = 1
        }
        END { print p + 0, t + 0, found + 0 }' "$log")
    ;;
java)
    # Gradle JUnit XML: <testsuite ... tests="N" skipped="S" failures="F" errors="E" ...>
    read -r passed total parsed < <(cat ${WORK:-/app}/build/test-results/test/*.xml 2>/dev/null | awk '
        function attr(name,   r) {
            if (match($0, name "=\"[0-9]+\"")) {
                r = substr($0, RSTART + length(name) + 2, RLENGTH - length(name) - 3)
                return r + 0
            }
            return 0
        }
        /<testsuite / {
            found = 1
            n = attr("tests"); bad = attr("skipped") + attr("failures") + attr("errors")
            t += n; p += n - bad
        }
        END { print p + 0, t + 0, found + 0 }')
    ;;
esac

reward=$([ "$exit_code" -eq 0 ] && echo 1 || echo 0)
frac=$(awk -v p="$passed" -v t="$total" 'BEGIN { printf "%.4f", (t > 0 ? p / t : 0) }')
# (fork) test-calls tasks: the agent's run-tests history (one line per call:
# n, timestamp, exit code, passed, total -- written by /opt/run-tests/run-tests-inner)
calls_json=""
if [ -f /opt/run-tests/limit ]; then
    limit=$(cat /opt/run-tests/limit)
    log=/var/lib/run-tests/log.tsv
    read -r calls first_pass < <(awk -F'\t' '
        { n++ } $3 == 0 && !fp { fp = $1 }
        END { print n + 0, fp + 0 }' "$log" 2>/dev/null || echo "0 0")
    calls_json=$(printf ', "test_limit": %d, "test_calls": %d, "test_limit_hit": %d, "first_pass_call": %d' \
        "$limit" "$calls" "$([ "$calls" -ge "$limit" ] && echo 1 || echo 0)" "$first_pass")
fi
printf '{"reward": %d, "tests_passed": %d, "tests_total": %d, "tests_frac": %s, "tests_parsed": %d%s}\n' \
    "$reward" "$passed" "$total" "$frac" "$parsed" "$calls_json"
