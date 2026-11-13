#!/usr/bin/env python3
"""Per-trial issue findings from an ATIF trajectory, per docs/issue-taxonomy.md.

Every failing tool call gets a (category, layer, target, note):
  categories  malformed / hallucinated / attention / mis-wired / overreach / env gap
              (+ leak, harness, unknown, unclassified -- harness-side or unrecognised)
  layers      tool-call / edit / shell
A repeat of something the model was already told fails becomes attention/"forgot feedback".

collect.py calls trial_issues() per trial for the iss_* columns and write_alerts() once
for results/harness-alerts.tsv: findings no rule recognises, new missing tools, leaks,
harness bugs -- so harness gaps and classifier blind spots surface instead of being scored.
Prototype / sampling tool: tmp/issue_classify.py --samples <note>.
"""
import collections
import difflib
import json
import os
import re
from pathlib import Path

SHELL = {"Bash", "bash", "execute_bash", "run_command", "shell"}
EDIT = {"Edit", "MultiEdit", "Write", "NotebookEdit", "str_replace_editor"}
CC_TOOLS = {"Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "LS", "TodoWrite", "Task", "Agent",
            "WebFetch", "WebSearch", "NotebookEdit", "NotebookRead", "BashOutput", "KillShell", "KillBash",
            "ExitPlanMode", "EnterPlanMode", "Skill", "SlashCommand", "AskUserQuestion", "TaskOutput",
            "TaskStop", "ToolSearch", "LSP", "Monitor", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet"}
MARKUP = re.compile(r"<tool_call>|<function[=>]|<parameter[=>]|<invoke\b")
EXIT = re.compile(r"\[exit_code\] (-?\d+)|Exit code (-?\d+)")
ENV_GAP_CMDS = {"xxd", "file", "unzip", "hexdump", "strings", "tree", "bc", "jq", "zip", "valgrind", "strace", "ltrace",
                "netcat", "nc", "wget", "rg", "fd", "ag", "iconv", "patch", "dos2unix", "git", "python", "python3", "pip", "pip3",
                "node", "gdb", "make", "gcc", "g++", "sqlite3", "perl", "ruby"}
INTERACTIVE = {"nano", "vim", "vi", "less", "more", "top", "htop", "emacs"}
KNOWN_CMDS = ENV_GAP_CMDS | INTERACTIVE | {"python", "python3", "node", "npm", "npx", "cargo", "rustc", "go", "gcc", "g++",
             "make", "cmake", "java", "javac", "mvn", "gradle", "pytest", "echo", "cat", "ls", "grep", "sed", "awk",
             "find", "head", "tail", "cd", "mkdir", "rm", "cp", "mv", "chmod", "touch", "diff", "sort", "uniq", "wc",
             "tr", "cut", "printf", "timeout", "sleep", "kill", "pkill", "ps", "env", "which", "git", "tar", "curl"}
TESTBUILD = re.compile(r"(^|[;&|(]\s*|\s)(run-tests|cargo (test|build|check|run|clippy)|pytest|python3? -m (pytest|unittest)|"
                       r"go (test|build|vet|run)|npm (test|run|t)\b|npx |jest|mvn|gradle|make\b|cmake|g\+\+|gcc|clang|rustc|"
                       r"javac|tsc|ctest|cobc)")
COMPILE = re.compile(r"(^|[;&|(]\s*|\s)(cargo (build|check|test|run)|rustc|go (build|vet|test|run)|g\+\+|gcc|clang|javac|tsc|cobc|make\b|mvn|gradle)")
COMPILE_ERR = re.compile(r"error(\[E\d+\])?:|undefined reference|cannot find symbol|compilation failed|"
                         r"could not compile|\.go:\d+:\d+: |BUILD FAILURE|error TS\d+")
PEEK = re.compile(r"hidden-tests|/opt/run-tests|/var/lib/run-tests|run-tests-inner|\bsudo\b(?!\s+(-\S+\s+)*(apt|apt-get|pkg-install)\b)|\bsu\b")
SYSWRITE = re.compile(r"pip3? install|npm (i|install) -g|/usr/local/lib|site-packages|apt(-get)? install")
NOISE = re.compile(r"Error disabling address space randomization")
LEAK = re.compile(r"\b(python|node|cargo)_ro\b")


PROBE_CMDS = re.compile(r"(cd [^;&|\n]+(&&|;)\s*)?(ls|find|grep|egrep|rg|test|\[|which|command -v|diff|cmp|stat|cat|head|tail|pgrep)\b")
# repeats of these mean the model ignored what it was already told
FORGETTABLE = {
    "env gap": lambda n: n.startswith("missing common tool"),
    "hallucinated": lambda n: n in ("unknown command", "path doesn't exist", "Read missing file", "import / module"),
    "malformed": lambda n: n == "path typo",
    "mis-wired": lambda n: n == "interactive tool",
}
# a program the model ran failed (dev loop), not the shell call itself
PROGRAM_FAIL = re.compile(r"Traceback \(most recent|AssertionError|\bFAIL|panicked|Error: |error(\[E\d+\])?:|"
                          r"Segmentation fault|Aborted|throw |Exception|libcob: |assert")


def out_body(out):
    """tool output minus the harness's own markers (Exit code N, [error] tool reported failure, [stdout]...)"""
    return "\n".join(l for l in out.splitlines() if not re.match(r"\s*(Exit code -?\d+|\[(error|stdout|stderr|metadata|exit_code)\])", l))


def text(x):
    return x if isinstance(x, str) else json.dumps(x) if x is not None else ""


def first_word(cmd):
    c = re.sub(r"^\s*((cd|export|source) [^;&|\n]+(&&|;)\s*)+", "", cmd.strip())
    c = re.sub(r"^(timeout\s+\d+\S*\s+|sudo\s+(-\S+\s+)*|env\s+(\w+=\S+\s+)*|\w+=\S+\s+)+", "", c)
    m = re.match(r"[\w./+-]+", c)
    return (m.group(0).rsplit("/", 1)[-1]) if m else ""


def near(word, pool, cutoff=0.8):
    return bool(difflib.get_close_matches(word, [p for p in pool if p != word], n=1, cutoff=cutoff))


def is_typo(p, pool):
    """same-shape path with a small in-place change (/temp vs /tmp), not a sibling guess
    (foo.spec.js next to foo.js) or a parent/child of a known path"""
    for q in pool:
        if q == p or q.startswith(p) or p.startswith(q) or abs(len(q) - len(p)) > 2:
            continue
        if p.count("/") == q.count("/") and difflib.SequenceMatcher(None, p, q).ratio() >= 0.9:
            return True
    return False


def paths_in(s):
    return set(re.findall(r"(?<![\w.])(/(?:app|tmp|opt|usr|var|home|etc|root)[\w./+-]*)", s))


def classify_shell(cmd, out, code, seen_paths, prior):
    """-> (category, layer, target, note) or None for clean."""
    fw = first_word(cmd)
    if LEAK.search(cmd):
        return ("leak", "shell", fw, "personal CLAUDE.md tool")
    # only the shell's own message, not a test program printing "not found"
    m = re.search(r"(?m)^(?:/bin/)?(?:ba)?sh: (?:line \d+: )?([\w./+-]+): (?:command )?not found", out)
    if m:
        name = m.group(1).rsplit("/", 1)[-1]
        if name in INTERACTIVE:
            return ("mis-wired", "shell", name, "interactive tool")
        if name in ENV_GAP_CMDS:
            return ("env gap", "shell", name, "missing common tool")
        if near(name, KNOWN_CMDS, 0.75) or re.fullmatch(r"[a-z]{1,3}", name) and name not in KNOWN_CMDS:
            return ("malformed", "shell", name, "typo/garbled command")
        return ("hallucinated", "shell", name, "unknown command")   # also alerts (not in any list)
    if PEEK.search(cmd) and (code != 0 or re.search(r"Permission denied|password|not allowed", out)):
        return ("overreach", "shell", "peek", "hidden tests / sudo")
    if re.search(r"Permission denied|Read-only file system|Operation not permitted", out) and not NOISE.search(out):
        if SYSWRITE.search(cmd):
            return ("env gap", "shell", "syswrite", "install into system dir")
        if "run-tests" in cmd:
            return ("harness", "shell", "run-tests-perm", "run-tests copy perms")
        return ("unknown", "shell", "perm:" + fw, "permission denied: " + first_err(out))
    if re.search(r"\b(pkill|killall)\b", cmd) and code in (124, 137, 143, 144, -1):
        return ("mis-wired", "shell", "pkill", "killed own shell")
    if code in (124, 137) and not TESTBUILD.search(cmd):
        pass  # timeout of own probe: ambiguous, fall through
    if re.search(r"syntax error|unexpected EOF|bad substitution|ambiguous redirect", out) and not TESTBUILD.search(cmd):
        return ("malformed", "shell", fw, "bash syntax")
    if re.search(r"unrecognized option|invalid option|unknown option|bad option", out):
        return ("hallucinated", "shell", fw, "CLI flag (placeholder: may be typo)")
    for p in re.findall(r"/tmp/run-tests\.\w+[\w./-]*", cmd):
        if re.search(r"No such file|cannot (open|access|stat)", out):
            return ("attention", "shell", "/tmp/run-tests.*", "stale state: old run-tests tmpdir")
    m = re.search(r"(?:cannot (?:open|access|stat) |: )'?(/[\w./+-]+)'?:? No such file or directory|"
                  r"No such file or directory: '(/?[\w./+-]+)'", out)
    if m and (m.group(1) or m.group(2)) in cmd:   # path the model wrote, not one a program printed
        p = m.group(1) or m.group(2)
        if re.search(r"-o\s+" + re.escape(p) + r"\b|> ?" + re.escape(p) + r"\.\w+ <<", cmd) or re.search(re.escape(p) + r"\.(rs|c|cpp|cbl|go|java)\b", cmd):
            return None   # the binary a failed compile in this same command didn't produce
        if is_typo(p, seen_paths):
            return ("malformed", "shell", p, "path typo")
        if code != 0 or not TESTBUILD.search(cmd):
            return ("hallucinated", "shell", p, "path doesn't exist")
    if code != 0 and "2>/dev/null" in cmd and not re.search(r"(?i)error|denied|not found|no such|traceback", out_body(out)):
        # stderr thrown away: the failure is invisible to us *and* to the model
        if code == 127:
            gone = [w for w in re.findall(r"(?:^|[;&|]\s*)(\w[\w+-]*)", cmd) if w in ENV_GAP_CMDS]
            if gone:
                return ("env gap", "shell", gone[0], "missing common tool (silenced)")
        if not TESTBUILD.search(cmd):
            return ("hallucinated", "shell", fw, "silenced failure (2>/dev/null, likely missing path)")
    if re.search(r"ModuleNotFoundError|Cannot find module", out) and not TESTBUILD.search(cmd):
        return ("hallucinated", "shell", fw, "import / module")
    return None


def edit_split(old, versions, out, fp):
    """failed edit -> (category, layer, target, note); see docs/issue-taxonomy.md 'Edit failure split'"""
    if re.search(r"has not been read|Read it first", out, re.I):
        return ("attention", "edit", fp, "lost instruction: edit before read")
    if re.search(r"modified since|has been modified|changed since", out, re.I):
        return ("attention", "edit", fp, "stale state: file changed since read")
    if re.search(r"Found \d+ matches|not unique|multiple (matches|occurrences)", out, re.I):
        return ("malformed", "edit", fp, "old_string not unique")
    if re.search(r"No changes to make", out):
        return ("attention", "edit", fp, "no-op edit (old == new)")
    if not re.search(r"String to replace not found|not found in file|No match|did not match", out, re.I):
        return ("unknown", "edit", fp, "edit error: " + first_err(out))
    if not old or not versions:
        return ("malformed", "edit", fp, "no match, file never seen")
    cur = versions[-1]
    if any(old in v for v in versions[:-1]):
        return ("attention", "edit", fp, "stale state: matched an earlier version")
    if " ".join(old.split()) in " ".join(cur.split()):
        return ("malformed", "edit", fp, "typo: whitespace only")
    lines = [l.strip() for l in old.splitlines() if l.strip()]
    have = {l.strip() for l in cur.splitlines()}
    if lines and sum(l in have for l in lines) / len(lines) >= 0.6:
        return ("malformed", "edit", fp, "typo: near-match")
    if len(lines) == 1 and difflib.get_close_matches(lines[0], list(have), n=1, cutoff=0.85):
        return ("malformed", "edit", fp, "typo: near-match")
    return ("hallucinated", "edit", fp, "no close match (invented code)")


def first_err(out):
    for l in out.splitlines():
        l = l.strip()
        if l and not l.startswith(("Exit code", "[")):
            return norm_sig(l)
    return ""


def norm_sig(l):
    l = re.sub(r"/[\w./+-]+", "PATH", l)
    l = re.sub(r"'[^']*'|\"[^\"]*\"", "Q", l)
    l = re.sub(r"\d+", "N", l)
    return l[:90]


def masked(cmd, out, code):
    if code != 0:
        return False
    m = re.search(r"\|\|\s*(?:\{\s*)?echo\s+(?:-n\s+)?(?:\"([^\"$`]{3,})\"|'([^']{3,})')", cmd)
    if m and (m.group(1) or m.group(2)) in out:
        return True
    if re.search(r"echo\s[^;\n]*\$\?", cmd) and any(int(x) for x in re.findall(r"(?:exit|rc|status|code)[\w ]*[:=]\s*(\d+)", out, re.I)):
        return True
    return False


def scan(path):
    t = json.load(open(path))
    steps = t.get("steps") or []
    model = next((s.get("model_name") for s in steps if s.get("model_name")), "?")
    seen_paths, files = set(), collections.defaultdict(list)  # file -> past contents
    prior = {"missing_paths": set(), "failed": collections.Counter()}
    finds, rates = [], collections.Counter()
    last_fail_target = None
    for st in steps:
        if st.get("source") != "agent":
            continue
        obs = (st.get("observation") or {}).get("results") or []
        res = {r.get("source_call_id"): r for r in obs}
        for i, tc in enumerate(st.get("tool_calls") or []):
            name, args = tc.get("function_name") or "", tc.get("arguments") or {}
            r = res.get(tc.get("tool_call_id")) or (obs[i] if i < len(obs) else {})
            out = text(r.get("content"))
            extra = r.get("extra") or {}
            is_err = bool(extra.get("tool_result_is_error") or extra.get("is_error"))
            f = None
            if "InputValidationError" in out or MARKUP.search(text(args)):
                f = ("malformed", "tool-call", name, "invalid tool call")
            elif name and name not in CC_TOOLS and is_err and re.search(r"No such tool|not available|Unknown tool", out):
                f = ("hallucinated", "tool-call", name, "unknown tool")
            elif name in EDIT:
                fp = str(args.get("file_path") or args.get("path") or "")
                old = text(args.get("old_string") or "")
                if is_err:
                    f = edit_split(old, files[fp], out, fp)
                body = text(args.get("content") or "")
                if name == "Write" and body:
                    files[fp].append(body)
                elif files[fp] and not is_err and old:
                    files[fp].append(files[fp][-1].replace(old, text(args.get("new_string") or ""), 1))
                seen_paths.add(fp)
            elif name == "Read":
                fp = str(args.get("file_path") or "")
                if not is_err:
                    seen_paths.add(fp)
                    files[fp].append(re.sub(r"(?m)^\s*\d+\t", "", out))
                elif re.search(r"does not exist|No such file", out):
                    f = ("malformed" if is_typo(fp, seen_paths) else "hallucinated", "tool-call", fp, "Read missing file")
                    prior["missing_paths"].add(fp)
            elif name in SHELL:
                cmd = text(args.get("command") or args.get("keystrokes") or "")
                m = EXIT.search(out)
                code = int(m.group(1) or m.group(2)) if m else 0
                f = classify_shell(cmd, out, code, seen_paths, prior)
                if f and f[3] == "path doesn't exist":
                    prior["missing_paths"].add(f[2])
                if not f and code != 0 and PROBE_CMDS.match(cmd.strip()) and not re.search(r"(?i)error|denied|not found|no such", out_body(out)):
                    rates["not_an_issue"] += 1      # negative probe / trailing probe set the exit code
                elif not f and code != 0 and not TESTBUILD.search(cmd) and not PROGRAM_FAIL.search(out):
                    f = ("unclassified", "shell", first_word(cmd), first_err(out))
                if COMPILE.search(cmd):
                    rates["compile_runs"] += 1
                    rates["compile_fails"] += bool(COMPILE_ERR.search(out))
                if masked(cmd, out, code):
                    rates["masked"] += 1
                seen_paths |= {p for p in paths_in(out)}
                if code == 0:
                    seen_paths |= paths_in(cmd)
            rates["calls"] += 1
            if f:
                key = (f[0], f[2])
                rep = prior["failed"][key] > 0
                prior["failed"][key] += 1
                cat, note = f[0], f[3]
                if rep and FORGETTABLE.get(cat, lambda n: False)(note) and f[2] != "syswrite":
                    cat, note = "attention", "forgot feedback: " + note   # told once, tried again
                finds.append(dict(cat=cat, layer=f[1], target=f[2], note=note, repeat=rep, model=model, trial=path,
                                  out=out[:200], args=text(args)[:200]))
    return model, finds, rates




CATS = ["malformed", "hallucinated", "attention", "mis-wired", "overreach", "env gap", "unclassified"]
ISSUE_COLUMNS = ["iss_" + c.replace(" ", "_").replace("-", "") for c in CATS] + [
    "iss_repeats", "iss_detail", "prompt_leak", "harness_issues", "bash_masked", "compile_runs", "compile_fails"]


def trial_issues(path) -> tuple[dict, list[dict]]:
    """-> (iss_* columns, findings) for one trajectory.json"""
    _, finds, rates = scan(str(path))
    cols = dict.fromkeys(ISSUE_COLUMNS, 0)
    detail = collections.Counter()
    for f in finds:
        c = f["cat"]
        if c in CATS:
            cols["iss_" + c.replace(" ", "_").replace("-", "")] += 1
        cols["iss_repeats"] += f["repeat"]
        cols["prompt_leak"] += c == "leak"
        cols["harness_issues"] += c == "harness"
        if c in CATS:
            detail[f"{c}/{f['layer']}/{re.sub(r'[;=]', ',', f['note'])[:60]}"] += 1
    cols["iss_detail"] = ";".join(f"{k}={v}" for k, v in detail.most_common())
    cols["bash_masked"] = rates["masked"]
    cols["compile_runs"], cols["compile_fails"] = rates["compile_runs"], rates["compile_fails"]
    return cols, finds


ALERT_NOTES = ("unknown command", "typo/garbled command", "personal CLAUDE.md tool", "run-tests copy perms",
               "missing common tool", "missing common tool (silenced)")
KNOWN = Path(__file__).with_name("issue-known.tsv")


def known_rules() -> list[tuple[re.Pattern, str]]:
    """issue-known.tsv: <regex on signature> TAB <verdict>; a matching alert is resolved"""
    out = []
    if KNOWN.exists():
        for line in KNOWN.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                pat, _, verdict = line.partition("\t")
                out.append((re.compile(pat), verdict.strip()))
    return out


def write_alerts(finds: list[dict], out: Path, min_trials: int = 3) -> None:
    """group alert-worthy findings by signature -> out TSV + short stdout summary.
    Harness/leak hits always alert; other signatures once they reach min_trials trials."""
    sig = collections.defaultdict(lambda: {"n": 0, "trials": set(), "ex": "", "cat": "", "jobs": set()})
    for f in finds:
        if f["cat"] in ("unknown", "unclassified", "harness", "leak") or f["note"] in ALERT_NOTES:
            k = (f"{f['cat']}: {f['target']} | {f['note']}" if f["cat"] in ("unclassified", "unknown")
                 else f"{f['cat']}: {f['note']} | {f['target']}")
            g = sig[k]
            g["n"] += 1
            g["trials"].add(f["trial"])
            g["cat"] = f["cat"]
            g["ex"] = g["ex"] or f["trial"]
            g["jobs"].add(f.get("job", ""))
    rules = known_rules()
    rows = []
    for k, g in sig.items():
        verdict = next((v for p, v in rules if p.search(k)), "")
        if len(g["trials"]) >= min_trials or g["cat"] in ("harness", "leak"):
            rows.append((k, g, verdict))
    rows.sort(key=lambda x: (bool(x[2]), -len(x[1]["trials"])))
    with out.open("w") as fh:
        fh.write("signature\tcategory\tverdict\tcalls\ttrials\tfirst_job\tlast_job\texample\n")
        for k, g, v in rows:
            jobs = sorted(j for j in g["jobs"] if j)
            fh.write(f"{k}\t{g['cat']}\t{v}\t{g['n']}\t{len(g['trials'])}\t{jobs[0] if jobs else ''}\t"
                     f"{jobs[-1] if jobs else ''}\t{g['ex']}\n")
    new = [r for r in rows if not r[2]]
    print(f"{len(new)} open harness alerts ({len(rows) - len(new)} known) -> {out}")
    for k, g, _ in new[:10]:
        print(f"  {len(g['trials']):4d} trials {g['n']:4d} calls  {k[:110]}")
