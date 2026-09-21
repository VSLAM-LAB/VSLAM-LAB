#!/usr/bin/env python3
"""
PreToolUse hook (matcher: Bash) for Claude Code.

Rejects Bash tool calls that Claude Code's permission parser cannot analyze
statically. Such commands bypass every allow rule and fall through to a manual
permission prompt when permissions.blockReadsOutsideWorkingDirectories is on.
Denying them here sends a rewrite hint back to Claude instead of a prompt to
the user.

Rejected constructs (outside single quotes and quoted heredocs, where the
shell does not expand anything):
  - command substitution: $(...) and backticks
  - parameter expansion: $VAR, ${VAR}, $1, $?, ...
  - eval
  - cd (relative paths after a cd cannot be resolved by the parser)
  - exec-wrappers such as setsid, xargs, sudo, exec, find -exec (the parser
    only sees through timeout/time/nice/nohup/stdbuf/command/builtin)
  - a newline followed by # inside a double-quoted argument (a comment line
    in an inline python3 -c block), which the parser refuses to analyze

Input: hook JSON on stdin. Output: nothing (allow) or a PreToolUse deny JSON.
"""
import json
import re
import sys

CHECKS = [
    ("command substitution $(...)", re.compile(r"\$\(")),
    ("backticks", re.compile(r"`")),
    ("variable expansion like $VAR or ${VAR}", re.compile(r"\$[A-Za-z_{@*#?!$0-9-]")),
    ("eval", re.compile(r"(?<![A-Za-z0-9_])eval(?![A-Za-z0-9_])")),
    # cd makes every later relative path unresolvable to the parser, and the
    # Bash tool's working directory persists anyway, so cd is never needed.
    ("cd (use absolute paths instead)", re.compile(r"(?:^|[;&|\n])\s*cd(?:\s|$)")),
    # Programs that run their argument as a command. The parser only strips a
    # fixed wrapper set (timeout, time, nice, nohup, stdbuf, command, builtin);
    # anything else that wraps a command is opaque to it.
    (
        "an exec-wrapper the parser cannot see through (setsid/xargs/sudo/exec/...)",
        re.compile(
            r"(?:^|[;&|\n]|\s)(?:setsid|xargs|sudo|su|doas|exec|flock|unbuffer|parallel|"
            r"watch|strace|ltrace|chroot)(?:\s|$)|\s-(?:exec|execdir|ok|okdir)\s"
        ),
    ),
]

QUOTED_HEREDOC = re.compile(r"<<-?\s*(['\"])(\w+)\1[^\n]*\n")


def strip_quoted_heredocs(cmd: str) -> str:
    """Replace the body of every <<'EOF' / <<"EOF" heredoc (never expanded)."""
    while True:
        m = QUOTED_HEREDOC.search(cmd)
        if not m:
            break
        delim = m.group(2)
        body_start = m.end()
        end = re.compile(rf"^\s*{re.escape(delim)}\s*$", re.M).search(cmd, body_start)
        body_end = end.end() if end else len(cmd)
        cmd = cmd[: m.start()] + "<<HEREDOC\n" + cmd[body_end:]
    return cmd


def analyze(cmd: str) -> tuple[str, bool]:
    """Return (visible_text, newline_hash_in_double_quotes).

    visible_text has everything the shell treats literally removed:
    quoted-heredoc bodies, single-quoted spans (outside double quotes), and
    backslash-escaped characters. The flag is set when a newline followed by
    optional blanks and '#' occurs inside a double-quoted argument, which the
    permission parser refuses to analyze ("can hide arguments from path
    validation").
    """
    cmd = strip_quoted_heredocs(cmd)
    out = []
    i, n = 0, len(cmd)
    in_dq = False
    nl_hash = False
    while i < n:
        c = cmd[i]
        if c == "\\":
            if not in_dq or (i + 1 < n and cmd[i + 1] in '"\\$`\n'):
                i += 2  # escaped char: literal, drop both
                continue
            out.append(c)  # backslash is literal inside double quotes otherwise
            i += 1
            continue
        if c == '"':
            in_dq = not in_dq
            out.append(c)
            i += 1
            continue
        if c == "'" and not in_dq:
            j = cmd.find("'", i + 1)
            i = n if j == -1 else j + 1
            continue
        if c == "\n" and in_dq:
            k = i + 1
            while k < n and cmd[k] in " \t":
                k += 1
            if k < n and cmd[k] == "#":
                nl_hash = True
        out.append(c)
        i += 1
    return "".join(out), nl_hash


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    visible, nl_hash = analyze(command)
    hits = [label for label, rx in CHECKS if rx.search(visible)]
    if nl_hash:
        hits.append("a comment line (newline then #) inside a double-quoted argument")
    if not hits:
        return 0
    reason = (
        "Rejected by the project hook .claude/hooks/block_unparseable_bash.py: the command "
        f"contains {', '.join(hits)}. Claude Code's permission parser cannot analyze these, "
        "so they bypass Bash(*) and force a manual permission prompt. Rewrite it with no "
        "expansion or substitution: write full literal paths instead of $VAR, use "
        "grep -n -A<N> 'pattern' file or awk '/pattern/{f=1} f{print; if(++n><N>) exit}' file "
        "instead of $(...), never prefix with cd (use absolute paths; the tool's working "
        "directory already persists), and split multi-step lookups into separate plain "
        "commands. For detached or long-running work use the Bash tool's run_in_background "
        "parameter instead of setsid/nohup/&; if a wrapper is unavoidable, only timeout, "
        "time, nice, nohup and stdbuf are understood by the parser. Instead of xargs or "
        "find -exec, write a for loop over a literal glob or a small Python script. "
        "Prefer writing multi-line Python to a .py file in the scratchpad (a declared "
        "working directory) with the Write tool and running that file, over inline "
        "python3 -c \"...\" blocks; never put # comment lines inside a quoted argument. "
        "Single quotes and <<'EOF' heredocs are fine."
    )
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
