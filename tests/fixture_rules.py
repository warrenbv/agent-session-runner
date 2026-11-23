"""Rules the recorded fixtures must keep: no secret-shaped values, no home-directory detail, and the exact
commands the exit-code recordings ran. Secret prefixes are built from fragments so this file holds none."""

from __future__ import annotations

import re

_ANTHROPIC = "sk" + "-ant-"
_OPENAI = "sk" + "-"
_GITHUB = "gh" + "[pousr]_"
_GITHUB_PAT = "github" + "_pat_"
_JWT = "ey" + "J"

SECRET_VALUES = re.compile(
    rf"{_ANTHROPIC}[A-Za-z0-9_-]+|{_OPENAI}[A-Za-z0-9]{{20,}}|{_GITHUB}[A-Za-z0-9]{{20,}}|"
    rf"{_GITHUB_PAT}[A-Za-z0-9_]+|{_JWT}[A-Za-z0-9_-]{{8,}}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")

HOME_DETAIL = re.compile(r"/home/[a-z_][a-z0-9_-]*/\.(?:claude|codex|config)/\S*|/tmp/(?:claude-\d+|cc-socks)/\S*")

EXIT_COMMANDS = ("sh -c 'exit 3'",
                 "sh -c 'echo to-stdout; echo to-stderr 1>&2; exit 4'",
                 "sh -c 'echo fine'",
                 "test -e /nonexistent-runner-probe-path",
                 "grep -q runner-probe-no-such-text /etc/hostname")
