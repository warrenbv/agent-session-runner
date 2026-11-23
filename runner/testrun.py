"""The project's test command as the last turn of a run, and the verdict read from the tool record.

The runner never executes the test command itself: code a session wrote runs only inside the provider's
sandbox. After changed work has completed, the driver sends one more turn asking the agent to run the
exact command, then reads the verdict from the provider's own record of that shell call, never from
what the model says or what the command printed:

- Claude Code: the main thread's `Bash` call whose command equals the configured one (only surrounding
  whitespace trimmed), and its tool result. This counts only when the provider confirms that it launched
  the CLI with each Bash call starting in the worktree (providers/claude.py, PROJECT_CWD_ENV). Without
  that confirmation the verdict is `not_run`. A non-zero exit is a result with `is_error` true whose first line is
  the CLI's own "Exit code N". A pass has no exit code. Some commands (`test`, `grep -q`) exiting 1 are
  reported as ordinary answers with `returnCodeInterpretation`, so that field also means failed.
- Codex: the completed `commandExecution` item whose command, unwrapped from the `bash -lc '...'`
  the app-server reports, equals the configured one and whose own `cwd` is an absolute path that
  resolves to the worktree, with the item's own `exitCode`. A missing or different `cwd` is `not_run`.

The prompt asks for the bare command from the directory the session started in. Asking for
`cd <worktree> && <command>` does not work: a model drops a `cd` to the directory it is already in.
So the evidence of where the command ran comes from the provider, not from a prefix the model keeps.

Verdicts: `passed`, `failed`, `not_run` (no such call, denied, or no exit status) and `tree_changed`
(it ran, but the worktree differs across the turn). See docs/security-model.md, "Test verdicts".
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any

log = logging.getLogger("runner.tests")

TEST_CMD_CAP = 500
NO_CHANGES = "no_changes"
SKIPPED_NO_CHANGES = "Tests were skipped because no file changed."
SHELLS = ("bash", "sh", "zsh")
EXIT_LINE = re.compile(r"^Exit code (-?[0-9]{1,6})\Z")
# Fields the Bash tool result carries when the command did not finish in the foreground, or exited
# non-zero but was reported as an ordinary answer. Any of them means the run was not a pass.
NOT_A_FOREGROUND_PASS = ("backgroundTaskId", "timedOutAfterMs", "backgroundedByUser", "backgroundedByTurnAbort",
                         "backgroundedToDeliverMessage", "returnCodeInterpretation")

PROMPT = (
    "The project requires a final check before this work is handed off. Run exactly this command once, "
    "as a single shell command from the repository root, the directory this session started in, "
    "with nothing added before or after it:\n"
    "\n"
    "{command}\n"
    "\n"
    "Do not create, change or delete any file in this turn, and run no other command. Stop once the "
    "command has run, whatever its result: do not try to fix anything. If a structured result is "
    "required, give your previous result again, unchanged.")


def sentence(tests: dict[str, Any]) -> str:
    command, status, code = tests.get("command"), tests.get("status"), tests.get("exit_code")
    if status == "not_run" and tests.get("reason") == NO_CHANGES:
        return SKIPPED_NO_CHANGES
    if status == "passed":
        return f"Tests passed: `{command}`."
    if status == "failed":
        if isinstance(code, int) and not isinstance(code, bool):
            return f"Tests failed with exit code {code}: `{command}`."
        return f"Tests failed: `{command}`."
    if status == "tree_changed":
        return f"The test run changed files, so its result does not count: `{command}`."
    return f"Tests not run: `{command}`."


def check_test_cmd(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("test_cmd is a string")
    if not value.strip():
        return None
    if len(value) > TEST_CMD_CAP:
        raise ValueError(f"test_cmd is at most {TEST_CMD_CAP} characters")
    if "\0" in value or "\n" in value or "\r" in value or len(value.splitlines()) != 1:
        raise ValueError("test_cmd is one line without NUL")
    return value


def prompt(command: str) -> str:
    return PROMPT.format(command=command)


# Shell text is compared as written: only whitespace around one line is trimmed, so a change inside
# quotes or a second line never matches.
def same(a: Any, b: str) -> bool:
    return isinstance(a, str) and a.splitlines() == [a] and a.strip() == b.strip()


def unwrap(command: Any) -> str | None:
    if not isinstance(command, str):
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if len(parts) == 3 and os.path.basename(parts[0]) in SHELLS and parts[1] in ("-lc", "-c"):
        return parts[2]
    return command


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def _claude_calls(raws: list[dict[str, Any]], command: str, denied: set[str]) -> list[tuple[str, int | None]]:
    calls: dict[str, dict[str, Any]] = {}
    refused: set[str] = set(denied)
    for raw in raws:
        kind = raw.get("type")
        if raw.get("parent_tool_use_id"):
            continue
        if kind == "AssistantMessage":
            for block in raw.get("content") or []:
                if block.get("type") == "ToolUseBlock" and block.get("name") == "Bash" \
                        and same((block.get("input") or {}).get("command"), command) and isinstance(block.get("id"), str):
                    calls[block["id"]] = {"background": bool((block.get("input") or {}).get("run_in_background"))}
        elif kind == "UserMessage" and isinstance(raw.get("content"), list):
            for block in raw["content"]:
                if block.get("type") == "ToolResultBlock" and block.get("tool_use_id") in calls:
                    calls[block["tool_use_id"]]["result"] = (block, raw.get("tool_use_result"))
        elif kind == "ResultMessage":
            for denial in raw.get("permission_denials") or []:
                if isinstance(denial, dict) and isinstance(denial.get("tool_use_id"), str):
                    refused.add(denial["tool_use_id"])
    out = []
    for call_id, call in calls.items():
        if call_id in refused or call["background"] or "result" not in call:
            out.append(("not_run", None))
            continue
        block, output = call["result"]
        if block.get("is_error") is True:
            found = EXIT_LINE.match(_text(block.get("content")).split("\n", 1)[0].strip())
            out.append(("failed", int(found.group(1))) if found else ("not_run", None))
        elif not isinstance(output, dict) or output.get("interrupted") is not False:
            out.append(("not_run", None))
        elif output.get("returnCodeInterpretation"):
            out.append(("failed", None))
        elif any(output.get(key) for key in NOT_A_FOREGROUND_PASS):
            out.append(("not_run", None))
        else:
            out.append(("passed", None))
    return out


def _codex_calls(raws: list[dict[str, Any]], command: str, denied: set[str],
                 cwd: str) -> list[tuple[str, int | None]]:
    out = []
    expected_cwd = os.path.realpath(cwd)
    for raw in raws:
        item = (raw.get("params") or {}).get("item") or {}
        if raw.get("method") != "item/completed" or item.get("type") != "commandExecution":
            continue
        if not same(unwrap(item.get("command")), command):
            continue
        actual_cwd = item.get("cwd")
        if not isinstance(actual_cwd, str) or not os.path.isabs(actual_cwd) or "\0" in actual_cwd \
                or os.path.realpath(actual_cwd) != expected_cwd:
            out.append(("not_run", None))
            continue
        code, status = item.get("exitCode"), item.get("status")
        if item.get("id") in denied or status == "declined" or not isinstance(code, int) or isinstance(code, bool):
            out.append(("not_run", None))
        elif code == 0 and status == "completed":
            out.append(("passed", 0))
        else:
            out.append(("failed", code))
    return out


# The verdict comes from the provider's record of the tool call, never from the model's words or the
# command's output (docs/security-model.md, "Test verdicts").
def verdict(provider: str, raws: list[dict[str, Any]], command: str, denied: set[str],
            before: str | None, after: str | None, cwd: str, *, claude_project_cwd: bool = False) -> dict[str, Any]:
    if provider == "codex":
        calls = _codex_calls(raws, command, denied, cwd)
    elif provider == "claude-code" and claude_project_cwd is True:
        calls = _claude_calls(raws, command, denied)
    else:
        log.warning("test turn: no provider confirmation that Bash starts in the worktree")
        return {"command": command, "status": "not_run", "exit_code": None}
    if not calls:
        status, code = "not_run", None
        log.warning("test turn: no shell call of the test command in the provider's record")
    elif any(v == "failed" for v, _ in calls):
        status, code = next((v, c) for v, c in calls if v == "failed")
    elif all(v == "passed" for v, _ in calls):
        status, code = calls[-1]
    else:
        status, code = "not_run", None
        log.warning("test turn: the test command was denied, not finished or has no exit status")
    if status in ("passed", "failed"):
        if not before or not after:
            log.warning("test turn: no worktree snapshot to compare; the verdict is not_run")
            status, code = "not_run", None
        elif before != after:
            status = "tree_changed"
    if len(calls) > 1:
        log.info("test turn: the test command was called %d times; every call counts", len(calls))
    log.info("test turn: %s (exit code %s)", status, code)
    return {"command": command, "status": status, "exit_code": code}

