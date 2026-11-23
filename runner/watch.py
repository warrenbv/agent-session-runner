"""Coding-session liveness and loop detection, read from raw provider frames.

The watch notices a session whose provider has sent nothing for a while (first quiet, then stuck)
and the same failing command or the same refusal repeated many times. It posts a runner note on the
stream for each, and when a bounded session loops it is wrapped up (session.py) rather than left to
spend its whole budget.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from runner.events import Event, clip

QUIET_SECONDS = 10 * 60
STUCK_SECONDS = 20 * 60
OPEN_STUCK_SECONDS = 30 * 60
WATCH_INTERVAL = 1.0
FAILURES = 8
REFUSALS = 5
COMMAND_CAP = 300
LOOP_NOTE = (
    "The runner has seen this command fail {count} times with no file changed in between: "
    "`{command}`. Change approach, or stop and write in your summary what is blocking you.")
REFUSAL_LOOP_NOTE = (
    "The runner has refused this command {count} times: `{command}`. It will be refused again. "
    "Change approach, or stop and write in your summary what is blocking you.")
LOOP_WRAP_REASON = "The session is wrapping up after looping."
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "fileChange"})
CODEX_TOOLS = frozenset({"commandExecution", "fileChange", "collabAgentToolCall", "mcpToolCall",
                         "dynamicToolCall", "webSearch", "subAgentActivity"})


@dataclass
class Call:

    tool: str
    began: float
    command: str = ""


class Watch:

    def __init__(self, provider: str, clock: Callable[[], float] = time.monotonic) -> None:
        self.provider, self.clock = provider, clock
        self.last_frame = clock()
        self.began = self.last_frame
        self.silent_from = self.last_frame
        self.open: dict[str, Call] = {}
        self.stages: set[str] = set()
        self.counts: dict[tuple[str, str], int] = {}
        self.notices: deque[tuple[Event, str | None]] = deque()
        self.wrap_requested = False

    def begin(self) -> None:
        self.began = self.clock()
        self.open.clear()

    def _post(self, purpose: str, **fields: Any) -> Event:
        event = Event("message", {"role": "runner", "purpose": purpose, **fields})
        self.notices.append((event, None))
        return event

    def tick(self) -> None:
        now = self.clock()
        seconds = int(now - max(self.last_frame, self.began))
        stage = "stuck" if seconds >= (OPEN_STUCK_SECONDS if self.open else STUCK_SECONDS) else "quiet"
        if seconds < QUIET_SECONDS or (self.open and stage == "quiet") or stage in self.stages:
            return
        opened = min(self.open.values(), key=lambda call: call.began) if self.open else None
        self.stages.add(stage)
        self.silent_from = max(self.last_frame, self.began)
        self._post("quiet", stage=stage, seconds=seconds,
                   open={"tool": opened.tool, "seconds": int(now - opened.began)} if opened else None)

    def frame(self, raw: dict[str, Any], *, count: bool = True) -> None:
        now = self.clock()
        if self.stages:
            self._post("alive", quiet_seconds=int(now - self.silent_from))
            self.stages.clear()
        self.last_frame = now
        if self.provider == "claude-code":
            for block in raw.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "ToolUseBlock":
                    inputs = block.get("input") or {}
                    self._start(block.get("id"), block.get("name", ""), inputs.get("command", ""))
                elif block.get("type") == "ToolResultBlock":
                    self._complete(block.get("tool_use_id"), bool(block.get("is_error")), count=count)
        else:
            item = (raw.get("params") or {}).get("item") or {}
            kind, method = item.get("type"), raw.get("method")
            if method == "item/started" and kind in CODEX_TOOLS:
                tool = "command" if kind == "commandExecution" else item.get("tool") or kind
                self._start(item.get("id"), tool, item.get("command", ""))
            elif method == "item/completed":
                failed = item.get("status") in ("failed", "declined") or item.get("exitCode") not in (None, 0)
                self._complete(item.get("id"), failed, count=count)

    def _start(self, key: Any, tool: str, command: Any) -> None:
        if key and key not in self.open:
            self.open[key] = Call(tool, self.clock(), command.strip() if isinstance(command, str) else "")

    def _complete(self, key: Any, failed: bool, *, count: bool) -> None:
        call = self.open.pop(key, None)
        if not call or not count:
            return
        if call.tool in EDIT_TOOLS and not failed:
            self.counts = {k: v for k, v in self.counts.items() if k[0] != "failing"}
        elif failed and call.tool in ("Bash", "command"):
            self.repeated("failing", call.command)

    def repeated(self, kind: str, command: Any) -> None:
        if not isinstance(command, str) or not command.strip() or self.wrap_requested:
            return
        command = command.strip()
        key = kind, command
        count = self.counts[key] = self.counts.get(key, 0) + 1
        threshold = FAILURES if kind == "failing" else REFUSALS
        if count not in (threshold, 2 * threshold):
            return
        wrapping = count == 2 * threshold
        self.wrap_requested = wrapping
        event = self._post("looping", kind=kind, count=count, command=clip(command, COMMAND_CAP),
                           action="wrapping_up" if wrapping else "nudged")
        if not wrapping:
            template = LOOP_NOTE if kind == "failing" else REFUSAL_LOOP_NOTE
            self.notices[-1] = (event, template.format(count=count, command=command))

    def drain(self) -> list[tuple[Event, str | None]]:
        found = list(self.notices)
        self.notices.clear()
        return found

