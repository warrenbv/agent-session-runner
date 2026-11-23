"""The scripted provider used by every test and by the demo.

It replays raw messages in each vendor's real shape from a JSON script, so the real normaliser,
relay, driver, test-turn and hand-off code run on them. Script keys:

    provider          "claude-code" or "codex"
    turns             a list of steps per turn, and default_turn for turns past the list
    follows           steps replayed when the driver reads on after a turn (Claude)
    on_interrupt      raw messages replayed when a turn is interrupted
    steer             whether steer() succeeds (default true)
    schema_per_turn   behave as Codex does and read the schema per turn
    refuse_settings   {field: reason} for change_settings to refuse
    ids, thread_id    report a conversation id. refuse_resume and cost_carries_over act as the real providers

A step is a raw message or a directive: `sleep`, `hang`, `wait_steer`, `write` (a file in the
worktree), `approval` (with `on_allow` and `on_deny` branches) and `refusal`.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any, AsyncIterator

from runner import normalise
from runner import refusals
from runner import settings as words
from runner.events import Event
from runner.providers.base import CODING_KIND, Change, ProviderContext, ToolNames, applied, refused

DEFAULT_ON_INTERRUPT = {
    "claude-code": [{"type": "ResultMessage", "subtype": "error_during_execution", "is_error": True,
                     "num_turns": 1, "duration_ms": 1, "duration_api_ms": 1, "session_id": "fake",
                     "terminal_reason": "aborted_streaming"}],
    "codex": [{"method": "turn/completed", "params": {"threadId": "fake", "turn": {
        "id": "fake-turn", "items": [], "status": "interrupted"}}}],
}


def _replace(value: Any, needle: str, text: str) -> Any:
    if isinstance(value, str):
        return value.replace(needle, text)
    if isinstance(value, list):
        return [_replace(v, needle, text) for v in value]
    if isinstance(value, dict):
        return {k: _replace(v, needle, text) for k, v in value.items()}
    return value


class RefusedResume(RuntimeError):

    def __init__(self, text: str, errors: list[str] | None = None) -> None:
        super().__init__(text)
        self.errors = errors


class FakeProvider:
    def __init__(self, fixture: dict[str, Any] | str | pathlib.Path) -> None:
        if not isinstance(fixture, dict):
            fixture = json.loads(pathlib.Path(fixture).read_text(encoding="utf-8"))
        self.name = fixture["provider"]
        self.enforces_max_turns = fixture.get("enforces_max_turns", self.name == "claude-code")
        self.steer_ok = fixture.get("steer", True)
        self.schema_per_turn = fixture.get("schema_per_turn", False)
        self.schemas: list[Any] = []
        self.turns = fixture.get("turns") or []
        self.default_turn = fixture.get("default_turn") or []
        self.follows = fixture.get("follows") or []
        self.followed = 0
        self.on_interrupt = fixture.get("on_interrupt") or DEFAULT_ON_INTERRUPT[self.name]
        self.ctx: ProviderContext | None = None
        self.bash_starts_in_worktree = False
        self.tool_names = ToolNames()
        self.context = normalise.ContextMeter(self.name)
        self.sent: list[str] = []
        self.steered: list[str] = []
        self.modes: list[str] = []
        self.settings_calls: list[dict[str, str]] = []
        self.refuse_settings: dict[str, str] = fixture.get("refuse_settings") or {}
        self._mode = words.CODEX_START_MODE
        self.interrupts = 0
        self.terminated = False
        self.closed = False
        self._interrupted = asyncio.Event()
        self._terminated = asyncio.Event()
        self._steer = asyncio.Event()
        self.ids = bool(fixture.get("ids"))
        self.thread_id = str(fixture.get("thread_id") or "thread-fake-0001")
        self.refuse_resume = bool(fixture.get("refuse_resume"))
        self.cost_carries_over = fixture.get("cost_carries_over", self.name == "claude-code")
        self.session_id: str | None = None
        self.starts: list[str | None] = []

    async def start(self, ctx: ProviderContext) -> None:
        self.starts.append(ctx.resume)
        if ctx.resume and self.refuse_resume:
            if self.name == "claude-code":
                raise RefusedResume(f"Claude Code returned an error result: No conversation found with session ID: "
                                    f"{ctx.resume} (exit code: 1)",
                                    [f"No conversation found with session ID: {ctx.resume}"])
            raise RefusedResume(f"JSON-RPC error -32600: no rollout found for thread id {ctx.resume}")
        self.ctx = ctx
        # Stands in for the Claude provider's launch setting: a coding session's Bash starts in the worktree.
        self.bash_starts_in_worktree = self.name == "claude-code" and ctx.kind == CODING_KIND
        if self.ids and self.name == "codex":
            self.session_id = ctx.resume or self.thread_id

    def conversation_id(self) -> str | None:
        return self.session_id

    def resume_refused(self, exc: BaseException) -> bool:
        from runner.providers import claude, codex
        return (claude if self.name == "claude-code" else codex).resume_refused(exc)

    def normalise(self, raw: dict[str, Any]) -> list[Event]:
        ids = bool(self.ctx and self.ctx.drafts)
        self.context.see(raw)
        return self.context.annotate(self.tool_names.annotate(normalise.NORMALISERS[self.name](raw, ids=ids)))

    async def _wait(self, event: asyncio.Event, seconds: float) -> bool:
        try:
            await asyncio.wait_for(event.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    def _approval_payload(self, step: dict[str, Any]) -> dict[str, Any]:
        if self.name == "claude-code":
            return normalise.claude_approval(step["tool_name"], step.get("input") or {}, step["tool_use_id"],
                                             step.get("agent_id"))
        return normalise.codex_approval(step["method"], step.get("params") or {})

    async def turn(self, text: str) -> AsyncIterator[dict[str, Any]]:
        assert self.ctx is not None, "start() first"
        self.sent.append(text)
        self.schemas.append(self.ctx.output_schema)
        index = len(self.sent) - 1
        script = self.turns[index] if index < len(self.turns) else self.default_turn
        if self.name == "claude-code":
            yield {"type": "QuerySent"}
        async for raw in self._play(script):
            if self.ctx.raw_frame is not None:
                self.ctx.raw_frame(raw)
            yield raw

    async def follow(self) -> AsyncIterator[dict[str, Any]]:
        assert self.ctx is not None, "start() first"
        index = self.followed
        self.followed += 1
        script = self.follows[index] if index < len(self.follows) else [{"fake": "sleep", "seconds": 3600}]
        async for raw in self._play(script):
            if self.ctx.raw_frame is not None:
                self.ctx.raw_frame(raw)
            yield raw

    async def _play(self, script: list[Any]) -> AsyncIterator[dict[str, Any]]:
        self._interrupted.clear()
        self._steer.clear()
        for step in script:
            if self._terminated.is_set() or self._interrupted.is_set():
                break
            directive = step.get("fake") if isinstance(step, dict) else None
            if directive == "sleep":
                await self._wait(self._interrupted, float(step["seconds"]))
            elif directive == "hang":
                await self._terminated.wait()
            elif directive == "wait_steer":
                await self._wait(self._steer, float(step.get("timeout", 5)))
            elif directive == "refusal":
                await refusals.note(self.ctx, refusals.line("Refused.", "command", step["command"]))
            elif directive == "write":
                target = pathlib.Path(self.ctx.cwd, step["path"]).resolve()
                if not target.is_relative_to(pathlib.Path(self.ctx.cwd).resolve()):
                    raise ValueError("a fake write stays in the worktree")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(step["text"], encoding="utf-8")
            elif directive == "approval":
                decision = await self.ctx.request_approval(self._approval_payload(step))
                for raw in step.get("on_allow" if decision.allow else "on_deny") or []:
                    yield _replace(raw, "$DENY_MESSAGE", decision.message)
            else:
                if self.ids and self.name == "claude-code" and isinstance(step, dict) \
                        and isinstance(step.get("session_id"), str) and step["session_id"]:
                    self.session_id = step["session_id"]
                yield step
        if self._terminated.is_set():
            return
        if self._interrupted.is_set():
            for raw in self.on_interrupt:
                yield raw

    async def steer(self, text: str) -> bool:
        if not self.steer_ok:
            return False
        self.steered.append(text)
        self._steer.set()
        return True

    async def interrupt(self) -> None:
        self.interrupts += 1
        self._interrupted.set()

    async def set_permission_mode(self, mode: str) -> None:
        self.modes.append(mode)

    def settings_now(self) -> dict[str, str | None]:
        assert self.ctx is not None
        if self.name == "claude-code":
            mode = words.CLAUDE_MODE_WORD.get(self.ctx.permission_mode, self.ctx.permission_mode)
        else:
            mode = self._mode
        return {"model": self.ctx.model, "effort": self.ctx.effort, "mode": mode,
                "thinking": "on" if self.ctx.thinking else "off"}

    async def change_settings(self, changes: dict[str, str]) -> dict[str, Change]:
        assert self.ctx is not None
        self.settings_calls.append(dict(changes))
        out: dict[str, Change] = {}
        for field, word in changes.items():
            if field in self.refuse_settings:
                out[field] = refused(self.refuse_settings[field])
                continue
            if field == "model":
                value = word
                self.ctx.model = word
            elif field == "effort":
                value = words.CLAUDE_EFFORT[word] if self.name == "claude-code" \
                    else words.CODEX_EFFORT.get(word) or words.codex_max_effort(None)
                self.ctx.effort = value
            elif field == "thinking":
                self.ctx.thinking = word == "on"
                value = word
            else:
                if self.name == "claude-code":
                    value = words.CLAUDE_MODE[word]
                    self.ctx.permission_mode = value
                else:
                    value = words.codex_mode_value(word)
                    self._mode = word
            out[field] = applied(value)
        return out

    async def terminate(self) -> None:
        self.terminated = True
        self._terminated.set()

    async def close(self) -> None:
        self.closed = True

