"""Helpers shared by several test modules: recorded-message builders for both providers, fake forges, the
event contract table, a fake clock, and drivers for a session, a research session and the provider stubs."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import shlex
import time
import uuid
from typing import Any

import pytest

from conftest import fast_timing, typed_result
from runner import handoff as handoff_mod
from runner.control_plane import ResearchClient
from runner.providers.base import RESEARCH_KIND, ProviderContext
from runner.providers.fake import FakeProvider
from runner.relay import Decision
from runner.research_out import ResearchOut
from runner.session import Caps, Options, Session


CLAUDE, CODEX = {"provider": "claude-code"}, {"provider": "codex"}
NUM = (int, float)
S, I, B, D, L = (str,), (int,), (bool,), (dict,), (list,)

CONTRACT: list[tuple[str, dict, str, tuple, bool]] = [
    ("turn", {}, "phase", S, True),
    ("turn", {"phase": "started"}, "model", S, False),
    ("turn", {"phase": "started"}, "effort", S, False),
    ("turn", {"phase": "started"}, "mode", S, False),
    ("turn", {"phase": "completed"}, "interrupted", B, True),
    ("turn", {"phase": "completed"}, "is_error", B, True),
    ("turn", {"phase": "completed"}, "duration_ms", NUM, False),
    ("turn", {"phase": "completed", **CLAUDE}, "duration_ms", NUM, True),
    ("turn", {"phase": "completed", **CLAUDE}, "subtype", S, True),
    ("turn", {"phase": "completed", **CLAUDE}, "session_id", S, True),
    ("turn", {"phase": "completed", **CLAUDE}, "structured_output", D, False),
    ("turn", {"phase": "completed", **CODEX}, "status", S, True),
    ("turn", {"phase": "completed", **CODEX}, "turn_id", S, True),
    ("message", {}, "role", S, True),
    ("message", {}, "text", S, True),
    ("message", {"role": "assistant", **CLAUDE}, "parent_tool_use_id", S, False),
    ("message", {"role": "assistant", **CODEX}, "item_id", S, True),
    ("message", {"role": "assistant"}, "message_id", S, False),
    ("message", {"role": "assistant"}, "part", I, False),
    ("message", {"role": "assistant"}, "parts", I, False),
    ("message", {"role": "user"}, "author", S, True),
    ("message", {"role": "user"}, "mode", S, True),
    ("message", {"role": "user"}, "entry_id", I, False),
    ("message", {"role": "user"}, "purpose", S, False),
    ("message", {"role": "user", "purpose": "tests-verdict"}, "reason", S, False),
    ("message", {"role": "runner"}, "purpose", S, True),
    ("message", {"role": "runner", "purpose": "withdraw"}, "outcome", S, True),
    ("message", {"role": "runner", "purpose": "withdraw"}, "entry_id", I, True),
    ("message", {"role": "runner", "purpose": "settings"}, "applied", D, True),
    ("message", {"role": "runner", "purpose": "settings"}, "held", D, True),
    ("message", {"role": "runner", "purpose": "settings"}, "refused", D, True),
    ("message", {"role": "runner", "purpose": "settings"}, "control_id", I, False),
    ("message", {"role": "runner", "purpose": "provider_session"}, "id", S, True),
    ("message", {"role": "runner", "purpose": "resumed_fresh"}, "role", S, True),
    ("message", {"role": "user", "purpose": "resume"}, "text", S, True),
    ("message", {"role": "runner", "purpose": "baseline"}, "version", I, True),
    ("message", {"role": "runner", "purpose": "baseline"}, "sha", S, True),
    ("message", {"role": "runner", "purpose": "baseline"}, "applied", S, True),
    ("message", {"role": "runner", "purpose": "baseline"}, "project_file", S, False),
    ("message", {"role": "runner", "purpose": "baseline"}, "project_file_bytes", I, True),
    ("message", {"role": "runner", "purpose": "baseline"}, "cut", B, True),
    ("message", {"role": "user", "purpose": "wrap_up"}, "why", S, True),
    ("message", {"role": "runner", "purpose": "quiet"}, "stage", S, True),
    ("message", {"role": "runner", "purpose": "quiet"}, "seconds", I, True),
    ("message", {"role": "runner", "purpose": "quiet"}, "open", (dict, type(None)), False),
    ("message", {"role": "runner", "purpose": "alive"}, "quiet_seconds", I, True),
    ("message", {"role": "runner", "purpose": "looping"}, "kind", S, True),
    ("message", {"role": "runner", "purpose": "looping"}, "count", I, True),
    ("message", {"role": "runner", "purpose": "looping"}, "command", S, True),
    ("message", {"role": "runner", "purpose": "looping"}, "action", S, True),
    ("tool", {}, "phase", S, True),
    ("tool", {}, "id", S, True),
    ("tool", {}, "name", S, True),
    ("tool", {"phase": "started"}, "input_preview", S, True),
    ("tool", {"phase": "completed"}, "is_error", B, True),
    ("tool", {"phase": "completed"}, "output_preview", S, False),
    ("tool", {"phase": "started", "name": "AskUserQuestion"}, "questions", L, True),
    ("approval", {}, "request_id", S, True),
    ("approval", {}, "tool", S, True),
    ("approval", {}, "input_preview", S, False),
    ("approval", {}, "requester", S, False),
    ("approval", {"tool": "AskUserQuestion"}, "input_preview", S, True),
    ("approval", {"tool": "AskUserQuestion"}, "questions", L, True),
    ("approval", {"tool": "request_user_input"}, "input_preview", S, True),
    ("approval", {"tool": "request_user_input"}, "questions", L, True),
    ("subagent", {}, "phase", S, True),
    ("diff", {}, "scope", S, True),
    ("diff", {"scope": "turn"}, "files", I, True),
    ("diff", {"scope": "turn"}, "paths", L, True),
    ("diff", {"scope": "turn"}, "stat", S, True),
    ("diff", {"scope": "turn"}, "patch", S, True),
    ("diff", {"scope": "turn"}, "patch_truncated", B, True),
    ("diff", {"scope": "session_file"}, "path", S, True),
    ("diff", {"scope": "session_file"}, "old_path", S, False),
    ("diff", {"scope": "session_file"}, "status", S, True),
    ("diff", {"scope": "session_file"}, "added", I, True),
    ("diff", {"scope": "session_file"}, "removed", I, True),
    ("diff", {"scope": "session_file"}, "binary", B, True),
    ("diff", {"scope": "session_file"}, "sig", S, True),
    ("diff", {"scope": "session_file"}, "patch", S, True),
    ("diff", {"scope": "session_file"}, "too_large", B, True),
    ("diff", {"scope": "session"}, "base", S, True),
    ("diff", {"scope": "session"}, "to", S, True),
    ("diff", {"scope": "session"}, "files", I, True),
    ("diff", {"scope": "session"}, "added", I, True),
    ("diff", {"scope": "session"}, "removed", I, True),
    ("diff", {"scope": "session"}, "listed", L, True),
    ("diff", {"scope": "session"}, "complete", B, True),
    ("usage", CLAUDE, "total_cost_usd", NUM, True),
    ("usage", CLAUDE, "model_usage", D, False),
    ("usage", CODEX, "total", D, True),
    ("limit", {}, "limit", S, True),
    ("limit", {}, "source", S, True),
    ("limit", {"limit": "rate_window", **CLAUDE}, "window", S, True),
    ("limit", {"limit": "rate_window", **CLAUDE}, "status", S, True),
    ("limit", {"limit": "rate_window", **CLAUDE}, "utilization", NUM, False),
    ("limit", {"limit": "rate_window", **CODEX}, "primary", D, True),
    ("limit", {"limit": "rate_window", **CODEX}, "limit_id", S, False),
    ("end", {}, "reason", S, True),
    ("end", {}, "turns", I, True),
    ("end", {}, "cost_usd", NUM, False),
    ("end", {}, "ended_by", S, False),
    ("end", {}, "wrapped_up", S, False),
    ("end", {}, "tests_skipped", S, False),
    ("end", {"reason": "completed"}, "checks", D, True),
]


TEXTLESS = ("provider_session", "resumed_fresh", "baseline", "quiet", "alive", "looping")


def contract_matches(when: dict, provider: str, payload: dict) -> bool:
    return all((provider if k == "provider" else payload.get(k)) == v for k, v in when.items())


def violations(kind: str, provider: str, payload: dict) -> list[str]:
    out = []
    for row_kind, when, field, types, required in CONTRACT:
        if row_kind != kind or not contract_matches(when, provider, payload):
            continue
        if (kind, field) == ("message", "text") and payload.get("role") == "runner" \
                and payload.get("purpose") in TEXTLESS:
            continue
        value = payload.get(field)
        if value is None:
            if required:
                out.append(f"{kind} {when}: {field} missing")
        elif not isinstance(value, types) or (types is NUM and isinstance(value, bool)):
            out.append(f"{kind} {when}: {field} is {type(value).__name__}")
    return out


COMMAND = "pytest tests/test_backup.py"


REFUSAL_NUDGE = (
    "The runner has refused this command 5 times: `pytest tests/test_backup.py`. It will be refused again. "
    "Change approach, or stop and write in your summary what is blocking you.")


class Clock:

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def call(provider, key="a", tool=None, command=COMMAND):
    if provider == "claude-code":
        return {"type": "AssistantMessage", "content": [{"type": "ToolUseBlock", "id": key,
                "name": tool or "Bash", "input": {"command": command}}]}
    kind = tool or "commandExecution"
    return {"method": "item/started", "params": {"item": {"id": key, "type": kind, "command": command,
                                                           "tool": "wait" if kind == "collabAgentToolCall" else None}}}


def complete(provider, key="a", tool=None, failed=True, command=COMMAND):
    if provider == "claude-code":
        return {"type": "UserMessage", "content": [{"type": "ToolResultBlock", "tool_use_id": key,
                                                    "is_error": failed, "content": "exit"}]}
    return {"method": "item/completed", "params": {"item": {"id": key, "type": tool or "commandExecution",
            "command": command, "status": "failed" if failed else "completed", "exitCode": 1 if failed else 0}}}


def partial(provider):
    if provider == "claude-code":
        return {"type": "StreamEvent", "event": {"type": "content_block_delta", "delta": {"text": "x"}}}
    return {"method": "item/commandExecution/outputDelta", "params": {"itemId": "a", "delta": "x"}}


def watch_done(provider):
    if provider == "claude-code":
        return {"type": "ResultMessage", "subtype": "success", "session_id": "s", "num_turns": 1,
                "is_error": False, "duration_ms": 1, "structured_output": typed_result("wrapped")}
    return {"method": "turn/completed", "params": {"turn": {"id": "t", "status": "completed"}}}


def failures(provider, n, command=COMMAND):
    return [raw for i in range(n) for raw in (call(provider, str(i), command=command),
                                             complete(provider, str(i), command=command))]


def result(structured=None, subtype="success", cost=0.01):
    return {"type": "ResultMessage", "subtype": subtype, "duration_ms": 1, "duration_api_ms": 1,
            "is_error": subtype != "success", "num_turns": 1, "session_id": "s", "total_cost_usd": cost,
            "terminal_reason": "completed", "structured_output": structured}


def write(path="a.txt", text="work\n"):
    return {"fake": "write", "path": path, "text": text}


def claude(*turns, default=None):
    return {"provider": "claude-code", "turns": [list(t) for t in turns],
            "default_turn": default or [say("Nothing more."), result()]}


def codex_turn(*items, tid="turn_w"):
    return ([{"method": "turn/started", "params": {"threadId": "thr", "turn": {"id": tid, "items": [],
                                                                                 "status": "inProgress"}}}]
            + list(items)
            + [{"method": "turn/completed", "params": {"threadId": "thr", "turn": {"id": tid, "items": [],
                                                                                     "status": "completed"}}}])


def codex_msg(text, item_id="m"):
    return {"method": "item/completed", "params": {"threadId": "thr", "turnId": "t",
                                                   "item": {"type": "agentMessage", "id": item_id, "text": text}}}


TEST_CMD = "uv run pytest -q"


class FakeForge:
    def __init__(self):
        self.pushed, self.prs = [], []

    def push(self, cwd, branch):
        self.pushed.append(branch)

    def open_pr(self, branch, base, title, body):
        self.prs.append({"title": title, "body": body})
        return handoff_mod.Opened("https://github.example/owner/repo/pull/7", 7)


def bash(command, tool_id="toolu_t1", parent=None, **extra_input):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": parent,
            "content": [{"type": "ToolUseBlock", "id": tool_id, "name": "Bash",
                         "input": {"command": command, "description": "Run the tests", **extra_input}}]}


def bash_result(tool_id="toolu_t1", ok=True, code=1, parent=None, output=None):
    if ok:
        content, is_error = "3 passed", False
        tur = output or {"stdout": "3 passed", "stderr": "", "interrupted": False, "isImage": False,
                         "noOutputExpected": False}
    else:
        content, is_error = f"Exit code {code}\nFAILED tests/test_x.py::test_y\nExit code 0", True
        tur = f"Error: {content}"
    return {"type": "UserMessage", "parent_tool_use_id": parent, "tool_use_result": tur,
            "content": [{"type": "ToolResultBlock", "tool_use_id": tool_id, "content": content,
                         "is_error": is_error}]}


def codex_cmd(command, status="completed", code=0, cwd=None, item_id="call_t", phase="completed"):
    item = {"type": "commandExecution", "id": item_id, "command": f"/usr/bin/bash -lc {shlex.quote(command)}",
            "commandActions": [{"command": command, "type": "unknown"}], "status": status, "exitCode": code,
            "source": "agent"}
    if cwd:
        item["cwd"] = cwd
    return {"method": f"item/{phase}", "params": {"threadId": "thr", "turnId": "t", "item": item}}


class RecordingForge:
    def __init__(self):
        self.pushed, self.prs = [], []

    def push(self, cwd, branch):
        self.pushed.append(branch)

    def open_pr(self, branch, base, title, body):
        self.prs.append({"branch": branch, "base": base, "title": title, "body": body})
        return handoff_mod.Opened("https://github.example/owner/repo/pull/7", 7)


class ReusingForge(RecordingForge):
    def __init__(self, found=None, push_error=None):
        super().__init__()
        self.found, self.push_error, self.asked = found, push_error, []
        self.updated = []

    def push(self, cwd, branch):
        if self.push_error is not None:
            raise self.push_error
        super().push(cwd, branch)

    def open_pr_for(self, branch):
        self.asked.append(branch)
        return self.found

    def update_pr(self, number, body):
        self.updated.append((number, body))


def handoff_typed(n=1, **fields):
    return {"summary": "Changed the backup", "files_changed": [], "tests_run": [], "tests_claim": None,
            "open_questions": [],
            "next_steps": [], "branch": "agent/fix-backup", "items": [
                {"label": f"Request {i}", "status": "unverified", "where": ""} for i in range(n)], **fields}


def driven_result(cost=0.01, subtype="success", **extra):
    return {"type": "ResultMessage", "subtype": subtype, "duration_ms": 1, "duration_api_ms": 1,
            "is_error": subtype != "success", "num_turns": 1, "session_id": "s", "total_cost_usd": cost,
            "terminal_reason": "completed", "structured_output": driven_typed("Work"), **extra}


def driven_typed(summary, files=()):
    return typed_result(summary, files_changed=list(files))


def tool(tool_id="toolu_9", name="Read"):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "ToolUseBlock", "id": tool_id, "name": name, "input": {"file_path": "x"}}]}


def driven_claude(*turns, default=None, **extra):
    return {"provider": "claude-code", "turns": [list(t) for t in turns],
            "default_turn": default or [say("Done."), driven_result()], **extra}


def drive(fixture, client, cwd, *, driven=True, idle=30.0, caps=None, timing=None, forge=None, test_cmd=None,
          prompt="Fix the backup job.", timeout=30):
    provider = FakeProvider(fixture)
    hand = None
    if forge is not None:
        def hand(res):
            return handoff_mod.hand_off(str(cwd), "fix-backup", "main", res, forge)
    session = Session(provider, client, Options(cwd=str(cwd), driven=driven, idle_seconds=idle, test_cmd=test_cmd),
                      caps or Caps(wall_clock=30, grace=1), timing or fast_timing(), handoff=hand)
    outcome = asyncio.run(asyncio.wait_for(session.run(prompt), timeout=timeout))
    return outcome, provider, session


def on_waiting(orch, n, action):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == n, lambda e: action())


def codex_text_turn(text, n):
    return [{"method": "turn/started", "params": {"threadId": "t", "turn": {"id": n, "items": [], "status": "inProgress"}}},
            {"method": "item/completed", "params": {"threadId": "t", "turnId": n,
                                                    "item": {"type": "agentMessage", "id": f"m{n}", "text": text}}},
            {"method": "turn/completed", "params": {"threadId": "t", "turn": {"id": n, "items": [],
                                                                              "status": "completed"}}}]


def typed_codex():
    return {"provider": "codex", "schema_per_turn": True,
            "default_turn": codex_text_turn(json.dumps(typed_result()), "1")}


def typed_claude():
    return {"provider": "claude-code", "default_turn": [
        {"type": "ResultMessage", "subtype": "success", "duration_ms": 1, "duration_api_ms": 1, "is_error": False,
         "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01, "terminal_reason": "completed",
         "structured_output": typed_result()}]}


def research_result(subtype="success"):
    return {"type": "ResultMessage", "subtype": subtype, "duration_ms": 1, "duration_api_ms": 1,
            "is_error": subtype != "success", "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01,
            "terminal_reason": "completed"}


SYNTHESIS = ("# Synthesis\n\nIntro.\n\n## C1 The first claim\n\n" + "Evidence é. " * 9000
             + "\n\n## C2: The second claim\n\nMore.\n\n## Methods and limits\n\nSearched.\n")
CLAIMS = [{"id": f"C{n}", "text": f"Claim {n}", "confidence": "high", "how_found": "search",
           "sources": [{"url": "https://example.com/a", "title": "A", "accessed": "2026-11-12"}]} for n in (1, 2)]


def provider_files(claims=CLAIMS, synthesis=SYNTHESIS):
    return [write("synthesis.md", synthesis), write("claims.json", json.dumps(claims))]


def research_run(orch, client, tmp_path, steps, role="provider", caps=None, max_subagents=3, max_searches=10,
        provider_cls=FakeProvider, name="claude-code", later=(), follows=(), timing=None):
    scratch = tmp_path / "research" / "5"
    scratch.mkdir(parents=True, exist_ok=True)
    provider = provider_cls({"provider": name, "turns": [steps, *later], "follows": list(follows)})
    out = ResearchOut(ResearchClient(client, 5, sleep=lambda _s: None), role, str(scratch))
    session = Session(provider, client, Options(cwd=str(scratch), output_schema=None, diff=False, kind=RESEARCH_KIND,
                                                max_subagents=max_subagents, max_searches=max_searches),
                      caps or Caps(max_turns=10, wall_clock=30, grace=1), timing or fast_timing(), research=out)
    began = time.monotonic()
    outcome = asyncio.run(asyncio.wait_for(session.run("Research the question."), timeout=30))
    return outcome, provider, session, scratch, time.monotonic() - began


EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def say(text, message_id=None):
    out = {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
           "content": [{"type": "TextBlock", "text": text}]}
    if message_id:
        out["message_id"] = message_id
    return out


def settings_done():
    return {"type": "ResultMessage", "subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
            "is_error": False, "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01,
            "terminal_reason": "completed", "structured_output": typed_result()}


def run_settings_session(fixture, client, cwd, driven=True, drafts=False, **options):
    provider = FakeProvider(fixture)
    session = Session(provider, client, Options(cwd=str(cwd), driven=driven, idle_seconds=30, diff=False,
                                                drafts=drafts, **options),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("first"), timeout=30))
    return provider, session


def settings_acks(orch):
    return orch.of("message", role="runner", purpose="settings")


def turn_starts(orch):
    return orch.of("turn", phase="started")


@dataclasses.dataclass
class ResultMessage:
    subtype: str
    session_id: str


SESSION = str(uuid.uuid4())


class StubClient:
    made: list["StubClient"] = []
    refuse_modes: tuple = ()

    def __init__(self, options):
        self.options, self.queries, self.calls = options, [], []
        StubClient.made.append(self)

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def query(self, text):
        self.queries.append(text)

    async def receive_response(self):
        yield ResultMessage("success", SESSION)

    async def set_model(self, model):
        self.calls.append(("set_model", model))

    async def set_permission_mode(self, mode):
        if mode in StubClient.refuse_modes:
            raise Exception("Auto mode is unavailable for your plan")
        self.calls.append(("set_permission_mode", mode))


async def deny_approval(_payload):
    raise AssertionError("no approval expected")


@pytest.fixture
def claude_sdk_stub(monkeypatch, tmp_path, managed_policy):
    import claude_agent_sdk
    from runner.providers.claude import ClaudeProvider

    StubClient.made, StubClient.refuse_modes = [], ()
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", StubClient)
    provider = ClaudeProvider()
    provider.managed_settings = str(managed_policy)
    ctx = ProviderContext(request_approval=deny_approval, cwd=str(tmp_path), model="claude-opus-5-5", effort="high")
    return provider, ctx


def claude_go(provider, ctx, *steps):
    async def run():
        await provider.start(ctx)
        out = []
        for step in steps:
            if step == "turn":
                [r async for r in provider.turn("hello")]
            else:
                out.append(await provider.change_settings(step))
        return out
    return asyncio.run(run())


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class CodexStub:
    def __init__(self, models=None):
        self.started, self.models = [], models

    def turn_start(self, thread_id, text, params):
        self.started.append(params)
        return Obj(turn=Obj(id="t"))

    def next_turn_notification(self, turn_id):
        return Obj(method="turn/completed", payload=Obj(params={"turn": {"id": turn_id, "status": "completed"}}))

    def model_list(self, include_hidden=False):
        if self.models is None:
            raise RuntimeError("no list")
        return Obj(data=[Obj(model=name, id=name, is_default=default,
                             supported_reasoning_efforts=[Obj(reasoning_effort=e) for e in efforts])
                         for name, default, efforts in self.models])


def codex_on_stub(stub, model=None):
    from runner.providers.codex import CodexProvider
    provider = CodexProvider()
    provider.client, provider.thread_id = stub, "th"
    provider.ctx = ProviderContext(request_approval=deny_approval, cwd="/tmp", model=model)
    return provider


def codex_go(provider, *steps):
    async def run():
        out = []
        for step in steps:
            if step == "turn":
                [r async for r in provider.turn("hello")]
            else:
                out.append(await provider.change_settings(step))
        return out
    return asyncio.run(run())


ESCALATION = {
    "kind": "command", "threadId": "t-1", "turnId": "u-1", "itemId": "exec-1", "startedAtMs": 1,
    "environmentId": "local", "reason": "May I create the requested file outside the workspace?",
    "command": "/usr/bin/bash -lc 'touch /home/agent/outside.txt'", "cwd": "/tmp/wt",
    "commandActions": [{"type": "unknown", "command": "touch /home/agent/outside.txt"}],
    "proposedExecpolicyAmendment": ["touch", "/home/agent/outside.txt"],
    "availableDecisions": ["accept", "cancel"]}


class Seen:

    def __init__(self, decision: Decision = Decision(True)) -> None:
        self.cards: list[dict[str, Any]] = []
        self.lines: list[dict[str, Any]] = []
        self.decision = decision

    async def card(self, payload: dict[str, Any]) -> Decision:
        self.cards.append(payload)
        return self.decision

    async def line(self, payload: dict[str, Any]) -> None:
        self.lines.append(payload)

    def ctx(self, **kw: Any) -> ProviderContext:
        return ProviderContext(request_approval=self.card, cwd="/tmp/wt", refusal_note=self.line, **kw)


def codex_answer(seen: Seen, method: str, params: dict[str, Any], mode: str | None = None, **kw: Any):
    from runner.providers.codex import CodexProvider

    provider = CodexProvider()

    async def go():
        provider.ctx = seen.ctx(**kw)
        provider.loop = asyncio.get_running_loop()
        if mode is not None:
            await provider.change_settings({"mode": mode})
        answer = await asyncio.to_thread(provider._on_server_request, method, params)
        await asyncio.sleep(0)
        return answer
    return asyncio.run(go()), provider
