"""The event stream's contract: every recorded fixture and every runner-written event carries the fields a
control plane reads for its kind."""

from __future__ import annotations

import asyncio
import json
import pathlib

import pytest

from conftest import fast_timing, git
from helpers import CONTRACT, contract_matches as _matches, violations
from runner import normalise
from runner.events import KINDS, PAYLOAD_CAP, size
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_handoff_checks_and_next_steps_are_the_plans_literal_contract():
    from runner import handoff_check
    from runner.providers.base import RESULT_SCHEMA

    assert handoff_check.check("Work", None, handoff_check.Status()) == {
        "version": 2, "findings": [{"key": "result_missing", "text": "The session returned no typed result."}]}
    assert RESULT_SCHEMA["properties"]["next_steps"] == {"type": "array", "items": {"type": "string"}}
    assert "next_steps" in RESULT_SCHEMA["required"]
    assert "tests_claim" in RESULT_SCHEMA["required"]
    assert RESULT_SCHEMA["properties"]["tests_claim"]["type"] == "object"


def _question(n=4):
    return {"questions": [{"question": f"Which layout for part {q}?", "header": f"Part {q}", "multiSelect": False,
                           "options": [{"label": f"Layout {o}", "description": "A layout. " * 30,
                                        "preview": "<div>" + "x" * 3000 + "</div>"} for o in range(4)]}
                          for q in range(n)]}


EXTRA = {
    "claude-code": [{"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None, "message_id": "msg_x",
                     "content": [{"type": "ToolUseBlock", "id": "toolu_q", "name": "AskUserQuestion",
                                  "input": _question()},
                                 {"type": "TextBlock", "text": "A long reply.\n\n" * 2000}]},
                    {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": "toolu_agent",
                     "content": [{"type": "TextBlock", "text": "A subagent's words."}]},
                    {"type": "UserMessage", "parent_tool_use_id": None,
                     "content": [{"type": "ToolResultBlock", "tool_use_id": "toolu_q", "content": "Layout 1"}]}],
    "codex": [{"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": "m_long",
                                                               "text": "line\n" * 5000, "phase": "final_answer"}}}],
}


def _corpus() -> list[tuple[str, str, str, dict]]:
    out = []
    for path in sorted(FIXTURES.glob("*.raw.json")):
        provider = "claude-code" if path.name.startswith("claude") else "codex"
        raw = json.loads(path.read_text(encoding="utf-8"))
        names = FakeProvider({"provider": provider})
        for message in raw["messages"] + EXTRA[provider]:
            out += [(path.name, provider, e.kind, e.payload) for e in names.normalise(message)]
        for request in raw["approvals"]:
            if provider == "claude-code":
                ctx = request["context"]
                payload = normalise.claude_approval(request["tool_name"], request["input"], ctx["tool_use_id"],
                                                    ctx.get("agent_id"), ctx.get("title"))
            else:
                payload = normalise.codex_approval(request["method"], request["params"])
            out.append((path.name, provider, "approval", payload))
    out.append(("synthetic", "claude-code", "approval",
                normalise.claude_approval("AskUserQuestion", _question(), "toolu_q")))
    asked = json.loads((FIXTURES / "question_recorded.json").read_text(encoding="utf-8"))["codex"]["request"]
    out.append(("question_recorded.json", "codex", "approval", normalise.codex_question(asked["params"])))
    return out


def _session_events(orch, client, worktree) -> list[tuple[str, str, str, dict]]:
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 1,
              lambda e: (orch.control("settings", model="claude-opus-5-5", effort="high", mode="plan"),
                         orch.queue("second"), orch.control("withdraw", 99)))
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 2, lambda e: orch.control("finish"))
    say = {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
           "content": [{"type": "TextBlock", "text": "Done."}]}
    done = {"type": "ResultMessage", "subtype": "success", "duration_ms": 5, "duration_api_ms": 1, "is_error": False,
            "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01, "terminal_reason": "completed"}
    fixture = {"provider": "claude-code",
               "turns": [[{"fake": "write", "path": "a.txt", "text": "a\n"}, say, done]], "default_turn": [say, done]}
    session = Session(FakeProvider(fixture), client, Options(cwd=str(worktree), driven=True, idle_seconds=30,
                                                             drafts=True, file_diffs=True),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("first"), timeout=30))
    return [("session", e["provider"], e["kind"], e["payload"]) for e in orch.events]


RESUME_ID = "0b7c4a52-8f1e-4d3a-9c6b-2e5f7a8d9e10"


def _resumed_events(orch, client, worktree) -> list[tuple[str, str, str, dict]]:
    before = len(orch.events)
    done = {"type": "ResultMessage", "subtype": "success", "duration_ms": 5, "duration_api_ms": 1, "is_error": False,
            "num_turns": 1, "session_id": "5d1e9f30-2c4b-4a7e-8f6d-1b3c5e7a9d20", "total_cost_usd": 0.02,
            "terminal_reason": "completed"}
    fixture = {"provider": "claude-code", "ids": True, "refuse_resume": True, "default_turn": [done]}
    session = Session(FakeProvider(fixture), client,
                      Options(cwd=str(worktree), resume=RESUME_ID, resume_text="And the restore script.",
                              carry_over="The brief and the last result.", cost_before=0.01,
                              first_sequence=before + 1), Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("first"), timeout=30))
    assert [e["sequence"] for e in orch.events[before:]] == list(range(before + 1, len(orch.events) + 1))
    return [("resumed", e["provider"], e["kind"], e["payload"]) for e in orch.events[before:]]


def _wrapped_events(orch, client, worktree) -> list[tuple[str, str, str, dict]]:
    before = len(orch.events)
    say = {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
           "content": [{"type": "TextBlock", "text": "working"}]}
    done = {"type": "ResultMessage", "subtype": "success", "duration_ms": 5, "duration_api_ms": 1, "is_error": False,
            "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01, "terminal_reason": "completed"}
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "working",
              lambda e: orch.control("finish"))
    fixture = {"provider": "claude-code", "turns": [[say, {"fake": "sleep", "seconds": 20}, done]],
               "default_turn": [done]}
    session = Session(FakeProvider(fixture), client, Options(cwd=str(worktree), first_sequence=before + 1),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("bounded"), timeout=30))
    return [("wrapped", e["provider"], e["kind"], e["payload"]) for e in orch.events[before:]]


def _unchanged_events(orch, client, worktree) -> list[tuple[str, str, str, dict]]:
    clean = worktree.parent / "unchanged"
    git(worktree.parent, "clone", "--quiet", str(worktree), str(clean))
    before = len(orch.events)
    done = {"type": "ResultMessage", "subtype": "success", "duration_ms": 5, "duration_api_ms": 1, "is_error": False,
            "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01, "terminal_reason": "completed"}
    session = Session(FakeProvider({"provider": "claude-code", "default_turn": [done]}), client,
                      Options(cwd=str(clean), test_cmd="true", first_sequence=before + 1),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("bounded"), timeout=30))
    return [("unchanged", e["provider"], e["kind"], e["payload"]) for e in orch.events[before:]]


@pytest.fixture
def stream(orch, client, worktree):
    return (_corpus() + _session_events(orch, client, worktree) + _resumed_events(orch, client, worktree)
            + _wrapped_events(orch, client, worktree) + _unchanged_events(orch, client, worktree) + _watched_events())


def test_an_unchanged_run_says_its_tests_were_skipped_and_sends_no_test_turn(stream):
    from runner import testrun

    events = [(kind, p) for source, _, kind, p in stream if source == "unchanged"]
    [verdict] = [p for kind, p in events if kind == "message" and p.get("purpose") == "tests-verdict"]
    assert verdict["reason"] == testrun.NO_CHANGES and verdict["text"] == testrun.SKIPPED_NO_CHANGES
    assert not [p for kind, p in events if kind == "message" and p.get("purpose") == "tests"]
    [end] = [p for kind, p in events if kind == "end"]
    assert end["tests_skipped"] == testrun.NO_CHANGES


def _watched_events():
    from runner.watch import Watch
    from helpers import Clock, call, complete, failures, partial

    out = []
    for provider in ("claude-code", "codex"):
        clock = Clock()
        w = Watch(provider, clock)
        for seconds in (600, 600):
            clock.advance(seconds)
            w.tick()
        w.frame(partial(provider))
        w.frame(call(provider))
        clock.advance(1800)
        w.tick()
        w.frame(complete(provider, failed=False))
        for raw in failures(provider, 16):
            w.frame(raw)
        refused = Watch(provider, clock)
        for _ in range(10):
            refused.repeated("refused", "pytest tests/test_backup.py")
        out.extend(("watched", provider, event.kind, event.payload) for event, _ in w.drain() + refused.drain())
    return out


def test_watch_messages_are_the_same_literals_as_the_plans_contract(stream):
    expected = [
        {"role": "runner", "purpose": "quiet", "stage": "quiet", "seconds": 600, "open": None},
        {"role": "runner", "purpose": "quiet", "stage": "stuck", "seconds": 1200, "open": None},
        {"role": "runner", "purpose": "alive", "quiet_seconds": 1200},
        {"role": "runner", "purpose": "quiet", "stage": "stuck", "seconds": 1800,
         "open": {"tool": "Bash", "seconds": 1800}},
        {"role": "runner", "purpose": "alive", "quiet_seconds": 1800},
        {"role": "runner", "purpose": "looping", "kind": "failing", "count": 8,
         "command": "pytest tests/test_backup.py", "action": "nudged"},
        {"role": "runner", "purpose": "looping", "kind": "failing", "count": 16,
         "command": "pytest tests/test_backup.py", "action": "wrapping_up"},
        {"role": "runner", "purpose": "looping", "kind": "refused", "count": 5,
         "command": "pytest tests/test_backup.py", "action": "nudged"},
        {"role": "runner", "purpose": "looping", "kind": "refused", "count": 10,
         "command": "pytest tests/test_backup.py", "action": "wrapping_up"},
    ]
    for provider in ("claude-code", "codex"):
        wanted = [dict(p) for p in expected]
        if provider == "codex":
            wanted[3] = {**wanted[3], "open": {"tool": "command", "seconds": 1800}}
        assert [p for source, name, _, p in stream if source == "watched" and name == provider] == wanted


def test_the_wrap_up_carries_exactly_the_contracts_fields(stream):
    from runner.session import wrap_up_prompt

    said = [p for source, _, kind, p in stream if source == "wrapped" and kind == "message"
            and p.get("purpose") == "wrap_up"]
    assert said == [{"role": "user", "text": wrap_up_prompt("asked"), "mode": "queue", "entry_id": None,
                     "author": "runner", "purpose": "wrap_up", "why": "asked"}]
    [end] = [p for source, _, kind, p in stream if source == "wrapped" and kind == "end"]
    assert end["reason"] == "completed" and end["wrapped_up"] == "asked"
    assert all("wrapped_up" not in p for source, _, kind, p in stream if kind == "end" and source != "wrapped")


def test_the_two_resume_messages_carry_exactly_the_contracts_fields(stream):
    said = [p for source, _, kind, p in stream if source == "resumed" and kind == "message"
            and p.get("purpose") in ("provider_session", "resumed_fresh")]
    assert said == [{"role": "runner", "purpose": "resumed_fresh"},
                    {"role": "runner", "purpose": "provider_session", "id": "5d1e9f30-2c4b-4a7e-8f6d-1b3c5e7a9d20"}]


def test_the_baseline_message_carries_exactly_the_contracts_fields_once_per_run(stream):
    from runner import baseline

    for source in ("session", "resumed"):
        events = [(kind, p) for s, _, kind, p in stream if s == source]
        said = [p for kind, p in events if kind == "message" and p.get("purpose") == "baseline"]
        assert said == [{"role": "runner", "purpose": "baseline", "version": baseline.load().version,
                         "sha": baseline.load().sha[:12], "applied": "append", "project_file": None,
                         "project_file_bytes": 0, "cut": False}]
        first_turn = next(i for i, (kind, _) in enumerate(events) if kind == "turn")
        assert any(p is said[0] for _, p in events[:first_turn])


def test_every_event_keeps_the_contract(stream):
    broken = [f"{source}: {v}" for source, provider, kind, payload in stream
              for v in violations(kind, provider, payload)]
    assert not broken, "\n".join(broken[:40])


def test_every_kind_and_every_row_is_exercised(stream):
    assert {kind for _, _, kind, _ in stream} == set(KINDS)
    unused = [(k, w, f) for k, w, f, _, _ in CONTRACT
              if not any(kind == k and _matches(w, p, payload) and f in payload for _, p, kind, payload in stream)]
    assert not unused, unused


def test_a_reply_in_parts_carries_all_three_fields_or_none(stream):
    alone = 0
    for _, _, kind, payload in stream:
        if kind == "message":
            present = {f for f in ("message_id", "part", "parts") if f in payload}
            assert present in (set(), {"message_id"}, {"message_id", "part", "parts"})
            alone += present == {"message_id"}
            if "part" in present:
                assert 1 <= payload["part"] <= payload["parts"]
    assert alone


def test_the_settings_answer_and_the_turn_after_it(stream):
    [ack] = [p for source, _, kind, p in stream if source == "session" and kind == "message"
             and p.get("purpose") == "settings"]
    assert ack["applied"] == {"model": "claude-opus-5-5", "effort": "high", "mode": "plan"}
    starts = [p for source, _, kind, p in stream if source == "session" and kind == "turn" and p["phase"] == "started"]
    assert starts[0] == {"phase": "started"}
    assert all(s == {"phase": "started", "model": "claude-opus-5-5", "effort": "high", "mode": "plan"}
               for s in starts[1:])


def test_a_question_preview_parses_and_every_payload_fits(stream):
    for _, _, kind, payload in stream:
        assert size(payload) <= PAYLOAD_CAP
        if kind in ("tool", "approval") and (payload.get("name") or payload.get("tool")) in normalise.QUESTION_TOOLS \
                and "input_preview" in payload:
            parsed = json.loads(payload["input_preview"])
            assert len(payload["input_preview"]) <= normalise.PREVIEW_CAP
            for q in parsed["questions"] + payload["questions"]:
                assert isinstance(q["question"], str) and all(isinstance(o["label"], str) for o in q["options"])

