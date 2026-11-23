"""What a session page needs from the stream: thinking events when asked for, context use per turn, and
the result's items."""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import FIXTURES, fast_timing, typed_result
from runner.stub_control_plane import OLD_KINDS
from runner import handoff, normalise
from runner import settings as words
from runner.events import PAYLOAD_CAP, size
from runner.providers.base import ITEMS_MAX, RESULT_SCHEMA, ProviderContext, tidy_items, tidy_result
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session
from helpers import (CodexStub, StubClient, claude_go as _go, claude_sdk_stub as claude, codex_go as _codex_go,
                     codex_on_stub as _codex)


def _raw(stem):
    return json.loads((FIXTURES / f"{stem}.raw.json").read_text(encoding="utf-8"))["messages"]


def think(text, mid="msg_1"):
    return {"type": "AssistantMessage", "model": "claude-fable-5-2", "parent_tool_use_id": None, "message_id": mid,
            "content": [{"type": "ThinkingBlock", "thinking": text, "signature": "sig"},
                        {"type": "TextBlock", "text": "answer"}]}


def done():
    return {"type": "ResultMessage", "subtype": "success", "duration_ms": 1, "is_error": False, "num_turns": 1,
            "session_id": "s", "total_cost_usd": 0.01, "terminal_reason": "completed",
            "structured_output": typed_result()}


def _run(fixture, client, cwd, driven=True, **options):
    provider = FakeProvider(fixture)
    session = Session(provider, client, Options(cwd=str(cwd), driven=driven, idle_seconds=30, diff=False, **options),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("first"), timeout=30))
    return provider, session


def test_thinking_is_a_settings_word_on_or_off():
    assert words.check("thinking", "on") is None and words.check("thinking", "off") is None
    assert words.check("thinking", True) == "thinking must be one of on, off"
    assert "thinking" in words.FIELDS and "thinking" not in words.TURN_FIELDS


def test_a_thinking_block_with_text_is_its_own_event_in_order_and_never_reply_text():
    events = normalise.claude(think("Check the script first."))
    assert [e.kind for e in events] == ["thinking", "message"]
    assert events[0].payload == {"text": "Check the script first.", "id": "msg_1:0"}
    assert events[1].payload["text"] == "answer"


def test_an_omitted_thinking_block_makes_nothing():
    blocks = [b for m in _raw("claude_recorded_2_1_321") if m.get("type") == "AssistantMessage"
              for b in m.get("content") or [] if b.get("type") == "ThinkingBlock"]
    assert blocks and all(b["thinking"] == "" for b in blocks)
    assert not [e for m in _raw("claude_recorded_2_1_321") for e in normalise.claude(m) if e.kind == "thinking"]


def test_a_codex_reasoning_summary_is_one_thinking_event_and_an_empty_one_nothing():
    item = {"type": "reasoning", "id": "rs_1", "summary": ["**Plan** read it", "then fix"], "content": []}
    [event] = normalise.codex({"method": "item/completed", "params": {"item": item}})
    assert event.kind == "thinking" and event.payload == {"text": "**Plan** read it\n\nthen fix", "id": "rs_1"}
    assert normalise.codex({"method": "item/started", "params": {"item": item}}) == []
    assert not [e for m in _raw("codex_recorded") for e in normalise.codex(m) if e.kind == "thinking"]


def test_long_thinking_is_cut_and_marked_within_the_payload_cap():
    [event, _] = normalise.claude(think("é" * 20000))
    assert event.payload["truncated"] is True and len(event.payload["text"]) <= normalise.THINKING_CAP
    assert size(event.payload) <= PAYLOAD_CAP


def test_thinking_off_posts_none_even_when_the_provider_sends_some(orch, client, worktree):
    fixture = {"provider": "claude-code", "default_turn": [think("unasked"), done()]}
    _run(fixture, client, worktree, driven=False)
    assert "thinking" not in orch.kinds()
    assert orch.of("turn", phase="started") == [{"phase": "started"}]


def test_thinking_from_the_start_is_posted_and_the_turn_says_so(orch, client, worktree):
    fixture = {"provider": "claude-code", "default_turn": [think("Read it first."), done()]}
    _run(fixture, client, worktree, driven=False, thinking=True)
    kinds = orch.kinds()
    assert kinds.index("thinking") < kinds.index("end")
    assert orch.of("thinking") == [{"text": "Read it first.", "id": "msg_1:0"}]
    assert orch.of("turn", phase="started") == [{"phase": "started", "thinking": "on"}]
    texts = [(e["kind"], e["payload"].get("text")) for e in orch.events]
    assert texts.index(("thinking", "Read it first.")) < texts.index(("message", "answer"))


def test_thinking_turned_on_between_turns_applies_from_the_next_turn(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 1,
              lambda e: (orch.control("settings", thinking="on"), orch.queue("second")))
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 2, lambda e: orch.control("finish"))
    fixture = {"provider": "claude-code", "turns": [[think("first thoughts", "m1"), done()],
                                                     [think("second thoughts", "m2"), done()]],
               "default_turn": [done()]}
    provider, _ = _run(fixture, client, worktree)
    [ack] = orch.of("message", role="runner", purpose="settings")
    assert ack["applied"] == {"thinking": "on"} and ack["text"] == "From the next turn: thinking on."
    assert [p["text"] for p in orch.of("thinking")] == ["second thoughts"]
    starts = orch.of("turn", phase="started")
    assert starts[0] == {"phase": "started"} and starts[1]["thinking"] == "on"
    assert provider.ctx.thinking is True


def test_thinking_refused_by_a_provider_says_so_and_stores_nothing(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 1,
              lambda e: (orch.control("settings", thinking="on"), orch.queue("second")))
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 2, lambda e: orch.control("finish"))
    fixture = {"provider": "claude-code", "default_turn": [think("thoughts"), done()],
               "refuse_settings": {"thinking": "this model returns no readable thinking"}}
    _run(fixture, client, worktree)
    [ack] = orch.of("message", role="runner", purpose="settings")
    assert ack["refused"] == {"thinking": "this model returns no readable thinking"}
    assert "thinking" not in orch.kinds()


def test_a_new_runner_against_an_older_control_plane_posts_nothing_it_refuses(orch, client, worktree):
    orch.known_kinds = OLD_KINDS
    fixture = {"provider": "claude-code", "default_turn": [think("unasked"), done()]}
    _run(fixture, client, worktree, driven=False)
    assert orch.kinds()[-1] == "end" and "thinking" not in orch.kinds()


def test_claude_thinking_reconnects_with_the_summarized_display(claude):
    provider, ctx = claude
    answers = _go(provider, ctx, "turn", {"thinking": "on"}, "turn", {"thinking": "off"}, "turn")
    assert [a["thinking"].value for a in answers] == ["display summarized", "display omitted (the default)"]
    first, second, third = StubClient.made
    assert first.options.thinking is None
    assert second.options.thinking == {"type": "adaptive", "display": "summarized"} and second.options.resume
    assert third.options.thinking is None and provider.reconnects == 2


def test_claude_thinking_asked_again_when_already_on_does_not_reconnect(claude):
    provider, ctx = claude
    ctx.thinking = True
    _go(provider, ctx, "turn", {"thinking": "on"}, "turn")
    assert len(StubClient.made) == 1 and StubClient.made[0].options.thinking["display"] == "summarized"
    assert provider.settings_now()["thinking"] == "on"


def test_codex_thinking_rides_the_next_turn_start_once():
    stub = CodexStub()
    provider = _codex(stub)
    answers = _codex_go(provider, {"thinking": "on"}, "turn", "turn", {"thinking": "off"}, "turn")
    assert [a["thinking"].value for a in answers] == ["reasoning summary auto", "reasoning summary none"]
    assert stub.started == [{"summary": "auto"}, None, {"summary": "none"}]
    assert provider.settings_now()["thinking"] == "off"


def test_codex_thinking_from_the_start_rides_the_first_turn():
    stub = CodexStub()
    provider = _codex(stub)
    provider.ctx.thinking = True
    provider.start_thinking()
    _codex_go(provider, "turn", "turn")
    assert stub.started == [{"summary": "auto"}, None] and provider.settings_now()["thinking"] == "on"


def test_the_runner_takes_thinking_from_its_command_line():
    from runner.__main__ import parse
    base = ["--provider", "fake", "--cwd", "/tmp", "--prompt-file", "/tmp/p"]
    assert parse(base + ["--thinking"]).thinking is True and parse(base).thinking is False


def _completed(provider_cls, stem):
    provider = provider_cls()
    return [e.payload for m in _raw(stem) for e in provider.normalise(m)
            if e.kind == "turn" and e.payload.get("phase") == "completed"]


def test_claude_context_is_the_last_main_thread_call_and_the_models_window():
    from runner.providers.claude import ClaudeProvider
    first, second = _completed(ClaudeProvider, "claude_recorded_2_1_321")
    assert (first["context_tokens"], first["context_window"]) == (19577, 1000000)
    assert (second["context_tokens"], second["context_window"]) == (18457, 1000000)


def test_codex_context_is_the_last_token_usage_of_the_turn():
    from runner.providers.codex import CodexProvider
    [turn] = _completed(CodexProvider, "codex_recorded")
    assert (turn["context_tokens"], turn["context_window"]) == (17249, 475000)


def test_context_sends_what_is_known_and_never_a_percentage():
    meter = normalise.ContextMeter("claude-code")
    say = {"type": "AssistantMessage", "model": "claude-opus-5-5", "parent_tool_use_id": None,
           "usage": {"input_tokens": 3, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 100}}
    for raw in ({"type": "QuerySent"}, say, {**say, "parent_tool_use_id": "sub", "usage": {
            "input_tokens": 1, "cache_creation_input_tokens": 1, "cache_read_input_tokens": 1}},
            {"type": "ResultMessage", "model_usage": {"claude-haiku-5-5": {"contextWindow": 200000}}}):
        meter.see(raw)
    [event] = meter.annotate([normalise._event("turn", {"phase": "completed"})])
    assert event.payload == {"phase": "completed", "context_tokens": 113}
    meter.see({"type": "QuerySent"})
    [event] = meter.annotate([normalise._event("turn", {"phase": "completed"})])
    assert event.payload == {"phase": "completed"}


@pytest.mark.parametrize("usage,model,expected", [
    ({"claude-fable-5-2": {"contextWindow": 1000000}}, "claude-fable-5-2[1m]", 1000000),
    ({"x": {"canonicalModel": "claude-opus-5-5", "contextWindow": 200000}}, "claude-opus-5-5", 200000),
    ({"claude-opus-5-5": {"contextWindow": True}}, "claude-opus-5-5", None),
    (None, "claude-opus-5-5", None)])
def test_the_claude_window_is_read_only_for_the_turns_model(usage, model, expected):
    assert normalise.claude_window(usage, model) == expected


def test_the_schema_asks_for_items_with_three_statuses_and_stays_strict():
    assert RESULT_SCHEMA["properties"]["next_steps"] == {"type": "array", "items": {"type": "string"}}
    assert "next_steps" in RESULT_SCHEMA["required"]
    assert "tests_claim" in RESULT_SCHEMA["required"]
    items = RESULT_SCHEMA["properties"]["items"]
    assert "items" in RESULT_SCHEMA["required"] and RESULT_SCHEMA["additionalProperties"] is False
    entry = items["items"]
    assert entry["required"] == ["label", "status", "where"] and entry["additionalProperties"] is False
    assert entry["properties"]["status"]["enum"] == ["done", "not_done", "unverified"]
    assert "honest" in entry["properties"]["status"]["description"]
    assert "empty" in items["description"]


def test_items_are_tidied_capped_and_bad_ones_dropped():
    found = [{"label": " Fix   the\nbackup ", "status": "done", "where": "scripts/backup.sh"},
             {"label": "x", "status": "maybe", "where": ""}, {"label": "", "status": "done"}, "nope",
             {"label": "L" * 500, "status": "unverified"}]
    items, dropped = tidy_items(found)
    assert items[0] == {"label": "Fix the backup", "status": "done", "where": "scripts/backup.sh"}
    assert items[1]["where"] == "" and len(items[1]["label"]) == 200 and dropped == 3
    many, more = tidy_items([{"label": str(i), "status": "done", "where": ""} for i in range(40)])
    assert len(many) == ITEMS_MAX and more == 10
    assert tidy_result({"summary": "s"}) == {"summary": "s"}
    assert tidy_result({"summary": "s", "items": "bad"}) == {"summary": "s", "items": []}


def test_a_claude_result_carries_tidied_items_on_its_turn_event():
    raw = {**done(), "structured_output": {"summary": "s", "items": [
        {"label": "a", "status": "done", "where": "f.py"}, {"label": "b", "status": "wrong", "where": ""}]}}
    [turn, _usage] = normalise.claude(raw)
    assert turn.payload["structured_output"] == {"summary": "s", "items": [
        {"label": "a", "status": "done", "where": "f.py"}], "items_more": 1}


def test_a_codex_result_carries_its_items_to_the_end_event(orch, client, worktree):
    result = {"summary": "Fixed it", "files_changed": [], "tests_run": [], "branch": "b", "open_questions": [],
              "items": [{"label": "Fix the backup", "status": "unverified", "where": "scripts/backup.sh"}]}
    fixture = {"provider": "codex", "default_turn": [
        {"method": "turn/started", "params": {"turn": {"id": "t1"}}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": "m1",
                                                         "text": json.dumps(result)}}},
        {"method": "turn/completed", "params": {"turn": {"id": "t1", "status": "completed"}}}]}
    _run(fixture, client, worktree, driven=False)
    [end] = orch.of("end")
    assert end["result"]["items"] == result["items"]


def test_the_pull_request_body_lists_the_items_and_an_old_result_renders_as_before():
    body = handoff.render({"summary": "S", "items": [
        {"label": "Fix the backup", "status": "done", "where": "scripts/backup.sh"},
        {"label": "Add a test", "status": "unverified", "where": ""}]})
    assert "## What was asked (as the session reports it)" in body
    assert "- Done: Fix the backup (scripts/backup.sh)" in body and "- Unverified: Add a test" in body
    assert "What was asked" not in handoff.render({"summary": "S", "files_changed": ["a"]})
    assert "What was asked" not in handoff.render({"summary": "S", "items": []})


def test_the_prompts_ask_for_items_without_asking_for_more_than_was_done():
    for text in (handoff.BRIEF_NOTE, handoff.SUMMARY_PROMPT):
        assert "`items`" in text and "unverified" in text and "done only where you saw it work" in text \
            or "mark an item done only where you saw it work" in text


def test_the_provider_context_has_thinking_off_by_default():
    async def deny(_p):
        raise AssertionError
    assert ProviderContext(request_approval=deny, cwd="/tmp").thinking is False

