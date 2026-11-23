"""Stream shapes: questions that arrive whole, long replies in parts, the result schema per turn in a
driven Claude session, and the withdraw control."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid

import pytest

from conftest import fast_timing, typed_result
from runner import normalise
from runner.events import MARK, PAYLOAD_CAP, size
from runner.control_plane import Control, pull_all
from runner.providers.base import RESULT_SCHEMA, ProviderContext
from runner.providers.fake import FakeProvider
from runner.session import WITHDRAW_WORDS, Caps, Options, Session


def _question_input(n_questions=4, n_options=4, preview=5000, description=400, question=300):
    return {"questions": [{
        "question": f"Question {q}: " + "which layout? " * (question // 14),
        "header": f"Layout {q}",
        "multiSelect": q % 2 == 1,
        "options": [{"label": f"Option {q}.{o}", "description": "d" * description,
                     "preview": f"<preview {q}.{o}>" + "x" * preview} for o in range(n_options)],
    } for q in range(n_questions)]}


def _ask(tool_input):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "ToolUseBlock", "id": "toolu_q", "name": "AskUserQuestion", "input": tool_input}]}


@pytest.mark.parametrize("shape", [dict(), dict(preview=20000), dict(n_questions=1, n_options=2, preview=50),
                                   dict(description=3000, question=4000), dict(n_questions=1, n_options=12, preview=3000),
                                   dict(n_questions=8, n_options=12, preview=3000)])
def test_a_question_preview_is_valid_json_within_the_cap_and_keeps_its_structure(shape):
    tool_input = _question_input(**shape)
    [event] = normalise.claude(_ask(tool_input))
    approval = normalise.claude_approval("AskUserQuestion", tool_input, "toolu_q")
    for payload in (event.payload, approval):
        preview = payload["input_preview"]
        assert len(preview) <= normalise.PREVIEW_CAP
        parsed = json.loads(preview)
        assert size(payload) <= PAYLOAD_CAP
        for got, cut, more in ((parsed["questions"], parsed.get("cut"), parsed.get("questions_more", 0)),
                               (payload["questions"], payload.get("questions_cut"), payload.get("questions_more", 0))):
            assert len(got) + more == len(tool_input["questions"]) and (cut or not more)
            for q, want in zip(got, tool_input["questions"]):
                assert [o["label"] for o in q["options"]] == [o["label"] for o in want["options"]][:len(q["options"])]
                assert len(q["options"]) + q.get("options_more", 0) == len(want["options"])
                assert q["header"] == want["header"] and q["multiSelect"] == want["multiSelect"]
                assert q["question"] == want["question"] or q["question"].endswith(MARK)
                for o, wo in zip(q["options"], want["options"]):
                    for key in ("description", "preview"):
                        assert o.get(key) == wo[key] or (o.get(key) or MARK).endswith(MARK)
                        assert key in o or cut


def test_four_questions_of_four_options_keep_every_label_whole():
    tool_input = _question_input(preview=20000, description=3000, question=4000)
    payload = normalise.question_payload(tool_input)
    for got in (json.loads(payload["input_preview"])["questions"], payload["questions"]):
        assert [[o["label"] for o in q["options"]] for q in got] == \
            [[o["label"] for o in q["options"]] for q in tool_input["questions"]]
    assert all(o["description"] for q in payload["questions"] for o in q["options"])


def test_previews_give_way_before_descriptions_and_labels():
    tool_input = _question_input(n_questions=1, n_options=4, preview=3000, description=200)
    preview = json.loads(normalise.question_payload(tool_input)["input_preview"])
    options = preview["questions"][0]["options"]
    assert all(o["description"] == "d" * 200 for o in options)
    assert all(o["preview"].endswith(MARK) for o in options)
    roomy = normalise.question_payload(tool_input)["questions"][0]["options"]
    assert all(len(o["preview"]) > len(p["preview"]) for o, p in zip(roomy, options))


def test_a_short_question_is_unchanged_but_for_key_order():
    tool_input = {"questions": [{"question": "Ship it?", "header": "Ship", "multiSelect": False,
                                 "options": [{"label": "Yes", "description": "Now."}, {"label": "No"}]}]}
    payload = normalise.question_payload(tool_input)
    assert json.loads(payload["input_preview"]) == tool_input
    assert payload["questions"] == tool_input["questions"]


def test_the_control_planes_reader_parses_the_preview_without_its_fallback():
    tool_input = _question_input(preview=8000)
    preview = normalise.claude_approval("AskUserQuestion", tool_input, "toolu_q")["input_preview"]
    json.loads(preview)
    assert preview[0] == "{" and preview[-1] == "}"


def test_other_tools_and_odd_question_inputs_keep_the_plain_preview():
    big = {"file_path": "a.txt", "content": "x" * 5000}
    [event] = normalise.claude({**_ask(big), "content": [{"type": "ToolUseBlock", "id": "t", "name": "Write",
                                                          "input": big}]})
    assert event.payload["input_preview"].endswith(MARK) and "questions" not in event.payload
    [odd] = normalise.claude(_ask({"prompt": "not question-shaped"}))
    assert odd.payload["input_preview"] == '{"prompt": "not question-shaped"}' and "questions" not in odd.payload


def test_a_bare_string_option_is_a_label():
    payload = normalise.question_payload({"questions": [{"question": "Pick", "options": ["A", "B"]}]})
    assert payload["questions"][0]["options"] == [{"label": "A"}, {"label": "B"}]


def _say(text, message_id="msg_1"):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None, "message_id": message_id,
            "content": [{"type": "TextBlock", "text": text}]}


def _paragraphs(n, width=900):
    return "\n\n".join(f"Paragraph {i}. " + "word " * (width // 5) for i in range(n))


@pytest.mark.parametrize("text", [
    _paragraphs(40),
    "\n".join("line " * 30 for _ in range(400)),
    "x" * 30000,
    "日本語のテキスト。" * 4000,
    "🙂" * 9000,
    "é" * 9000,
    '"quoted"\\ ' * 3000 + "\x01" * 2000,
])
def test_a_long_reply_is_posted_whole_in_ordered_parts(text):
    events = normalise.claude(_say(text))
    assert len(events) > 1
    parts = [e.payload for e in events]
    assert "".join(p["text"] for p in parts) == text
    assert [p["part"] for p in parts] == list(range(1, len(parts) + 1))
    assert {p["parts"] for p in parts} == {len(parts)} and {p["message_id"] for p in parts} == {"msg_1:0"}
    for p in parts:
        assert p["role"] == "assistant" and "truncated" not in p and size(p) <= PAYLOAD_CAP
        assert len(p["text"]) <= normalise.TEXT_CAP
    for left, right in zip(parts, parts[1:]):
        assert not normalise._joins(right["text"][0])


@pytest.mark.parametrize("ch", ["\u200d", "\ufe0f", "\ufe0e", "\U000e0100", "\U000e01ef", "\u0301"])
def test_joiners_and_variation_selectors_attach_to_the_character_before(ch):
    assert normalise._joins(ch)


@pytest.mark.parametrize("ch", ["a", " ", "\n", "\u200b", "\ufe0d", "\U000e00ff", "\U000e01f0", "\U0001f642"])
def test_other_characters_do_not_join(ch):
    assert not normalise._joins(ch)


def test_a_long_reply_of_joined_emoji_never_starts_a_part_on_a_joiner():
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    text = (family + "\u2764\ufe0f" + "\u845b\U000e0100") * 3000
    parts = normalise.split_text(text)
    assert len(parts) > 1 and "".join(parts) == text
    assert not any(normalise._joins(part[0]) for part in parts[1:])


def test_parts_end_at_a_paragraph_when_one_is_near():
    parts = normalise.split_text(_paragraphs(40))
    assert all(p.endswith("\n\n") for p in parts[:-1])


def test_a_reply_that_fits_is_one_event_as_before():
    [event] = normalise.claude(_say("short"))
    assert event.payload == {"role": "assistant", "text": "short"}


def test_codex_replies_are_split_by_item_id():
    item = {"type": "agentMessage", "id": "msg_c", "text": _paragraphs(20), "phase": "final_answer"}
    events = normalise.codex({"method": "item/completed", "params": {"item": item}})
    assert len(events) > 1 and {e.payload["message_id"] for e in events} == {"msg_c"}
    assert all(e.payload["item_id"] == "msg_c" and e.payload["phase"] == "final_answer" for e in events)
    assert "".join(e.payload["text"] for e in events) == item["text"]


def test_the_session_joins_parts_for_codex_typed_result(orch, client, worktree):
    result = typed_result("s" * 20000, branch="b")
    text = json.dumps(result)
    fixture = {"provider": "codex", "turns": [[
        {"method": "turn/started", "params": {"turn": {"id": "1"}}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": "m1", "text": text}}},
        {"method": "turn/completed", "params": {"turn": {"id": "1", "status": "completed"}}}]]}
    session = Session(FakeProvider(fixture), client, Options(cwd=str(worktree)), Caps(wall_clock=30, grace=1),
                      fast_timing())
    outcome = asyncio.run(asyncio.wait_for(session.run("go"), timeout=30))
    assert outcome.result == result
    assert len(orch.of("message", role="assistant")) > 1


@dataclasses.dataclass
class ResultMessage:
    subtype: str
    session_id: str


class StubClient:
    made: list["StubClient"] = []

    def __init__(self, options):
        self.options, self.queries, self.closed = options, [], False
        StubClient.made.append(self)

    async def connect(self):
        pass

    async def disconnect(self):
        self.closed = True

    async def query(self, text):
        self.queries.append(text)

    async def receive_response(self):
        yield ResultMessage("success", SESSION)


SESSION = str(uuid.uuid4())


async def _deny(_payload):
    raise AssertionError("no approval expected")


def _turns(provider, ctx, schemas):
    async def go():
        await provider.start(ctx)
        for schema in schemas:
            ctx.output_schema = schema
            [r async for r in provider.turn("hello")]
    asyncio.run(go())


def test_claude_reconnects_with_the_schema_only_when_it_changes(monkeypatch, tmp_path):
    import claude_agent_sdk
    from runner.providers.claude import ClaudeProvider

    StubClient.made = []
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", StubClient)
    provider = ClaudeProvider()
    ctx = ProviderContext(request_approval=_deny, cwd=str(tmp_path), output_schema=None)
    _turns(provider, ctx, [None, None, RESULT_SCHEMA])
    first, second = StubClient.made
    assert first.options.output_format is None and first.options.resume is None
    assert len(first.queries) == 2 and first.closed
    assert second.options.output_format == {"type": "json_schema", "schema": RESULT_SCHEMA}
    assert "tests_claim" in second.options.output_format["schema"]["required"]
    assert second.options.resume == SESSION and second.queries == ["hello"]
    assert provider.reconnects == 1


def test_a_bounded_claude_session_never_reconnects(monkeypatch, tmp_path):
    import claude_agent_sdk
    from runner.providers.claude import ClaudeProvider

    StubClient.made = []
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", StubClient)
    provider = ClaudeProvider()
    ctx = ProviderContext(request_approval=_deny, cwd=str(tmp_path), output_schema=RESULT_SCHEMA)
    _turns(provider, ctx, [RESULT_SCHEMA, RESULT_SCHEMA])
    [only] = StubClient.made
    assert only.options.output_format == {"type": "json_schema", "schema": RESULT_SCHEMA}
    assert len(only.queries) == 2 and provider.reconnects == 0


def say(text):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "TextBlock", "text": text}]}


def done():
    return {"type": "ResultMessage", "subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
            "is_error": False, "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01,
            "terminal_reason": "completed", "structured_output": typed_result()}


def _drive(fixture, client, cwd, driven=True):
    provider = FakeProvider(fixture)
    session = Session(provider, client, Options(cwd=str(cwd), driven=driven, idle_seconds=30, diff=False),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("first"), timeout=30))
    return provider


def _withdrawals(orch):
    return orch.of("message", role="runner", purpose="withdraw")


@pytest.mark.parametrize("driven", [True, False])
def test_a_held_message_is_withdrawn_before_the_session_reads_it(orch, client, worktree, driven):
    def queue_then_withdraw(_event):
        entry = orch.queue("second")
        orch.control("withdraw", entry)
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "one", queue_then_withdraw)
    if driven:
        orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting",
                  lambda e: orch.control("finish"))
    fixture = {"provider": "claude-code", "turns": [[say("one"), {"fake": "sleep", "seconds": 0.3}, done()]],
               "default_turn": [say("more"), done()]}
    fake = _drive(fixture, client, worktree, driven)
    assert fake.sent == ["first"]
    [w] = _withdrawals(orch)
    assert w == {"role": "runner", "purpose": "withdraw", "text": WITHDRAW_WORDS["withdrawn"],
                 "outcome": "withdrawn", "entry_id": 1, "control_id": 1}
    assert not orch.of("message", role="user", entry_id=1)


def test_a_message_already_handed_over_is_too_late(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 1, lambda e: orch.queue("second"))
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("entry_id") == 1,
              lambda e: orch.control("withdraw", 1))
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 2, lambda e: orch.control("finish"))
    fixture = {"provider": "claude-code", "turns": [[say("one"), done()],
                                                    [say("two"), {"fake": "sleep", "seconds": 0.3}, done()]]}
    fake = _drive(fixture, client, worktree)
    assert fake.sent[:2] == ["first", "second"]
    [w] = _withdrawals(orch)
    assert w["outcome"] == "too_late" and w["entry_id"] == 1 and w["text"] == WITHDRAW_WORDS["too_late"]


def test_a_steer_taken_is_too_late_and_an_unknown_id_is_unknown(orch, client, worktree):
    def steer_then_withdraw(_event):
        entry = orch.queue("look here", mode="steer")
        orch.control("withdraw", entry)
        orch.control("withdraw", 99)
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "one", steer_then_withdraw)
    fixture = {"provider": "claude-code", "turns": [[say("one"), {"fake": "wait_steer", "timeout": 2}, done()]]}
    fake = _drive(fixture, client, worktree, driven=False)
    assert fake.steered == ["look here"]
    assert [(w["entry_id"], w["outcome"]) for w in _withdrawals(orch)] == [(1, "too_late"), (99, "unknown")]
    assert orch.of("message", role="user", entry_id=1, delivered="next step")


def test_pull_reads_withdraw_controls_and_drops_one_without_an_entry(orch, client):
    orch.control("withdraw", 5)
    orch.control("withdraw", None)
    orch.control("finish")
    _, _, controls = asyncio.run(pull_all(client))
    assert controls == [Control(1, "withdraw", 5), Control(3, "finish")]

