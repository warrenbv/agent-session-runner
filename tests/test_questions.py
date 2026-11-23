"""A session's question on both providers: one card for Claude and Codex, and the owner's reply returned
as each provider's answer."""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any

import pytest

from runner import normalise, refusals
from runner.providers.base import RESEARCH_KIND
from runner.relay import Decision

from helpers import Seen, codex_answer as _codex_answer

RECORDED = json.loads((pathlib.Path(__file__).parent / "fixtures" / "question_recorded.json").read_text(encoding="utf-8"))
CODEX, CLAUDE = RECORDED["codex"], RECORDED["claude"]
ASKED = CODEX["request"]["params"]
QUESTIONS = ASKED["questions"]
CLAUDE_INPUT = CLAUDE["can_use_tool"]["input"]
DISMISSED = CODEX["dismissed"]["relay_message"]


def reply(text: str) -> Decision:
    return Decision(False, normalise.REPLY_PREFIX + text)


def test_a_codex_question_raises_the_card_a_claude_question_raises():
    codex = normalise.codex_question(ASKED)
    claude = normalise.claude_approval("AskUserQuestion", CLAUDE_INPUT, "toolu_probe1")
    assert codex["tool"] == "request_user_input" and codex["request_id"] == "call_1"
    assert set(codex) == set(claude) == {"request_id", "tool", "input_preview", "questions"}
    preview = json.loads(codex["input_preview"])
    assert [q["question"] for q in preview["questions"]] == [q["question"] for q in QUESTIONS]
    assert [q["header"] for q in codex["questions"]] == ["Database", "Port"]
    assert [[o["label"] for o in q["options"]] for q in codex["questions"]] == \
        [["SQLite (Recommended)", "Postgres"], ["8080", "3000"]]
    assert all(q["multiSelect"] is False for q in codex["questions"])
    assert normalise.CODEX_QUESTION_TOOL in normalise.QUESTION_TOOLS


def test_the_owners_reply_is_the_answer_codex_handed_the_model():
    seen = Seen(Decision(False, CODEX["answered"]["relay_message"]))
    answer, provider = _codex_answer(seen, CODEX["request"]["method"], ASKED)
    assert json.dumps(answer, separators=(",", ":")) == CODEX["answered"]["function_call_output"]["output"]
    assert [c["tool"] for c in seen.cards] == ["request_user_input"]
    assert seen.lines == [] and provider._pending_feedback == []


def test_a_dismissal_is_no_answers_and_its_words_steered_in():
    seen = Seen(Decision(False, DISMISSED))
    answer, provider = _codex_answer(seen, CODEX["request"]["method"], ASKED)
    assert json.dumps(answer, separators=(",", ":")) == CODEX["dismissed"]["function_call_output"]["output"]
    assert provider._pending_feedback == [CODEX["dismissed"]["steered_user_text"]]
    assert len(seen.cards) == 1 and seen.lines == []


@pytest.mark.parametrize("decision", [Decision(False, "No answer within ten minutes; denied."),
                                      Decision(False, "The approval expired unanswered; denied."),
                                      Decision(True)])
def test_no_reply_is_never_the_empty_result_codex_cannot_read(decision):
    answer, _ = _codex_answer(Seen(decision), CODEX["request"]["method"], ASKED)
    assert answer == {"answers": {}}


@pytest.mark.parametrize("params,kw", [
    ({**ASKED, "questions": [{**QUESTIONS[0], "isSecret": True}, QUESTIONS[1]]}, {}),
    ({**ASKED, "questions": []}, {}),
    (ASKED, {"kind": RESEARCH_KIND}),
])
def test_a_question_not_put_on_a_card_is_said_on_the_stream(params, kw):
    seen = Seen(reply("SQLite"))
    answer, provider = _codex_answer(seen, CODEX["request"]["method"], params, **kw)
    assert answer == {"answers": {}} and seen.cards == []
    assert seen.lines == [{"text": refusals.CODEX_QUESTION_LINE, "tool": "request_user_input"}]


def test_without_a_session_a_question_still_gets_the_shape_codex_reads():
    from runner.providers.codex import CodexProvider

    assert CodexProvider()._on_server_request(CODEX["request"]["method"], ASKED) == {"answers": {}}


def test_the_runner_threads_offer_the_question_tool_and_research_threads_do_not():
    from runner.providers.codex import CONFIG_OVERRIDES, QUESTION_FEATURE, WEB_SEARCH_OFF, research_overrides

    assert QUESTION_FEATURE == "features.default_mode_request_user_input=true" and QUESTION_FEATURE in CONFIG_OVERRIDES
    assert CONFIG_OVERRIDES[-1] == WEB_SEARCH_OFF
    assert not any("request_user_input" in o for o in research_overrides(3))


def test_the_feature_warning_and_the_resolution_make_no_event():
    assert normalise.codex(CODEX["feature_warning"]) == [] and normalise.codex(CODEX["resolved"]) == []


@pytest.mark.parametrize("text,expected", [
    ("Postgres", {"db_choice": {"answers": ["Postgres"]}}),
    ("Database: SQLite (Recommended)\nPort: 8080", {"db_choice": {"answers": ["SQLite (Recommended)"]},
                                                     "port": {"answers": ["8080"]}}),
    ("Which port should it listen on?: 3000", {"port": {"answers": ["3000"]}}),
    ("Whatever is simplest", {"db_choice": {"answers": ["Whatever is simplest"]},
                              "port": {"answers": ["Whatever is simplest"]}}),
    ("Port: 3000\nuse SQLite unless it is slow", {"db_choice": {"answers": ["use SQLite unless it is slow"]},
                                                   "port": {"answers": ["3000", "user_note: use SQLite unless it is slow"]}}),
    ("Engine: Postgres", {"db_choice": {"answers": ["Engine: Postgres"]}, "port": {"answers": ["Engine: Postgres"]}}),
])
def test_a_reply_is_split_back_per_question(text, expected):
    assert normalise.codex_answers(text, QUESTIONS) == expected


def test_several_picked_labels_are_one_answer_each_and_other_commas_stay():
    multi = [{"id": "q", "header": "Pick", "question": "Which?",
              "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}]}]
    assert normalise.codex_answers("Pick: a, b", multi) == {"q": {"answers": ["a", "b"]}}
    assert normalise.codex_answers("Pick: a, or so", multi) == {"q": {"answers": ["a, or so"]}}


def test_only_a_reply_is_read_as_one():
    assert normalise.owner_reply(normalise.REPLY_PREFIX + "x") == "x"
    for other in (DISMISSED, "Denied by the session's owner.", "", None):
        assert normalise.owner_reply(other) is None


@pytest.mark.parametrize("method,expected", [
    ("item/permissions/requestApproval", {"permissions": {}, "scope": "turn"}),
    ("mcpServer/elicitation/request", {"action": "decline"}),
    ("item/tool/call", {}),
    ("currentTime/read", {}),
    ("execCommandApproval", {}),
])
def test_a_request_the_runner_does_not_card_is_answered_as_before_and_said(method, expected):
    seen = Seen()
    answer, provider = _codex_answer(seen, method, {"threadId": "t", "turnId": "u"})
    assert answer == expected and seen.cards == [] and provider._pending_feedback == []
    assert seen.lines == [{"text": refusals.CODEX_DECLINED_LINE.format(method=method), "tool": method}]
    assert method in seen.lines[0]["text"]


def test_the_escalation_rule_is_unchanged():
    from helpers import ESCALATION

    seen = Seen()
    answer, provider = _codex_answer(seen, refusals.CODEX_COMMAND_APPROVAL, ESCALATION, "auto")
    assert answer == {"decision": "decline"} and seen.cards == []
    assert provider._pending_feedback == [refusals.CODEX_ESCALATION]
    assert [line["text"] for line in seen.lines] == [refusals.CODEX_LINE]


def _claude_decide(decision: Decision, tool: str = "AskUserQuestion", tool_input: Any = None):
    from runner.providers.claude import ClaudeProvider

    seen = Seen(decision)
    provider = ClaudeProvider()
    provider.ctx = seen.ctx()
    answer = asyncio.run(provider._can_use_tool(tool, CLAUDE_INPUT if tool_input is None else tool_input, None))
    return answer, seen


def test_a_claude_reply_is_allowed_with_the_answers_the_cli_returns_as_the_result():
    from claude_agent_sdk import PermissionResultAllow

    answer, seen = _claude_decide(reply("Database: SQLite\nPort: 8080\nkeep it on this machine"))
    assert isinstance(answer, PermissionResultAllow)
    recorded = CLAUDE["allowed_with_answers"]["updated_input_extra"]
    assert answer.updated_input["questions"] == CLAUDE_INPUT["questions"]
    assert answer.updated_input["answers"] == recorded["answers"]
    assert answer.updated_input["annotations"] == {q: {"notes": "keep it on this machine"} for q in recorded["answers"]}
    assert [c["tool"] for c in seen.cards] == ["AskUserQuestion"]


def test_a_claude_option_button_and_a_written_answer():
    one = {"questions": [CLAUDE_INPUT["questions"][0]]}
    answer, _ = _claude_decide(reply("Postgres"), tool_input=one)
    assert answer.updated_input == {**one, "answers": {"Which database should the app use?": "Postgres"}}
    answer, _ = _claude_decide(reply("Something lighter than both"), tool_input=one)
    assert answer.updated_input["answers"] == {"Which database should the app use?": "Something lighter than both"}


@pytest.mark.parametrize("decision", [Decision(False, DISMISSED), Decision(False, "No answer within ten minutes; denied.")])
def test_a_claude_dismissal_or_expiry_is_still_a_denial_with_its_words(decision):
    from claude_agent_sdk import PermissionResultDeny

    answer, _ = _claude_decide(decision)
    assert isinstance(answer, PermissionResultDeny) and answer.message == decision.message


def test_a_permission_denial_is_never_read_as_an_answer():
    from claude_agent_sdk import PermissionResultDeny

    answer, _ = _claude_decide(reply("yes"), tool="Write", tool_input={"file_path": "/tmp/wt/a", "content": "x"})
    assert isinstance(answer, PermissionResultDeny)
    answer, _ = _claude_decide(reply("yes"), tool_input={"questions": "not a list"})
    assert isinstance(answer, PermissionResultDeny)

