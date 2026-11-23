"""Requests the runner refuses itself: Claude Code's shell in auto mode and every Codex request to run
outside the sandbox. Each is refused with words the model can act on and leaves one line on the stream."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from conftest import fast_timing
from helpers import ESCALATION, Seen, codex_answer as _codex_answer
from runner import refusals
from runner.providers.base import CODING_KIND, RESEARCH_KIND, ProviderContext
from runner.providers.fake import FakeProvider
from runner.relay import Decision
from runner.session import Caps, Options, Session

ORDINARY = {k: v for k, v in ESCALATION.items() if k not in ("reason", "proposedExecpolicyAmendment")}
SED = "TEST_WORKERS=0 sh scripts/run-tests.sh"


def _claude(seen: Seen, **kw: Any):
    from runner.providers.claude import ClaudeProvider

    provider = ClaudeProvider()
    provider.ctx = seen.ctx(**kw)
    return provider


def _decide(provider, tool: str, tool_input: dict[str, Any]):
    return asyncio.run(provider._can_use_tool(tool, tool_input, None))


def test_an_auto_mode_shell_request_is_refused_with_the_message_and_no_card():
    from claude_agent_sdk import PermissionResultDeny

    seen = Seen()
    answer = _decide(_claude(seen, permission_mode="auto"), "Bash", {"command": f"cd /tmp/wt; {SED}"})
    assert isinstance(answer, PermissionResultDeny) and answer.interrupt is False
    assert answer.message == refusals.CLAUDE_SHELL
    assert seen.cards == []
    assert seen.lines == [{"text": refusals.CLAUDE_LINE, "tool": "Bash", "command": f"cd /tmp/wt; {SED}"}]


def test_the_claude_message_says_what_happened_and_how_to_go_on():
    text = refusals.CLAUDE_SHELL
    for words in ("needed an approval", "in Auto mode nobody is asked", "Go on without the command",
                  "say in your summary what you could not do without it"):
        assert words in text
    for words in ("`env`", "`cd`", "subshell", "plainly", "retry"):
        assert words not in text and words not in refusals.CLAUDE_LINE


def test_a_request_to_leave_the_sandbox_is_refused_with_its_own_message():
    from claude_agent_sdk import PermissionResultDeny

    seen = Seen()
    asked = {"command": "uv sync", "dangerouslyDisableSandbox": True}
    answer = _decide(_claude(seen, permission_mode="auto"), "Bash", asked)
    assert isinstance(answer, PermissionResultDeny) and answer.interrupt is False
    assert answer.message == refusals.CLAUDE_UNSANDBOXED != refusals.CLAUDE_SHELL
    for words in ("never runs a command outside its sandbox", "Go on without the command",
                  "say in your summary what you could not do without it"):
        assert words in answer.message
    assert seen.cards == []
    assert seen.lines == [{"text": refusals.CLAUDE_UNSANDBOXED_LINE, "tool": "Bash", "command": "uv sync"}]


@pytest.mark.parametrize("flag", [False, "true", None])
def test_only_a_literal_true_flag_is_a_request_to_leave_the_sandbox(flag):
    seen = Seen()
    answer = _decide(_claude(seen, permission_mode="auto"), "Bash", {"command": SED, "dangerouslyDisableSandbox": flag})
    assert answer.message == refusals.CLAUDE_SHELL
    assert [line["text"] for line in seen.lines] == [refusals.CLAUDE_LINE]


def test_the_line_keeps_why_the_cli_asked():
    from types import SimpleNamespace

    seen = Seen()
    context = SimpleNamespace(tool_use_id="toolu_1", agent_id=None, title=None,
                              decision_reason="Auto mode paused after repeated blocks")
    asyncio.run(_claude(seen, permission_mode="auto")._can_use_tool("Bash", {"command": SED}, context))
    assert seen.lines == [{"text": refusals.CLAUDE_LINE, "tool": "Bash", "command": SED,
                           "reason": "Auto mode paused after repeated blocks"}]


@pytest.mark.parametrize("mode", ["default", "acceptEdits", "plan"])
def test_the_same_request_in_another_mode_still_raises_a_card(mode):
    from claude_agent_sdk import PermissionResultAllow

    seen = Seen()
    answer = _decide(_claude(seen, permission_mode=mode), "Bash", {"command": SED})
    assert isinstance(answer, PermissionResultAllow)
    assert [c["tool"] for c in seen.cards] == ["Bash"] and seen.lines == []


@pytest.mark.parametrize("tool, tool_input", [("Write", {"file_path": "/tmp/wt/.git/x", "content": "y"}),
                                              ("SandboxNetworkAccess", {"host": "example.com"}),
                                              ("AskUserQuestion", {"questions": []})])
def test_a_tool_other_than_the_shell_still_raises_a_card_in_auto_mode(tool, tool_input):
    seen = Seen(Decision(False, "Denied by the session's owner."))
    _decide(_claude(seen, permission_mode="auto"), tool, tool_input)
    assert [c["tool"] for c in seen.cards] == [tool] and seen.lines == []


def test_a_mode_changed_to_auto_mid_session_is_followed():
    from claude_agent_sdk import PermissionResultDeny

    seen = Seen()
    provider = _claude(seen, permission_mode="default")
    provider.ctx.permission_mode = "auto"
    assert isinstance(_decide(provider, "Bash", {"command": SED}), PermissionResultDeny)
    assert seen.cards == []


def test_a_research_session_is_untouched():
    from claude_agent_sdk import PermissionResultDeny
    from runner.providers.claude import RESEARCH_REFUSED

    seen = Seen()
    answer = _decide(_claude(seen, kind=RESEARCH_KIND, permission_mode="auto"), "Bash", {"command": SED})
    assert isinstance(answer, PermissionResultDeny) and answer.message == RESEARCH_REFUSED
    assert seen.cards == [] and seen.lines == []
    assert not refusals.claude_refuses(seen.ctx(kind=RESEARCH_KIND, permission_mode="auto"), "Bash")
    assert refusals.claude_refuses(seen.ctx(kind=CODING_KIND, permission_mode="auto"), "Bash")


@pytest.mark.parametrize("mode", [None, "manual", "accept-edits", "plan", "auto"])
def test_a_codex_escalation_is_declined_in_every_mode_with_no_card(mode):
    seen = Seen()
    answer, provider = _codex_answer(seen, refusals.CODEX_COMMAND_APPROVAL, ESCALATION, mode)
    assert answer == {"decision": "decline"}
    assert seen.cards == []
    assert provider._pending_feedback == [refusals.CODEX_ESCALATION]
    assert seen.lines == [{"text": refusals.CODEX_LINE, "tool": "command", "command": ESCALATION["command"],
                           "reason": ESCALATION["reason"]}]


def test_a_request_for_additional_permissions_is_an_escalation_too():
    seen = Seen()
    asked = {**ORDINARY, "additionalPermissions": {"fileSystem": {"read": ["/home/agent"]}}}
    answer, _ = _codex_answer(seen, refusals.CODEX_COMMAND_APPROVAL, asked)
    assert answer == {"decision": "decline"} and seen.cards == [] and len(seen.lines) == 1


@pytest.mark.parametrize("params", [ORDINARY,
                                    {**ESCALATION, "networkApprovalContext": {"host": "example.com", "protocol": "https"}},
                                    {**ESCALATION, "kind": "writeStdin"}])
def test_an_ordinary_codex_command_approval_behaves_as_before(params):
    seen = Seen()
    answer, provider = _codex_answer(seen, refusals.CODEX_COMMAND_APPROVAL, params, "manual")
    assert answer == {"decision": "accept"}
    assert [c["tool"] for c in seen.cards] == ["command"] and seen.lines == []
    assert provider._pending_feedback == []


def test_a_codex_file_change_approval_still_raises_a_card():
    seen = Seen(Decision(False, "Not there."))
    answer, provider = _codex_answer(seen, "item/fileChange/requestApproval",
                                     {"threadId": "t", "turnId": "u", "itemId": "exec-2", "startedAtMs": 1,
                                      "reason": "write outside", "grantRoot": None})
    assert answer == {"decision": "decline"} and [c["tool"] for c in seen.cards] == ["fileChange"]
    assert seen.lines == [] and provider._pending_feedback == ["Not there."]


def test_a_codex_research_session_is_untouched():
    assert not refusals.codex_refuses(Seen().ctx(kind=RESEARCH_KIND), refusals.CODEX_COMMAND_APPROVAL, ESCALATION)
    assert refusals.codex_refuses(Seen().ctx(), refusals.CODEX_COMMAND_APPROVAL, ESCALATION)


def test_a_permissions_request_is_answered_with_an_empty_grant():
    from runner.providers.codex import CodexProvider

    assert CodexProvider()._on_server_request("item/permissions/requestApproval", {}) == {"permissions": {},
                                                                                         "scope": "turn"}


def test_a_refusal_is_a_runner_line_on_the_stream_and_no_approval(orch, client, tmp_path):
    from claude_agent_sdk import PermissionResultDeny
    from runner.providers.claude import ClaudeProvider

    session = Session(FakeProvider({"provider": "claude-code"}), client, Options(cwd=str(tmp_path)),
                      Caps(wall_clock=30, grace=1), fast_timing(approval_ceiling=0.2))
    provider = ClaudeProvider()
    provider.ctx = ProviderContext(request_approval=session.relay.request, cwd=str(tmp_path), permission_mode="auto",
                                   refusal_note=session._refusal_note)

    async def go():
        refused = await provider._can_use_tool("Bash", {"command": SED}, None)
        provider.ctx.permission_mode = "default"
        asked = await provider._can_use_tool("Bash", {"command": SED}, None)
        return refused, asked
    refused, asked = asyncio.run(go())
    assert isinstance(refused, PermissionResultDeny) and refused.message == refusals.CLAUDE_SHELL
    lines = orch.of("message", role="runner", purpose="refused")
    assert lines == [{"role": "runner", "purpose": "refused", "text": refusals.CLAUDE_LINE, "tool": "Bash",
                      "command": SED}]
    assert len(orch.of("approval")) == 1
    assert orch.kinds() == ["message", "approval"]

