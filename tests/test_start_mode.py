"""The mode a session starts in: --start-mode reaches either provider before the first turn."""

from __future__ import annotations

import asyncio

import pytest

from runner import settings as words
from helpers import (CodexStub, StubClient, claude_go as _go, claude_sdk_stub as claude, codex_on_stub as _codex,
                     run_settings_session as _run, say, settings_acks as _acks, settings_done as done,
                     turn_starts as _starts)


def test_a_start_in_auto_mode_is_applied_before_the_first_turn_and_named_by_it(orch, client, worktree):
    fixture = {"provider": "claude-code", "default_turn": [say("ok"), done()]}
    provider, _ = _run(fixture, client, worktree, driven=False, start_mode="auto")
    [ack] = _acks(orch)
    assert ack == {"role": "runner", "purpose": "settings", "control_id": None, "start": True,
                   "asked": {"mode": "auto"}, "mode": "auto", "text": "Started with: mode auto.",
                   "applied": {"mode": "auto"}, "held": {}, "refused": {}}
    assert provider.settings_calls == [{"mode": "auto"}]
    kinds = [(e["kind"], e["payload"].get("purpose") or e["payload"].get("role")) for e in orch.events]
    assert kinds.index(("message", "settings")) < kinds.index(("message", "user"))
    [first] = _starts(orch)
    assert first == {"phase": "started", "mode": "auto"}


@pytest.mark.parametrize("word, value", [("accept-edits", "acceptEdits"), ("plan", "plan"), ("manual", "default")])
def test_each_mode_maps_as_session_settings_maps_it(orch, client, worktree, word, value):
    fixture = {"provider": "claude-code", "default_turn": [say("ok"), done()]}
    provider, _ = _run(fixture, client, worktree, driven=False, start_mode=word)
    assert _acks(orch)[0]["applied"] == {"mode": value} and provider.ctx.permission_mode == value
    assert _starts(orch)[0]["mode"] == word


def test_a_mode_the_box_refuses_leaves_the_providers_own_start_and_says_why(orch, client, worktree):
    fixture = {"provider": "claude-code", "default_turn": [say("ok"), done()],
               "refuse_settings": {"mode": "auto mode is not available on this machine: test"}}
    provider, _ = _run(fixture, client, worktree, driven=False, start_mode="auto")
    [ack] = _acks(orch)
    assert ack["start"] is True and ack["applied"] == {} and ack["mode"] == "manual"
    assert ack["refused"] == {"mode": "auto mode is not available on this machine: test"}
    assert ack["text"] == "Not changed: mode (auto mode is not available on this machine: test)."
    assert provider.ctx.permission_mode == "default"
    assert _starts(orch)[0]["mode"] == "manual"
    assert orch.kinds()[-1] == "end"


def test_without_a_start_mode_the_session_posts_what_it_did_before(orch, client, worktree):
    fixture = {"provider": "claude-code", "default_turn": [say("ok"), done()]}
    provider, _ = _run(fixture, client, worktree, driven=False)
    assert _acks(orch) == [] and provider.settings_calls == []
    assert _starts(orch) == [{"phase": "started"}]


def test_claude_codes_auto_at_the_start_is_the_live_switch_before_the_first_query(claude):
    provider, ctx = claude
    [answer] = _go(provider, ctx, {"mode": "auto"}, "turn")
    assert answer["mode"].applied
    made = StubClient.made[0]
    assert made.calls == [("set_permission_mode", "auto")] and made.queries == ["hello"]
    assert made.options.permission_mode == "default"


@pytest.mark.parametrize("word", words.MODES)
def test_codex_takes_the_start_mode_on_its_first_turn_start(word):
    stub = CodexStub()
    provider = _codex(stub)

    async def run():
        await provider.change_settings({"mode": word})
        [r async for r in provider.turn("hello")]
    asyncio.run(run())
    policy, sandbox = words.CODEX_MODE[word]
    profile = "session_read" if sandbox["type"] == "readOnly" else "session_write"
    assert stub.started[0] == {"approvalPolicy": policy, "permissions": profile}
    assert provider.settings_now()["mode"] == word


def test_the_runner_takes_the_start_mode_from_its_command_line():
    from runner.__main__ import parse
    base = ["--provider", "fake", "--cwd", "/tmp", "--prompt-file", "/tmp/p"]
    assert parse(base + ["--start-mode", "accept-edits"]).start_mode == "accept-edits"
    assert parse(base).start_mode is None
    with pytest.raises(SystemExit):
        parse(base + ["--start-mode", "bypassPermissions"])

