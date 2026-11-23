"""The settings control (model, effort, mode and thinking of a running session, from the next turn) and
drafts (replies as they are written, best-effort, outside the stream)."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from helpers import (CodexStub, SESSION, StubClient, claude_go as _go, claude_sdk_stub as claude, codex_go as _codex_go,
                     codex_on_stub as _codex, run_settings_session as _run, say, settings_acks as _acks,
                     settings_done as done, turn_starts as _starts)
from runner import drafts as drafts_mod
from runner import settings as words
from runner.control_plane import Control, pull_all
from runner.providers.base import CODING_KIND, RESEARCH_KIND


def _finish_on_second_wait(orch):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
              and len(orch.of("turn", phase="waiting")) == 2, lambda e: orch.control("finish"))


def test_pull_reads_a_settings_control_and_drops_one_that_names_nothing(orch, client):
    orch.control("settings", model="claude-opus-5-5", effort="high", mode="plan")
    orch.control("settings")
    orch.control("settings", effort=None, colour="blue")
    orch.control("settings", effort="turbo")
    _, _, controls = asyncio.run(pull_all(client))
    assert controls == [Control(1, "settings", None, {"model": "claude-opus-5-5", "effort": "high", "mode": "plan"}),
                        Control(4, "settings", None, {"effort": "turbo"})]


@pytest.mark.parametrize("field,value,reason", [
    ("model", "bad id!", "not a well-formed model id"), ("model", 5, "not a well-formed model id"),
    ("effort", "turbo", "effort must be one of low, medium, high, xhigh, max"),
    ("effort", "ultra", "effort must be one of low, medium, high, xhigh, max"),
    ("mode", "bypass", "mode must be one of manual, accept-edits, plan, auto"),
    ("mode", "bypassPermissions", "mode must be one of manual, accept-edits, plan, auto")])
def test_a_malformed_value_is_refused_before_any_provider(field, value, reason):
    assert words.check(field, value) == reason


def test_between_turns_a_change_applies_at_once_and_the_next_turn_names_it(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting",
              lambda e: (orch.control("settings", model="claude-opus-5-5", effort="max", mode="accept-edits"),
                         orch.queue("second")))
    _finish_on_second_wait(orch)
    fixture = {"provider": "claude-code", "default_turn": [say("ok"), done()]}
    provider, _ = _run(fixture, client, worktree)
    [ack] = _acks(orch)
    assert ack == {"role": "runner", "purpose": "settings", "control_id": 1,
                   "text": "From the next turn: model claude-opus-5-5, effort max, mode acceptEdits.",
                   "applied": {"model": "claude-opus-5-5", "effort": "max", "mode": "acceptEdits"},
                   "held": {}, "refused": {}}
    first, second = _starts(orch)[:2]
    assert first == {"phase": "started"}
    assert second == {"phase": "started", "model": "claude-opus-5-5", "effort": "max", "mode": "accept-edits"}
    assert provider.settings_calls == [{"model": "claude-opus-5-5", "effort": "max", "mode": "accept-edits"}]
    seqs = [e["sequence"] for e in orch.events]
    assert seqs == list(range(1, len(seqs) + 1))


def test_during_a_turn_a_change_is_held_and_applied_when_the_next_turn_starts(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "working",
              lambda e: (orch.control("settings", effort="low", mode="plan"), orch.queue("second")))
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting",
              lambda e: orch.control("finish"))
    fixture = {"provider": "codex", "turns": [[
        {"method": "turn/started", "params": {"turn": {"id": "t1"}}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": "m1", "text": "working"}}},
        {"fake": "sleep", "seconds": 0.3},
        {"method": "turn/completed", "params": {"turn": {"id": "t1", "status": "completed"}}}]],
        "default_turn": [{"method": "turn/started", "params": {"turn": {"id": "t2"}}},
                         {"method": "turn/completed", "params": {"turn": {"id": "t2", "status": "completed"}}}]}
    provider, _ = _run(fixture, client, worktree)
    held, applied = _acks(orch)
    assert held["held"] == {"effort": "low", "mode": "plan"} and held["applied"] == {} and held["control_id"] == 1
    assert held["text"] == "Held until the running turn ends, then applied: effort low, mode plan."
    assert applied["applied"] == {"effort": "low", "mode": "approval on-request, sandbox read-only, network off"}
    assert applied["held"] == {} and applied["control_id"] == 1
    starts = _starts(orch)
    assert starts[0] == {"phase": "started", "turn_id": "t1"}
    assert starts[1] == {"phase": "started", "turn_id": "t2", "effort": "low", "mode": "plan"}
    kinds = [(e["kind"], e["payload"].get("purpose"), e["payload"].get("text")) for e in orch.events]
    assert kinds.index(("message", "settings", applied["text"])) < kinds.index(("message", None, "second"))
    assert provider.settings_calls == [{"effort": "low", "mode": "plan"}]


def test_a_refused_field_changes_nothing_and_says_why(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting",
              lambda e: (orch.control("settings", model="no good", effort="turbo", mode="auto"),
                         orch.queue("second")))
    _finish_on_second_wait(orch)
    fixture = {"provider": "claude-code", "default_turn": [say("ok"), done()],
               "refuse_settings": {"mode": "auto mode is not available on this machine: test"}}
    provider, _ = _run(fixture, client, worktree)
    [ack] = _acks(orch)
    assert ack["applied"] == {} and ack["held"] == {}
    assert ack["refused"] == {"model": "not a well-formed model id",
                              "effort": "effort must be one of low, medium, high, xhigh, max",
                              "mode": "auto mode is not available on this machine: test"}
    assert ack["text"].startswith("Not changed: model (not a well-formed model id); effort")
    assert all(set(s) <= {"phase"} for s in _starts(orch))
    assert provider.ctx.permission_mode == "default"


def test_a_bounded_session_accepts_the_control(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "one",
              lambda e: orch.control("settings", model="claude-sonnet-5-5"))
    fixture = {"provider": "claude-code", "turns": [[say("one"), {"fake": "sleep", "seconds": 0.3}, done()]]}
    provider, _ = _run(fixture, client, worktree, driven=False)
    [ack] = _acks(orch)
    assert ack["held"] == {"model": "claude-sonnet-5-5"}
    assert provider.settings_calls == [] and orch.kinds()[-1] == "end"


def test_a_session_without_the_control_posts_what_it_did_before(orch, client, worktree):
    fixture = {"provider": "claude-code", "default_turn": [say("ok"), done()]}
    _run(fixture, client, worktree, driven=False)
    assert _starts(orch) == [{"phase": "started"}]
    assert orch.of("message", role="assistant") == [{"role": "assistant", "text": "ok"}]
    assert not [r for r in orch.requests if r["op"] == "sessions.draft_post"]


@pytest.mark.parametrize("kind", [CODING_KIND, RESEARCH_KIND])
def test_claude_provider_refuses_to_construct_a_client_without_managed_settings(claude, managed_policy, kind):
    from runner.claude_policy import ClaudePolicyError

    provider, ctx = claude
    ctx.kind = kind
    managed_policy.unlink()
    with pytest.raises(ClaudePolicyError, match="managed settings are missing"):
        asyncio.run(provider.start(ctx))
    assert StubClient.made == []


DIRECTORY = "sit in a directory that root does not own or that group or other users can write to"
DROP_IN_DIRECTORY = ("have a managed-settings.d beside them that is not a root-owned directory writable only by "
                     "root, or that the runner cannot list")


# One case for each problem the start check names, each refused with the full sentence before any client
# exists. Ownership and mode come from the opened file and directories, so the fixture's fake owner is replaced
# here by one keyed to each file's inode.
@pytest.mark.parametrize("problem,arrange", [
    ("are missing", lambda path, policy, owner: path.unlink()),
    ("could not be read as a JSON object", lambda path, policy, owner: path.write_text("[1, 2]")),
    ("are not owned by root", lambda path, policy, owner: owner.update({path.stat().st_ino: 4242})),
    ("are writable by group or other users", lambda path, policy, owner: path.chmod(0o664)),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"].update(enabled=False), path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"].update(failIfUnavailable=False), path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"].update(allowUnsandboxedCommands=True), path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"].pop("network"), path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"]["network"].pop("allowedDomains"),
        policy.update(notes={"allowedDomains": ["example.invalid"]}),
        policy["sandbox"].update(filesystem={"allowedDomains": ["example.invalid"]}),
        path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"]["network"].update(allowedDomains=[]), path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"]["network"].update(allowedDomains="example.invalid"), path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"]["network"].update(allowedDomains=["example.invalid", ""]),
        path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"]["network"].update(allowedDomains=["example.invalid", 7]),
        path.write_text(json.dumps(policy)))),
    ("lack the required sandbox settings or network allowlist", lambda path, policy, owner: (
        policy["sandbox"]["network"].pop("allowManagedDomainsOnly"), path.write_text(json.dumps(policy)))),
    ("are extended by files in managed-settings.d, which the runner does not check", lambda path, policy, owner: (
        (path.parent / "managed-settings.d").mkdir(),
        (path.parent / "managed-settings.d" / "50-extra.json").write_text("{}"))),
    ("are a symbolic link, which the runner does not follow", lambda path, policy, owner: (
        (path.parent / "real.json").write_text(path.read_text()), path.unlink(),
        path.symlink_to(path.parent / "real.json"))),
    (DIRECTORY, lambda path, policy, owner: path.parent.chmod(0o770)),
    (DIRECTORY, lambda path, policy, owner: path.parent.chmod(0o707)),
    (DIRECTORY, lambda path, policy, owner: owner.update({path.parent.stat().st_ino: 4242})),
    ("name a policyHelper program, whose output the runner cannot check", lambda path, policy, owner: (
        policy.update(policyHelper={"path": "/usr/local/libexec/policy-helper"}), path.write_text(json.dumps(policy)))),
    ("name a policyHelper program, whose output the runner cannot check", lambda path, policy, owner: (
        policy.update(policyHelpers=[{"path": "/usr/local/libexec/policy-helper"}]),
        path.write_text(json.dumps(policy)))),
    (DROP_IN_DIRECTORY, lambda path, policy, owner: (path.parent / "managed-settings.d").write_text("{}")),
    (DROP_IN_DIRECTORY, lambda path, policy, owner: (
        (path.parent / "managed-settings.d").mkdir(), (path.parent / "managed-settings.d").chmod(0o777))),
    (DROP_IN_DIRECTORY, lambda path, policy, owner: (
        (path.parent / "managed-settings.d").mkdir(),
        owner.update({(path.parent / "managed-settings.d").stat().st_ino: 4242}))),
    (DROP_IN_DIRECTORY, lambda path, policy, owner: (
        (path.parent / "elsewhere").mkdir(),
        (path.parent / "managed-settings.d").symlink_to(path.parent / "elsewhere"))),
], ids=["missing", "not-an-object", "not-root", "group-writable", "sandbox-off", "may-run-without-sandbox",
        "unsandboxed-allowed", "no-allowlist", "allowlist-word-elsewhere", "allowlist-empty", "allowlist-a-string",
        "allowlist-empty-entry", "allowlist-non-string-entry", "allowlist-widenable", "drop-in-file", "file-a-link",
        "directory-group-writable", "directory-other-writable", "directory-not-root", "policy-helper",
        "policy-helpers", "drop-in-a-regular-file", "drop-in-writable", "drop-in-not-root", "drop-in-a-link"])
def test_claude_start_is_refused_for_each_managed_settings_problem(claude, managed_policy, monkeypatch,
                                                                   problem, arrange):
    from os import fstat
    from types import SimpleNamespace

    from runner import claude_policy

    provider, ctx = claude
    owner: dict[int, int] = {}
    monkeypatch.setattr(claude_policy, "fstat", lambda fd: SimpleNamespace(
        st_mode=fstat(fd).st_mode, st_uid=owner.get(fstat(fd).st_ino, 0)))
    arrange(managed_policy, json.loads(managed_policy.read_text()), owner)
    with pytest.raises(claude_policy.ClaudePolicyError) as refused:
        asyncio.run(provider.start(ctx))
    assert str(refused.value) == (
        f"Claude Code could not start because its managed settings {problem}, so install them at "
        f"{claude_policy.managed_settings_path()} as a root-owned file writable only by root, with the sandbox "
        "settings and network allowlist that examples/managed-settings.json shows.")
    assert StubClient.made == []


# The file checked is the one Claude Code reads on the platform the runner reports, wherever the tests run.
@pytest.mark.parametrize("platform,path", [
    ("linux", "/etc/claude-code/managed-settings.json"),
    ("darwin", "/Library/Application Support/ClaudeCode/managed-settings.json"),
])
def test_the_checked_file_is_the_one_claude_code_reads_on_the_platform(monkeypatch, platform, path):
    from runner import claude_policy

    opened = []

    def absent(name, *args, **kwargs):
        opened.append(name)
        raise FileNotFoundError(name)

    monkeypatch.setattr(claude_policy, "PLATFORM", platform)
    monkeypatch.setattr(claude_policy, "MANAGED_SETTINGS", None)
    monkeypatch.setattr(claude_policy, "_open_directory", absent)
    with pytest.raises(claude_policy.ClaudePolicyError) as refused:
        claude_policy.require_managed_settings()
    assert opened == [os.path.dirname(path)]
    assert str(refused.value) == (
        f"Claude Code could not start because its managed settings are missing, so install them at {path} as a "
        "root-owned file writable only by root, with the sandbox settings and network allowlist that "
        "examples/managed-settings.json shows.")


@pytest.mark.parametrize("platform", ["win32", "cygwin", "freebsd14"])
def test_claude_start_is_refused_on_a_platform_the_check_has_no_path_for(claude, monkeypatch, platform):
    from runner import claude_policy

    provider, ctx = claude
    monkeypatch.setattr(claude_policy, "PLATFORM", platform)
    with pytest.raises(claude_policy.ClaudePolicyError) as refused:
        asyncio.run(provider.start(ctx))
    assert str(refused.value) == (
        "Claude Code could not start because the runner can check its managed settings only on Linux and macOS, "
        f"and this machine reports {platform}.")
    assert StubClient.made == []


# Only a *.json drop-in is merged by Claude Code, so other files beside the settings do not matter. A
# drop-in directory that cannot be listed is refused, because it may hold one, and its sentence says so.
def test_drop_in_files_other_than_json_pass_and_an_unlistable_directory_is_refused(claude, managed_policy):
    from runner import claude_policy

    provider, ctx = claude
    drop_ins = managed_policy.parent / "managed-settings.d"
    drop_ins.mkdir()
    (drop_ins / "README.txt").write_text("notes\n")
    asyncio.run(provider.start(ctx))
    assert len(StubClient.made) == 1
    for name in ("10-extra.json", ".hidden.JSON"):
        (drop_ins / name).write_text("{}")
        with pytest.raises(claude_policy.ClaudePolicyError, match="are extended by files in managed-settings.d"):
            claude_policy.require_managed_settings()
        (drop_ins / name).unlink()
    drop_ins.chmod(0)
    try:
        if os.access(drop_ins, os.R_OK):
            pytest.skip("running as a user who can list any directory")
        with pytest.raises(claude_policy.ClaudePolicyError) as refused:
            claude_policy.require_managed_settings()
        assert claude_policy.DROP_IN_DIRECTORY in str(refused.value)
    finally:
        drop_ins.chmod(0o755)


# managedSourcesBehavior only decides how Claude Code combines managed sources ranked above or beside the
# file. A higher source wins over the file under either value, so the key changes nothing the check vouches
# for and is not refused.
@pytest.mark.parametrize("behaviour", ["first-wins", "merge"])
def test_managed_sources_behaviour_is_not_refused(claude, managed_policy, behaviour):
    provider, ctx = claude
    policy = json.loads(managed_policy.read_text())
    managed_policy.write_text(json.dumps({**policy, "managedSourcesBehavior": behaviour}))
    asyncio.run(provider.start(ctx))
    assert len(StubClient.made) == 1


def test_a_reconnect_checks_the_managed_settings_again(claude, managed_policy):
    from runner.claude_policy import ClaudePolicyError

    provider, ctx = claude

    async def reconnect():
        await provider.start(ctx)
        managed_policy.unlink()
        await provider.change_settings({"effort": "max"})
        [r async for r in provider.turn("hello")]

    with pytest.raises(ClaudePolicyError, match="managed settings are missing"):
        asyncio.run(reconnect())
    assert len(StubClient.made) == 1


# The example is the documented content: installed root-owned and not group-writable, it passes the check.
def test_the_example_managed_settings_pass_the_start_check(tmp_path):
    from runner import claude_policy
    from helpers import EXAMPLES

    installed = tmp_path / "managed-settings.json"
    installed.write_text((EXAMPLES / "managed-settings.json").read_text(encoding="utf-8"), encoding="utf-8")
    installed.chmod(0o644)
    info, data = claude_policy.read_managed_settings(str(installed))
    assert claude_policy.sandbox_problem(info) is None
    assert {k: data["sandbox"][k] for k in claude_policy.SANDBOX_FLAGS} == claude_policy.SANDBOX_FLAGS
    claude_policy.require_managed_settings(str(installed))


def test_claude_changes_model_and_mode_live_and_effort_by_a_reconnect(claude):
    provider, ctx = claude
    [answer] = _go(provider, ctx, "turn", {"model": "claude-sonnet-5-5", "mode": "accept-edits", "effort": "max"},
                   "turn")
    assert {k: (v.applied, v.value) for k, v in answer.items()} == {
        "model": (True, "claude-sonnet-5-5"), "mode": (True, "acceptEdits"), "effort": (True, "max")}
    first, second = StubClient.made
    assert first.calls == [("set_model", "claude-sonnet-5-5"), ("set_permission_mode", "acceptEdits")]
    assert second.options.effort == "max" and second.options.model == "claude-sonnet-5-5"
    assert second.options.permission_mode == "acceptEdits" and second.options.resume == SESSION
    assert provider.reconnects == 1
    assert provider.settings_now() == {"model": "claude-sonnet-5-5", "effort": "max", "mode": "accept-edits",
                                       "thinking": "off"}
    assert second.options.thinking is None


def test_claude_model_changes_alone_never_reconnect_but_fable_in_or_out_does(claude):
    provider, ctx = claude
    _go(provider, ctx, "turn", {"model": "claude-sonnet-5-5"}, "turn", {"model": "claude-fable-5-2"}, "turn")
    assert provider.reconnects == 1
    assert StubClient.made[1].options.model == "claude-fable-5-2"
    assert StubClient.made[1].options.env["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-opus-5-5"


@pytest.mark.parametrize("model,why", [("fable", "not a full Claude model id"),
                                       ("claude-fable-5-1", "not Fable 5.2")])
def test_claude_refuses_an_alias_or_another_fable(claude, model, why):
    provider, ctx = claude
    [answer] = _go(provider, ctx, {"model": model})
    assert not answer["model"].applied and why in answer["model"].reason
    assert ctx.model == "claude-opus-5-5" and StubClient.made[0].calls == []


def test_auto_is_allowed_by_the_machines_managed_settings_and_set_live(claude):
    provider, ctx = claude
    [answer] = _go(provider, ctx, {"mode": "auto"})
    assert answer["mode"].applied and answer["mode"].value == "auto"
    assert StubClient.made[0].calls == [("set_permission_mode", "auto")]


def test_auto_is_refused_when_managed_settings_turn_it_off(claude, tmp_path, managed_policy):
    provider, ctx = claude
    managed = tmp_path / "managed.json"
    policy = json.loads(managed_policy.read_text())
    policy["permissions"] = {"disableAutoMode": "disable"}
    managed.write_text(json.dumps(policy))
    managed.chmod(0o644)
    provider.managed_settings = str(managed)
    [answer] = _go(provider, ctx, {"mode": "auto", "effort": "low"})
    assert not answer["mode"].applied and "disableAutoMode" in answer["mode"].reason
    assert answer["effort"].applied
    assert StubClient.made[0].calls == [] and ctx.permission_mode == "default"


def test_auto_is_refused_when_managed_settings_go_missing(claude, managed_policy):
    from runner import claude_policy

    provider, ctx = claude

    async def switch_mode():
        await provider.start(ctx)
        managed_policy.unlink()
        return await provider.change_settings({"mode": "auto"})

    answer = asyncio.run(switch_mode())["mode"]
    assert not answer.applied and answer.reason == (
        "auto mode is not available on this machine: its managed settings are missing, so install them at "
        f"{claude_policy.managed_settings_path()} (examples/managed-settings.json shows the required content)")
    assert StubClient.made[0].calls == [] and ctx.permission_mode == "default"


def test_auto_is_refused_when_claude_code_refuses_it(claude):
    provider, ctx = claude
    StubClient.refuse_modes = ("auto",)
    [answer] = _go(provider, ctx, {"mode": "auto"})
    assert answer["mode"].reason == ("auto mode is not available on this machine: Claude Code refused it "
                                     "(Auto mode is unavailable for your plan)")
    assert ctx.permission_mode == "default"


def test_claude_never_offers_bypass():
    from runner.providers import claude
    assert "bypassPermissions" not in claude.MODES and "bypassPermissions" not in words.CLAUDE_MODE.values()


def test_claude_asks_for_partial_frames_only_with_drafts(claude):
    provider, ctx = claude
    assert provider.options(ctx).include_partial_messages is False
    ctx.drafts = True
    assert provider.options(ctx).include_partial_messages is True


@pytest.mark.parametrize("schema", [None, {"type": "object"}])
def test_every_client_the_claude_provider_opens_carries_the_partial_frames_flag(claude, schema):
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    def command(resume=None):
        transport = SubprocessCLITransport("", provider.options(ctx, resume=resume))
        transport._cli_path = "claude"
        return transport._build_command()

    provider, ctx = claude
    ctx.drafts, ctx.output_schema = True, schema
    for resume in (None, SESSION):
        args = command(resume)
        assert "--include-partial-messages" in args and "--verbose" in args
        assert args[args.index("--output-format") + 1] == "stream-json"
        assert ("--json-schema" in args) is (schema is not None)
    ctx.drafts = False
    assert "--include-partial-messages" not in command()


def test_the_claude_reconnects_keep_asking_for_partial_frames(claude):
    provider, ctx = claude
    ctx.drafts, ctx.output_schema = True, None

    async def run():
        await provider.start(ctx)
        [r async for r in provider.turn("hello")]
        ctx.output_schema = {"type": "object"}
        [r async for r in provider.turn("summary")]
        await provider.change_settings({"effort": "max"})
        [r async for r in provider.turn("again")]
    asyncio.run(run())
    assert provider.reconnects == 2
    assert [c.options.include_partial_messages for c in StubClient.made] == [True, True, True]


@pytest.mark.parametrize("word,policy,sandbox", [
    ("manual", "untrusted", "workspaceWrite"), ("accept-edits", "on-request", "workspaceWrite"),
    ("plan", "on-request", "readOnly"), ("auto", "never", "workspaceWrite")])
def test_codex_modes_keep_the_sandbox_and_the_network_off(word, policy, sandbox):
    stub = CodexStub()
    _codex_go(_codex(stub), "turn", {"mode": word}, "turn", "turn")
    assert stub.started[0] is None and stub.started[2] is None
    profile = "session_read" if sandbox == "readOnly" else "session_write"
    assert stub.started[1] == {"approvalPolicy": policy, "permissions": profile}
    for policy_value, box in words.CODEX_MODE.values():
        assert policy_value in ("untrusted", "on-request", "never") and box["networkAccess"] is False
        assert box["type"] in ("workspaceWrite", "readOnly")


def test_codex_model_and_effort_ride_the_next_turn_start_with_the_schema():
    stub = CodexStub()
    provider = _codex(stub)
    provider.ctx.output_schema = {"type": "object"}
    [answer] = _codex_go(provider, {"model": "gpt-5.5-codex", "effort": "medium"}, "turn")
    assert stub.started == [{"outputSchema": {"type": "object"}, "model": "gpt-5.5-codex", "effort": "medium"}]
    assert answer["model"].value == "gpt-5.5-codex" and answer["effort"].value == "medium"
    assert provider.settings_now() == {"model": "gpt-5.5-codex", "effort": "medium", "mode": "accept-edits",
                                       "thinking": "off"}


def test_codex_refuses_a_model_it_does_not_list_and_sends_nothing():
    stub = CodexStub([("gpt-a", True, ["low"])])
    provider = _codex(stub)
    [answer] = _codex_go(provider, {"model": "gpt-zzz", "mode": "auto"}, "turn")
    assert not answer["model"].applied and "model/list" in answer["model"].reason
    assert answer["mode"].applied and "model" not in stub.started[0] and provider.ctx.model is None


@pytest.mark.parametrize("models,model,expected", [
    ([("gpt-a", True, ["low", "medium", "high", "xhigh"])], None, "xhigh"),
    ([("gpt-a", True, ["low", "high"]), ("gpt-b", False, ["low", "max", "ultra"])], "gpt-b", "ultra"),
    ([("gpt-a", True, ["minimal", "low"])], "gpt-a", "low"),
    (None, None, "xhigh"),
    ([("gpt-a", True, [])], "gpt-z", "xhigh")])
def test_codex_max_is_the_highest_effort_the_model_lists(models, model, expected):
    stub = CodexStub(models)
    [answer] = _codex_go(_codex(stub, model), {"effort": "max"})
    assert answer["effort"].value == expected


@pytest.mark.parametrize("models,word,expected", [
    ([("gpt-a", True, ["low", "medium", "high"])], "high", "high"),
    ([("gpt-a", True, ["low", "medium", "high", "xhigh", "max"])], "xhigh", "xhigh"),
    ([("gpt-a", True, ["low", "medium", "high", "xhigh"])], "max", "xhigh"),
    (None, "medium", "medium")])
def test_codex_start_effort_rides_the_first_turn_once(models, word, expected):
    stub = CodexStub(models)
    provider = _codex(stub)
    provider.ctx.effort = word

    async def run():
        await provider.start_effort()
        for _ in range(2):
            [r async for r in provider.turn("hello")]
    asyncio.run(run())
    assert stub.started == [{"effort": expected}, None]
    assert provider.settings_now()["effort"] == expected and provider.start_refused == {}


def test_codex_start_effort_the_model_does_not_support_runs_on_the_default_and_says_why():
    stub = CodexStub([("gpt-a", True, ["medium", "high"])])
    provider = _codex(stub)
    provider.ctx.effort = "low"

    async def run():
        await provider.start_effort()
        [r async for r in provider.turn("hello")]
    asyncio.run(run())
    assert stub.started == [None] and provider.effort is None
    assert provider.start_refused == {"effort": "the model does not support low; the session runs on the default effort"}


def test_codex_start_effort_xhigh_on_a_model_without_it_runs_on_the_default_and_says_why():
    stub = CodexStub([("gpt-a", True, ["low", "medium", "high"])])
    provider = _codex(stub)
    provider.ctx.effort = "xhigh"
    asyncio.run(provider.start_effort())
    assert provider.overrides == {} and provider.effort is None
    assert provider.start_refused == {"effort": "the model does not support xhigh; the session runs on the default effort"}


def test_xhigh_is_a_settings_word_for_both_providers():
    from runner.providers.claude import EFFORTS as CLAUDE_EFFORTS
    assert words.EFFORTS == ("low", "medium", "high", "xhigh", "max") and words.check("effort", "xhigh") is None
    assert words.CLAUDE_EFFORT["xhigh"] == "xhigh" and "xhigh" in CLAUDE_EFFORTS
    assert words.CODEX_EFFORT["xhigh"] == "xhigh"
    stub = CodexStub([("gpt-a", True, ["low", "xhigh"])])
    [answer] = _codex_go(_codex(stub), {"effort": "xhigh"}, "turn")
    assert answer["effort"].value == "xhigh" and stub.started == [{"effort": "xhigh"}]


def test_codex_without_a_start_effort_sends_nothing():
    stub = CodexStub([("gpt-a", True, ["low"])])
    provider = _codex(stub)
    asyncio.run(provider.start_effort())
    assert provider.overrides == {} and provider.start_refused == {}


def test_the_runner_takes_drafts_from_its_command_line():
    from runner.__main__ import parse
    base = ["--provider", "fake", "--cwd", "/tmp", "--prompt-file", "/tmp/p"]
    assert parse(base + ["--drafts"]).drafts is True and parse(base).drafts is False


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _stream(event, parent=None):
    return {"type": "StreamEvent", "uuid": "u", "session_id": "s", "event": event, "parent_tool_use_id": parent}


def _drafter(provider):
    clock, sent = Clock(), []

    async def send(body):
        sent.append(body)
    return drafts_mod.Drafter(provider, send, clock=clock), clock, sent


def test_claude_text_is_drafted_at_most_once_a_second_and_closed_by_its_message():
    async def go():
        drafter, clock, sent = _drafter("claude-code")
        drafter.feed(_stream({"type": "message_start", "message": {"id": "msg_1"}}))
        drafter.feed(_stream({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}))
        drafter.feed(_stream({"type": "content_block_delta", "index": 0,
                              "delta": {"type": "thinking_delta", "thinking": "secret"}}))
        drafter.feed(_stream({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}))
        for i, word in enumerate(["Hello", " there", ",", " friend"]):
            clock.now = 0.4 * (i + 1)
            drafter.feed(_stream({"type": "content_block_delta", "index": 1,
                                  "delta": {"type": "text_delta", "text": word}}))
            await asyncio.sleep(0)
        drafter.feed(_stream({"type": "content_block_delta", "index": 0,
                              "delta": {"type": "text_delta", "text": "sub"}}, parent="toolu_a"))
        drafter.feed(say("Hello there, friend", "msg_1"))
        clock.now = 9
        drafter.feed(_stream({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "!"}}))
        await asyncio.sleep(0)
        return sent
    sent = asyncio.run(go())
    assert sent == [{"message_id": "msg_1:0", "offset": 0, "text": "Hello there,"}]


def test_codex_deltas_are_drafted_by_item_id():
    async def go():
        drafter, clock, sent = _drafter("codex")
        drafter.feed({"method": "item/started", "params": {"item": {"type": "agentMessage", "id": "m1"}}})
        drafter.feed({"method": "item/started", "params": {"item": {"type": "commandExecution", "id": "c1"}}})
        for i in range(5):
            clock.now = 0.6 * (i + 1)
            drafter.feed({"method": "item/agentMessage/delta", "params": {"itemId": "m1", "delta": f"w{i} "}})
            await asyncio.sleep(0)
        drafter.feed({"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": "m1", "text": "x"}}})
        clock.now = 10
        drafter.feed({"method": "item/agentMessage/delta", "params": {"itemId": "m1", "delta": "late"}})
        await asyncio.sleep(0)
        return sent
    sent = asyncio.run(go())
    assert sent == [{"message_id": "m1", "offset": 0, "text": "w0 w1 "},
                    {"message_id": "m1", "offset": 0, "text": "w0 w1 w2 w3 "}]


def test_a_long_reply_is_drafted_as_its_tail_with_the_offset():
    text = "é" * 5000 + "x" * 20000
    offset, tail = drafts_mod.window(text)
    assert text[offset:] == tail and len(json.dumps(tail, ensure_ascii=False).encode()) - 2 <= drafts_mod.DRAFT_BYTES
    assert offset == len(text) - drafts_mod.DRAFT_BYTES
    assert drafts_mod.window("short") == (0, "short")


def test_a_draft_in_flight_skips_the_next_and_a_failure_is_ignored():
    async def go():
        gate = asyncio.Event()
        sent = []

        async def send(body):
            sent.append(body)
            await gate.wait()
            raise RuntimeError("refused")
        clock = Clock()
        drafter = drafts_mod.Drafter("codex", send, clock=clock)
        drafter.feed({"method": "item/started", "params": {"item": {"type": "agentMessage", "id": "m"}}})
        for i in range(4):
            clock.now = 1.5 * (i + 1)
            drafter.feed({"method": "item/agentMessage/delta", "params": {"itemId": "m", "delta": "a"}})
            await asyncio.sleep(0)
        assert [body["text"] for body in sent] == ["a"] and not drafter._task.done()
        gate.set()
        await drafter._task
        clock.now = 20
        drafter.feed({"method": "item/agentMessage/delta", "params": {"itemId": "m", "delta": "b"}})
        await drafter._task
        await drafter.close()
        return sent
    sent = asyncio.run(go())
    assert [b["text"] for b in sent] == ["a", "aaaab"]


def _codex_writing(text_parts, item="m1", pause=0.0):
    steps = [{"method": "turn/started", "params": {"turn": {"id": "t1"}}},
             {"method": "item/started", "params": {"item": {"type": "agentMessage", "id": item}}}]
    for part in text_parts:
        steps.append({"method": "item/agentMessage/delta", "params": {"itemId": item, "delta": part}})
        if pause:
            steps.append({"fake": "sleep", "seconds": pause})
    steps += [{"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": item,
                                                               "text": "".join(text_parts)}}},
              {"method": "turn/completed", "params": {"turn": {"id": "t1", "status": "completed"}}}]
    return {"provider": "codex", "turns": [steps]}


def test_a_session_with_drafts_posts_them_outside_the_stream(orch, client, worktree, monkeypatch):
    monkeypatch.setattr(drafts_mod, "INTERVAL", 0.05)
    orch.draft_reply = (200, {"stored": True})
    _run(_codex_writing([f"word{i} " for i in range(8)], pause=0.03), client, worktree, driven=False, drafts=True)
    assert orch.drafts, "drafts were posted"
    for body in orch.drafts:
        assert body["message_id"] == "m1" and body["offset"] == 0
        assert "".join(f"word{i} " for i in range(8)).startswith(body["text"])
    [message] = orch.of("message", role="assistant")
    assert message["message_id"] == "m1" and "part" not in message
    seqs = [e["sequence"] for e in orch.events]
    assert seqs == list(range(1, len(seqs) + 1))
    posts = [r for r in orch.requests if r["op"] == "sessions.event_post"]
    assert len(posts) == len(orch.events)


def test_drafts_against_todays_control_plane_cost_the_run_nothing(orch, client, worktree, monkeypatch):
    monkeypatch.setattr(drafts_mod, "INTERVAL", 0.05)
    _, session = _run(_codex_writing([f"w{i} " for i in range(6)], pause=0.03), client, worktree, driven=False,
                      drafts=True)
    assert [r for r in orch.requests if r["op"] == "sessions.draft_post"] and not orch.drafts
    assert orch.kinds()[-1] == "end" and orch.of("end")[0]["reason"] == "completed"


def test_draft_bodies_carry_exactly_three_fields():
    async def go():
        drafter, clock, sent = _drafter("codex")
        drafter.feed({"method": "item/started", "params": {"item": {"type": "agentMessage", "id": "m"}}})
        clock.now = 2
        drafter.feed({"method": "item/agentMessage/delta", "params": {"itemId": "m", "delta": "x"}})
        await asyncio.sleep(0)
        return sent
    [body] = asyncio.run(go())
    assert set(body) == {"message_id", "offset", "text"}


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "runner.drafts" and r.levelname == "INFO"]


def test_each_finished_reply_logs_its_drafts_and_failures_without_text(caplog):
    from runner.control_plane import Rejected

    caplog.set_level("INFO", logger="runner.drafts")

    async def go():
        async def send(body):
            raise Rejected(404, "not_found")
        clock = Clock()
        drafter = drafts_mod.Drafter("codex", send, clock=clock)
        drafter.feed({"method": "item/started", "params": {"item": {"type": "agentMessage", "id": "m1"}}})
        for i in range(3):
            clock.now = 1.5 * (i + 1)
            drafter.feed({"method": "item/agentMessage/delta", "params": {"itemId": "m1", "delta": "secret words "}})
            await drafter._task
        drafter.feed({"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": "m1"}}})
    asyncio.run(go())
    [line] = _lines(caplog)
    assert line == ("reply finished: 3 drafts sent, 3 failed (last: Rejected 404), 3 partial frames, "
                    "3 text deltas over 3.0 s, post in flight no (message m1)")
    assert "secret" not in line


def test_a_reply_with_no_partial_frames_still_leaves_its_line(caplog):
    caplog.set_level("INFO", logger="runner.drafts")
    drafter, _clock, sent = _drafter("claude-code")
    drafter.feed(say("Hello there", "msg_9"))
    assert _lines(caplog) == ["reply finished: 0 drafts sent, 0 failed, 0 partial frames, 0 text deltas over "
                              "0.0 s, post in flight no (message msg_9:0)"]
    assert sent == []


def _block(index, btype, deltas, delta_type, clock=None, step=0.0):
    key = {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json"}[delta_type]
    start = {"type": btype, **({"text": ""} if btype == "text" else {})}
    out = [("frame", _stream({"type": "content_block_start", "index": index, "content_block": start}))]
    for piece in deltas:
        out.append(("tick", step))
        out.append(("frame", _stream({"type": "content_block_delta", "index": index,
                                      "delta": {"type": delta_type, key: piece}})))
    out.append(("frame", _stream({"type": "content_block_stop", "index": index})))
    return out


def _assistant(mid, *blocks):
    shapes = {"thinking": {"type": "ThinkingBlock", "thinking": "t", "signature": "s"},
              "tool_use": {"type": "ToolUseBlock", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}}
    content = [shapes.get(b) or {"type": "TextBlock", "text": b} for b in blocks]
    return ("frame", {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None, "message_id": mid,
                      "content": content})


REPLY = [f"word{i} " for i in range(12)]


def _tool_turn(*, preamble=True, second_start=True, second_thinking=True):
    steps = [("frame", _stream({"type": "message_start", "message": {"id": "msg_A"}}))]
    steps += _block(0, "thinking", ["plan"], "thinking_delta") + [_assistant("msg_A", "thinking")]
    index = 1
    if preamble:
        steps += _block(1, "text", ["I'll list", " the files."], "text_delta", step=0.1)
        steps += [_assistant("msg_A", "I'll list the files.")]
        index = 2
    steps += _block(index, "tool_use", ['{"command":', ' "ls"}'], "input_json_delta")
    steps += [_assistant("msg_A", "tool_use"),
              ("frame", _stream({"type": "message_delta", "delta": {"stop_reason": "tool_use"}})),
              ("frame", _stream({"type": "message_stop"})),
              ("tick", 3.0),
              ("frame", {"type": "UserMessage", "parent_tool_use_id": None,
                         "content": [{"type": "ToolResultBlock", "tool_use_id": "toolu_1", "content": "a.txt"}]})]
    if second_start:
        steps.append(("frame", _stream({"type": "message_start", "message": {"id": "msg_B"}})))
    index = 0
    if second_thinking:
        steps += _block(0, "thinking", ["so"], "thinking_delta") + [_assistant("msg_B", "thinking")]
        index = 1
    steps += _block(index, "text", REPLY, "text_delta", step=0.4)
    steps += [_assistant("msg_B", "".join(REPLY)),
              ("frame", _stream({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})),
              ("frame", _stream({"type": "message_stop"})),
              ("frame", {"type": "ResultMessage", "subtype": "success", "is_error": False, "num_turns": 2,
                         "duration_ms": 1, "duration_api_ms": 1, "session_id": "s"})]
    return steps


def _play(steps, caplog=None):
    async def go():
        drafter, clock, sent = _drafter("claude-code")
        for kind, value in steps:
            if kind == "tick":
                clock.now += value
            else:
                drafter.feed(value)
                await asyncio.sleep(0)
        return sent
    return asyncio.run(go())


@pytest.mark.parametrize("preamble", [True, False])
def test_a_reply_after_tool_use_is_drafted_under_its_own_message_id(preamble, caplog):
    caplog.set_level("INFO", logger="runner.drafts")
    sent = _play(_tool_turn(preamble=preamble))
    assert {b["message_id"] for b in sent} == {"msg_B:0"} and len(sent) >= 3
    assert "".join(REPLY).startswith(sent[-1]["text"])
    lines = _lines(caplog)
    assert lines[-1].startswith(f"reply finished: {len(sent)} drafts sent, 0 failed") and "(message msg_B:0)" in lines[-1]
    assert "12 text deltas over 4.4 s" in lines[-1]


def test_a_response_without_its_message_start_is_never_drafted_under_the_previous_id(caplog):
    caplog.set_level("INFO", logger="runner.drafts")
    for preamble in (True, False):
        caplog.clear()
        sent = _play(_tool_turn(preamble=preamble, second_start=False, second_thinking=False))
        assert not [b for b in sent if b["message_id"].startswith("msg_A")]
        assert "1 text blocks with no message_start" in _lines(caplog)[-1]


def test_a_text_delta_whose_block_start_was_not_seen_still_opens_the_draft():
    steps = [s for s in _tool_turn() if not (s[0] == "frame" and s[1].get("type") == "StreamEvent"
                                            and s[1]["event"].get("type") == "content_block_start"
                                            and s[1]["event"]["content_block"]["type"] == "text"
                                            and s[1]["event"]["index"] == 1)]
    sent = _play(steps)
    assert {b["message_id"] for b in sent} == {"msg_B:0"} and len(sent) >= 3


def test_a_message_that_holds_its_text_at_another_index_closes_the_draft_and_names_both(caplog):
    caplog.set_level("INFO", logger="runner.drafts")
    steps = [s for s in _tool_turn(preamble=False)
             if not (s[0] == "frame" and s[1].get("type") == "AssistantMessage" and s[1]["message_id"] == "msg_B")]
    at = next(i for i, s in enumerate(steps) if s[0] == "frame" and s[1].get("type") == "StreamEvent"
              and s[1]["event"].get("type") == "message_delta" and s[1]["event"]["delta"]["stop_reason"] == "end_turn")
    whole = _assistant("msg_B", "thinking", "".join(REPLY))
    steps[at:at] = [whole, ("tick", 5.0),
                    ("frame", _stream({"type": "content_block_delta", "index": 1,
                                       "delta": {"type": "text_delta", "text": "late"}}))]
    sent = _play(steps)
    assert {b["message_id"] for b in sent} == {"msg_B:0"}
    assert not any("late" in b["text"] for b in sent)
    line = _lines(caplog)[-1]
    assert "(message msg_B:1, drafted as msg_B:0)" in line and line.startswith(f"reply finished: {len(sent)} drafts")


def test_a_failure_says_its_class_and_status_or_reason_only():
    from runner.control_plane import SessionEnded, TooLarge, Unreachable

    assert drafts_mod.why(SessionEnded(403, "forbidden", "a message")) == "SessionEnded 403"
    assert drafts_mod.why(Unreachable("URLError")) == "Unreachable URLError"
    assert drafts_mod.why(TooLarge()) == "TooLarge"

