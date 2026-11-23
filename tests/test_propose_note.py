"""Session proposals: the standing sentence and the sessions_propose tool are given only to a coding
session started with --tools and --propose, and the lead sentence only with --lead."""

from __future__ import annotations

import asyncio

import pytest

from conftest import fast_timing
from runner import __main__ as cli
from runner import config
from runner import handoff as handoff_mod
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Outcome, Session
from helpers import provider_files, research_result as result, say, typed_claude as claude, typed_codex as codex


def test_the_standing_note_adds_it_after_the_edit_note_only_on_a_literal_true():
    assert handoff_mod.standing_note() == handoff_mod.standing_note(False) == handoff_mod.EDIT_NOTE
    assert handoff_mod.standing_note("yes") == handoff_mod.EDIT_NOTE
    assert handoff_mod.standing_note(True) == f"{handoff_mod.EDIT_NOTE}\n\n{handoff_mod.PROPOSE_NOTE}"
    assert handoff_mod.brief_note("fix-backup", True).endswith(f"{handoff_mod.EDIT_NOTE}\n\n{handoff_mod.PROPOSE_NOTE}")
    assert handoff_mod.driven_note("fix-backup", False, True).endswith(handoff_mod.PROPOSE_NOTE)
    assert handoff_mod.PROPOSE_NOTE not in handoff_mod.brief_note("fix-backup")
    assert handoff_mod.PROPOSE_NOTE not in handoff_mod.driven_note("fix-backup", True)


def test_the_standing_note_adds_the_lead_sentence_only_on_a_literal_true_and_never_the_propose_one():
    edit, lead, propose = handoff_mod.EDIT_NOTE, handoff_mod.LEAD_NOTE, handoff_mod.PROPOSE_NOTE
    for flag in (False, None, "true", 1):
        assert lead not in handoff_mod.standing_note(False, flag)
        assert handoff_mod.standing_note(True, flag) == f"{edit}\n\n{propose}"
    assert handoff_mod.standing_note(False, True) == handoff_mod.standing_note(True, True) == f"{edit}\n\n{lead}"
    assert handoff_mod.brief_note("fix-backup", True, True).endswith(f"{edit}\n\n{lead}")
    assert handoff_mod.driven_note("fix-backup", False, True, True).endswith(f"{edit}\n\n{lead}")
    assert propose not in handoff_mod.brief_note("fix-backup", True, True)
    assert lead not in handoff_mod.brief_note("fix-backup", True) and lead not in handoff_mod.driven_note("x", True)


def _main_notes(monkeypatch, tmp_path, provider, flags_list):
    seen = []

    class StubSession:
        def __init__(self, provider, client, options, caps, timing, handoff=None):
            seen.append((options.brief_note, options.propose, options.lead))

        async def run(self, prompt):
            return Outcome("completed", 1, None, None)

    monkeypatch.setattr(cli, "Session", StubSession)
    monkeypatch.setattr(cli.config, "take_secrets", lambda: config.Settings(token="t", github_token="g"))
    brief = tmp_path / "brief.md"
    brief.write_text("Do it.\n")
    common = ["--provider", provider, "--cwd", str(tmp_path), "--prompt-file", str(brief)]
    for flags in flags_list:
        assert cli.main([*common, *flags]) == 0
    return seen


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_main_tells_only_a_session_with_tools_and_proposals(monkeypatch, tmp_path, provider):
    handed = ["--slug", "fix-backup", "--repo", "owner/repo"]
    seen = _main_notes(monkeypatch, tmp_path, provider, [
        [*handed, "--tools", "--propose"], [*handed, "--driven", "--tools", "--propose"], ["--tools", "--propose"],
        [*handed, "--tools"], [*handed, "--propose"], [*handed, "--driven", "--tools"], ["--tools"]])
    told = [(note.count(handoff_mod.PROPOSE_NOTE), propose) for note, propose, _ in seen]
    assert told == [(1, True)] * 3 + [(0, False)] * 4
    assert seen[0][0].endswith(f"\n\n{handoff_mod.EDIT_NOTE}\n\n{handoff_mod.PROPOSE_NOTE}")
    assert seen[1][0].startswith("How this session works:") and seen[1][0].endswith(handoff_mod.PROPOSE_NOTE)
    assert seen[2][0] == f"{handoff_mod.EDIT_NOTE}\n\n{handoff_mod.PROPOSE_NOTE}"
    assert seen[6][0] == handoff_mod.EDIT_NOTE
    assert all(note.count(handoff_mod.EDIT_NOTE) == 1 for note, _, _ in seen)
    assert not any(lead for _, _, lead in seen) and not any(handoff_mod.LEAD_NOTE in note for note, _, _ in seen)


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_main_tells_only_a_session_with_tools_and_lead_and_never_of_proposals(monkeypatch, tmp_path, provider):
    handed = ["--slug", "fix-backup", "--repo", "owner/repo"]
    seen = _main_notes(monkeypatch, tmp_path, provider, [
        [*handed, "--tools", "--lead"], [*handed, "--driven", "--tools", "--lead", "--propose"], ["--tools", "--lead"],
        [*handed, "--lead"], [*handed, "--lead", "--propose"], [*handed, "--tools"], [*handed, "--tools", "--propose"]])
    told = [(note.count(handoff_mod.LEAD_NOTE), note.count(handoff_mod.PROPOSE_NOTE), propose, lead)
            for note, propose, lead in seen]
    assert told == [(1, 0, False, True)] * 3 + [(0, 0, False, False)] * 3 + [(0, 1, True, False)]
    assert seen[0][0].endswith(f"\n\n{handoff_mod.EDIT_NOTE}\n\n{handoff_mod.LEAD_NOTE}")
    assert seen[1][0].startswith("How this session works:") and seen[1][0].endswith(handoff_mod.LEAD_NOTE)
    assert seen[2][0] == f"{handoff_mod.EDIT_NOTE}\n\n{handoff_mod.LEAD_NOTE}"
    assert seen[5][0].endswith(handoff_mod.EDIT_NOTE)
    assert all(note.count(handoff_mod.EDIT_NOTE) == 1 for note, _, _ in seen)


def test_a_research_session_refuses_propose_and_is_never_told():
    args = cli.parse(["--provider", "claude", "--kind", "research", "--cwd", "/tmp/r", "--prompt-file", "/tmp/p",
                      "--role", "provider", "--session-id", "5", "--research-run-id", "4", "--max-subagents", "1",
                      "--max-searches", "5", "--max-turns", "3", "--propose"])
    with pytest.raises(SystemExit, match="proposals"):
        cli.check_research(args)


def test_a_research_session_refuses_lead():
    args = cli.parse(["--provider", "claude", "--kind", "research", "--cwd", "/tmp/r", "--prompt-file", "/tmp/p",
                      "--role", "provider", "--session-id", "5", "--research-run-id", "4", "--max-subagents", "1",
                      "--max-searches", "5", "--max-turns", "3", "--lead"])
    with pytest.raises(SystemExit, match="lead"):
        cli.check_research(args)


def test_a_research_session_is_not_given_the_sentence(orch, client, tmp_path):
    scratch = tmp_path / "research"
    scratch.mkdir()
    args = cli.parse(["--provider", "claude", "--kind", "research", "--cwd", str(scratch), "--prompt-file",
                      "/tmp/p", "--role", "provider", "--session-id", "5", "--research-run-id", "4",
                      "--max-subagents", "1", "--max-searches", "5", "--max-turns", "3", "--wall-clock", "30"])
    cli.check_research(args)
    provider = FakeProvider({"provider": "claude-code", "turns": [[*provider_files(), say("Found."), result()]]})
    session = cli.research_session(args, provider, client)
    asyncio.run(asyncio.wait_for(session.run("Research the question."), timeout=30))
    assert provider.sent == ["Research the question."] and provider.ctx.propose is False


@pytest.mark.parametrize("fixture", [claude, codex])
@pytest.mark.parametrize("propose", [False, True])
def test_the_session_passes_propose_to_the_provider_and_the_sentence_to_its_first_prompt(orch, client, worktree,
                                                                                        fixture, propose):
    note = handoff_mod.brief_note("fix-backup", propose)
    provider = FakeProvider(fixture())
    session = Session(provider, client, Options(cwd=str(worktree), brief_note=note, propose=propose),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    assert provider.ctx.propose is propose
    assert (handoff_mod.PROPOSE_NOTE in provider.sent[0]) is propose
    assert provider.ctx.lead is False and handoff_mod.LEAD_NOTE not in provider.sent[0]


@pytest.mark.parametrize("fixture", [claude, codex])
def test_the_session_passes_lead_to_the_provider_and_the_lead_sentence_to_its_first_prompt(orch, client, worktree,
                                                                                         fixture):
    note = handoff_mod.brief_note("fix-backup", False, True)
    provider = FakeProvider(fixture())
    session = Session(provider, client, Options(cwd=str(worktree), brief_note=note, lead=True),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    assert provider.ctx.lead is True and provider.ctx.propose is False
    assert handoff_mod.LEAD_NOTE in provider.sent[0] and handoff_mod.PROPOSE_NOTE not in provider.sent[0]

