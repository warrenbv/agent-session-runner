"""The standing edit note: every coding session is told to change files with its own edit tool, never
with an in-place shell edit. Research sessions are not."""

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

BRIEF = "Fix the backup job."


def test_the_sentence_names_the_tool_the_three_shell_forms_and_why():
    note = handoff_mod.EDIT_NOTE
    assert note.count(". ") + note.endswith(".") == 3
    for words in ("file-editing tool", "apply_patch", "`sed -i`", "`perl -i`", "output redirected over the file",
                  "permission card that pages the owner"):
        assert words in note


def test_the_note_says_only_what_holds_about_the_sandbox_and_permissions():
    sentences = handoff_mod.EDIT_NOTE.split(". ")
    assert "runs inside your sandbox" in sentences[1] and "outside it is never granted" in sentences[1]
    assert "refused without asking anyone" in sentences[2]
    for words in ("`env`", "`cd`", "subshell", "plainly"):
        assert words not in handoff_mod.EDIT_NOTE


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_main_gives_every_coding_session_the_edit_note(monkeypatch, tmp_path, provider):
    seen = []

    class StubSession:
        def __init__(self, provider, client, options, caps, timing, handoff=None):
            seen.append(options.brief_note)

        async def run(self, prompt):
            return Outcome("completed", 1, None, None)

    monkeypatch.setattr(cli, "Session", StubSession)
    monkeypatch.setattr(cli.config, "take_secrets", lambda: config.Settings(token="t", github_token="g"))
    brief = tmp_path / "brief.md"
    brief.write_text("Do it.\n")
    common = ["--provider", provider, "--cwd", str(tmp_path), "--prompt-file", str(brief)]
    handed = ["--slug", "fix-backup", "--repo", "owner/repo"]
    for extra in (handed, [*handed, "--driven"], []):
        assert cli.main([*common, *extra]) == 0
    bounded, driven, alone = seen
    assert bounded.startswith("How this run is handed off:") and bounded.endswith(f"\n\n{handoff_mod.EDIT_NOTE}")
    assert driven.startswith("How this session works:") and driven.endswith(f"\n\n{handoff_mod.EDIT_NOTE}")
    assert alone == handoff_mod.EDIT_NOTE
    assert all(note.count(handoff_mod.EDIT_NOTE) == 1 for note in seen)


@pytest.mark.parametrize("fixture", [claude, codex])
def test_the_sentence_reaches_the_providers_first_prompt(orch, client, worktree, fixture):
    note = handoff_mod.brief_note("fix-backup")
    provider = FakeProvider(fixture())
    session = Session(provider, client, Options(cwd=str(worktree), brief_note=note), Caps(wall_clock=30, grace=1),
                      fast_timing())
    asyncio.run(asyncio.wait_for(session.run(BRIEF), timeout=30))
    assert provider.sent == [handoff_mod.with_note(BRIEF, note)]
    assert handoff_mod.EDIT_NOTE in provider.sent[0]


def test_a_research_session_is_not_given_the_sentence(orch, client, tmp_path):
    scratch = tmp_path / "research"
    scratch.mkdir()
    args = cli.parse(["--provider", "claude", "--kind", "research", "--cwd", str(scratch), "--prompt-file",
                      "/tmp/p", "--role", "provider", "--session-id", "5", "--research-run-id", "4",
                      "--max-subagents", "1", "--max-searches", "5", "--max-turns", "3", "--wall-clock", "30"])
    cli.check_research(args)
    provider = FakeProvider({"provider": "claude-code", "turns": [[*provider_files(), say("Found."), result()]]})
    session = cli.research_session(args, provider, client)
    assert session.options.brief_note is None
    asyncio.run(asyncio.wait_for(session.run("Research the question."), timeout=30))
    assert provider.sent == ["Research the question."]
    assert not any("sed -i" in text for text in provider.sent)

