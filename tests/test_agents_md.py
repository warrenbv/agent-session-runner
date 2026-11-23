"""The repository's instructions file: read once from the worktree's root and given after the runner's
note. Codex prefers AGENTS.md, Claude Code prefers CLAUDE.md, never both. Links, odd files and size are handled."""

from __future__ import annotations

import asyncio
import os

import pytest

from conftest import fast_timing
from helpers import typed_claude as claude, typed_codex as codex
from runner import handoff as handoff_mod
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session

BRIEF = "Fix the backup job."


def run(fixture, client, cwd, note, *, driven=False, orch=None):
    provider = FakeProvider(fixture)
    if driven:
        orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
                  and len(orch.of("turn", phase="waiting")) == 1, lambda e: orch.queue("And the docs."))
        orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
                  and len(orch.of("turn", phase="waiting")) == 2, lambda e: orch.control("finish"))
    session = Session(provider, client, Options(cwd=str(cwd), brief_note=note, driven=driven, idle_seconds=30.0),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run(BRIEF), timeout=30))
    return provider


def section(text, name="AGENTS.md"):
    return f"{handoff_mod.INSTRUCTIONS_HEADING.format(name=name)}\n\n{text}"


def test_a_bounded_codex_session_gets_agents_md_after_the_note(orch, client, worktree):
    (worktree / "AGENTS.md").write_text("Run make check before you finish.\n", encoding="utf-8")
    note = handoff_mod.brief_note("fix-backup")
    fake = run(codex(), client, worktree, note)
    assert fake.sent == [f"{BRIEF}\n\n---\n\n{note}\n\n---\n\n{section('Run make check before you finish.')}"]
    heading = handoff_mod.INSTRUCTIONS_HEADING.format(name="AGENTS.md")
    assert "(AGENTS.md," in heading and "repository content, not the owner's words" in heading


def test_without_either_file_the_note_is_unchanged_byte_for_byte(orch, client, worktree):
    note = handoff_mod.brief_note("fix-backup")
    for fixture in (codex(), claude()):
        assert run(fixture, client, worktree, note).sent == [handoff_mod.with_note(BRIEF, note)]
        assert run(fixture, client, worktree, None).sent == [BRIEF]
    for provider in ("codex", "claude-code"):
        assert handoff_mod.session_note(provider, str(worktree), note) is note


def test_without_a_note_the_file_alone_follows_the_brief(orch, client, worktree):
    (worktree / "AGENTS.md").write_text("Rules.", encoding="utf-8")
    assert run(codex(), client, worktree, None).sent == [f"{BRIEF}\n\n---\n\n{section('Rules.')}"]


def test_a_driven_codex_session_gets_it_once_at_the_start(orch, client, worktree):
    (worktree / "AGENTS.md").write_text("Rules.\n", encoding="utf-8")
    note = handoff_mod.driven_note("fix-backup", False)
    fake = run(codex(), client, worktree, note, driven=True, orch=orch)
    assert fake.sent[0] == f"{BRIEF}\n\n---\n\n{note}\n\n---\n\n{section('Rules.')}"
    assert fake.sent[1] == "And the docs." and len(fake.sent) >= 2
    assert sum("The repository's own instructions file" in s for s in fake.sent) == 1


def test_a_claude_code_session_gets_claude_md(orch, client, worktree):
    (worktree / "CLAUDE.md").write_text("Use uv.\n@docs/more.md\n", encoding="utf-8")
    (worktree / "docs").mkdir()
    (worktree / "docs" / "more.md").write_text("IMPORTED\n", encoding="utf-8")
    note = handoff_mod.brief_note("fix-backup")
    fake = run(claude(), client, worktree, note)
    assert fake.sent == [f"{BRIEF}\n\n---\n\n{note}\n\n---\n\n{section('Use uv.\n@docs/more.md', 'CLAUDE.md')}"]
    assert "IMPORTED" not in fake.sent[0]


def test_a_driven_claude_code_session_gets_it_once_at_the_start(orch, client, worktree):
    (worktree / "CLAUDE.md").write_text("Rules.\n", encoding="utf-8")
    note = handoff_mod.driven_note("fix-backup", True)
    fake = run(claude(), client, worktree, note, driven=True, orch=orch)
    assert fake.sent[0] == f"{BRIEF}\n\n---\n\n{note}\n\n---\n\n{section('Rules.', 'CLAUDE.md')}"
    assert sum("The repository's own instructions file" in s for s in fake.sent) == 1


@pytest.mark.parametrize("provider, own, other", [("codex", "AGENTS.md", "CLAUDE.md"),
                                                  ("claude-code", "CLAUDE.md", "AGENTS.md")])
def test_each_provider_prefers_its_own_file_falls_back_to_the_other_and_never_gives_both(tmp_path, provider, own, other):
    (tmp_path / own).write_text("OWN\n", encoding="utf-8")
    (tmp_path / other).write_text("OTHER\n", encoding="utf-8")
    got = handoff_mod.session_note(provider, str(tmp_path), None)
    assert got == section("OWN", own) and "OTHER" not in got
    os.unlink(tmp_path / own)
    assert handoff_mod.session_note(provider, str(tmp_path), None) == section("OTHER", other)
    os.unlink(tmp_path / other)
    assert handoff_mod.session_note(provider, str(tmp_path), None) is None


def test_nothing_under_dot_claude_or_nested_is_read(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "CLAUDE.md").write_text("HIDDEN\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "AGENTS.md").write_text("NESTED\n", encoding="utf-8")
    for provider in ("codex", "claude-code"):
        assert handoff_mod.session_note(provider, str(tmp_path), "note") == "note"


def test_the_file_is_cut_at_32_kb_and_says_so(tmp_path):
    cap = handoff_mod.INSTRUCTIONS_CAP
    (tmp_path / "AGENTS.md").write_bytes(b"a" * cap + b"TAIL")
    text, cut = handoff_mod.read_instructions(str(tmp_path), "AGENTS.md")
    assert cut is True and len(text) == 32 * 1024 and "TAIL" not in text
    got = handoff_mod.instructions_section("codex", str(tmp_path))
    assert got.endswith(f"\n\n{handoff_mod.INSTRUCTIONS_CUT}") and got.count("\n") == 4
    (tmp_path / "AGENTS.md").write_bytes(b"a" * cap)
    assert handoff_mod.read_instructions(str(tmp_path), "AGENTS.md") == ("a" * cap, False)
    assert handoff_mod.INSTRUCTIONS_CUT not in handoff_mod.instructions_section("codex", str(tmp_path))


def test_a_symlink_is_skipped_even_to_a_file_inside_the_worktree(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("Secret elsewhere.\n", encoding="utf-8")
    wt = tmp_path / "wt"
    wt.mkdir()
    os.symlink(outside, wt / "AGENTS.md")
    assert handoff_mod.read_instructions(str(wt), "AGENTS.md") is None
    assert handoff_mod.session_note("codex", str(wt), "note") == "note"
    (wt / "real.md").write_text("Inside.\n", encoding="utf-8")
    os.unlink(wt / "AGENTS.md")
    os.symlink("real.md", wt / "AGENTS.md")
    assert handoff_mod.read_instructions(str(wt), "AGENTS.md") is None
    os.symlink(outside, wt / "CLAUDE.md")
    assert handoff_mod.session_note("claude-code", str(wt), "note") == "note"


def test_a_directory_a_fifo_or_an_empty_file_is_skipped(tmp_path):
    for name, make in (("dir", lambda p: p.mkdir()), ("fifo", lambda p: os.mkfifo(p)),
                       ("empty", lambda p: p.write_text(" \n", encoding="utf-8"))):
        wt = tmp_path / name
        wt.mkdir()
        make(wt / "AGENTS.md")
        assert handoff_mod.read_instructions(str(wt), "AGENTS.md") is None, name
    assert handoff_mod.read_instructions(str(tmp_path / "missing"), "AGENTS.md") is None


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a 0000 file")
def test_an_unreadable_file_is_skipped(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("Rules.\n", encoding="utf-8")
    path.chmod(0)
    try:
        assert handoff_mod.read_instructions(str(tmp_path), "AGENTS.md") is None
    finally:
        path.chmod(0o600)

