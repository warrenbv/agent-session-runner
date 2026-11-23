"""The report-only hand-off checks, the junk exclusions, and the one result repair turn."""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import fast_timing, git
from runner import handoff, handoff_check as hc, testrun
from runner.providers.base import RESULT_SCHEMA
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session, wrap_up_prompt
from helpers import FakeForge, claude, codex_msg, codex_turn, handoff_typed as typed, result, say, write


def keys(result=None, brief="Work", status=hc.Status()):
    return [f["key"] for f in hc.check(brief, result, status)["findings"]]


def test_missing_and_thin_sentences_are_fixed():
    assert hc.check("Work", None, hc.Status()) == {"version": 2, "findings": [
        {"key": "result_missing", "text": "The session returned no typed result."}]}
    for res in (typed(summary=" \n"), typed(0), {}):
        assert hc.result_findings("Work", res) == [
            {"key": "result_thin", "text": "The result has an empty summary or items."}]
    assert not keys(typed())
    assert hc.repair_prompt("Work", None) == (
        "Your result lacks a typed result, a non-empty summary, non-empty items; give it again, covering "
        "the whole brief and every numbered request, with all required result fields; change no files. "
        "Report only work you did and checks you saw. This repair turn is cut off after 300 seconds.")
    assert hc.REPAIR_TIMEOUT == "The session's result repair ran out of time."


def test_seven_requests_and_five_items():
    brief = "\n\n".join(f"{n}. Request {n}" for n in range(1, 8))
    assert hc.result_findings(brief, typed(5)) == [{"key": "items_short", "text":
        "The brief has 7 numbered items; the result reports 5."}]
    assert not keys(typed(7), brief)
    assert "items covering all 7 numbered requests" in hc.repair_prompt(brief, typed(5))


@pytest.mark.parametrize("brief,expected", [
    ("  1. Nested\n  2. Nested", 0), ("2. Starts at two\n3. Three", 0),
    ("1. One\n  1. Nested\n  2. Nested\n2) Two\n\nLater:\n1. Other\n2. Other", 2),
    ("1. One\n2. Two\n\nLater:\n3. Other\n4. Other", 2),
    ("1. One", 0), ("1. One\n3. Skipped", 0), ("100. Too large\n101. Too large", 0)])
def test_numbered_list_edges(brief, expected):
    assert hc.numbered_items(brief) == expected


def shell(provider, command):
    if provider == "claude-code":
        return {"type": "AssistantMessage", "content": [
            {"type": "ToolUseBlock", "name": "Bash", "id": "b", "input": {"command": command}}]}
    return {"method": "item/completed", "params": {"item": {
        "type": "commandExecution", "id": "b", "command": command, "exitCode": 0, "status": "completed"}}}


def test_unlisted_files_name_twenty_then_count():
    paths = tuple(f"file{i:02}" for i in range(24))
    findings = hc.check("Work", typed(files_changed=["file00"]), hc.Status(changed=paths))["findings"]
    assert findings == [{"key": "files_unlisted", "text": "Changed files not named in files_changed: " +
                         ", ".join(paths[1:21]) + "; and 3 more."}]


JUNK = ("pkg/__pycache__/x", "pkg/cache.pyc", ".pytest_cache/x", ".mypy_cache/x", ".ruff_cache/x",
        "node_modules/pkg/x", "nested/.DS_Store", "patch.orig", "patch.rej")


@pytest.mark.parametrize("path", JUNK)
def test_each_junk_pattern_is_left_out_named_and_never_deleted(worktree, path):
    file = worktree / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("keep on disk\n")
    (worktree / "real.txt").write_text("real work\n")
    status = handoff.worktree_status(str(worktree), "main")
    checks = hc.check("Work", typed(files_changed=["real.txt"]), status)
    assert checks["findings"] == [{"key": "junk_left_out", "text":
        f"Untracked junk left out of the hand-off commit, kept on disk: {path}."}]
    forge = FakeForge()
    handoff.hand_off(str(worktree), "fix-backup", "main", typed(), forge, checks=checks)
    assert git(worktree, "diff", "--name-only", "main", "HEAD").strip() == "real.txt"
    assert file.read_text() == "keep on disk\n" and path in forge.prs[0]["body"]


def test_tracked_node_modules_and_base_tracked_untracked_files_are_kept(worktree):
    tracked = worktree / "node_modules/kept"
    tracked.parent.mkdir()
    tracked.write_text("base\n")
    git(worktree, "add", ".")
    git(worktree, "commit", "-qm", "tracked dependency")
    tracked.write_text("changed\n")
    assert hc.junk_paths(handoff.worktree_status(str(worktree))) == ()
    assert hc.junk_paths(hc.Status(untracked=("node_modules/kept",),
                                   base_tracked=frozenset({"node_modules/kept"}))) == ()
    forge = FakeForge()
    handoff.hand_off(str(worktree), "fix-backup", "main", typed(), forge)
    assert "node_modules/kept" in git(worktree, "diff", "--name-only", "main", "HEAD")


def drive(fixture, client, worktree, *, brief="Work", driven=False, test_cmd=None, caps=None):
    fake, forge = FakeProvider(fixture), FakeForge()
    session = Session(fake, client, Options(cwd=str(worktree), driven=driven, idle_seconds=0.01,
                                           test_cmd=test_cmd), caps or Caps(wall_clock=30, grace=0.05),
                      fast_timing(), handoff=lambda res: handoff.hand_off(
                          str(worktree), "fix-backup", "main", res, forge, checks=session.checks))
    outcome = asyncio.run(asyncio.wait_for(session.run(brief), timeout=10))
    return outcome, fake, session, forge


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("bad", [None, typed(0), typed(summary=""), typed(1)])
def test_one_repair_before_tests_and_end_payload(orch, client, worktree, provider, bad):
    brief = "1. First\n2. Second"
    good = typed(2, files_changed=["a.txt"])
    if provider == "claude-code":
        fixture = claude([write(), result(bad)], [result(good)], [shell(provider, "pytest -q"), result()])
    else:
        fixture = {"provider": provider, "turns": [
            [write(), *codex_turn(codex_msg(json.dumps(bad)))] if bad else [write(), *codex_turn()],
            codex_turn(codex_msg(json.dumps(good))), codex_turn(shell(provider, "pytest -q"))]}
    outcome, fake, session, forge = drive(fixture, client, worktree, brief=brief, test_cmd="pytest -q")
    assert outcome.reason == "completed" and outcome.result == good and outcome.turns == 1
    assert fake.sent == [brief, hc.repair_prompt(brief, bad), testrun.prompt("pytest -q")]
    assert len(orch.of("message", purpose="handoff_repair")) == 1
    assert orch.of("end")[0]["checks"] == {"version": 2, "findings": []} == session.checks
    assert "## What the runner checked\nNothing found." in forge.prs[0]["body"]


def test_repair_never_twice_and_a_finding_does_not_stop_handoff(orch, client, worktree):
    outcome, fake, session, forge = drive(claude([write(), result()], [result()]), client, worktree)
    assert len(fake.sent) == 2 and outcome.reason == "completed" and len(forge.prs) == 1
    assert session.checks["findings"][0]["key"] == "result_missing"
    assert orch.of("end")[0]["checks"] == session.checks


def test_report_only_findings_do_not_ask_for_repair(orch, client, worktree):
    res = typed(tests_run=["pytest -q passed"])
    _, fake, session, forge = drive(claude([write(), write("junk.pyc"), result(res)]), client, worktree)
    assert fake.sent == ["Work"] and forge.prs
    assert [f["key"] for f in session.checks["findings"]] == ["files_unlisted", "junk_left_out"]


def test_driven_check_follows_summary_and_keeps_schema_for_repair(orch, client, worktree):
    fixture = {**claude([result()], [result(typed(0))], [result(typed())]), "schema_per_turn": True}
    _, fake, _, _ = drive(fixture, client, worktree, driven=True)
    assert fake.sent[1] == handoff.SUMMARY_PROMPT and fake.sent[2] == hc.repair_prompt("Work", typed(0))
    assert fake.schemas == [None, RESULT_SCHEMA, RESULT_SCHEMA]


@pytest.mark.parametrize("why", ["max_turns", "asked"])
def test_no_repair_after_wrap_up(orch, client, worktree, why):
    work = [write(), result(subtype="error_max_turns")]
    if why == "asked":
        orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "working",
                  lambda e: orch.control("finish"))
        work = [write(), say("working"), {"fake": "sleep", "seconds": 1}, result()]
    _, fake, session, _ = drive(claude(work, [result()]), client, worktree)
    assert fake.sent == ["Work", wrap_up_prompt(why)]
    assert session.checks["findings"][0]["key"] == "result_missing"


def test_no_repair_on_stop(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "working",
              lambda e: orch.revoke())
    outcome, fake, _, forge = drive(claude([say("working"), {"fake": "sleep", "seconds": 1}, result()]),
                                    client, worktree)
    assert outcome.reason == "revoked" and fake.sent == ["Work"] and not forge.prs


def test_repair_has_a_300_second_bound_and_terminates_if_needed(orch, client, worktree, monkeypatch):
    assert hc.REPAIR_SECONDS == 300
    monkeypatch.setattr(hc, "REPAIR_SECONDS", 0.2)
    fixture = claude([write(), result()], [{"fake": "hang"}])
    outcome, fake, _, forge = drive(fixture, client, worktree, test_cmd="pytest -q")
    assert len(fake.sent) == 2 and fake.interrupts == 1 and fake.terminated
    assert outcome.reason == "completed" and forge.prs


def test_body_sections_and_next_steps_contract():
    assert RESULT_SCHEMA["properties"]["next_steps"] == {"type": "array", "items": {"type": "string"}}
    assert set(RESULT_SCHEMA["required"]) == set(RESULT_SCHEMA["properties"])
    assert RESULT_SCHEMA["additionalProperties"] is False
    assert "`next_steps`" in handoff.SUMMARY_PROMPT and "`next_steps`" in wrap_up_prompt("asked")
    checks = hc.check("Work", None, hc.Status())
    body = handoff.render(typed(next_steps=["Check the restore job"]), checks=checks)
    assert "## What the runner checked\n- The session returned no typed result." in body
    assert "## For a later run\n- Check the restore job" in body
    assert "## What the runner checked\nNothing found." in handoff.render(typed())
    assert "## What the runner checked" in handoff.render(None, checks=checks)


def test_body_with_empty_next_steps_ends_with_runner_line():
    body = handoff.render(typed(next_steps=[]))
    assert "## For a later run" not in body
    assert "## What the runner checked\nNothing found." in body
    assert body.endswith("Opened by the session runner.")

