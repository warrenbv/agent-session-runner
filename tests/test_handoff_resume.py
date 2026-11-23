"""Hand-off descriptions on a resumed run and request coverage through both provider fakes."""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import fast_timing, git
from runner import handoff, handoff_check as hc, testrun
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session, wrap_up_prompt
from helpers import (FakeForge, ReusingForge, TEST_CMD as CMD, bash, bash_result, codex_cmd, codex_msg, codex_turn,
                     handoff_typed as typed, result, write)

ORIGINAL = "1. Original first\n2. Original second\n3. Original third"
LATEST = "1. Fix the description\n2. Fix the coverage"
PR = handoff.Opened("https://github.example/owner/repo/pull/3", 3)


class LocalClient:

    def __init__(self, finish_on=None):
        self.events = []
        self.finish_on = finish_on
        self.finish_pending = False

    def call(self, op, body):
        if op == "sessions.event_post":
            self.events.append(body)
            if body["kind"] == "turn" and self.finish_on and body["payload"].get("phase") == self.finish_on:
                self.finish_pending, self.finish_on = True, None
            return {}
        assert op == "sessions.pull"
        controls = [{"id": 1, "kind": "finish"}] if self.finish_pending else []
        self.finish_pending = False
        return {"prompts": [], "answers": [], "controls": controls}


def frames(provider, res, *before):
    return [*before, result(res)] if provider == "claude-code" else [
        *before, *codex_turn(codex_msg(json.dumps(res)))]


def close_with_tests(worktree, provider, *, entry="start", ending="bounded", changed=False, claim=None):
    base = git(worktree, "rev-parse", "HEAD").strip()
    prior_commit = entry == "resume" and changed
    if prior_commit:
        (worktree / "a.txt").write_text("earlier run\n")
        git(worktree, "add", "a.txt")
        git(worktree, "commit", "-m", "Earlier run")
    res = typed(files_changed=["a.txt"] if changed else [], tests_claim=claim,
                tests_run=[CMD] if changed else [])
    edits = [write()] if changed and not prior_commit else []
    turns = [frames(provider, res, *edits)]
    if ending != "bounded":
        turns.append(frames(provider, res))
    if changed:
        turns.append([bash(CMD), bash_result(), result()] if provider == "claude-code"
                     else codex_turn(codex_cmd(CMD, cwd=str(worktree))))
    fake = FakeProvider({"provider": provider, "turns": turns, "schema_per_turn": True})
    client = LocalClient({"driven": "waiting", "wrap": "completed"}.get(ending))
    forge = FakeForge()
    options = Options(cwd=str(worktree), base_ref=base, test_cmd=CMD, driven=ending == "driven",
                      resume_text="Resume this work" if entry == "resume" else None)
    brief = "Continues session 1.\n\nWhat to do now:\nCheck the work." if entry == "continue" else "Check the work."
    session = Session(fake, client, options, Caps(wall_clock=30, grace=0.05), fast_timing(),
                      handoff=lambda res: handoff.hand_off(
                          str(worktree), "fix-backup", "main", res, forge, since=base, checks=session.checks))
    outcome = asyncio.run(asyncio.wait_for(session.run(brief), timeout=10))
    return outcome, fake, client, forge


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("entry,ending", [
    ("start", "bounded"), ("start", "driven"), ("start", "wrap"), ("resume", "bounded"), ("continue", "bounded")])
@pytest.mark.parametrize("changed", [False, True])
def test_closing_tests_depend_on_whole_branch_and_worktree(worktree, provider, entry, ending, changed):
    outcome, fake, client, forge = close_with_tests(worktree, provider, entry=entry, ending=ending, changed=changed)
    assert outcome.reason == "completed" and outcome.turns == 1
    end = client.events[-1]["payload"]
    assert end.get("wrapped_up") == ("asked" if ending == "wrap" else None)
    assert end.get("ended_by") == ("finish" if ending == "driven" else None)
    expected = {"driven": [handoff.SUMMARY_PROMPT], "wrap": [wrap_up_prompt("asked")]}.get(ending, [])
    notes = [e["payload"] for e in client.events if e["payload"].get("purpose") == "tests-verdict"]
    assert end["checks"] == {"version": 2, "findings": []}
    if changed:
        assert fake.sent[1:] == [*expected, testrun.prompt(CMD)]
        assert outcome.handoff["handoff"] == "pull request" and forge.prs
        assert outcome.handoff["tests"] == {
            "command": CMD, "status": "passed", "exit_code": None if provider == "claude-code" else 0}
        assert notes[0]["text"] == f"Tests passed: `{CMD}`."
        assert f"## Tests run\n- {CMD}\n" in forge.prs[0]["body"]
        assert "cd " not in forge.prs[0]["body"]
        assert end["result"]["tests_claim"] is None and len(notes) == 1
        assert "reason" not in notes[0] and "tests_skipped" not in end
    else:
        assert fake.sent[1:] == expected
        assert end["handoff"] == {"handoff": "nothing to push", "branch": "agent/fix-backup"}
        assert not forge.prs and not forge.pushed
        assert end["tests_skipped"] == "no_changes"
        assert end["result"]["tests_claim"] == {"command": CMD, "status": "not_run"}
        assert [(note["text"], note["reason"]) for note in notes] == [
            ("Tests were skipped because no file changed.", "no_changes")]
        assert not any(e["payload"].get("purpose") == "tests" for e in client.events)


def test_skipping_the_test_turn_still_flags_an_unverified_pass(worktree):
    claim = {"command": CMD, "status": "passed"}
    outcome, fake, client, forge = close_with_tests(worktree, "codex", claim=claim)
    end = client.events[-1]["payload"]
    assert len(fake.sent) == 1 and not forge.prs
    assert end["result"]["tests_claim"] == outcome.result["tests_claim"] == claim
    assert end["checks"] == {"version": 2, "findings": [{
        "key": "tests_claim_not_run",
        "text": "The session reports that tests passed, but the runner recorded no verified final test run."}]}


def test_unavailable_git_state_keeps_the_test_turn(worktree, monkeypatch):
    def unavailable(cwd, base):
        raise handoff.GitError("git status exited 128")

    monkeypatch.setattr(handoff, "has_changes", unavailable)
    outcome, fake, client, _ = close_with_tests(worktree, "claude-code")
    assert outcome.reason == "completed" and fake.sent[1:] == [testrun.prompt(CMD)]
    notes = [e["payload"] for e in client.events if e["payload"].get("purpose") == "tests-verdict"]
    assert len(notes) == 1 and "reason" not in notes[0]


def run(provider, worktree, results, *, message=LATEST, continued=False, fresh=False, forge=None):
    fixture = {"provider": provider, "turns": [frames(provider, res, *([write()] if n == 0 else []))
                                              for n, res in enumerate(results)]}
    fake, client = FakeProvider(fixture), LocalClient()
    forge = forge or ReusingForge(found=PR)
    brief = ORIGINAL
    if continued:
        brief = f"Continues session 1.\n\nWhat to do now:\n{message}\n\nIts brief:\n{ORIGINAL}"
    options = Options(cwd=str(worktree), diff=False, resume_text=None if continued else message,
                      resume=None if fresh or continued else "0b7c4a52-8f1e-4d3a-9c6b-2e5f7a8d9e10",
                      carry_over="Old records:\n" + ORIGINAL)
    session = Session(fake, client, options, Caps(wall_clock=30, grace=0.05), fast_timing(),
                      handoff=lambda res: handoff.hand_off(
                          str(worktree), "fix-backup", "main", res, forge, reuse=not continued,
                          run=3 if not continued else 1, checks=session.checks))
    outcome = asyncio.run(asyncio.wait_for(session.run(brief), timeout=10))
    return outcome, fake, client, session, forge


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("continued,fresh", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("message,expected", [(LATEST, 2), ("Also fix the description.", 3),
    (LATEST + "\n3. Third\n4. Fourth\n5. Fifth", 5)])
def test_latest_numbered_request_drives_repair_and_pr_items(worktree, provider, continued, fresh, message, expected):
    labels = [line.split(". ", 1)[1] for line in (message if expected != 3 else ORIGINAL).splitlines()]
    good = typed(expected, files_changed=["a.txt"], items=[
        {"label": label, "status": "done", "where": "test"} for label in labels])
    bad = typed(1, files_changed=["a.txt"])
    outcome, fake, client, session, forge = run(
        provider, worktree, [bad, good], continued=continued, fresh=fresh, message=message)
    effective = message if expected != 3 else (session._brief if continued else ORIGINAL)
    assert outcome.reason == "completed" and outcome.handoff["handoff"] == "pull request"
    assert fake.sent[-1] == hc.repair_prompt(effective, bad) and len(fake.sent) == 2
    assert session.checks == {"version": 2, "findings": []}
    assert client.events[-1]["payload"]["checks"] == session.checks
    body = forge.prs[0]["body"] if continued else forge.updated[0][1]
    assert "## What was asked (as the session reports it)" in body
    assert all(f"- Done: {label} (test)" in body for label in labels)
    if expected != 3:
        assert "Original first" not in body
    if not continued:
        assert forge.updated == [(3, "Result from run 3.\n\n" + handoff.render(good, checks=session.checks))]


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
def test_complete_latest_result_ignores_old_count_and_prose_test_claims(worktree, provider):
    res = typed(2, files_changed=["a.txt"], tests_run=["All the tests passed: " + "long prose " * 100])
    outcome, fake, _, session, forge = run(provider, worktree, [res])
    assert outcome.result == res and len(fake.sent) == 1
    assert session.checks == {"version": 2, "findings": []}
    assert res["tests_run"][0] in forge.updated[0][1]


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
def test_description_update_failure_is_a_finding_after_a_successful_push(worktree, provider):
    class RefusingForge(ReusingForge):

        def update_pr(self, number, body):
            raise handoff.ForgeError("update_pr: HTTP 403 private detail")

    forge = RefusingForge(found=PR)
    res = typed(2, files_changed=["a.txt"])
    outcome, fake, client, session, _ = run(provider, worktree, [res], forge=forge)
    assert outcome.reason == "completed" and len(fake.sent) == 1
    assert outcome.handoff["handoff"] == "pull request" and outcome.handoff["pr_number"] == 3
    assert forge.pushed == ["agent/fix-backup"] and not forge.prs
    assert handoff.PR_UPDATE_FAILED == (
        "The runner could not update the pull request description; it may still show an earlier result.")
    assert session.checks == {"version": 2, "findings": [
        {"key": "pr_update_failed", "text": handoff.PR_UPDATE_FAILED}]}
    assert client.events[-1]["payload"]["checks"] == session.checks
    assert "private detail" not in json.dumps(client.events)


def test_run_line_is_fixed_and_github_patches_only_the_body(monkeypatch):
    assert handoff.RUN_LINE == "Result from run {run}."
    calls = []
    github = handoff.GitHub("owner/repo", "t0k")
    monkeypatch.setattr(github, "_send", lambda *args: calls.append(args))
    github.update_pr(3, "Result from run 3.\n\nLatest result")
    assert calls == [("update_pr", "PATCH", "/pulls/3", {"body": "Result from run 3.\n\nLatest result"})]
    for number in (True, 0, -1, "3"):
        with pytest.raises(handoff.ForgeError, match="bad pull request number"):
            github.update_pr(number, "body")


def test_exhausted_pr_creation_reports_the_deadline_and_keeps_the_pushed_branch(worktree, tmp_path, caplog):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(remote))

    class LimitedForge(handoff.GitHub):

        def push(self, cwd, branch):
            git(cwd, "push", str(remote), branch)

        def open_pr(self, branch, base, title, body):
            raise handoff.RateLimited(1794929400)

    forge = LimitedForge("owner/repo", "token")
    outcome, _, client, _, _ = run("claude-code", worktree, [typed(2, files_changed=["a.txt"])],
                                    continued=True, forge=forge)
    assert outcome.handoff == client.events[-1]["payload"]["handoff"] == {
        "handoff": "failed", "step": "github_rate_limit 1794929400", "error": "RateLimited"}
    assert git(remote, "rev-parse", "agent/fix-backup") == git(worktree, "rev-parse", "HEAD")
    assert "GitHub API rate limit reached. Retry after Tue 17 Nov at 15:30:00 UTC." in caplog.text

