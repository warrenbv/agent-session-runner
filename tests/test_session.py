"""The fake provider driving the real Session against the stub control plane: every kind posted, the
sequence rules, the mid-turn modes, the approval relay, each cap, the turn diff and the hand-off."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time

import pytest

from conftest import fast_timing, git, load, run_session, typed_result
from helpers import RecordingForge as FakeForge, ReusingForge
from runner.stub_control_plane import TOKEN
from runner import config, gitdiff
from runner import handoff as handoff_mod
from runner.events import KINDS, MARK, PAYLOAD_CAP, size
from runner.providers.fake import FakeProvider
from runner.relay import EXPIRED, NO_ANSWER
from runner.session import Caps, Options, Outcome, Session


def allow_every_approval(orch):
    orch.when(lambda e: e["kind"] == "approval", lambda e: orch.answer(e["payload"]["request_id"], "allow"))


def result(cost=0.01, subtype="success", **extra):
    return {"type": "ResultMessage", "subtype": subtype, "duration_ms": 1, "duration_api_ms": 1,
            "is_error": subtype != "success", "num_turns": 1, "session_id": "s", "total_cost_usd": cost,
            "terminal_reason": "completed", "structured_output": typed_result(), **extra}


def say(text):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "TextBlock", "text": text}]}


def tool(tool_id="toolu_9", name="Read"):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "ToolUseBlock", "id": tool_id, "name": name, "input": {"file_path": "x"}}]}


def claude(*turns, default=None, **extra):
    return {"provider": "claude-code", "turns": [list(t) for t in turns],
            "default_turn": default or [say("Done."), result()], **extra}


def users(orch):
    return [p["text"] for p in orch.of("message", role="user")]


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("abandoned", [False, True])
def test_watch_sees_raw_frames_before_normalising_and_pulls_do_not_refresh_it(orch, client, tmp_path,
                                                                           monkeypatch, provider, abandoned):
    from runner import watch
    from helpers import Clock, call, partial, watch_done as done

    clock = Clock()
    monkeypatch.setattr(watch, "WATCH_INTERVAL", 0.005)

    class ClockProvider(FakeProvider):

        async def _wait(self, event, seconds):
            clock.advance(seconds)
            await asyncio.sleep(0.08)
            return False

        def normalise(self, raw):
            if raw.get("type") != "QuerySent":
                assert session.watch.last_frame == clock()
            return super().normalise(raw)

    turns = [[
        partial(provider), {"fake": "sleep", "seconds": 600},
        {"fake": "sleep", "seconds": 600}, partial(provider), partial(provider), done(provider)]]
    if abandoned:
        turns.insert(0, [call(provider, "abandoned"), done(provider)])
        orch.queue("next turn")
    fake = ClockProvider({"provider": provider, "turns": turns})
    session = Session(fake, client, Options(cwd=str(tmp_path), diff=False),
                      Caps(wall_clock=30, grace=1), fast_timing(pull_interval=0.005))
    session.watch = watch.Watch(provider, clock)
    outcome = asyncio.run(asyncio.wait_for(session.run("work"), 5))
    assert outcome.reason == "completed"
    assert orch.of("message", purpose="quiet") == [
        {"role": "runner", "purpose": "quiet", "stage": "quiet", "seconds": 600, "open": None},
        {"role": "runner", "purpose": "quiet", "stage": "stuck", "seconds": 1200, "open": None}]
    assert orch.of("message", purpose="alive") == [{"role": "runner", "purpose": "alive", "quiet_seconds": 1200}]
    assert len([r for r in orch.requests if r["op"] == "sessions.pull"]) > 2
    assert sequences_are_consecutive(orch)


def test_runner_nudge_between_turns_becomes_the_next_prompt(orch, client, tmp_path):
    from runner import refusals
    from helpers import COMMAND, REFUSAL_NUDGE, watch_done as done

    fake = FakeProvider({"provider": "claude-code", "default_turn": [done("claude-code")]})
    session = Session(fake, client, Options(cwd=str(tmp_path), diff=False), Caps(wall_clock=30), fast_timing())

    async def go():
        for _ in range(5):
            await session._refusal_note(refusals.line("Refused.", "Bash", COMMAND))
        return await session.run("work")

    outcome = asyncio.run(asyncio.wait_for(go(), 5))
    assert outcome.reason == "completed" and not fake.steered
    assert len(fake.sent) == 2 and fake.sent[0] == REFUSAL_NUDGE
    assert orch.of("message", purpose="looping")[0]["action"] == "nudged"


def sequences_are_consecutive(orch):
    return [e["sequence"] for e in orch.events] == list(range(1, len(orch.events) + 1))


@pytest.mark.parametrize("fixture,provider", [("fake_claude_full.json", "claude-code"),
                                              ("fake_codex_full.json", "codex")])
def test_a_full_session_posts_every_kind(orch, client, worktree, fixture, provider):
    allow_every_approval(orch)
    outcome, fake, _ = run_session(fixture, client, worktree)
    assert outcome.reason == "completed" and outcome.exit_code == 0 and outcome.turns == 1
    assert set(orch.kinds()) == set(KINDS) - {"thinking"}
    assert sequences_are_consecutive(orch)
    assert {e["provider"] for e in orch.events} == {provider}
    assert orch.kinds()[-1] == "end" and orch.kinds().count("end") == 1
    end = orch.of("end")[0]
    assert end["reason"] == "completed" and end["result"]["files_changed"] == ["scripts/backup.sh"]
    diff = orch.of("diff")[0]
    assert diff["paths"] == ["scripts/backup.sh"] and "+rsync -a --exclude cache" in diff["patch"]
    assert orch.approvals[orch.of("approval")[0]["request_id"]]["status"] == "allowed"
    assert all(r["auth"] == f"Bearer {TOKEN}" for r in orch.requests)
    assert fake.closed


def test_the_token_is_never_logged_or_posted(orch, client, worktree, caplog):
    caplog.set_level("DEBUG")
    allow_every_approval(orch)
    orch.faults.extend(["drop_before", "503"])
    run_session("fake_claude_full.json", client, worktree)
    assert TOKEN not in caplog.text
    assert all(TOKEN not in str(e) for e in orch.events)
    assert TOKEN not in repr(client)


def test_a_resend_after_a_network_error_reuses_the_number(orch, client, worktree):
    allow_every_approval(orch)
    orch.faults.extend(["drop_before", None, "drop_after", "503"])
    outcome, _, session = run_session("fake_claude_full.json", client, worktree)
    assert outcome.reason == "completed"
    assert sequences_are_consecutive(orch)
    posted = [r["body"]["sequence"] for r in orch.requests if r["op"] == "sessions.event_post"]
    assert posted[:6] == [1, 1, 2, 2, 2, 3]
    bodies = [r["body"] for r in orch.requests if r["op"] == "sessions.event_post" and r["body"]["sequence"] == 2]
    assert bodies[0] == bodies[1] == bodies[2]
    assert [s for s, _ in session.poster.posted] == list(range(1, len(orch.events) + 1))


def test_a_413_resends_the_same_number_with_a_stub(orch, client, worktree):
    orch.faults.extend([None, None, "413"])
    outcome, _, _ = run_session(claude([say("first"), say("second"), result()]), client, worktree, diff=False)
    assert outcome.reason == "completed" and sequences_are_consecutive(orch)
    assert orch.events[2]["payload"]["truncated"] is True and orch.events[3]["payload"]["text"] == "first"


def test_oversized_payloads_are_capped_before_posting(orch, client, worktree):
    huge = {"type": "UserMessage", "parent_tool_use_id": None, "content": [
        {"type": "ToolResultBlock", "tool_use_id": "t", "content": "z" * 100000}]}
    big_result = result(structured_output=typed_result("s" * 40000, branch="b"))
    outcome, _, _ = run_session(claude([tool("t"), huge, big_result]), client, worktree, diff=False)
    assert outcome.reason == "completed"
    assert all(size(e["payload"]) <= PAYLOAD_CAP for e in orch.events)
    assert orch.of("turn", phase="completed")[0]["truncated"] is True
    assert not [r for r in orch.requests if r["op"] == "sessions.event_post"
                and size(r["body"]["payload"]) > PAYLOAD_CAP]


def test_a_gap_is_a_bug_that_ends_the_run(orch, client, worktree):
    orch.when(lambda e: e["sequence"] == 2, lambda e: orch.events.pop())
    outcome, _, _ = run_session(claude([say("a"), say("b"), say("c"), result()]), client, worktree, diff=False)
    assert outcome.reason == "error" and outcome.exit_code == 1


def test_a_revoked_token_ends_the_session(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "tool", lambda e: orch.revoke())
    outcome, fake, _ = run_session(claude([tool(), {"fake": "sleep", "seconds": 5}, result()]), client, worktree,
                                   diff=False)
    assert outcome.reason == "revoked" and outcome.exit_code == 3 and fake.closed
    assert "end" not in orch.kinds()


def test_interrupt_now_interrupts_drains_and_sends_first(orch, client, worktree):
    def mid_turn(_):
        orch.queue("after the turn", "queue")
        orch.queue("stop and look at the cron entry", "now")

    orch.when(lambda e: e["kind"] == "tool", mid_turn)
    started = time.monotonic()
    outcome, fake, _ = run_session(claude([tool(), {"fake": "sleep", "seconds": 10}, say("never"), result()]),
                                   client, worktree, diff=False)
    assert time.monotonic() - started < 5
    assert fake.interrupts == 1
    assert fake.sent == ["Fix the backup job.", "stop and look at the cron entry", "after the turn"]
    assert orch.of("turn", phase="completed")[0]["interrupted"] is True
    assert "never" not in [p.get("text") for p in orch.of("message", role="assistant")]
    assert outcome.reason == "completed" and outcome.turns == 3


def test_after_this_turn_holds_until_the_end_event(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "tool", lambda e: orch.queue("then run the linter", "queue"))
    outcome, fake, _ = run_session(claude([tool(), {"fake": "sleep", "seconds": 0.3}, say("finished"), result()]),
                                   client, worktree, diff=False)
    assert fake.interrupts == 0 and fake.steered == []
    assert fake.sent == ["Fix the backup job.", "then run the linter"]
    kinds = [(e["kind"], e["payload"].get("phase"), e["payload"].get("text")) for e in orch.events]
    first_end = kinds.index(("turn", "completed", None))
    assert kinds.index(("message", None, "then run the linter")) > first_end
    assert ("message", None, "finished") in kinds[:first_end]
    assert outcome.turns == 2


def test_steer_at_the_next_step_goes_into_the_running_turn(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "tool", lambda e: orch.queue("also check the cache path", "steer"))
    outcome, fake, _ = run_session(claude([tool(), {"fake": "wait_steer", "timeout": 5}, say("steered"), result()]),
                                   client, worktree, diff=False)
    assert fake.steered == ["also check the cache path"] and fake.interrupts == 0
    assert fake.sent == ["Fix the backup job."] and outcome.turns == 1
    steer = orch.of("message", role="user", mode="steer")[0]
    assert steer["delivered"] == "next step"


def test_a_steer_the_provider_refuses_waits_for_the_turn_end(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "tool", lambda e: orch.queue("also check the cache path", "steer"))
    fixture = claude([tool(), {"fake": "sleep", "seconds": 0.3}, result()], steer=False)
    outcome, fake, _ = run_session(fixture, client, worktree, diff=False)
    assert fake.steered == [] and fake.sent == ["Fix the backup job.", "also check the cache path"]
    assert outcome.turns == 2


def test_a_now_prompt_that_replaces_a_held_one_runs_the_text_once(orch, client, worktree):
    def mid_turn(_):
        first = orch.queue("look at the cron entry", "queue")
        orch.queue("look at the cron entry", "now", replaces=first)

    orch.when(lambda e: e["kind"] == "tool", mid_turn)
    outcome, fake, _ = run_session(claude([tool(), {"fake": "sleep", "seconds": 10}, say("never"), result()]),
                                   client, worktree, diff=False)
    assert fake.interrupts == 1
    assert fake.sent == ["Fix the backup job.", "look at the cron entry"] and outcome.turns == 2


def test_a_replaces_naming_an_unknown_prompt_changes_nothing(orch, client, worktree):
    def mid_turn(_):
        orch.queue("after the turn", "queue")
        orch.queue("stop and look at the cron entry", "now", replaces=999)

    orch.when(lambda e: e["kind"] == "tool", mid_turn)
    outcome, fake, _ = run_session(claude([tool(), {"fake": "sleep", "seconds": 10}, say("never"), result()]),
                                   client, worktree, diff=False)
    assert fake.sent == ["Fix the backup job.", "stop and look at the cron entry", "after the turn"]
    assert outcome.turns == 3


def test_pulled_prompts_read_replaces_as_a_positive_int_or_none():
    from runner import control_plane

    class Fixed:
        def call(self, method, params):
            return {"prompts": [{"id": 1, "text": "a", "mode": "now", "author": "x"},
                                {"id": 2, "text": "b", "mode": "now", "author": "x", "replaces": 1},
                                {"id": 3, "text": "c", "mode": "now", "author": "x", "replaces": None},
                                {"id": 4, "text": "d", "mode": "now", "author": "x", "replaces": "1"},
                                {"id": 5, "text": "e", "mode": "now", "author": "x", "replaces": True},
                                {"id": 6, "text": "f", "mode": "now", "author": "x", "replaces": 0}]}

    prompts, _ = asyncio.run(control_plane.pull(Fixed()))
    assert [p.replaces for p in prompts] == [None, 1, None, None, None, None]


def test_prompts_between_turns_wait_for_linger(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "completed",
              lambda e: orch.queue("one more thing", "now"))
    outcome, fake, _ = run_session(claude([say("a"), result()]), client, worktree, diff=False,
                                   timing=fast_timing(linger=1.0))
    assert fake.sent == ["Fix the backup job.", "one more thing"] and fake.interrupts == 0


def approval_fixture(provider="claude-code"):
    return load("fake_claude_full.json" if provider == "claude-code" else "fake_codex_full.json")


def test_a_denial_returns_the_feedback_as_the_reason(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "approval",
              lambda e: orch.answer(e["payload"]["request_id"], "deny", "Use make clean instead."))
    outcome, _, session = run_session("fake_claude_full.json", client, worktree)
    assert session.relay.decisions[0][1].allow is False
    denied = [p for p in orch.of("tool", phase="completed") if p["id"] == "toolu_01"][0]
    assert denied["is_error"] is True and denied["output_preview"] == "Use make clean instead."
    assert outcome.reason == "completed"


def test_codex_denial(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "approval", lambda e: orch.answer(e["payload"]["request_id"], "deny"))
    run_session("fake_codex_full.json", client, worktree)
    assert orch.of("approval")[0] == {"request_id": "call_1", "tool": "command",
                                      "input_preview": '{"command": "pytest -q", "cwd": "/w"}'}
    assert [p["status"] for p in orch.of("tool", id="call_1")] == ["inProgress", "declined"]


def test_silence_denies_at_the_ceiling(orch, client, worktree):
    outcome, _, session = run_session("fake_claude_full.json", client, worktree,
                                      timing=fast_timing(approval_ceiling=0.3))
    decision = session.relay.decisions[0][1]
    assert decision.allow is False and decision.message == NO_ANSWER
    assert [p for p in orch.of("tool", phase="completed") if p["id"] == "toolu_01"][0]["output_preview"] == NO_ANSWER
    assert outcome.reason == "completed"


def test_an_expired_answer_denies(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "approval", lambda e: orch.expire(e["payload"]["request_id"]))
    _, _, session = run_session("fake_claude_full.json", client, worktree)
    assert session.relay.decisions[0][1].message == EXPIRED


def test_interrupt_now_denies_a_waiting_approval(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "approval", lambda e: orch.queue("never mind, stop", "now"))
    outcome, fake, session = run_session("fake_claude_full.json", client, worktree,
                                         timing=fast_timing(approval_ceiling=10))
    assert session.relay.decisions[0][1].allow is False
    assert fake.interrupts == 1 and fake.sent[1] == "never mind, stop"


def wrap_prompt(why):
    from runner.session import wrap_up_prompt
    return wrap_up_prompt(why)


def test_claude_max_turns_is_reported_and_wraps_up(orch, client, worktree):
    orch.queue("a queued follow-up", "queue")
    outcome, fake, _ = run_session(claude([{"fake": "sleep", "seconds": 0.1}, say("a"),
                                           result(subtype="error_max_turns")]),
                                   client, worktree, caps=Caps(max_turns=3), diff=False)
    assert outcome.reason == "completed" and outcome.exit_code == 0
    assert fake.sent == ["Fix the backup job.", wrap_prompt("max_turns")]
    assert orch.of("limit", limit="max_turns")[0]["source"] == "provider"
    assert orch.of("end")[0]["reason"] == "completed" and orch.of("end")[0]["wrapped_up"] == "max_turns"


def test_codex_max_turns_is_counted_by_the_runner(orch, client, worktree):
    orch.queue("second", "queue")
    orch.queue("third", "queue")
    fixture = load("fake_codex_full.json")
    fixture["turns"] = []
    outcome, fake, _ = run_session(fixture, client, worktree, caps=Caps(max_turns=2), diff=False,
                                   timing=fast_timing(linger=0.5))
    assert fake.sent == ["Fix the backup job.", "second", wrap_prompt("max_turns")]
    assert outcome.reason == "completed" and outcome.turns == 2
    assert orch.of("limit", limit="max_turns") == [{"limit": "max_turns", "source": "runner", "turns": 2}]
    assert orch.of("end")[0]["wrapped_up"] == "max_turns"


def test_the_budget_is_checked_after_each_turn(orch, client, worktree):
    orch.queue("keep going", "queue")
    outcome, fake, _ = run_session(claude([say("a"), result(cost=0.6)], default=[say("Done."), result(cost=0.61)]),
                                   client, worktree, caps=Caps(max_budget_usd=0.5), diff=False)
    assert outcome.reason == "completed" and fake.sent == ["Fix the backup job.", wrap_prompt("budget")]
    assert orch.of("limit", limit="budget")[0] == {"limit": "budget", "source": "runner", "cost_usd": 0.6,
                                                   "max_budget_usd": 0.5}
    assert orch.of("end")[0]["wrapped_up"] == "budget" and len(orch.of("limit")) == 1


def test_claude_s_own_budget_stop_is_reported(orch, client, worktree):
    outcome, _, _ = run_session(claude([say("a"), result(cost=0.9, subtype="error_max_budget_usd")]), client,
                                worktree, caps=Caps(max_budget_usd=1.0), diff=False)
    assert outcome.reason == "completed" and orch.of("limit", limit="budget")[0]["source"] == "provider"
    assert orch.of("end")[0]["wrapped_up"] == "budget"


def test_the_wall_clock_interrupts_then_wraps_up(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "tool", lambda e: orch.queue("later", "queue"))
    started = time.monotonic()
    outcome, fake, _ = run_session(claude([tool(), {"fake": "sleep", "seconds": 20}, result()]), client, worktree,
                                   caps=Caps(wall_clock=0.5, grace=2), diff=False)
    assert time.monotonic() - started < 5
    assert outcome.reason == "completed" and outcome.exit_code == 0
    assert fake.interrupts == 1 and not fake.terminated
    assert fake.sent == ["Fix the backup job.", wrap_prompt("wall_clock")]
    assert orch.of("limit", limit="wall_clock")[0]["source"] == "runner"
    assert orch.of("turn", phase="completed")[0]["interrupted"] is True
    assert orch.of("end")[0]["reason"] == "completed" and orch.of("end")[0]["wrapped_up"] == "wall_clock"


def test_the_wall_clock_terminates_after_the_grace_and_still_ends_wrapped_up(orch, client, worktree):
    started = time.monotonic()
    outcome, fake, _ = run_session(claude([tool(), {"fake": "hang"}, result()]), client, worktree,
                                   caps=Caps(wall_clock=0.3, grace=0.3), diff=False)
    assert time.monotonic() - started < 5
    assert outcome.reason == "completed" and fake.interrupts == 1 and fake.terminated
    assert fake.sent == ["Fix the backup job."]
    assert orch.of("end")[0]["wrapped_up"] == "wall_clock"
    assert orch.kinds()[-1] == "end" and sequences_are_consecutive(orch)


def test_a_completed_run_is_handed_off_as_a_pull_request(orch, client, worktree):
    allow_every_approval(orch)
    forge = FakeForge()
    outcome, _, _ = run_session("fake_claude_full.json", client, worktree,
                                handoff=lambda res: handoff_mod.hand_off(str(worktree), "fix-backup", "main", res,
                                                                         forge))
    assert forge.pushed == ["agent/fix-backup"]
    pr = forge.prs[0]
    assert pr["title"] == "Exclude the cache directory from the backup" and pr["base"] == "main"
    assert "- scripts/backup.sh" in pr["body"] and "- pytest -q" in pr["body"]
    end = orch.of("end")[0]
    assert end["pr_url"] == "https://github.example/owner/repo/pull/7" and end["handoff"]["commits"] == 1
    assert git(worktree, "status", "--porcelain") == ""
    assert outcome.handoff["handoff"] == "pull request"


REPORT = {"summary": "## What was built\n\nThe routine proposals card, with tests.", "files_changed": ["a.txt"]}


def test_the_pull_request_and_commit_take_the_sessions_title_not_the_reports_heading(worktree):
    (worktree / "a.txt").write_text("a\n")
    forge = FakeForge()
    handoff_mod.hand_off(str(worktree), "fix-backup", "main", REPORT, forge, title="Routine proposals",
                         brief="## Brief\n\nPropose routines from the week's sessions.")
    assert forge.prs[0]["title"] == "Routine proposals"
    assert git(worktree, "log", "-1", "--format=%s").strip() == "Routine proposals"
    assert forge.prs[0]["body"].startswith("## What was built")


def test_without_a_title_the_pull_request_takes_a_plain_line_of_the_brief(worktree):
    (worktree / "a.txt").write_text("a\n")
    forge = FakeForge()
    brief = ("## Context\n\n```python\n- **Search** steps 2 and 3 of the [plan](plans/search.md): the index and"
             " the ranking, with a test for each and a note in the log when both pass\n")
    handoff_mod.hand_off(str(worktree), "fix-backup", "main", REPORT, forge, title=None, brief=brief)
    title = forge.prs[0]["title"]
    assert title == "Search steps 2 and 3 of the plan: the index and the ranking, with a…"
    assert len(title) <= handoff_mod.BRIEF_LINE_CAP


@pytest.mark.parametrize("title,brief,result,expected", [
    ("Search steps 2 and 3", None, REPORT, "Search steps 2 and 3"),
    ("  ## Routine\n proposals ", None, None, "Routine proposals"),
    ("", "# Fix the backup\n", None, "Fix the backup"),
    ("---", "> - [ ] *Tidy* the `cache` dir", None, "Tidy the cache dir"),
    (None, "```\n---\n", REPORT, "The routine proposals card, with tests."),
    (None, None, {"summary": "## What was built"}, "What was built"),
    (None, None, None, handoff_mod.FALLBACK_TITLE),
    ("x" * 300, None, None, "x" * (handoff_mod.PR_TITLE_CAP - 1) + "…"),
])
def test_the_pull_request_title_never_starts_with_a_markdown_mark(title, brief, result, expected):
    got = handoff_mod.pr_title(title, brief, result)
    assert got == expected and got[0] not in "#>*-+|`~=_[" and "\n" not in got


@pytest.mark.parametrize("latest,title,brief,expected", [
    ("Backup: retry", "Fix the backup", None, "Backup: retry"),
    ("  ## Backup\n\tretry\x07 ", "Fix the backup", None, "Backup retry"),
    ("", "Fix the backup", None, "Fix the backup"),
    ("  \n ", "Fix the backup", None, "Fix the backup"),
    ("---", "Fix the backup", None, "Fix the backup"),
    ("## ", None, "Retry the backup job\n", "Retry the backup job"),
    (None, "Fix the backup", None, "Fix the backup"),
    ("y" * 300, "Fix the backup", None, "y" * (handoff_mod.PR_TITLE_CAP - 1) + "…"),
])
def test_a_later_title_wins_once_cleaned_and_an_unusable_one_falls_back(latest, title, brief, expected):
    assert handoff_mod.pr_title(title, brief, None, latest=latest) == expected


def test_a_rename_while_the_session_runs_titles_its_pull_request(orch, client, worktree):
    allow_every_approval(orch)
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "started",
              lambda e: orch.pull_extra.update(title="Backup: retry with a cap"))
    forge = FakeForge()
    session = None

    def hand(res):
        return handoff_mod.hand_off(str(worktree), "fix-backup", "main", res, forge, title="Fix the backup job",
                                    latest=session.title)

    session = Session(FakeProvider(load("fake_claude_full.json")), client, Options(cwd=str(worktree)),
                      Caps(wall_clock=30, grace=1), fast_timing(), handoff=hand)
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    assert outcome.handoff["handoff"] == "pull request"
    assert forge.prs[0]["title"] == "Backup: retry with a cap"
    assert git(worktree, "log", "-1", "--format=%s").strip() == "Backup: retry with a cap"


def test_an_control_plane_without_the_title_leaves_the_start_title(orch, client, worktree):
    allow_every_approval(orch)
    orch.pull_extra.update(title=7)
    forge = FakeForge()
    session = None

    def hand(res):
        return handoff_mod.hand_off(str(worktree), "fix-backup", "main", res, forge, title="Fix the backup job",
                                    latest=session.title)

    session = Session(FakeProvider(load("fake_claude_full.json")), client, Options(cwd=str(worktree)),
                      Caps(wall_clock=30, grace=1), fast_timing(), handoff=hand)
    asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    assert session.title is None and forge.prs[0]["title"] == "Fix the backup job"


def test_nothing_to_hand_off(worktree):
    forge = FakeForge()
    assert handoff_mod.hand_off(str(worktree), "fix-backup", "main", None, forge)["handoff"] == "nothing to push"
    assert forge.pushed == [] and forge.prs == []


def test_a_capped_run_is_not_handed_off(orch, client, worktree):
    forge = FakeForge()
    run_session(claude([say("a"), result(subtype="error_max_turns")]), client, worktree,
                handoff=lambda res: handoff_mod.hand_off(str(worktree), "fix-backup", "main", res, forge))
    assert forge.pushed == []


def test_a_failed_handoff_is_reported(orch, client, worktree):
    git(worktree, "switch", "--quiet", "main")
    outcome, _, _ = run_session(claude([{"fake": "write", "path": "a.txt", "text": "a"}, result()]), client, worktree,
                                handoff=lambda res: handoff_mod.hand_off(str(worktree), "fix-backup", "main", res,
                                                                         FakeForge()))
    assert outcome.reason == "completed" and orch.of("end")[0]["handoff"] == {
        "handoff": "failed", "error": "GitError", "step": "the worktree is on 'main', not 'agent/fix-backup'"}


def test_a_failed_git_step_is_named_without_its_output(orch, client, worktree, monkeypatch):
    real_git = handoff_mod.git

    def fail_commit(cwd, *args, **kw):
        if args[0] == "commit":
            raise gitdiff.GitError("git commit exited 128")
        return real_git(cwd, *args, **kw)
    monkeypatch.setattr(handoff_mod, "git", fail_commit)
    run_session(claude([{"fake": "write", "path": "a.txt", "text": "a"}, result()]), client, worktree,
                handoff=lambda res: handoff_mod.hand_off(str(worktree), "fix-backup", "main", res, FakeForge()))
    assert orch.of("end")[0]["handoff"] == {"handoff": "failed", "error": "GitError", "step": "git commit exited 128"}


class RefusingForge(FakeForge):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def open_pr(self, branch, base, title, body):
        raise self.exc


@pytest.mark.parametrize("exc,expected", [
    (handoff_mod.ForgeError("open_pr: HTTP 422"), {"step": "open_pr: HTTP 422", "error": "ForgeError"}),
    (handoff_mod.ForgeError("x" * 500), {"step": "x" * (200 - len(MARK)) + MARK, "error": "ForgeError"}),
    (ValueError("https://x-access-token:t0k@github.com/o/r"), {"error": "ValueError"}),
])
def test_a_failed_pull_request_reports_only_safe_fields(orch, client, worktree, exc, expected):
    run_session(claude([{"fake": "write", "path": "a.txt", "text": "a"}, result()]), client, worktree,
                handoff=lambda res: handoff_mod.hand_off(str(worktree), "fix-backup", "main", res,
                                                         RefusingForge(exc)))
    assert orch.of("end")[0]["handoff"] == {"handoff": "failed", **expected}
    assert "t0k" not in str(orch.events)


def test_open_pr_errors_carry_the_status_or_type_only(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    class Opener:
        def __init__(self, exc):
            self.exc = exc

        def open(self, request, timeout):
            raise self.exc

    gh = handoff_mod.GitHub("owner/repo", "t0k")
    for exc, message in ((urllib.error.HTTPError("https://api.github.com/x", 422, "t0k", {}, io.BytesIO()),
                          "open_pr: HTTP 422"),
                         (urllib.error.URLError("https://t0k@example.invalid"), "open_pr: URLError"),
                         (TimeoutError("t0k"), "open_pr: TimeoutError")):
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a, exc=exc: Opener(exc))
        with pytest.raises(handoff_mod.ForgeError) as caught:
            gh.open_pr("agent/x", "main", "title", "body")
        assert str(caught.value) == message


@pytest.fixture(params=["control_plane", "github"])
def assert_redirect_refused(request):
    from runner.control_plane import Client, Rejected

    def call(url, status):
        if request.param == "control_plane":
            client = Client(url, "test-token", timeout=2, host="tasks.test")
            with pytest.raises(Rejected) as caught:
                client.call("sessions.pull", {})
            assert caught.value.status == status and caught.value.code == "redirect"
            assert str(caught.value) == f"control plane refused the request: {status} redirect"
        else:
            client = handoff_mod.GitHub("owner/repo", "test-token", api=url)
            with pytest.raises(handoff_mod.ForgeError) as caught:
                client.open_pr("agent/x", "main", "title", "body")
            assert str(caught.value) == f"open_pr: HTTP {status}"

    return call


# Two loopback servers: the first answers the authenticated POST with a redirect to the second, which
# must receive nothing.
@pytest.mark.needs_sockets
def test_authenticated_clients_never_request_the_redirect_target(assert_redirect_refused):
    from contextlib import ExitStack
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    source, target = [], []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            target.append(self.command)
            self.send_response(200)
            self.end_headers()

        do_POST = do_GET

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            source.append(self.headers.get("Authorization"))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(302)
            self.send_header("Location", f"{target_url}/private-url")
            self.end_headers()
            self.wfile.write(b'{"error":{"code":"private-body"}}')

    with ExitStack() as stack:
        def serve(handler):
            server = stack.enter_context(HTTPServer(("127.0.0.1", 0), handler))
            thread = Thread(target=server.serve_forever)
            thread.start()
            stack.callback(thread.join)
            stack.callback(server.shutdown)
            return f"http://127.0.0.1:{server.server_port}"

        target_url = serve(Target)
        assert_redirect_refused(serve(Redirect), 302)
    assert source == ["Bearer test-token"] and not target


# The real urllib redirect processing, with the transport stubbed below it, so it runs without sockets.
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_authenticated_clients_reject_redirect_statuses_without_sockets(monkeypatch, assert_redirect_refused, status):
    from email.message import Message
    from io import BytesIO
    from urllib.request import HTTPHandler
    from urllib.response import addinfourl

    requests = []

    def reply(self, req):
        assert not requests, "attempted a second HTTP request"
        requests.append(req)
        headers = Message()
        headers["Location"] = "http://target.invalid/private-url"
        answer = addinfourl(BytesIO(b'{"error":{"code":"private-body"}}'), headers, req.full_url, status)
        answer.msg = "test-token"
        return answer

    monkeypatch.setattr(HTTPHandler, "http_open", reply)
    assert_redirect_refused("http://origin.invalid", status)
    assert [req.get_header("Authorization") for req in requests] == ["Bearer test-token"]


def test_the_handoff_commit_needs_no_git_identity(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    real_env = gitdiff.git_environment
    monkeypatch.setattr(gitdiff, "git_environment",
                        lambda extra=None: real_env({"GIT_CONFIG_NOSYSTEM": "1", **(extra or {})}))
    isolated = {"PATH": __import__("os").environ["PATH"], "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1"}

    def bare_git(*args, check=True):
        return subprocess.run(["git", *args], cwd=repo, env=isolated, check=check, capture_output=True, text=True)

    repo = tmp_path / "repo"
    repo.mkdir()
    bare_git("init", "--quiet", "--initial-branch=main")
    (repo / "README.txt").write_text("hello\n")
    bare_git("add", "README.txt")
    bare_git("-c", "user.name=Seed", "-c", "user.email=seed@example.invalid", "-c", "commit.gpgsign=false",
             "commit", "--quiet", "-m", "initial")
    bare_git("switch", "--quiet", "-c", "agent/fix-backup")
    for key in ("user.name", "user.email"):
        assert bare_git("config", key, check=False).returncode == 1
    (repo / "README.md").write_text("# new\n")

    forge = FakeForge()
    out = handoff_mod.hand_off(str(repo), "fix-backup", "main", {"summary": "Add a README"}, forge)
    tip = bare_git("rev-parse", "agent/fix-backup").stdout.strip()
    assert out == {"handoff": "pull request", "branch": "agent/fix-backup", "commits": 1,
                   "pr_url": "https://github.example/owner/repo/pull/7", "pr_number": 7, "head_sha": tip,
                   "base": "main", "stats": {"files": 1, "insertions": 1, "deletions": 0, "paths": ["README.md"],
                                             "paths_truncated": False}}
    who = bare_git("log", "-1", "--format=%an <%ae>|%cn <%ce>").stdout.strip()
    agent = f"{handoff_mod.AGENT_NAME} <{handoff_mod.AGENT_EMAIL}>"
    assert who == f"{agent}|{agent}"
    assert handoff_mod.AGENT_NAME == "Session Agent" and handoff_mod.AGENT_EMAIL == "agent@example.invalid"


def test_the_brief_carries_the_handoff_note_only_when_a_handoff_is_configured(orch, client, worktree):
    note = handoff_mod.brief_note("fix-backup")
    assert "`agent/fix-backup`" in note and note.startswith("How this run is handed off:")
    provider = FakeProvider(claude())
    session = Session(provider, client, Options(cwd=str(worktree), brief_note=note), Caps(wall_clock=30, grace=1),
                      fast_timing())
    asyncio.run(asyncio.wait_for(session.run("Fix the backup job.\n"), timeout=30))
    assert provider.sent == [f"Fix the backup job.\n\n---\n\n{note}"]
    assert users(orch) == ["Fix the backup job.\n"]

    _, plain, _ = run_session(claude(), client, worktree)
    assert plain.sent == ["Fix the backup job."]


def test_main_passes_the_note_with_a_handoff_and_not_without(monkeypatch, tmp_path):
    from runner import __main__ as cli

    seen = []

    class StubSession:
        def __init__(self, provider, client, options, caps, timing, handoff=None):
            seen.append((options.brief_note, handoff))

        async def run(self, prompt):
            return Outcome("completed", 1, None, None)

    monkeypatch.setattr(cli, "Session", StubSession)
    monkeypatch.setattr(cli.config, "take_secrets", lambda: config.Settings(token="t", github_token="g"))
    brief = tmp_path / "brief.md"
    brief.write_text("Do it.\n")
    common = ["--cwd", str(tmp_path), "--prompt-file", str(brief)]
    for provider in ("claude", "codex"):
        assert cli.main(["--provider", provider, *common, "--slug", "fix-backup", "--repo", "owner/repo"]) == 0
        assert cli.main(["--provider", provider, *common]) == 0
    note = handoff_mod.brief_note("fix-backup")
    assert [(n, h is not None) for n, h in seen] == [(note, True), (handoff_mod.EDIT_NOTE, False)] * 2


def test_slugs_and_repos_are_checked():
    for bad in ("", "-x", "Fix", "a/b", "a" * 64, "a b", "a..b/"):
        with pytest.raises(ValueError):
            handoff_mod.branch_for(bad)
    with pytest.raises(ValueError):
        handoff_mod.GitHub("owner/repo/../x", "t")
    with pytest.raises(ValueError):
        handoff_mod.GitHub("owner/repo", "")
    assert "t0k" not in repr(handoff_mod.GitHub("owner/repo", "t0k"))


UUID_1 = "0b7c4a52-8f1e-4d3a-9c6b-2e5f7a8d9e10"
UUID_2 = "5d1e9f30-2c4b-4a7e-8f6d-1b3c5e7a9d20"
HIS = "Also fix the restore script."
CARRY = "Carry-over: the brief, the last result and the branch's files."
NOTE = "How this run is handed off: (the runner's note)"


def codex_done(n="u1"):
    return {"method": "turn/completed", "params": {"threadId": "t", "turn": {"id": n, "items": [],
                                                                           "status": "completed"}}}


def codex(**extra):
    return {"provider": "codex", "default_turn": [
        {"method": "item/completed", "params": {"item": {
            "type": "agentMessage", "id": "r", "text": json.dumps(typed_result())}}}, codex_done()], **extra}


def resumed(fixture, client, cwd, *, resume=UUID_1, text=HIS, carry=None, cost_before=None, first=1,
            caps=None, handoff=None, driven=False, provider=None, base_fetch=None, merge_note=None):
    provider = provider or FakeProvider(fixture)
    options = Options(cwd=str(cwd), diff=False, resume=resume, resume_text=text, carry_over=carry,
                      cost_before=cost_before, first_sequence=first, brief_note=NOTE, driven=driven,
                      base_fetch=base_fetch, merge_note=merge_note)
    session = Session(provider, client, options, caps or Caps(wall_clock=30, grace=1), fast_timing(), handoff=handoff)
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    return outcome, provider, session


def runner_said(orch, purpose):
    return [p for p in orch.of("message", role="runner") if p.get("purpose") == purpose]


def test_a_resumed_runs_events_are_numbered_from_its_first_sequence(orch, client, worktree):
    orch.sequence_base = 40
    outcome, _, _ = resumed(claude(ids=True), client, worktree, first=41)
    assert outcome.reason == "completed"
    assert [e["sequence"] for e in orch.events] == list(range(41, 41 + len(orch.events)))
    assert orch.kinds()[-1] == "end"


def test_claudes_conversation_id_is_posted_once_known_and_again_when_it_changes(orch, client, worktree):
    orch.queue("second", "queue")
    fixture = claude([say("a"), result(session_id=UUID_1)], [say("b"), result(session_id=UUID_1)],
                     [say("c"), result(session_id=UUID_2)], ids=True)
    orch.queue("third", "queue")
    run_session(fixture, client, worktree, diff=False)
    said = runner_said(orch, "provider_session")
    assert said == [{"role": "runner", "purpose": "provider_session", "id": UUID_1},
                    {"role": "runner", "purpose": "provider_session", "id": UUID_2}]
    kinds = [(e["kind"], e["payload"].get("purpose") or e["payload"].get("phase")) for e in orch.events]
    assert kinds.index(("message", "provider_session")) < kinds.index(("turn", "completed"))
    assert sequences_are_consecutive(orch)


def test_codexs_thread_id_is_posted_as_soon_as_the_thread_starts(orch, client, worktree):
    run_session(codex(ids=True, thread_id="019a7f3e-thread_9"), client, worktree, diff=False)
    assert orch.events[0]["payload"] == {"role": "runner", "purpose": "provider_session", "id": "019a7f3e-thread_9"}
    assert orch.events[1]["payload"]["purpose"] == "baseline"
    assert orch.events[2]["payload"]["role"] == "user"


def test_an_id_the_control_plane_would_not_store_is_not_posted_and_no_id_posts_nothing(orch, client, worktree):
    run_session(claude(ids=True), client, worktree, diff=False)
    run_session(claude(), client, worktree, diff=False)
    run_session(codex(ids=True, thread_id="has a space"), client, worktree, diff=False)
    assert not runner_said(orch, "provider_session")


def test_a_resume_by_id_sends_his_message_alone_and_shows_it_as_the_resume(orch, client, worktree):
    outcome, fake, session = resumed(claude([say("ok"), result(session_id=UUID_1)], ids=True), client, worktree,
                                     carry=CARRY, caps=Caps(max_budget_usd=5.0, wall_clock=30, grace=1))
    assert outcome.reason == "completed" and fake.starts == [UUID_1]
    assert fake.sent == [HIS]
    assert fake.ctx.max_budget_usd is None
    [first] = orch.of("message", role="user")
    assert first == {"role": "user", "text": HIS, "mode": "queue", "entry_id": None, "author": "launcher",
                     "purpose": "resume"}
    assert not runner_said(orch, "resumed_fresh") and not session.fresh
    assert runner_said(orch, "provider_session") == [{"role": "runner", "purpose": "provider_session", "id": UUID_1}]


@pytest.mark.parametrize("fixture,stored", [(claude(refuse_resume=True), UUID_1),
                                            (codex(refuse_resume=True), "019a-old-thread")])
def test_a_refused_id_starts_a_new_conversation_with_the_carry_over_then_his_message(orch, client, worktree,
                                                                                      fixture, stored):
    outcome, fake, session = resumed(fixture, client, worktree, resume=stored, carry=CARRY)
    assert outcome.reason == "completed" and session.fresh
    assert fake.starts == [stored, None]
    assert fake.sent == [f"{CARRY}\n\n---\n\n{HIS}\n\n---\n\n{NOTE}"]
    purposes = [(e["kind"], e["payload"].get("role"), e["payload"].get("purpose")) for e in orch.events]
    assert purposes.index(("message", "runner", "resumed_fresh")) < purposes.index(("message", "user", "resume"))
    assert runner_said(orch, "resumed_fresh") == [{"role": "runner", "purpose": "resumed_fresh"}]
    assert orch.of("message", role="user")[0]["text"] == HIS


@pytest.mark.parametrize("outcome", ["fetched", "failed"])
def test_a_resumed_run_is_told_how_the_default_branch_s_fetch_went_after_his_message(orch, client, worktree,
                                                                                      outcome):
    from runner.session import base_fetch_line
    line = base_fetch_line(outcome, "main")
    _, fake, _ = resumed(claude([say("ok"), result(session_id=UUID_1)], ids=True), client, worktree,
                         base_fetch=line)
    assert fake.sent == [f"{HIS}\n\n---\n\n{line}"]
    assert orch.of("message", role="user")[0]["text"] == HIS
    _, fresh, _ = resumed(claude(refuse_resume=True), client, worktree, carry=CARRY, base_fetch=line)
    assert fresh.sent == [f"{CARRY}\n\n---\n\n{HIS}\n\n---\n\n{line}\n\n---\n\n{NOTE}"]


def test_a_sent_back_run_is_told_of_the_merge_after_the_fetch_line(orch, client, worktree):
    from runner.session import base_fetch_line
    fetched = base_fetch_line("fetched", "main")
    merge = "Runner's note: the merge line, as the launcher wrote it."
    _, fake, _ = resumed(claude([say("ok"), result(session_id=UUID_1)], ids=True), client, worktree,
                         base_fetch=fetched, merge_note=merge)
    assert fake.sent == [f"{HIS}\n\n---\n\n{fetched}\n\n{merge}"]
    assert orch.of("message", role="user")[0]["text"] == HIS


def test_the_base_fetch_lines_are_pinned():
    from runner import session as session_mod
    assert session_mod.BASE_FETCH_OUTCOMES == ("fetched", "failed")
    assert session_mod.base_fetch_line("fetched", "main") == (
        "Runner's note: before this run the runner fetched `main` from GitHub into `origin/main`, so you can compare"
        " your branch with it and bring its latest changes into this working tree if the work needs them.")
    assert session_mod.base_fetch_line("failed", "main") == (
        "Runner's note: before this run the runner could not fetch `main` from GitHub, so `origin/main` is as this"
        " machine last fetched it, which may be older than GitHub's; carry on with the work, and say in your result if"
        " it needed the latest `main`.")
    assert session_mod.base_fetch_line(None, "main") is None


def test_a_new_codex_thread_after_a_refusal_posts_its_own_id_first(orch, client, worktree):
    resumed(codex(refuse_resume=True, ids=True, thread_id="019a-new-thread"), client, worktree,
            resume="019a-old-thread", carry=CARRY)
    assert [e["payload"].get("purpose") for e in orch.events[:4]] == ["provider_session", "resumed_fresh", "baseline",
                                                                      "resume"]
    assert orch.events[0]["payload"]["id"] == "019a-new-thread"


def test_with_no_stored_id_and_no_carry_over_the_original_brief_stands_in(orch, client, worktree):
    from runner.session import RESUMED_WITHOUT

    outcome, fake, _ = resumed(claude(), client, worktree, resume=None)
    assert outcome.reason == "completed" and fake.starts == [None]
    assert fake.sent == [f"{RESUMED_WITHOUT.format(brief='Fix the backup job.')}\n\n---\n\n{HIS}\n\n---\n\n{NOTE}"]
    assert runner_said(orch, "resumed_fresh") == [{"role": "runner", "purpose": "resumed_fresh"}]


def test_a_start_failure_that_is_not_a_refused_id_is_not_taken_for_one(orch, client, worktree):
    class Broken(FakeProvider):
        async def start(self, ctx):
            self.starts.append(ctx.resume)
            raise RuntimeError("Claude Code returned an error result: something else (exit code: 1)")

    outcome, fake, _ = resumed(None, client, worktree, provider=Broken(claude()))
    assert outcome.reason == "error" and fake.starts == [UUID_1]
    assert not runner_said(orch, "resumed_fresh") and not fake.sent


def test_the_providers_recognise_exactly_their_own_refusal():
    from runner.providers import claude as claude_mod
    from runner.providers import codex as codex_mod

    class ResultError(Exception):
        def __init__(self, text, errors):
            super().__init__(text)
            self.errors = errors

    gone = ResultError(f"Claude Code returned an error result: No conversation found with session ID: {UUID_1} "
                       "(exit code: 1)", [f"No conversation found with session ID: {UUID_1}"])
    assert claude_mod.resume_refused(gone)
    assert not claude_mod.resume_refused(RuntimeError("Claude Code returned an error result: overloaded"))
    assert codex_mod.resume_refused(RuntimeError(f"JSON-RPC error -32600: no rollout found for thread id {UUID_1}"))
    assert not codex_mod.resume_refused(RuntimeError("JSON-RPC error -32600: invalid params"))


@pytest.mark.parametrize("reported,limited", [(0.50, False), (0.70, True)])
def test_a_resumed_runs_budget_is_counted_from_the_cost_it_had_reached(orch, client, worktree, reported, limited):
    orch.queue("keep going", "queue")
    outcome, fake, _ = resumed(claude([say("a"), result(cost=reported)]), client, worktree, cost_before=0.40,
                               caps=Caps(max_budget_usd=0.25, wall_clock=30, grace=1))
    assert orch.of("usage")[0]["total_cost_usd"] == reported
    if limited:
        assert outcome.reason == "completed" and len(fake.sent) == 2 and "keep going" not in fake.sent
        assert orch.of("end")[0]["wrapped_up"] == "budget"
        [limit] = orch.of("limit", limit="budget")
        assert limit == {"limit": "budget", "source": "runner", "cost_usd": reported, "max_budget_usd": 0.25,
                         "spent_usd": pytest.approx(reported - 0.40)}
    else:
        assert outcome.reason == "completed" and fake.sent == [HIS, "keep going"]
        assert not orch.of("limit")
    assert orch.of("end")[0]["cost_usd"] == outcome.cost_usd


def test_a_new_conversations_figure_is_all_this_runs_and_posted_as_the_provider_gives_it(orch, client, worktree):
    orch.queue("keep going", "queue")
    outcome, fake, _ = resumed(claude([say("a"), result(cost=0.10)], [say("b"), result(cost=0.30)],
                                      [say("closing"), result(cost=0.32)], refuse_resume=True), client, worktree,
                               carry=CARRY, cost_before=0.40, caps=Caps(max_budget_usd=0.25, wall_clock=30, grace=1))
    assert [u["total_cost_usd"] for u in orch.of("usage")] == [0.10, 0.30, 0.32]
    assert outcome.reason == "completed" and len(fake.sent) == 3 and orch.of("end")[0]["wrapped_up"] == "budget"
    assert orch.of("limit", limit="budget")[0]["spent_usd"] == pytest.approx(0.30)
    assert orch.of("end")[0]["cost_usd"] == pytest.approx(0.32)


def test_a_figure_below_the_cost_before_counts_in_full(orch, client, worktree):
    outcome, _, _ = resumed(claude([say("a"), result(cost=0.028)]), client, worktree, cost_before=0.514,
                            caps=Caps(max_budget_usd=0.01, wall_clock=30, grace=1))
    assert outcome.reason == "completed" and orch.of("end")[0]["wrapped_up"] == "budget"
    assert orch.of("limit", limit="budget")[0]["spent_usd"] == pytest.approx(0.028)


def _work(worktree, name="a.txt"):
    (worktree / name).write_text("the resumed run's work\n")


def test_a_resumed_hand_off_pushes_to_the_open_pull_request_instead_of_opening_one(worktree):
    _work(worktree)
    forge = ReusingForge(found=handoff_mod.Opened("https://github.example/owner/repo/pull/3", 3))
    out = handoff_mod.hand_off(str(worktree), "fix-backup", "main", {"summary": "More"}, forge, reuse=True, run=2)
    assert forge.pushed == ["agent/fix-backup"] and forge.asked == ["agent/fix-backup"] and forge.prs == []
    assert (out["handoff"], out["pr_url"], out["pr_number"]) == ("pull request",
                                                                "https://github.example/owner/repo/pull/3", 3)
    assert out["head_sha"] == git(worktree, "rev-parse", "HEAD").strip()
    assert forge.updated == [(3, "Result from run 2.\n\n" + handoff_mod.render({"summary": "More"}))]


def test_a_resumed_hand_off_with_no_open_pull_request_opens_one(worktree):
    _work(worktree)
    forge = ReusingForge(found=None)
    out = handoff_mod.hand_off(str(worktree), "fix-backup", "main", {"summary": "More"}, forge, reuse=True)
    assert forge.asked == ["agent/fix-backup"] and len(forge.prs) == 1 and out["pr_number"] == 7


def test_a_first_run_never_looks_for_an_open_pull_request(worktree):
    _work(worktree)
    forge = ReusingForge(found=handoff_mod.Opened("https://github.example/owner/repo/pull/3", 3))
    out = handoff_mod.hand_off(str(worktree), "fix-backup", "main", {"summary": "Work"}, forge)
    assert forge.asked == [] and out["pr_number"] == 7
    assert forge.prs[0]["body"] == handoff_mod.render({"summary": "Work"})


def test_a_rejected_push_on_a_resumed_run_is_reported_as_today(orch, client, worktree):
    forge = ReusingForge(found=handoff_mod.Opened("https://github.example/owner/repo/pull/3", 3),
                         push_error=gitdiff.GitError("git push exited 1"))
    resumed(claude([{"fake": "write", "path": "a.txt", "text": "a"}, result()]), client, worktree,
            handoff=lambda res: handoff_mod.hand_off(str(worktree), "fix-backup", "main", res, forge, reuse=True))
    assert orch.of("end")[0]["handoff"] == {"handoff": "failed", "error": "GitError", "step": "git push exited 1"}
    assert forge.asked == [] and forge.prs == []


def test_the_hand_off_counts_from_the_merge_base_when_the_base_has_moved_on(worktree, tmp_path):
    _work(worktree, "branch.txt")
    git(worktree, "add", "branch.txt")
    git(worktree, "commit", "--quiet", "-m", "the first run's work")
    cut = git(worktree, "merge-base", "main", "agent/fix-backup").strip()
    other = tmp_path / "main-wt"
    git(worktree, "worktree", "add", "--quiet", str(other), "main")
    (other / "merged.txt").write_text("merged on main since\n")
    git(other, "add", "merged.txt")
    git(other, "commit", "--quiet", "-m", "someone else's pull request")
    moved = git(other, "rev-parse", "HEAD").strip()
    _work(worktree, "resumed.txt")
    out = handoff_mod.hand_off(str(worktree), "fix-backup", "main", {"summary": "More"}, ReusingForge(),
                               since=moved, reuse=True)
    assert out["commits"] == 2 and handoff_mod.merge_base(str(worktree), moved, "agent/fix-backup") == cut
    assert sorted(out["stats"]["paths"]) == ["branch.txt", "resumed.txt"]


def test_github_finds_the_open_pull_request_of_a_branch(monkeypatch):
    import io
    import json as json_mod
    import urllib.request

    asked = []

    class Answer(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Opener:
        def __init__(self, body):
            self.body = body

        def open(self, request, timeout):
            asked.append((request.get_method(), request.full_url))
            return Answer(json_mod.dumps(self.body).encode())

    gh = handoff_mod.GitHub("owner/repo", "t0k")
    pr = {"html_url": "https://github.com/owner/repo/pull/12", "number": 12, "state": "open"}
    for body, found in (([pr], handoff_mod.Opened("https://github.com/owner/repo/pull/12", 12)), ([], None)):
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a, body=body: Opener(body))
        assert gh.open_pr_for("agent/session-9") == found
    assert asked[0] == ("GET", "https://api.github.com/repos/owner/repo/pulls?head=owner%3Aagent%2Fsession-9"
                               "&state=open&per_page=5")


def test_main_reads_a_resumed_runs_arguments_into_its_options(monkeypatch, tmp_path):
    from runner import __main__ as cli

    seen = []

    class StubSession:
        def __init__(self, provider, client, options, caps, timing, handoff=None):
            seen.append(options)

        async def run(self, prompt):
            return Outcome("completed", 1, None, None)

    monkeypatch.setattr(cli, "Session", StubSession)
    monkeypatch.setattr(cli.config, "take_secrets", lambda: config.Settings(token="t", github_token="g"))
    brief, his, carry = tmp_path / "brief.txt", tmp_path / "resume.txt", tmp_path / "carry-over.txt"
    brief.write_text("Do it.\n")
    his.write_text(HIS)
    carry.write_text(CARRY)
    common = ["--provider", "claude", "--cwd", str(tmp_path), "--prompt-file", str(brief)]
    assert cli.main([*common, "--first-sequence", "41", "--resume-text-file", str(his), f"--resume={UUID_1}",
                     "--carry-over-file", str(carry), "--cost-before", "0.420000"]) == 0
    assert cli.main(common) == 0
    resumed_options, first = seen
    assert (resumed_options.first_sequence, resumed_options.resume, resumed_options.resume_text,
            resumed_options.carry_over, resumed_options.cost_before) == (41, UUID_1, HIS, CARRY, 0.42)
    assert (first.first_sequence, first.resume, first.resume_text, first.carry_over, first.cost_before) == (
        1, None, None, None, None)
    assert resumed_options.base_fetch is None and first.base_fetch is None
    from runner.session import base_fetch_line
    assert cli.main([*common, "--resume-text-file", str(his), "--base-fetch", "failed", "--base", "trunk"]) == 0
    assert seen[-1].base_fetch == base_fetch_line("failed", "trunk")
    assert cli.main([*common, "--base-fetch", "fetched"]) == 0
    assert seen[-1].base_fetch is None
    with pytest.raises(SystemExit):
        cli.main([*common, "--resume-text-file", str(his), "--base-fetch", "maybe"])
    note = tmp_path / "merge-note.txt"
    note.write_text("Runner's note: merged.\n")
    assert cli.main([*common, "--resume-text-file", str(his), "--merge-note-file", str(note)]) == 0
    assert seen[-1].merge_note == "Runner's note: merged."
    assert cli.main([*common, "--resume-text-file", str(his), "--merge-note-file", str(tmp_path / "gone")]) == 0
    assert seen[-1].merge_note is None
    with pytest.raises(SystemExit):
        cli.main([*common, "--merge-note-file", str(note)])
    with pytest.raises(SystemExit):
        cli.main([*common, "--carry-over-file", str(carry)])
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n")
    with pytest.raises(SystemExit):
        cli.main([*common, "--resume-text-file", str(empty)])
    with pytest.raises(SystemExit):
        cli.main([*common, "--first-sequence", "0"])

