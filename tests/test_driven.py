"""Driven sessions: the session waits after each turn for the owner's next message and ends on finish,
the idle limit or a stop."""

from __future__ import annotations

import asyncio
import threading
import time

from conftest import fast_timing, git, load
from helpers import (drive, driven_claude as claude, driven_result as result, driven_typed as typed, on_waiting, say,
                     tool)
from runner import handoff as handoff_mod
from runner import handoff_check
from runner import testrun
from runner.providers.base import RESULT_SCHEMA
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session, wrap_up_prompt


class FakeForge:
    def __init__(self):
        self.pushed, self.prs = [], []

    def push(self, cwd, branch):
        self.pushed.append(branch)

    def open_pr(self, branch, base, title, body):
        self.prs.append({"branch": branch, "base": base, "title": title, "body": body})
        return handoff_mod.Opened("https://github.example/owner/repo/pull/7", 7)


def waiting(orch):
    return orch.of("turn", phase="waiting")


def test_a_driven_session_waits_takes_prompts_and_finishes_on_the_control(orch, client, worktree):
    on_waiting(orch, 1, lambda: orch.queue("second"))
    on_waiting(orch, 2, lambda: orch.queue("third"))
    on_waiting(orch, 3, lambda: orch.control("finish"))
    fixture = claude([{"fake": "write", "path": "a.txt", "text": "a\n"}, say("one"), result(structured_output=typed("one"))],
                     [say("two"), result(structured_output=typed("two"))],
                     [say("three"), result(structured_output=typed("three"))],
                     [say("closing"), result(structured_output=typed("The whole session", ["a.txt"]))])
    forge = FakeForge()
    outcome, fake, session = drive(fixture, client, worktree, forge=forge, test_cmd="pytest -q")
    assert outcome.reason == "completed" and outcome.turns == 3
    assert fake.sent[1:] == ["second", "third", handoff_mod.SUMMARY_PROMPT, testrun.prompt("pytest -q")]
    assert len(waiting(orch)) == 3 and all(w == {"phase": "waiting"} for w in waiting(orch))
    summary = orch.of("message", role="user", purpose="summary")
    assert summary and summary[0]["author"] == "runner" and summary[0]["mode"] == "queue"
    assert orch.of("message", role="user", purpose="tests")
    assert outcome.result["summary"] == "The whole session"
    assert forge.prs[0]["title"] == "The whole session" and "- a.txt" in forge.prs[0]["body"]
    end = orch.of("end")[0]
    assert end["reason"] == "completed" and end["ended_by"] == "finish" and "limit" not in end
    assert end["handoff"]["handoff"] == "pull request" and end["handoff"]["tests"]
    kinds = [(e["kind"], e["payload"].get("phase"), e["payload"].get("text")) for e in orch.events]
    assert kinds.index(("turn", "waiting", None)) < kinds.index(("message", None, "second"))
    assert orch.kinds()[-1] == "end" and fake.closed and fake.interrupts == 0


def test_one_owner_turn_needs_no_closing_turn(orch, client, worktree):
    on_waiting(orch, 1, lambda: orch.control("finish"))
    forge = FakeForge()
    fixture = claude([{"fake": "write", "path": "a.txt", "text": "a\n"}, result(structured_output=typed("only"))])
    outcome, fake, _ = drive(fixture, client, worktree, forge=forge)
    assert fake.sent == ["Fix the backup job."]
    assert not orch.of("message", role="user", purpose="summary")
    assert outcome.result["summary"] == "only" and forge.prs[0]["title"] == "only"
    assert orch.of("end")[0]["ended_by"] == "finish"


def test_the_idle_limit_finishes_the_session(orch, client, worktree):
    forge = FakeForge()
    fixture = claude([{"fake": "write", "path": "a.txt", "text": "a\n"}, result(structured_output=typed("only"))])
    started = time.monotonic()
    outcome, fake, _ = drive(fixture, client, worktree, idle=0.4, forge=forge)
    assert time.monotonic() - started >= 0.4
    assert outcome.reason == "completed" and len(waiting(orch)) == 1
    end = orch.of("end")[0]
    assert end["ended_by"] == "idle" and end["handoff"]["handoff"] == "pull request"


def test_an_interrupt_during_a_turn_denies_cards_interrupts_and_then_waits(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "approval", lambda e: orch.control("interrupt"))
    on_waiting(orch, 1, lambda: orch.control("finish"))
    forge = FakeForge()
    fixture = claude([tool("toolu_1", "Bash"),
                      {"fake": "approval", "tool_name": "Bash", "input": {"command": "make"}, "tool_use_id": "toolu_1"},
                      {"fake": "sleep", "seconds": 10}, say("never"), result()])
    started = time.monotonic()
    outcome, fake, session = drive(fixture, client, worktree, forge=forge,
                                   timing=fast_timing(approval_ceiling=10))
    assert time.monotonic() - started < 5
    assert fake.interrupts == 1 and session.relay.decisions[0][1].allow is False
    from runner.handoff_check import repair_prompt
    assert fake.sent == ["Fix the backup job.", repair_prompt("Fix the backup job.", None)]
    assert orch.of("turn", phase="completed")[0]["interrupted"] is True
    seq = [(e["kind"], e["payload"].get("phase")) for e in orch.events]
    assert seq.index(("turn", "completed")) < seq.index(("turn", "waiting")) < seq.index(("end", None))
    assert "never" not in [p.get("text") for p in orch.of("message", role="assistant")]
    assert orch.of("end")[0]["ended_by"] == "finish" and outcome.reason == "completed"


def test_an_interrupt_while_waiting_does_nothing(orch, client, worktree):
    on_waiting(orch, 1, lambda: orch.control("interrupt"))
    outcome, fake, _ = drive(claude([say("a"), result()]), client, worktree, idle=0.5)
    assert fake.interrupts == 0 and fake.sent == ["Fix the backup job."]
    assert len(waiting(orch)) == 1 and orch.of("end")[0]["ended_by"] == "idle"


def test_a_finish_during_a_turn_lets_the_turn_complete(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "tool", lambda e: orch.control("finish"))
    fixture = claude([tool(), {"fake": "sleep", "seconds": 0.4}, say("finished"), result(structured_output=typed("f"))])
    outcome, fake, _ = drive(fixture, client, worktree)
    assert fake.interrupts == 0 and fake.sent == ["Fix the backup job."]
    assert "finished" in [p.get("text") for p in orch.of("message", role="assistant")]
    assert not waiting(orch) and orch.of("end")[0]["ended_by"] == "finish"
    assert outcome.result["summary"] == "f"


def test_prompts_held_when_finish_arrives_still_run_first(orch, client, worktree):
    def mid_turn(_):
        orch.queue("after the turn", "queue")
        orch.control("finish")

    orch.when(lambda e: e["kind"] == "tool", mid_turn)
    fixture = claude([tool(), {"fake": "sleep", "seconds": 0.3}, result()])
    outcome, fake, _ = drive(fixture, client, worktree)
    assert fake.sent == ["Fix the backup job.", "after the turn", handoff_mod.SUMMARY_PROMPT]
    assert not waiting(orch) and orch.of("end")[0]["ended_by"] == "finish"


def test_a_stop_while_waiting_ends_the_session(orch, client, worktree):
    forge = FakeForge()
    on_waiting(orch, 1, orch.revoke)
    outcome, fake, _ = drive(claude([{"fake": "write", "path": "a.txt", "text": "a\n"}, result()]), client, worktree,
                             forge=forge)
    assert outcome.reason == "revoked" and outcome.exit_code == 3 and fake.closed
    assert "end" not in orch.kinds() and forge.pushed == []


def test_the_wall_clock_counts_working_time_not_waiting(orch, client, worktree):
    on_waiting(orch, 1, lambda: threading.Timer(1.0, lambda: orch.queue("second")).start())
    on_waiting(orch, 2, lambda: orch.control("finish"))
    started = time.monotonic()
    outcome, fake, _ = drive(claude([say("a"), result()]), client, worktree, caps=Caps(wall_clock=0.6, grace=1))
    assert time.monotonic() - started > 1.0
    assert outcome.reason == "completed" and fake.sent[:2] == ["Fix the backup job.", "second"]
    assert not orch.of("limit") and orch.of("end")[0]["ended_by"] == "finish"


def test_the_wall_clock_sums_the_turns_and_wraps_up_a_long_one(orch, client, worktree):
    on_waiting(orch, 1, lambda: orch.queue("second"))
    fixture = claude([{"fake": "sleep", "seconds": 0.4}, result()],
                     [tool(), {"fake": "sleep", "seconds": 20}, result()])
    started = time.monotonic()
    outcome, fake, _ = drive(fixture, client, worktree, caps=Caps(wall_clock=0.7, grace=2))
    assert time.monotonic() - started < 5
    assert outcome.reason == "completed" and outcome.exit_code == 0
    assert fake.sent == ["Fix the backup job.", "second", wrap_up_prompt("wall_clock")]
    assert fake.interrupts == 1
    assert orch.of("limit", limit="wall_clock")[0]["source"] == "runner"
    end = orch.of("end")[0]
    assert end["reason"] == "completed" and end["wrapped_up"] == "wall_clock"
    assert "ended_by" not in end and "limit" not in end


def test_max_turns_is_checked_before_a_turn_then_the_session_is_finished(orch, client, worktree):
    on_waiting(orch, 1, lambda: orch.queue("second"))
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "second",
              lambda e: orch.queue("third"))
    forge = FakeForge()
    fixture = claude([{"fake": "write", "path": "a.txt", "text": "a\n"}, result(structured_output=typed("one"))],
                     [{"fake": "sleep", "seconds": 0.2}, result(structured_output=typed("two"))],
                     [result(structured_output=typed("both turns"))])
    outcome, fake, _ = drive(fixture, client, worktree, caps=Caps(max_turns=2, wall_clock=30, grace=1), forge=forge)
    assert fake.ctx.max_turns is None
    assert fake.sent == ["Fix the backup job.", "second", handoff_mod.SUMMARY_PROMPT]
    assert outcome.reason == "completed" and outcome.exit_code == 0 and outcome.turns == 2
    assert orch.of("limit") == [{"limit": "max_turns", "source": "runner", "turns": 2}]
    end = orch.of("end")[0]
    assert end["limit"] == "max_turns" and "ended_by" not in end and forge.prs[0]["title"] == "both turns"
    assert len(waiting(orch)) == 1


def codex_turn(text, n):
    return [{"method": "turn/started", "params": {"threadId": "t", "turn": {"id": n, "items": [], "status": "inProgress"}}},
            {"method": "item/completed", "params": {"threadId": "t", "turnId": n,
                                                    "item": {"type": "agentMessage", "id": f"m{n}", "text": text}}},
            {"method": "turn/completed", "params": {"threadId": "t", "turn": {"id": n, "items": [],
                                                                              "status": "completed"}}}]


def test_codex_owner_turns_go_without_the_schema_and_the_closing_turn_asks_for_it(orch, client, worktree):
    import json
    on_waiting(orch, 1, lambda: orch.control("finish"))
    fixture = {"provider": "codex", "schema_per_turn": True,
               "turns": [codex_turn(json.dumps(typed("looks like JSON")), "1"),
                         codex_turn(json.dumps(typed("closing")), "2")]}
    outcome, fake, _ = drive(fixture, client, worktree)
    assert fake.schemas == [None, RESULT_SCHEMA]
    assert fake.sent[1] == handoff_mod.SUMMARY_PROMPT
    assert outcome.result["summary"] == "closing"


def test_a_provider_without_a_per_turn_schema_keeps_it_on_every_turn(orch, client, worktree):
    on_waiting(orch, 1, lambda: orch.queue("second"))
    on_waiting(orch, 2, lambda: orch.control("finish"))
    _, fake, _ = drive(claude(), client, worktree)
    assert fake.schemas == [RESULT_SCHEMA] * 3


def test_a_claude_driven_session_answers_in_prose_and_only_the_closing_turn_takes_the_schema(orch, client, worktree):
    on_waiting(orch, 1, lambda: orch.queue("second"))
    on_waiting(orch, 2, lambda: orch.control("finish"))
    fixture = claude([{"fake": "write", "path": "a.txt", "text": "a\n"}, say("first, in prose"), result()],
                     [say("second, in prose"), result()],
                     [result(structured_output=typed("The whole session"))], schema_per_turn=True)
    outcome, fake, _ = drive(fixture, client, worktree, test_cmd="pytest -q")
    assert fake.schemas == [None, None, RESULT_SCHEMA, RESULT_SCHEMA]
    assert fake.sent[2:] == [handoff_mod.SUMMARY_PROMPT, testrun.prompt("pytest -q")]
    assert outcome.result["summary"] == "The whole session"
    assert [m["text"] for m in orch.of("message", role="assistant")][:2] == ["first, in prose", "second, in prose"]


def test_a_bounded_claude_session_keeps_the_schema_even_with_a_per_turn_provider(orch, client, worktree):
    fixture = claude([say("one"), result(structured_output=typed("one"))], schema_per_turn=True)
    _, fake, _ = drive(fixture, client, worktree, driven=False)
    assert fake.schemas == [RESULT_SCHEMA]


def test_main_passes_driven_options_and_the_driven_note(monkeypatch, tmp_path):
    from runner import __main__ as cli
    from runner import config
    from runner.session import Outcome

    seen = []

    class StubSession:
        def __init__(self, provider, client, options, caps, timing, handoff=None):
            seen.append(options)

        async def run(self, prompt):
            return Outcome("completed", 1, None, None)

    monkeypatch.setattr(cli, "Session", StubSession)
    monkeypatch.setattr(cli.config, "take_secrets", lambda: config.Settings(token="t", github_token="g"))
    brief = tmp_path / "brief.md"
    brief.write_text("Do it.\n")
    common = ["--cwd", str(tmp_path), "--prompt-file", str(brief), "--slug", "fix-backup", "--repo", "owner/repo"]
    cli.main(["--provider", "claude", *common])
    cli.main(["--provider", "claude", *common, "--driven", "--idle-seconds", "600"])
    cli.main(["--provider", "codex", *common, "--driven"])
    bounded, claude_driven, codex_driven = seen
    assert (bounded.driven, bounded.brief_note) == (False, handoff_mod.brief_note("fix-backup"))
    assert claude_driven.driven and claude_driven.idle_seconds == 600.0
    assert claude_driven.brief_note == handoff_mod.driven_note("fix-backup", False)
    assert handoff_mod.DRIVEN_EACH_TURN not in claude_driven.brief_note
    assert codex_driven.idle_seconds == 14400.0 and handoff_mod.DRIVEN_EACH_TURN not in codex_driven.brief_note
    assert "`agent/fix-backup`" in codex_driven.brief_note


def normal(events):
    out = []
    for e in events:
        p = {k: v for k, v in e["payload"].items() if k not in ("from", "to")}
        if isinstance(p.get("handoff"), dict):
            p["handoff"] = {k: v for k, v in p["handoff"].items() if k != "head_sha"}
        out.append([e["sequence"], e["kind"], e["provider"], p])
    return out


class GoldenForge(FakeForge):
    def open_pr(self, branch, base, title, body):
        self.prs.append({"branch": branch, "base": base, "title": title, "body": body})
        return handoff_mod.Opened("https://github.example/o/r/pull/7", 7)


def test_a_bounded_session_ignores_interrupt_and_its_events_are_unchanged(orch, client, worktree, caplog):
    caplog.set_level("INFO")
    orch.control("interrupt")
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == "one",
              lambda e: (orch.queue("after the turn"), orch.control("interrupt")))
    fixture = claude([{"fake": "write", "path": "a.txt", "text": "a\n"}, say("one"),
                      result(structured_output={**typed("b"), "branch": "x"})],
                     default=[say("Done."), result(structured_output=None)])
    outcome, fake, _ = drive(fixture, client, worktree, driven=False, forge=GoldenForge(), test_cmd="pytest -q")
    assert outcome.reason == "completed" and fake.interrupts == 0
    assert fake.sent[:2] == ["Fix the backup job.", "after the turn"] and len(fake.sent) == 3
    assert "ignored: this session is bounded" in caplog.text
    said, *rest = normal(orch.events)
    assert said[:3] == [1, "message", "claude-code"] and said[3]["purpose"] == "baseline"
    expected = load("bounded_run.golden.json")
    for _, kind, _, payload in expected:
        for key in ("result", "structured_output"):
            if key in payload:
                payload[key].update(next_steps=[], tests_claim=None, items=typed("b")["items"])
        if kind == "end":
            payload["checks"] = {"version": handoff_check.VERSION, "findings": [{"key": "files_unlisted", "text":
                "Changed files not named in files_changed: a.txt."}]}
    assert [[sequence - 1, *event] for sequence, *event in rest] == expected
    assert git(worktree, "status", "--porcelain") == ""


def test_a_resumed_driven_session_waits_for_him_with_a_fresh_idle_limit(orch, client, worktree):
    orch.sequence_base = 57
    on_waiting(orch, 1, lambda: orch.queue("and the docs"))
    provider = FakeProvider(claude(ids=True, default=[say("Done."), result(session_id="0b7c4a52-8f1e-4d3a-9c6b-"
                                                                                       "2e5f7a8d9e10")]))
    options = Options(cwd=str(worktree), driven=True, idle_seconds=0.5, resume="0b7c4a52-8f1e-4d3a-9c6b-2e5f7a8d9e10",
                      resume_text="Pick up where you left off: the restore script.", first_sequence=58)
    session = Session(provider, client, options, Caps(wall_clock=30, grace=1), fast_timing())
    started = time.monotonic()
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    second_wait = [e for e in orch.events if e["kind"] == "turn" and e["payload"].get("phase") == "waiting"][1]
    assert outcome.reason == "completed" and session.ended_by == "idle"
    assert provider.sent[:2] == ["Pick up where you left off: the restore script.", "and the docs"]
    assert orch.of("message", role="user")[0]["purpose"] == "resume" and len(waiting(orch)) == 2
    assert time.monotonic() - started >= 0.5 and second_wait["sequence"] > 58
    assert orch.events[0]["sequence"] == 58 and orch.of("end")[0]["ended_by"] == "idle"

