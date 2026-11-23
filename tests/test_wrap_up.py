"""Wrap-up of a bounded coding session: finish and the wall clock mid-turn, a closing turn that overruns,
the end payload and the pull request's line."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from conftest import fast_timing, git, typed_result
from helpers import FakeForge, claude, codex_msg, codex_turn, result, say, write
from runner import __main__ as cli
from runner import config, testrun
from runner import handoff as handoff_mod
from runner.providers.fake import FakeProvider
from runner.session import (WRAP_UP_PURPOSE, WRAP_UP_SECONDS, WRAP_UP_WHYS, Caps, Options, Outcome, Session,
                            wrap_up_prompt)

BRIEF = "Fix the backup job."
CMD = "pytest -q"
WORK = typed_result("Excluded the cache directory", files_changed=["a.txt"])
CLOSING = {**WORK, "summary": "Wrapped up: the cache exclusion is in, the restore script is not",
           "open_questions": ["The restore script is unfinished"]}


def sleep(seconds=20):
    return {"fake": "sleep", "seconds": seconds}


def run(fixture, client, worktree, *, caps=None, test_cmd=None, forge=None, driven=False):
    provider = FakeProvider(fixture)
    forge = forge or FakeForge()
    box: list[Session] = []

    def hand(res):
        return handoff_mod.hand_off(str(worktree), "fix-backup", "main", res, forge, wrapped_up=box[0].wrapped_up)

    session = Session(provider, client, Options(cwd=str(worktree), test_cmd=test_cmd, driven=driven, idle_seconds=30),
                      caps or Caps(wall_clock=30, grace=1), fast_timing(), handoff=hand)
    box.append(session)
    began = time.monotonic()
    outcome = asyncio.run(asyncio.wait_for(session.run(BRIEF), timeout=30))
    return outcome, provider, session, forge, time.monotonic() - began


def on_text(orch, text, action):
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("text") == text, lambda e: action())


def wrap_messages(orch):
    return orch.of("message", purpose=WRAP_UP_PURPOSE)


def test_finish_interrupts_the_turn_and_runs_one_closing_turn_then_the_tests_and_the_hand_off(orch, client,
                                                                                               worktree):
    on_text(orch, "working", lambda: (orch.control("finish"), orch.queue("and the docs too", "queue")))
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("purpose") == WRAP_UP_PURPOSE,
              lambda e: orch.control("finish"))
    fixture = claude([write(), say("working"), sleep(), result(WORK)],
                     [say("Wrapped."), result(CLOSING)],
                     [say("Ran them."), result(WORK)])
    outcome, fake, session, forge, took = run(fixture, client, worktree, test_cmd=CMD)
    assert took < 8
    assert fake.interrupts == 1 and not fake.terminated
    assert fake.sent == [BRIEF, wrap_up_prompt("asked"), testrun.prompt(CMD)]
    assert orch.of("turn", phase="completed")[0]["interrupted"] is True
    assert wrap_messages(orch) == [{"role": "user", "text": wrap_up_prompt("asked"), "mode": "queue",
                                    "entry_id": None, "author": "runner", "purpose": "wrap_up", "why": "asked"}]
    assert outcome.reason == "completed" and outcome.exit_code == 0 and outcome.turns == 1
    assert outcome.result == CLOSING
    end = orch.of("end")[0]
    assert end["reason"] == "completed" and end["wrapped_up"] == "asked" and end["result"] == CLOSING
    assert end["handoff"]["handoff"] == "pull request" and "tests" in end["handoff"]
    assert forge.pushed == ["agent/fix-backup"]
    assert not forge.prs[0]["body"].startswith(handoff_mod.WRAPPED_UP_LINE)
    assert forge.prs[0]["body"].startswith(CLOSING["summary"])
    assert git(worktree, "status", "--porcelain") == "" and not orch.of("limit")


def test_finish_on_a_codex_run_mid_turn(orch, client, worktree):
    on_text(orch, "working", lambda: orch.control("finish"))
    fixture = {"provider": "codex", "turns": [
        codex_turn(write(), codex_msg("working", "m1"), sleep(), codex_msg(json.dumps(WORK), "m2")),
        codex_turn(codex_msg(json.dumps(CLOSING)), tid="turn_c")]}
    outcome, fake, _, forge, took = run(fixture, client, worktree)
    assert took < 8 and fake.interrupts == 1
    assert fake.sent == [BRIEF, wrap_up_prompt("asked")]
    assert outcome.reason == "completed" and outcome.result == CLOSING
    assert orch.of("end")[0]["wrapped_up"] == "asked" and forge.pushed == ["agent/fix-backup"]


def test_the_wall_clock_mid_turn_wraps_up_and_the_pull_request_says_so(orch, client, worktree):
    fixture = claude([write(), say("working"), sleep(), result(WORK)], [say("Wrapped."), result(CLOSING)])
    outcome, fake, _, forge, took = run(fixture, client, worktree, caps=Caps(wall_clock=0.5, grace=2))
    assert took < 8 and fake.interrupts == 1 and not fake.terminated
    assert fake.sent == [BRIEF, wrap_up_prompt("wall_clock")]
    kinds = [(e["kind"], e["payload"].get("limit") or e["payload"].get("purpose")) for e in orch.events]
    assert kinds.index(("limit", "wall_clock")) < kinds.index(("message", "wrap_up"))
    assert wrap_messages(orch)[0]["why"] == "wall_clock"
    end = orch.of("end")[0]
    assert (end["reason"], end["wrapped_up"], end["result"]) == ("completed", "wall_clock", CLOSING)
    assert forge.prs[0]["body"].startswith(f"{handoff_mod.WRAPPED_UP_LINE}\n\n{CLOSING['summary']}")


def test_the_wall_clock_during_the_closing_turn_does_not_cut_it(orch, client, worktree):
    on_text(orch, "working", lambda: orch.control("finish"))
    fixture = claude([say("working"), sleep()], [sleep(0.6), say("Wrapped."), result(CLOSING)])
    outcome, fake, _, _, _ = run(fixture, client, worktree, caps=Caps(wall_clock=0.4, grace=1))
    assert fake.interrupts == 1 and outcome.result == CLOSING
    assert orch.of("end")[0]["wrapped_up"] == "asked" and not orch.of("limit")


def test_a_closing_turn_that_overruns_and_ignores_the_interrupt_is_terminated_and_still_handed_off(orch, client,
                                                                                                    worktree):
    on_text(orch, "working", lambda: orch.control("finish"))
    fixture = claude([write(), say("working"), sleep(), result(WORK)], [say("closing"), {"fake": "hang"}])
    outcome, fake, _, forge, took = run(fixture, client, worktree, test_cmd=CMD,
                                        caps=Caps(wall_clock=30, grace=0.2, wrap_up=0.4))
    assert took < 8
    assert fake.interrupts == 2 and fake.terminated
    assert fake.sent == [BRIEF, wrap_up_prompt("asked", 0.4)]
    assert outcome.reason == "completed" and forge.pushed == ["agent/fix-backup"]
    end = orch.of("end")[0]
    assert end["wrapped_up"] == "asked" and end["handoff"]["handoff"] == "pull request"
    assert "tests" not in end["handoff"]
    assert git(worktree, "show", "--stat", "--format=", "HEAD").strip().startswith("a.txt")


def test_a_closing_turn_that_overruns_but_stops_is_followed_by_the_test_turn(orch, client, worktree):
    on_text(orch, "working", lambda: orch.control("finish"))
    fixture = claude([write(), say("working"), sleep(), result(WORK)], [say("closing"), sleep(), result(CLOSING)],
                     [say("tests"), result(WORK)])
    outcome, fake, _, forge, took = run(fixture, client, worktree, test_cmd=CMD,
                                        caps=Caps(wall_clock=30, grace=1, wrap_up=0.4))
    assert took < 8 and fake.interrupts == 2 and not fake.terminated
    assert fake.sent[2] == testrun.prompt(CMD) and len(fake.sent) == 3
    assert outcome.reason == "completed" and forge.pushed == ["agent/fix-backup"]


def test_the_test_turn_after_a_wrap_up_is_bounded_too(orch, client, worktree):
    on_text(orch, "working", lambda: orch.control("finish"))
    fixture = claude([write(), say("working"), sleep()], [say("Wrapped."), result(CLOSING)], [say("testing"), sleep()])
    outcome, fake, _, _, took = run(fixture, client, worktree, test_cmd=CMD,
                                    caps=Caps(wall_clock=30, grace=1, wrap_up=0.4))
    assert took < 8 and fake.interrupts == 2 and len(fake.sent) == 3
    assert outcome.reason == "completed" and outcome.result == CLOSING


@pytest.mark.parametrize("why", ["max_turns", "budget"])
def test_the_end_payload_of_a_wrapped_up_run(orch, client, worktree, why):
    caps = Caps(max_turns=1, wall_clock=30, grace=1) if why == "max_turns" else \
        Caps(max_budget_usd=0.5, wall_clock=30, grace=1)
    first = result(WORK, subtype="error_max_turns") if why == "max_turns" else result(WORK, cost=0.6)
    run(claude([write(), first], [say("Wrapped."), result(CLOSING, cost=0.62)]), client, worktree, caps=caps)
    end = orch.of("end")[0]
    assert set(end) == {"reason", "turns", "cost_usd", "last_turn", "result", "handoff", "pr_url", "wrapped_up",
                        "checks"}
    assert (end["reason"], end["wrapped_up"], end["turns"], end["result"]) == ("completed", why, 1, CLOSING)
    assert end["pr_url"] == "https://github.example/owner/repo/pull/7"


def test_a_run_that_was_not_wrapped_up_carries_no_wrapped_up(orch, client, worktree):
    run(claude([write(), say("done"), result(WORK)]), client, worktree)
    assert "wrapped_up" not in orch.of("end")[0] and not wrap_messages(orch)


def test_the_contracts_whys_and_the_bound():
    assert WRAP_UP_WHYS == ("asked", "wall_clock", "looping", "max_turns", "budget")
    assert WRAP_UP_SECONDS == 600 and Caps().wrap_up == 600
    assert "This turn is cut off after 10 minutes." in wrap_up_prompt("asked")
    for words in ("Stop where you are", "Leave nothing half-edited", "give your final structured result"):
        assert words in wrap_up_prompt("wall_clock")
    assert len({wrap_up_prompt(why) for why in WRAP_UP_WHYS}) == len(WRAP_UP_WHYS)


@pytest.mark.parametrize("why,line", [(None, False), ("asked", False), ("wall_clock", True), ("max_turns", True),
                                      ("budget", True), ("looping", True)])
def test_the_body_line_unless_he_asked(why, line):
    assert handoff_mod.WRAPPED_UP_LINE == "Wrapped up at its limit; it may be unfinished."
    body = handoff_mod.render(WORK, wrapped_up=why)
    assert body.startswith(f"{handoff_mod.WRAPPED_UP_LINE}\n\n") is line
    assert body.count(handoff_mod.WRAPPED_UP_LINE) == int(line)
    bare = handoff_mod.render(None, wrapped_up=why)
    assert bare.startswith((f"{handoff_mod.WRAPPED_UP_LINE}\n\n" if line else "") +
                           "The run returned no typed result.")
    assert "## What the runner checked" in bare and "## For a later run" not in bare


def test_main_hands_the_sessions_why_to_the_hand_off(monkeypatch, tmp_path):
    seen = []

    class StubSession:
        def __init__(self, provider, client, options, caps, timing, handoff=None):
            self.handoff, self.existed, self.title, self.wrapped_up = handoff, None, None, "wall_clock"
            self.checks = {"version": 1, "findings": []}

        async def run(self, prompt):
            self.handoff({"summary": "s"})
            return Outcome("completed", 1, None, None)

    monkeypatch.setattr(cli, "Session", StubSession)
    monkeypatch.setattr(cli.config, "take_secrets", lambda: config.Settings(token="t", github_token="g"))
    monkeypatch.setattr(cli.handoff, "hand_off", lambda *a, **kw: seen.append(kw) or {})
    brief = tmp_path / "brief.md"
    brief.write_text("Do it.\n")
    assert cli.main(["--provider", "claude", "--cwd", str(tmp_path), "--prompt-file", str(brief),
                     "--slug", "fix-backup", "--repo", "owner/repo"]) == 0
    assert seen[0]["wrapped_up"] == "wall_clock"
    assert seen[0]["checks"] == {"version": 1, "findings": []}


def test_a_driven_sessions_finish_is_unchanged(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting",
              lambda e: orch.control("finish"))
    outcome, fake, session, forge, _ = run(claude([write(), say("done"), result(WORK)]), client, worktree, driven=True)
    assert outcome.reason == "completed" and session.ended_by == "finish" and session.wrapped_up is None
    assert fake.sent == [BRIEF] and fake.interrupts == 0 and not wrap_messages(orch)
    end = orch.of("end")[0]
    assert end["ended_by"] == "finish" and "wrapped_up" not in end and forge.pushed == ["agent/fix-backup"]


def test_a_driven_sessions_working_clock_mid_turn_wraps_up_tests_and_hands_off(orch, client, worktree):
    from helpers import violations

    fixture = claude([write(), say("working"), sleep()],
                     [sleep(0.5), result(CLOSING)], [result(WORK)])
    outcome, fake, session, forge, took = run(fixture, client, worktree, driven=True, test_cmd=CMD,
                                             caps=Caps(wall_clock=0.4, grace=1))
    assert took < 8 and fake.interrupts == 1 and not fake.terminated
    assert fake.sent == [BRIEF, wrap_up_prompt("wall_clock"), testrun.prompt(CMD)]
    assert orch.of("turn", phase="completed")[0]["interrupted"] is True
    assert session.wrapped_up == "wall_clock" and session.stop_reason is None
    assert outcome.reason == "completed" and outcome.exit_code == 0 and outcome.result == CLOSING
    assert orch.of("limit") == [{"limit": "wall_clock", "source": "runner", "seconds": 0.4}]
    assert wrap_messages(orch)[0]["why"] == "wall_clock"
    limit = next(i for i, e in enumerate(orch.events) if e["kind"] == "limit")
    interrupted = next(i for i, e in enumerate(orch.events)
                       if e["kind"] == "turn" and e["payload"].get("interrupted"))
    closing = next(i for i, e in enumerate(orch.events) if e["payload"].get("purpose") == "wrap_up")
    tests = next(i for i, e in enumerate(orch.events) if e["payload"].get("purpose") == "tests")
    assert limit < interrupted < closing < tests
    end = orch.of("end")[0]
    assert (end["reason"], end["wrapped_up"], end["result"]) == ("completed", "wall_clock", CLOSING)
    assert "ended_by" not in end and "limit" not in end
    assert end["handoff"]["handoff"] == "pull request" and "tests" in end["handoff"]
    assert forge.pushed == ["agent/fix-backup"]
    assert forge.prs[0]["body"].startswith(f"{handoff_mod.WRAPPED_UP_LINE}\n\n{CLOSING['summary']}")
    assert not [v for e in orch.events for v in violations(e["kind"], e["provider"], e["payload"])]


def test_a_research_session_never_wraps_up(orch, client, tmp_path, caplog):
    from helpers import provider_files, research_run

    caplog.set_level("INFO")
    orch.control("finish")
    outcome, fake, session, _, _ = research_run(orch, client, tmp_path, [*provider_files(), sleep(10), result()],
                                                caps=Caps(max_turns=10, wall_clock=0.5, grace=1))
    assert outcome.reason == "wall_clock" and session.wrapped_up is None
    assert len(fake.sent) == 1 and not wrap_messages(orch)
    assert "ignored: this session is bounded" in caplog.text
    assert "wrapped_up" not in orch.of("end")[0]


def test_a_research_sessions_turn_cap_still_ends_it(orch, client, tmp_path):
    from helpers import provider_files, research_result, research_run

    outcome, fake, session, _, _ = research_run(orch, client, tmp_path,
                                                [*provider_files(), research_result("error_max_turns")])
    assert outcome.reason == "max_turns" and session.wrapped_up is None and len(fake.sent) == 1

