"""The keep_open control: a waiting driven session's idle limit moves one idle period later, and anything
else is refused and changes nothing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import fast_timing
from runner import session as session_mod
from runner.control_plane import Control
from runner.providers.fake import FakeProvider
from runner.session import KEEP_OPEN_PURPOSE, KEEP_OPEN_WORDS, Caps, Options, Session
from helpers import (drive, driven_claude as claude, driven_result as result, driven_typed as typed, on_waiting, say,
                     tool)


def answers(orch):
    return orch.of("message", role="runner", purpose=KEEP_OPEN_PURPOSE)


def test_a_waiting_session_is_kept_open_one_more_idle_period(orch, client, worktree, monkeypatch):
    session = Session(FakeProvider(claude([say("one"), result(structured_output=typed("one"))])), client,
                      Options(cwd=str(worktree), driven=True, idle_seconds=0.6), Caps(wall_clock=30, grace=1),
                      fast_timing())
    waits: list[float] = []
    wait = session._await_input

    async def timed():
        now = 0.0
        pressed = False

        async def advance(awaitable, timeout):
            nonlocal now, pressed
            awaitable.close()
            assert session._idle_deadline is not None
            if not pressed:
                now += 0.3
                pressed = True
                await session._keep_open(Control(1, "keep_open"))
            else:
                now += timeout
                raise TimeoutError

        clocked = Mock(wraps=asyncio)
        clocked.get_running_loop.return_value = SimpleNamespace(time=lambda: now)
        clocked.wait_for = advance
        with monkeypatch.context() as patch:
            patch.setattr(session_mod, "asyncio", clocked)
            try:
                return await wait()
            finally:
                waits.append(now)

    session._await_input = timed
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=60))
    assert len(waits) == 1 and 1.15 <= waits[0] < 1.8, waits
    said = answers(orch)
    assert len(said) == 1
    a = said[0]
    assert a["outcome"] == "extended" and a["extensions"] == 1 and a["control_id"] == 1
    assert 0 <= a["idle_left"] <= 1 and a["idle_seconds"] == 0
    assert "step" not in a and a["text"].startswith("Kept open:")
    assert outcome.reason == "completed" and orch.of("end")[0]["ended_by"] == "idle"


def test_each_press_adds_one_period_and_a_new_wait_starts_afresh(orch, client, worktree):
    on_waiting(orch, 1, lambda: (orch.control("keep_open"), orch.control("keep_open")))
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("purpose") == KEEP_OPEN_PURPOSE
              and e["payload"].get("extensions") == 2, lambda e: orch.queue("second"))
    on_waiting(orch, 2, lambda: orch.control("keep_open"))
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("purpose") == KEEP_OPEN_PURPOSE
              and e["payload"].get("control_id") == 3, lambda e: orch.control("finish"))
    fixture = claude([say("one"), result()], [say("two"), result()])
    outcome, fake, _ = drive(fixture, client, worktree, idle=30.0)
    assert fake.sent[:2] == ["Fix the backup job.", "second"]
    assert [a["extensions"] for a in answers(orch)] == [1, 2, 1]
    first, second, third = answers(orch)
    assert second["idle_left"] > first["idle_left"] + 25
    assert 25 < third["idle_left"] <= 60
    assert orch.of("end")[0]["ended_by"] == "finish"


def test_a_press_pulled_while_the_waiting_event_is_posted_finds_the_session_waiting(orch, client, worktree):
    on_waiting(orch, 1, lambda: orch.control("keep_open"))
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("purpose") == KEEP_OPEN_PURPOSE,
              lambda e: orch.control("finish"))
    session = _session(orch, client, worktree)
    post = session.poster.post

    async def post_then_pull(event):
        out = await post(event)
        if event.kind == "turn" and event.payload.get("phase") == "waiting":
            await session._pull_once()
        return out

    session.poster.post = post_then_pull
    asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    assert [(a["control_id"], a["outcome"], a.get("extensions")) for a in answers(orch)] == [(1, "extended", 1)]
    assert orch.of("end")[0]["ended_by"] == "finish"


def drive_active_keep_open(orch, client, worktree, monkeypatch, *, driven):
    answered = asyncio.Event()
    keep_open = Session._keep_open

    async def answer(session, control):
        await keep_open(session, control)
        answered.set()

    async def wait_for_answer(provider, interrupted, _seconds):
        assert interrupted is provider._interrupted
        await answered.wait()
        return interrupted.is_set()

    monkeypatch.setattr(Session, "_keep_open", answer)
    monkeypatch.setattr(FakeProvider, "_wait", wait_for_answer)
    orch.when(lambda e: e["kind"] == "tool", lambda e: orch.control("keep_open"))
    if driven:
        on_waiting(orch, 1, lambda: orch.control("finish"))
    fixture = claude([tool(), {"fake": "sleep", "seconds": 0}, say("done"), result()])
    drive(fixture, client, worktree, driven=driven)


def test_a_press_while_a_turn_runs_is_refused_as_not_waiting(orch, client, worktree, monkeypatch):
    drive_active_keep_open(orch, client, worktree, monkeypatch, driven=True)
    said = answers(orch)
    assert len(said) == 1
    assert said[0]["outcome"] == "refused" and said[0]["step"] == "not_waiting"
    assert said[0]["text"] == KEEP_OPEN_WORDS["not_waiting"] and "idle_left" not in said[0]


def test_a_bounded_session_answers_not_waiting(orch, client, worktree, monkeypatch):
    drive_active_keep_open(orch, client, worktree, monkeypatch, driven=False)
    assert [(a["outcome"], a["step"]) for a in answers(orch)] == [("refused", "not_waiting")]


def test_a_press_after_finish_was_asked_is_refused_as_closed(orch, client, worktree):
    on_waiting(orch, 1, lambda: (orch.control("finish"), orch.control("keep_open")))
    drive(claude([say("one"), result()]), client, worktree)
    said = answers(orch)
    assert [(a["outcome"], a["step"]) for a in said] == [("refused", "closed")]
    assert said[0]["text"] == KEEP_OPEN_WORDS["closed"]
    assert orch.of("end")[0]["ended_by"] == "finish"


def _session(orch, client, worktree, idle=30.0):
    return Session(FakeProvider(claude([say("x"), result()])), client,
                   Options(cwd=str(worktree), driven=True, idle_seconds=idle), Caps(wall_clock=30, grace=1),
                   fast_timing())


# The clock is armed before the waiting event is posted, so a post that fails must disarm it again.
def test_waiting_post_failure_clears_the_deadline(worktree, monkeypatch):
    session = _session(None, None, worktree)

    async def broken_post(event):
        assert event.payload == {"phase": "waiting"}
        assert session._idle_deadline is not None
        raise RuntimeError("post failed")

    monkeypatch.setattr(session.poster, "post", broken_post)
    with pytest.raises(RuntimeError, match="post failed"):
        asyncio.run(session._await_input())
    assert session._idle_deadline is None


def test_a_press_at_or_past_the_limit_is_refused_as_closed(orch, client, worktree):
    session = _session(orch, client, worktree)

    async def go():
        loop = asyncio.get_running_loop()
        session._idle_deadline = loop.time() - 0.01
        await session._keep_open(Control(7, "keep_open"))
        session._idle_deadline = None
        session.stop_reason = "revoked"
        await session._keep_open(Control(8, "keep_open"))

    asyncio.run(go())
    assert [(a["control_id"], a["step"]) for a in answers(orch)] == [(7, "closed"), (8, "closed")]


def test_the_control_parses_from_the_pull(orch, client):
    from runner.control_plane import pull_all

    orch.control("keep_open")
    _, _, controls = asyncio.run(pull_all(client))
    assert controls == [Control(1, "keep_open")]

