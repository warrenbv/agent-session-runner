"""Ending what a session left running: TERM then KILL against a fake control group, the cgroup reader
against a fake /proc and /sys, and the order of steps at each kind of ending."""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

from conftest import fast_timing
from runner import placeholders, reaper
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session

SELF = 100


class FakeGroup:

    def __init__(self, pids, names=None, stubborn=(), unkillable=(), forks=None):
        self.pids = set(pids)
        self.names = names or {}
        self.stubborn, self.unkillable = set(stubborn), set(unkillable)
        self.forks = dict(forks or {})
        self.sent: list[tuple[int, int]] = []

    def members(self):
        return set(self.pids)

    def name(self, pid):
        return self.names.get(pid, "sleep")

    def signal(self, pid, sig):
        self.sent.append((pid, sig))
        if pid in self.unkillable:
            return
        if sig == signal.SIGTERM and pid in self.forks:
            self.pids.add(self.forks.pop(pid))
        if sig == signal.SIGKILL or pid not in self.stubborn:
            self.pids.discard(pid)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make(group, spare=(SELF,)):
    clock = Clock()
    return reaper.Reaper(group, spare=lambda: set(spare), grace=3.0, step=0.5, sleep=clock.sleep, clock=clock)


def test_term_ends_leftovers_and_spares_the_runner():
    group = FakeGroup({SELF, 201, 202}, names={201: "sleep", 202: "node"})
    swept = make(group).sweep()
    assert swept.ended == 2 and swept.killed == 0 and swept.left == 0
    assert swept.names == {"sleep": 1, "node": 1}
    assert group.pids == {SELF}
    assert all(pid != SELF for pid, _ in group.sent)
    assert {sig for _, sig in group.sent} == {signal.SIGTERM}


def test_term_then_kill_after_the_grace():
    group = FakeGroup({SELF, 201, 202}, stubborn={202}, names={201: "sleep", 202: "python3"})
    swept = make(group).sweep()
    assert group.pids == {SELF}
    assert swept.ended == 2 and swept.killed == 1 and swept.left == 0
    assert (202, signal.SIGTERM) in group.sent and (202, signal.SIGKILL) in group.sent
    assert group.sent.index((202, signal.SIGTERM)) < group.sent.index((202, signal.SIGKILL))
    assert (201, signal.SIGKILL) not in group.sent
    assert "needed SIGKILL" in swept.sentence()


def test_a_process_started_during_the_sweep_is_ended_too():
    group = FakeGroup({SELF, 201}, forks={201: 301}, names={201: "sh", 301: "sleep"})
    swept = make(group).sweep()
    assert group.pids == {SELF} and swept.ended == 2
    assert (301, signal.SIGTERM) in group.sent


def test_one_that_cannot_be_ended_is_counted_as_left():
    group = FakeGroup({SELF, 201}, unkillable={201})
    swept = make(group).sweep()
    assert swept.ended == 1 and swept.killed == 1 and swept.left == 1
    assert "could not be stopped" in swept.sentence()


def test_nothing_to_end_and_no_group():
    assert make(FakeGroup({SELF})).sweep() == reaper.Swept()

    class NoGroup:
        def members(self):
            return None

    assert make(NoGroup()).sweep() is None


def test_the_sentence_counts_repeated_names():
    swept = reaper.Swept(ended=3, names={"node": 2, "sleep": 1})
    assert swept.sentence() == ("At the session's end the runner stopped 3 processes it had left running: "
                                "node (2), sleep.")


def fake_host(tmp_path, rel, procs):
    proc, root = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text(f"0::{rel}\n")
    group = root / rel.lstrip("/")
    (group / "child").mkdir(parents=True)
    pids = sorted(procs)
    (group / "cgroup.procs").write_text("".join(f"{p}\n" for p in pids[:-1]))
    (group / "child" / "cgroup.procs").write_text(f"{pids[-1]}\n")
    for pid, (state, comm) in procs.items():
        (proc / str(pid)).mkdir()
        (proc / str(pid) / "stat").write_text(f"{pid} ({comm}) {state} 1 0 0\n")
        (proc / str(pid) / "comm").write_text(comm + "\n")
    return str(proc), str(root)


def test_the_group_reads_every_member_but_zombies(tmp_path):
    proc, root = fake_host(tmp_path, "/user.slice/user-1000.slice/user@1000.service/app.slice/runner-session-7.scope",
                           {11: ("S", "python3"), 12: ("S", "sleep"), 13: ("Z", "defunct"), 14: ("R", "node\x1b[31m")})
    group = reaper.Cgroup(proc=proc, root=root)
    assert group.members() == {11, 12, 14}
    assert group.name(14) == "node[31m" and group.name(99) == "?"


def test_outside_a_session_scope_nothing_is_read(tmp_path):
    proc, root = fake_host(tmp_path, "/user.slice/user-1000.slice/session-3.scope", {11: ("S", "bash")})
    assert reaper.Cgroup(proc=proc, root=root).members() is None
    assert reaper.Cgroup(proc=str(tmp_path / "none"), root=root).members() is None


def test_the_real_reaper_off_a_scope_does_nothing():
    group = reaper.Cgroup()
    if group.path() is not None:
        pytest.skip("this run is itself inside a session scope (a coding session the runner started), where a real "
                    "sweep would signal the session's own processes; it runs on a developer machine or a server's shell")
    assert reaper.Reaper(group).sweep() is None


def say(text):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "TextBlock", "text": text}]}


def result(subtype="success"):
    return {"type": "ResultMessage", "subtype": subtype, "duration_ms": 1, "duration_api_ms": 1,
            "is_error": subtype != "success", "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01,
            "terminal_reason": "completed"}


def claude(*turns):
    return {"provider": "claude-code", "turns": [list(t) for t in turns], "default_turn": [say("Done."), result()]}


class Recorder:

    def __init__(self, steps, session_ref, results):
        self.steps, self.session_ref, self.results = steps, session_ref, list(results)
        self.seen: list[tuple[int, bool]] = []

    def sweep(self):
        session = self.session_ref[0]
        self.seen.append((session.turns, session.provider.closed))
        self.steps.append("sweep")
        return self.results.pop(0) if self.results else reaper.Swept()


def drive(fixture, client, worktree, monkeypatch, *, results=(), handoff=True, caps=None, driven=False,
          test_cmd=None):
    steps: list[str] = []
    provider = FakeProvider(fixture)
    closing = provider.close

    async def close():
        steps.append("close")
        await closing()

    provider.close = close
    real_settle = placeholders.settle

    def settle(cwd, *a, **kw):
        steps.append("settle")
        return real_settle(cwd, *a, **kw)

    monkeypatch.setattr(placeholders, "settle", settle)

    def hand(_result):
        steps.append("handoff")
        return {"handoff": "nothing to push"}

    ref: list[Session] = []
    session = Session(provider, client, Options(cwd=str(worktree), driven=driven, idle_seconds=30, test_cmd=test_cmd),
                      caps or Caps(wall_clock=30, grace=0.2), fast_timing(), handoff=hand if handoff else None)
    ref.append(session)
    session.reaper = Recorder(steps, ref, results)
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    return outcome, steps, session


def cleanup_messages(orch):
    return orch.of("message", purpose="cleanup")


def test_finished_close_sweep_placeholders_handoff_sweep(orch, client, worktree, monkeypatch):
    outcome, steps, _ = drive(claude([say("ok"), result()]), client, worktree, monkeypatch)
    assert outcome.reason == "completed"
    assert steps == ["close", "sweep", "settle", "handoff", "sweep"]
    assert not cleanup_messages(orch)


def test_the_hand_off_and_test_turn_run_outside_the_sweep(orch, client, worktree, monkeypatch):
    (worktree / "change.txt").write_text("work\n")
    test_turn = [say("ran"), result()]
    outcome, steps, session = drive(claude([say("ok"), result()], test_turn), client, worktree, monkeypatch,
                                    test_cmd="pytest -q")
    assert outcome.reason == "completed"
    assert orch.of("message", purpose="tests")
    assert session.reaper.seen and all(closed for _, closed in session.reaper.seen)
    assert steps.index("handoff") == steps.index("settle") + 1 and steps[-1] == "sweep"


def test_stopped_by_a_revoked_token(orch, client, worktree, monkeypatch):
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("role") == "assistant", lambda e: orch.revoke())
    outcome, steps, _ = drive(claude([say("a"), {"fake": "sleep", "seconds": 5}, result()]), client, worktree,
                              monkeypatch, results=[reaper.Swept(1, {"sleep": 1})])
    assert outcome.reason == "revoked"
    assert steps == ["close", "sweep", "settle", "sweep"]
    assert not cleanup_messages(orch)


def test_a_limit(orch, client, worktree, monkeypatch):
    outcome, steps, _ = drive(claude([{"fake": "sleep", "seconds": 0.05}, say("a"), result(subtype="error_max_turns")]),
                              client, worktree, monkeypatch, caps=Caps(max_turns=1, wall_clock=30, grace=0.2))
    assert outcome.reason == "completed" and orch.of("end")[0]["wrapped_up"] == "max_turns"
    assert steps == ["close", "sweep", "settle", "handoff", "sweep"]


def test_the_wall_clock(orch, client, worktree, monkeypatch):
    outcome, steps, _ = drive(claude([{"fake": "hang"}]), client, worktree, monkeypatch,
                              caps=Caps(wall_clock=0.2, grace=0.1))
    assert outcome.reason == "completed" and orch.of("end")[0]["wrapped_up"] == "wall_clock"
    assert steps == ["close", "sweep", "settle", "handoff", "sweep"]


def test_a_failure(orch, client, worktree, monkeypatch):
    outcome, steps, _ = drive(claude([{"fake": "write", "path": "../outside.txt", "text": "x"}, result()]),
                              client, worktree, monkeypatch)
    assert outcome.reason == "error"
    assert steps == ["close", "sweep", "settle", "sweep"]


def test_the_event_when_something_was_ended(orch, client, worktree, monkeypatch):
    outcome, _, _ = drive(claude([say("ok"), result()]), client, worktree, monkeypatch,
                          results=[reaper.Swept(3, {"sleep": 1, "node": 2}, killed=1), reaper.Swept(1, {"git": 1})])
    assert outcome.reason == "completed"
    [event] = cleanup_messages(orch)
    assert event["author"] == "runner" and event["ended"] == 4 and event["killed"] == 1 and event["left"] == 0
    assert event["names"] == {"sleep": 1, "node": 2, "git": 1}
    assert "stopped 4 processes" in event["text"] and "node (2)" in event["text"]
    kinds = [e["kind"] for e in orch.events]
    assert kinds.index("end") > max(i for i, e in enumerate(orch.events) if e["payload"].get("purpose") == "cleanup")


def test_a_driven_session_is_swept_only_at_its_end(orch, client, worktree, monkeypatch):
    def on_waiting(n, action):
        orch.when(lambda e: e["kind"] == "turn" and e["payload"].get("phase") == "waiting"
                  and len(orch.of("turn", phase="waiting")) == n, lambda e: action())

    on_waiting(1, lambda: orch.queue("second"))
    on_waiting(2, lambda: orch.control("finish"))
    outcome, steps, session = drive(claude([say("one"), result()], [say("two"), result()]), client, worktree,
                                    monkeypatch, driven=True)
    assert outcome.reason == "completed" and session.turns == 2
    assert steps.count("sweep") == 2 and steps[:2] == ["close", "sweep"]
    assert session.reaper.seen == [(2, True), (2, True)]


def test_a_sweep_that_fails_never_fails_the_run(orch, client, worktree, monkeypatch):
    class Broken:
        def sweep(self):
            raise OSError("no group")

    provider = FakeProvider(claude([say("ok"), result()]))
    session = Session(provider, client, Options(cwd=str(worktree)), Caps(wall_clock=30, grace=0.2), fast_timing())
    session.reaper = Broken()
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix it."), timeout=30))
    assert outcome.reason == "completed" and orch.of("end")[0]["reason"] == "completed"


def test_the_real_reaper_spares_this_process():
    assert os.getpid() in reaper.ancestors()

