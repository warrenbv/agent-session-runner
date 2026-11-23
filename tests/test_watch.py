"""The quiet and loop watch on a fake clock, with both providers' raw frames."""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import fast_timing, typed_result
from helpers import COMMAND, Clock, REFUSAL_NUDGE, call, complete, failures, partial, watch_done as done
from runner import refusals, watch
from runner.providers.base import CODING_KIND, RESEARCH_KIND, ProviderContext
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session, wrap_up_prompt

PROVIDERS = ("claude-code", "codex")
NUDGE = (
    "The runner has seen this command fail 8 times with no file changed in between: "
    "`pytest tests/test_backup.py`. Change approach, or stop and write in your summary what is blocking you.")


def payloads(w):
    return [event.payload for event, _ in w.drain()]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_quiet_stuck_once_and_alive_at_first_partial(provider):
    clock = Clock()
    w = watch.Watch(provider, clock)
    clock.advance(599)
    w.tick()
    assert payloads(w) == []
    clock.advance(1)
    w.tick()
    w.tick()
    assert payloads(w) == [{"role": "runner", "purpose": "quiet", "stage": "quiet", "seconds": 600,
                            "open": None}]
    clock.advance(600)
    w.tick()
    w.tick()
    assert payloads(w) == [{"role": "runner", "purpose": "quiet", "stage": "stuck", "seconds": 1200,
                            "open": None}]
    clock.advance(3)
    w.frame(partial(provider))
    w.frame(partial(provider))
    assert payloads(w) == [{"role": "runner", "purpose": "alive", "quiet_seconds": 1203}]
    clock.advance(600)
    w.tick()
    assert payloads(w)[0]["stage"] == "quiet"


@pytest.mark.parametrize("provider,tool", [("claude-code", "Bash"), ("claude-code", "Agent"),
                                           ("codex", "commandExecution"), ("codex", "collabAgentToolCall")])
def test_open_calls_get_30_minutes_and_close_by_id(provider, tool):
    clock = Clock()
    w = watch.Watch(provider, clock)
    w.frame(call(provider, "old", tool))
    w.frame(call(provider, "other", tool))
    w.frame(complete(provider, "other", tool, failed=False))
    assert set(w.open) == {"old"}
    clock.advance(600)
    w.tick()
    clock.advance(600)
    w.tick()
    assert payloads(w) == []
    clock.advance(600)
    w.tick()
    assert payloads(w) == [{"role": "runner", "purpose": "quiet", "stage": "stuck", "seconds": 1800,
                            "open": {"tool": "command" if tool == "commandExecution" else
                                     "wait" if tool == "collabAgentToolCall" else tool, "seconds": 1800}}]
    w.frame(complete(provider, "old", tool, failed=False))
    assert not w.open
    assert payloads(w) == [{"role": "runner", "purpose": "alive", "quiet_seconds": 1800}]


@pytest.mark.parametrize("provider,tool", [("claude-code", "Bash"), ("claude-code", "Agent"),
                                           ("codex", "commandExecution"), ("codex", "collabAgentToolCall")])
def test_new_turn_forgets_unfinished_calls_and_uses_the_no_open_thresholds(provider, tool):
    clock = Clock()
    w = watch.Watch(provider, clock)
    w.frame(call(provider, "abandoned", tool))
    w.frame(done(provider))
    clock.advance(3600)
    w.begin()
    assert not w.open
    clock.advance(600)
    w.tick()
    assert payloads(w) == [{"role": "runner", "purpose": "quiet", "stage": "quiet", "seconds": 600,
                            "open": None}]
    clock.advance(600)
    w.tick()
    assert payloads(w) == [{"role": "runner", "purpose": "quiet", "stage": "stuck", "seconds": 1200,
                            "open": None}]
    w.frame(call(provider, "current", tool))
    assert payloads(w) == [{"role": "runner", "purpose": "alive", "quiet_seconds": 1200}]
    clock.advance(1200)
    w.tick()
    assert not w.drain() and set(w.open) == {"current"}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_partials_keep_it_alive(provider):
    clock = Clock()
    w = watch.Watch(provider, clock)
    for _ in range(20):
        clock.advance(599)
        w.frame(partial(provider))
        w.tick()
    assert payloads(w) == [] and w.last_frame == clock()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_nudge_exactly_at_8_and_wrap_at_16(provider):
    w = watch.Watch(provider)
    for raw in failures(provider, 7, command="  " + COMMAND + "\n"):
        w.frame(raw)
    assert not w.drain()
    for raw in failures(provider, 1):
        w.frame(raw)
    [(event, note)] = w.drain()
    assert note == NUDGE
    assert event.payload == {"role": "runner", "purpose": "looping", "kind": "failing", "count": 8,
                             "command": COMMAND, "action": "nudged"}
    for raw in failures(provider, 8):
        w.frame(raw)
    assert payloads(w) == [{"role": "runner", "purpose": "looping", "kind": "failing", "count": 16,
                            "command": COMMAND, "action": "wrapping_up"}]
    assert w.wrap_requested


@pytest.mark.parametrize("provider,edit", [("claude-code", "Edit"), ("claude-code", "Write"),
                                          ("claude-code", "MultiEdit"), ("codex", "fileChange")])
def test_edit_between_failures_resets_the_counter(provider, edit):
    w = watch.Watch(provider)
    for raw in failures(provider, 7):
        w.frame(raw)
    w.frame(call(provider, "edit", edit))
    w.frame(complete(provider, "edit", edit, failed=False))
    for raw in failures(provider, 7):
        w.frame(raw)
    assert not w.drain()


@pytest.mark.parametrize("provider,edit,status", [("claude-code", "Edit", None),
                                                 ("claude-code", "Write", None),
                                                 ("claude-code", "MultiEdit", None),
                                                 ("codex", "fileChange", "failed"),
                                                 ("codex", "fileChange", "declined")])
def test_failed_edit_between_eight_failures_still_nudges(provider, edit, status):
    w = watch.Watch(provider)
    for raw in failures(provider, 4):
        w.frame(raw)
    w.frame(call(provider, "edit", edit))
    raw = complete(provider, "edit", edit)
    if provider == "codex":
        raw["params"]["item"].pop("exitCode")
        raw["params"]["item"]["status"] = status
    w.frame(raw)
    assert "edit" not in w.open
    for raw in failures(provider, 4):
        w.frame(raw)
    [(event, note)] = w.drain()
    assert event.payload["count"] == 8 and event.payload["action"] == "nudged"
    assert note == NUDGE


def test_refusals_count_by_whole_command_and_wrap_at_10():
    w = watch.Watch("codex")
    command = "x" * 350
    for _ in range(4):
        w.repeated("refused", command + "a")
        w.repeated("refused", command + "b")
    assert not w.drain()
    w.repeated("refused", command + "a")
    [(event, note)] = w.drain()
    assert event.payload["kind"] == "refused" and event.payload["count"] == 5
    assert len(event.payload["command"]) == 300
    assert note == watch.REFUSAL_LOOP_NOTE.format(count=5, command=command + "a")
    for _ in range(5):
        w.repeated("refused", command + "a")
    assert payloads(w)[0]["action"] == "wrapping_up"


def run_watch(provider, client, cwd, steps, *, kind=CODING_KIND, driven=False, steer=True, test_cmd=None):
    output = [] if provider == "claude-code" else [{"method": "item/completed", "params": {"item": {
        "type": "agentMessage", "id": "result", "text": json.dumps(typed_result("wrapped"))}}}]
    steps = [*steps[:-1], *output, steps[-1]]
    fake = FakeProvider({"provider": provider, "turns": [steps],
                         "default_turn": [*output, done(provider)], "steer": steer,
                         "schema_per_turn": driven})
    session = Session(fake, client, Options(cwd=str(cwd), kind=kind, diff=False, driven=driven,
                                           idle_seconds=0.05, test_cmd=test_cmd),
                      Caps(wall_clock=30, grace=0.1), fast_timing(), handoff=lambda res: {"handoff": "pull request"})
    outcome = asyncio.run(asyncio.wait_for(session.run("work"), 5))
    return outcome, fake, session


@pytest.mark.parametrize("provider,edit", [("claude-code", "Edit"), ("claude-code", "Write"),
                                          ("claude-code", "MultiEdit"), ("codex", "fileChange")])
def test_failed_edits_do_not_hide_the_session_nudge(orch, client, tmp_path, provider, edit):
    steps = failures(provider, 4) + [call(provider, "edit", edit), complete(provider, "edit", edit)]
    steps += failures(provider, 4) + [done(provider)]
    outcome, fake, _ = run_watch(provider, client, tmp_path, steps)
    assert outcome.reason == "completed" and fake.steered == [NUDGE]
    assert [p["count"] for p in orch.of("message", purpose="looping")] == [8]


@pytest.mark.parametrize("provider,edit", [("claude-code", "Edit"), ("claude-code", "Write"),
                                          ("claude-code", "MultiEdit"), ("codex", "fileChange")])
def test_successful_edits_and_turn_boundaries_do_not_reset_refusals(provider, edit):
    w = watch.Watch(provider)
    for _ in range(4):
        w.repeated("refused", COMMAND)
    w.frame(call(provider, "edit", edit))
    w.frame(complete(provider, "edit", edit, failed=False))
    w.begin()
    w.repeated("refused", COMMAND)
    [(event, note)] = w.drain()
    assert event.payload["count"] == 5 and note == REFUSAL_NUDGE


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("driven", [False, True])
def test_both_providers_nudge_then_interrupt_and_wrap_up_a_looping_session(orch, client, tmp_path, provider, driven):
    steps = failures(provider, 16) + [{"fake": "sleep", "seconds": 20}, done(provider)]
    outcome, fake, session = run_watch(provider, client, tmp_path, steps, driven=driven, test_cmd="pytest -q")
    assert fake.steered == [NUDGE]
    assert fake.interrupts == 1 and not fake.terminated
    assert fake.sent[1] == wrap_up_prompt("looping") and len(fake.sent) == 3
    if driven:
        assert fake.schemas[0] is None and fake.schemas[1] == session.options.output_schema
    assert session.wrapped_up == "looping" and outcome.reason == "completed"
    assert orch.of("end")[0]["wrapped_up"] == "looping"
    assert orch.of("end")[0]["handoff"]["handoff"] == "pull request"
    assert [p["action"] for p in orch.of("message", purpose="looping")] == ["nudged", "wrapping_up"]
    assert orch.of("message", purpose="wrap_up")[0]["why"] == "looping"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_five_refusals_through_the_fakes(orch, client, tmp_path, provider):
    steps = [{"fake": "refusal", "command": "  " + COMMAND + "  "}] * 10
    steps += [{"fake": "sleep", "seconds": 20}, done(provider)]
    outcome, fake, _ = run_watch(provider, client, tmp_path, steps)
    assert outcome.reason == "completed" and fake.interrupts == 1
    assert fake.steered == [REFUSAL_NUDGE]
    assert [p["count"] for p in orch.of("message", purpose="looping")] == [5, 10]
    assert len(orch.of("message", purpose="refused")) == 10


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("kind", ["failing", "refused"])
def test_unaccepted_steer_is_the_next_prompt(orch, client, tmp_path, provider, kind):
    steps = failures(provider, 8) if kind == "failing" else [{"fake": "refusal", "command": COMMAND}] * 5
    note = NUDGE if kind == "failing" else REFUSAL_NUDGE
    _, fake, session = run_watch(provider, client, tmp_path, steps + [done(provider)], steer=False)
    assert fake.sent == ["work", note] and not fake.steered
    assert not session.wrapped_up
    said = orch.of("message", role="user")
    assert said[1]["author"] == "runner" and said[1]["text"] == note


@pytest.mark.parametrize("provider", PROVIDERS)
def test_research_excluded(orch, client, tmp_path, provider):
    steps = [{"fake": "refusal", "command": COMMAND}] * 12 + [partial(provider), done(provider)]
    _, fake, session = run_watch(provider, client, tmp_path, steps, kind=RESEARCH_KIND)
    assert session.watch is None and fake.ctx.raw_frame is None
    assert not fake.steered and not session.wrapped_up
    assert not any(orch.of("message", purpose=p) for p in ("quiet", "alive", "looping"))


def test_refusal_line_keeps_full_command_private():
    command = "x" * 400
    line = refusals.line("Refused.", "Bash", command)
    assert line.command == command
    assert set(line) == {"text", "tool", "command"} and len(line["command"]) == 300


@pytest.mark.parametrize("provider", PROVIDERS)
def test_test_turn_failures_do_not_escalate_the_work_counter(orch, client, tmp_path, provider):
    output = [] if provider == "claude-code" else [{"method": "item/completed", "params": {"item": {
        "type": "agentMessage", "id": "result", "text": json.dumps(typed_result())}}}]
    fake = FakeProvider({"provider": provider,
                         "turns": [failures(provider, 8) + output + [done(provider)],
                                   failures(provider, 8) + [done(provider)]]})
    session = Session(fake, client, Options(cwd=str(tmp_path), diff=False, test_cmd="pytest -q"),
                      Caps(wall_clock=30), fast_timing())
    outcome = asyncio.run(asyncio.wait_for(session.run("work"), 5))
    assert outcome.reason == "completed" and session.wrapped_up is None
    assert [p["action"] for p in orch.of("message", purpose="looping")] == ["nudged"]
    assert len(fake.sent) == 2


def test_thresholds_and_runner_words_are_pinned():
    assert (watch.QUIET_SECONDS, watch.STUCK_SECONDS, watch.OPEN_STUCK_SECONDS) == (600, 1200, 1800)
    assert (watch.FAILURES, watch.REFUSALS, watch.COMMAND_CAP) == (8, 5, 300)
    assert watch.LOOP_WRAP_REASON == "The session is wrapping up after looping."
    assert watch.LOOP_NOTE == (
        "The runner has seen this command fail {count} times with no file changed in between: "
        "`{command}`. Change approach, or stop and write in your summary what is blocking you.")
    assert watch.REFUSAL_LOOP_NOTE == (
        "The runner has refused this command {count} times: `{command}`. It will be refused again. "
        "Change approach, or stop and write in your summary what is blocking you.")


def test_idle_wait_does_not_rewrite_the_last_raw_frame_time():
    clock = Clock()
    w = watch.Watch("codex", clock)
    w.frame(partial("codex"))
    last = w.last_frame
    clock.advance(3600)
    w.begin()
    w.tick()
    assert not w.drain() and w.last_frame == last


def test_beginning_a_turn_cannot_clear_a_quiet_post_before_a_frame():
    clock = Clock()
    w = watch.Watch("claude-code", clock)
    clock.advance(600)
    w.tick()
    w.drain()
    w.begin()
    w.frame(partial("claude-code"))
    assert payloads(w) == [{"role": "runner", "purpose": "alive", "quiet_seconds": 600}]


def test_claude_receive_reports_partials_before_yield_and_steers_with_priority_next(monkeypatch, tmp_path):
    from runner.providers.claude import ClaudeProvider

    seen, sent = [], []
    raw = partial("claude-code")

    class Client:

        async def receive_response(self):
            yield raw

        async def query(self, messages):
            sent.extend([message async for message in messages])

    provider = ClaudeProvider()
    provider.client = Client()
    provider.ctx = ProviderContext(request_approval=None, cwd=str(tmp_path), raw_frame=seen.append)

    async def go():
        async for frame in provider._receive():
            assert seen == [frame] and not provider.normalise(frame)
        assert await provider.steer(NUDGE)

    asyncio.run(go())
    assert sent[0]["priority"] == "next" and sent[0]["message"]["content"] == NUDGE
    assert provider.options(provider.ctx).include_partial_messages is True
    provider.ctx.raw_frame = None
    assert provider.options(provider.ctx).include_partial_messages is False
    if provider.temp is not None:
        provider.temp.close()


def test_codex_adapter_observes_deltas_and_steers_the_current_turn():
    from types import SimpleNamespace

    from runner.providers.codex import CodexProvider

    seen, steered = [], []
    frames = [partial("codex"), done("codex")]

    class Client:

        def turn_start(self, thread, text, params):
            return SimpleNamespace(turn=SimpleNamespace(id="turn-id"))

        def next_turn_notification(self, turn):
            raw = frames.pop(0)
            return SimpleNamespace(method=raw["method"], payload=SimpleNamespace(params=raw["params"]))

        def turn_steer(self, thread, turn, text):
            steered.append((thread, turn, text))

    provider = CodexProvider()
    provider.client, provider.thread_id = Client(), "thread-id"
    provider.ctx = ProviderContext(request_approval=None, cwd="/tmp", raw_frame=seen.append)

    async def go():
        async for raw in provider.turn("work"):
            assert seen[-1] == raw
            if raw["method"].endswith("outputDelta"):
                assert not provider.normalise(raw)
                assert await provider.steer(NUDGE)
        assert not await provider.steer(NUDGE)

    asyncio.run(go())
    assert len(seen) == 2 and steered == [("thread-id", "turn-id", NUDGE)]


def test_codex_global_frames_are_observed_before_duplicate_filtering():
    from types import SimpleNamespace

    from runner.providers.codex import CodexProvider

    seen = []
    notification = SimpleNamespace(method="account/rateLimits/updated", payload=SimpleNamespace(params={"a": 1}))

    class Client:

        def __init__(self):
            self.n = 0

        def next_notification(self):
            self.n += 1
            if self.n > 2:
                raise EOFError
            return notification

    class Loop:

        def call_soon_threadsafe(self, callback, *args):
            callback(*args)

    provider = CodexProvider()
    provider.client, provider.loop = Client(), Loop()
    provider.ctx = ProviderContext(request_approval=None, cwd="/tmp", raw_frame=seen.append)
    provider._read_global()
    assert len(seen) == 2
    assert not provider.repeated(provider._global.get_nowait())
    assert provider.repeated(provider._global.get_nowait())

