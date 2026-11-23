"""The session driver: one provider, one session, one numbered event stream.

The driver sends the first prompt as a turn, posts every normalised event in order, and between and
during turns applies what the control plane's pull delivers: the owner's messages (`queue` for after
this turn, `steer` for the next step, `now` to interrupt), approval answers, and controls (`interrupt`,
`finish`, `withdraw`, `settings`, `keep_open`).

Caps are reported as `limit` and `end` events and never extended: turns, budget and wall clock. A
bounded coding session that reaches one, or is asked to finish, wraps up instead: its running turn is
interrupted and one closing turn asks for the typed result. A driven session waits for the owner's
next message after each turn until it is finished or idle.

After work that completed and changed files, one more turn asks the session to run the project's test
command, and the verdict is read from the provider's record of that tool call (testrun.py). Then the
hand-off checks (handoff_check.py) and the hand-off itself (handoff.py). At the end, by any route, the
tools socket closes, left-over processes are ended (reaper.py) and sandbox placeholders are removed
(placeholders.py). See docs/design.md, "How a session runs".
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from runner import baseline as baseline_mod
from runner import drafts as drafts_mod
from runner import gitdiff, placeholders, session_socket, testrun
from runner import handoff as handoff_mod
from runner import handoff_check
from runner import reaper as reaper_mod
from runner import refusals
from runner import settings as settings_mod
from runner import watch as watch_mod
from runner.claude_policy import ClaudePolicyError
from runner.events import Event, clip
from runner.control_plane import Client, Control, Poster, Prompt, Retry, SessionEnded, Unreachable, pull_full
from runner.providers.base import (CODING_KIND, RESEARCH_KIND, RESULT_SCHEMA, TEST_CLAIM_DESCRIPTION,
                                   Change, Provider, ProviderContext, tidy_result, usable_id)
from runner.relay import CEILING, Relay

log = logging.getLogger("runner.session")

PROMPT_PREVIEW = 4000
STEP_CAP = 200
_RUNNING = object()
RUNNER_AUTHOR = "runner"
EXIT = {"completed": 0, "max_turns": 2, "budget": 2, "wall_clock": 2, "max_subagents": 2, "revoked": 3, "error": 1}
SEARCH_TOOL = "WebSearch"
SHELL_ITEMS = ("command",)
CONTINUATIONS = 2
CONTINUE_PURPOSE = "continue"
FOLLOW_WORDS = ("The session's turn ended while {n} subagent{s} it started {were} still running and before it "
                "wrote {files}. The runner is waiting for their reports and the session's own next turn.")
CONTINUE_PROMPT = ("Your turn ended before you wrote {files}, and this session has no later turn unless the runner "
                   "gives one, so the runner gives you this one. {running}Write {files} in your working directory "
                   "now, in the shapes your instructions give, from what you and your subagents have found. Do not "
                   "end this turn while a subagent you started is still running.")


def subagent_start(payload: dict[str, Any]) -> str | None:
    phase = payload.get("phase")
    if phase == "requested":
        key = payload.get("tool_use_id")
    elif phase == "started" and "tool" not in payload:
        task_type = payload.get("task_type")
        if task_type is not None and "agent" not in str(task_type):
            return None
        key = payload.get("tool_use_id") or payload.get("task_id") or payload.get("agent_thread_id") \
            or payload.get("id")
    else:
        return None
    return str(key) if key else f"unnamed-{id(payload)}"
RESUME_PURPOSE = "resume"
PROVIDER_SESSION_PURPOSE = "provider_session"
RESUMED_FRESH_PURPOSE = "resumed_fresh"
RESUMED_WITHOUT = ("This session is being picked back up with a new message from its owner, but its earlier "
                   "conversation is no longer held on this machine, so you start without it. Its work so far is "
                   "on its branch in this working directory. Its original brief was:\n\n{brief}")
BASE_FETCH_OUTCOMES = ("fetched", "failed")
BASE_FETCHED = ("Runner's note: before this run the runner fetched `{base}` from GitHub into `origin/{base}`, so you "
                "can compare your branch with it and bring its latest changes into this working tree if the work "
                "needs them.")
BASE_FETCH_FAILED = ("Runner's note: before this run the runner could not fetch `{base}` from GitHub, so "
                     "`origin/{base}` is as this machine last fetched it, which may be older than GitHub's; carry on with "
                     "the work, and say in your result if it needed the latest `{base}`.")


def base_fetch_line(outcome: str | None, base: str) -> str | None:
    if outcome == "fetched":
        return BASE_FETCHED.format(base=base)
    if outcome == "failed":
        return BASE_FETCH_FAILED.format(base=base)
    return None
WRAP_UP_SECONDS = 600.0
WRAP_UP_PURPOSE = "wrap_up"
WRAP_UP_WHYS = ("asked", "wall_clock", "looping", "max_turns", "budget")
WRAP_UP_LIMITS = ("wall_clock", "max_turns", "budget")
WRAP_UP_BECAUSE = {
    "asked": "its owner asked for it to be wrapped up",
    "wall_clock": "it has reached its time limit",
    "looping": "it kept running the same failing command",
    "max_turns": "it has reached its turn limit",
    "budget": "it has reached its budget",
}
WRAP_UP_PROMPT = (
    "The runner is wrapping this session up now, because {because}, and this is its last turn of work. Stop "
    "where you are: do not start anything new. Leave nothing half-edited: finish or undo the change in hand so "
    "the working tree builds, and remove any scratch files. Do not commit or push; the runner commits your "
    "changes, has the project's tests run, pushes the branch and opens the pull request. Then give your final "
    "structured result for the whole session, not only this turn: `summary` says what you changed and why and "
    "what is left unfinished, `files_changed` every file changed, `tests_run` every test you ran with its "
    "outcome, `open_questions` what needs the owner, `next_steps` what a later run should do first (may be "
    "empty), and `items` each thing you were asked, done only where you "
    "saw it work, unverified where you did it but did not check it, not_done where you did not get to it. It "
    "becomes the pull request's description. This turn is cut off after {minutes} minutes. "
    + TEST_CLAIM_DESCRIPTION)


def wrap_up_prompt(why: str, seconds: float = WRAP_UP_SECONDS) -> str:
    minutes = max(int(round(seconds / 60)), 1)
    return WRAP_UP_PROMPT.format(because=WRAP_UP_BECAUSE.get(why, WRAP_UP_BECAUSE["asked"]), minutes=minutes)


KEEP_OPEN_PURPOSE = "keep_open"
KEEP_OPEN_WORDS = {
    "extended": "Kept open: the idle limit moved {span} later; it now closes in {left} if no message comes.",
    "not_waiting": "Not kept open: the session is not waiting for a message, so its idle clock is not running.",
    "closed": "Not kept open: the session is already closing (finishing, at its idle limit, or stopped).",
}
WITHDRAW_WORDS = {
    "withdrawn": "Message withdrawn: the session had not read it and will not see it.",
    "too_late": "Too late to withdraw this message: the session has already read it. Nothing was changed.",
    "unknown": "This message could not be withdrawn: the runner was not holding it (it was never taken from "
               "the queue, or the session had stopped taking messages). Nothing was changed.",
}


@dataclass
class Caps:
    max_turns: int | None = None
    max_budget_usd: float | None = None
    wall_clock: float = 1800.0
    grace: float = 30.0
    wrap_up: float = WRAP_UP_SECONDS


@dataclass
class Timing:
    pull_interval: float = 1.0
    approval_ceiling: float = CEILING
    linger: float = 0.0
    retry: Retry = field(default_factory=Retry)
    follow_quiet: float = 60.0
    follow_idle: float = 900.0


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _span(seconds: float) -> str:
    s = max(int(seconds), 0)
    if s < 60:
        return f"{s} s"
    h, m = divmod(s // 60, 60)
    return " ".join(part for part in (f"{h} h" if h else "", f"{m} min" if m else "") if part)


@dataclass
class Options:
    cwd: str
    model: str | None = None
    effort: str | None = None
    permission_mode: str = "default"
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    resume: str | None = None
    output_schema: dict[str, Any] | None = field(default_factory=lambda: RESULT_SCHEMA)
    diff: bool = True
    brief_note: str | None = None
    test_cmd: str | None = None
    tools: bool = False
    propose: bool = False
    lead: bool = False
    driven: bool = False
    idle_seconds: float = 14400.0
    drafts: bool = False
    file_diffs: bool = False
    base_ref: str | None = None
    start_mode: str | None = None
    thinking: bool = False
    web: bool = False
    kind: str = CODING_KIND
    max_subagents: int | None = None
    max_searches: int | None = None
    first_sequence: int = 1
    resume_text: str | None = None
    carry_over: str | None = None
    cost_before: float | None = None
    base_fetch: str | None = None
    merge_note: str | None = None
    baseline: bool = True
    worktree: gitdiff.Worktree | None = None


@dataclass
class Outcome:
    reason: str
    turns: int
    cost_usd: float | None
    result: dict[str, Any] | None
    handoff: dict[str, Any] | None = None

    @property
    def exit_code(self) -> int:
        return EXIT.get(self.reason, 1)


class Session:
    def __init__(self, provider: Provider, client: Client, options: Options, caps: Caps | None = None,
                 timing: Timing | None = None,
                 handoff: Callable[[dict[str, Any] | None], dict[str, Any]] | None = None,
                 research: Any = None) -> None:
        self.provider, self.client, self.options = provider, client, options
        self.caps, self.timing = caps or Caps(), timing or Timing()
        self.handoff = handoff
        self.research = research
        self.limits_hit: list[str] = []
        self._subagents: set[str] = set()
        self._cap_task: asyncio.Task | None = None
        self._shell_task: asyncio.Task | None = None
        self._running: set[str] = set()
        self._reported: set[str] = set()
        self._left_running = 0
        self._follows = 0
        self._continuations = 0
        self._no_more: str | None = None
        self._seen_at = 0.0
        self._turn_ends = 0
        self.searches = 0
        self.poster = Poster(client, provider.name, self.timing.retry)
        self.poster.next_sequence = max(int(options.first_sequence), 1)
        self.relay = Relay(self.poster.post, self.timing.approval_ceiling)
        self.resuming = options.resume_text is not None
        self.fresh = False
        self._posted_id: str | None = None
        self._first_shown: str | None = None
        self.held: deque[Prompt] = deque()
        self.urgent: deque[Prompt] = deque()
        self.turns = 0
        self.cost: float | None = None
        self.result: dict[str, Any] | None = None
        self.last_text = ""
        self.last_turn: dict[str, Any] = {}
        self.stop_reason: str | None = None
        self.detail: str | None = None
        self.turn_task: asyncio.Task | None = None
        self.tests: dict[str, Any] | None = None
        self._test_base = options.base_ref
        self.title: str | None = None
        self._testing = False
        self._test_raws: list[dict[str, Any]] = []
        self.checks: dict[str, Any] | None = None
        self._brief = ""
        self._latest_request: str | None = None
        self._repairing = False
        self._repaired = False
        self._first: Prompt | None = None
        self._note: str | None = options.brief_note
        self._wake = asyncio.Event()
        self._pull_lock = asyncio.Lock()
        self.existed: frozenset[str] | None = None
        self._diff_base: str | None = options.base_ref
        self._files_sent: dict[str, str] = {}
        self._file_sizes: dict[str, tuple[str, int]] = {}
        self.tools: session_socket.ToolSocket | None = None
        self.tools_parent: Callable[[], str | None] = session_socket.runtime_dir
        self.reaper: Any = reaper_mod.Reaper()
        self.swept = reaper_mod.Swept()
        self.ended_by: str | None = None
        self.capped: str | None = None
        self._closing = False
        self._summarised = 0
        self._schema_skipped = False
        self._schema_on = True
        self._worked = 0.0
        self._clock: asyncio.Task | None = None
        self._clock_fired = False
        self._idle_deadline: float | None = None
        self._extensions = 0
        self.wrapped_up: str | None = None
        self._wrapping = False
        self._wrap_stop: asyncio.Task | None = None
        self._terminated = False
        self.ctx: ProviderContext | None = None
        self.per_turn_schema = bool(options.driven and getattr(provider, "schema_per_turn", False))
        self._handed: set[int] = set()
        self._held_settings: dict[str, str] = {}
        self._held_control: int | None = None
        self._in_turn = False
        self._settings_lock = asyncio.Lock()
        self._settings_shown = False
        self._turn_settings: dict[str, Any] | None = None
        self._turn_thinking = False
        self.drafter: drafts_mod.Drafter | None = \
            drafts_mod.Drafter(provider.name, drafts_mod.poster(client)) if options.drafts else None
        self.watch = watch_mod.Watch(provider.name) if options.kind == CODING_KIND else None

    # Every git call of the session, including those in worker threads, uses the bound worktree paths.
    async def run(self, first_prompt: str) -> Outcome:
        with gitdiff.use_worktree(self.options.worktree):
            return await self._run(first_prompt)

    async def _run(self, first_prompt: str) -> Outcome:
        self._brief = first_prompt
        ctx = ProviderContext(
            request_approval=self.relay.request, cwd=self.options.cwd, model=self.options.model,
            effort=self.options.effort, max_turns=None if self.options.driven else self.caps.max_turns,
            # Not on a resumed run: Claude's own cap would count the earlier runs' cost, which carries over a
            # resume. The runner counts this run's spending itself.
            max_budget_usd=None if self.resuming else self.caps.max_budget_usd,
            permission_mode=self.options.permission_mode, allowed_tools=self.options.allowed_tools,
            disallowed_tools=self.options.disallowed_tools, resume=self.options.resume,
            output_schema=None if self.per_turn_schema else self.options.output_schema,
            drafts=self.options.drafts, thinking=self.options.thinking, web=self.options.web,
            propose=self.options.propose is True and self.options.kind == CODING_KIND,
            lead=self.options.lead is True and self.options.kind == CODING_KIND,
            kind=self.options.kind, max_subagents=self.options.max_subagents,
            max_searches=self.options.max_searches, refusal_note=self._refusal_note,
            baseline=baseline_mod.load().text if self.options.kind == CODING_KIND and self.options.baseline
            else None)
        self.ctx = ctx
        if self.watch is not None:
            ctx.raw_frame = self._raw_frame
        if self.options.kind == CODING_KIND:
            self._note = handoff_mod.session_note(self.provider.name, self.options.cwd, self.options.brief_note)
        if not self.resuming:
            self._first = Prompt(0, first_prompt, "queue", "launcher")
            self.held.append(self._first)
        background: list[asyncio.Task] = []
        handoff: dict[str, Any] | None = None
        try:
            self.existed = placeholders.present(self.options.cwd)
        except OSError:
            log.warning("could not record the worktree's start; sandbox placeholders will be kept")
        try:
            if self.options.kind == CODING_KIND:
                await asyncio.to_thread(gitdiff.check_repository, self.options.cwd)
            if self.options.test_cmd and self.options.kind == CODING_KIND and self._test_base is None:
                try:
                    self._test_base = (await asyncio.to_thread(
                        gitdiff.git, self.options.cwd, "rev-parse", "--verify", "HEAD^{commit}")).strip()
                except gitdiff.GitError:
                    log.warning("test check: starting commit unavailable")
            if self.research is not None:
                await asyncio.to_thread(self.research.fetch_inputs)
            if self.options.tools:
                self.tools = session_socket.ToolSocket(self.client)
                ctx.mcp_socket = await self.tools.open(self.tools_parent())
            if self.resuming:
                await self._start_resumed(ctx, first_prompt)
            else:
                await self.provider.start(ctx)
                await self._conversation_seen()
            await self._baseline_said()
            start_refused = getattr(self.provider, "start_refused", None)
            if start_refused:
                await self._settings_event(None, {}, {}, dict(start_refused))
            if self.options.start_mode:
                await self._start_in_mode(self.options.start_mode)
            background = [asyncio.create_task(self._pull_loop())]
            if self.watch is not None:
                background.append(asyncio.create_task(self._watch_loop()))
            if not self.options.driven:
                background.append(asyncio.create_task(self._wall_clock()))
            reason = await self._drive()
            if reason == "completed" and self.research is not None and self.options.kind == RESEARCH_KIND \
                    and not self.options.driven:
                reason = await self._research_continue()
            while reason == "completed" and not self.wrapped_up:
                if self._needs_closing():
                    await self._closing_turn()
                    if self.stop_reason:
                        break
                if self.options.kind == CODING_KIND:
                    await self._handoff_check(repair=True)
                    if self.stop_reason or self.wrapped_up:
                        break
                await self._test_before_handoff()
                if self.stop_reason or self.wrapped_up or self._terminated or (not self.urgent and not self.held):
                    break
                reason = await self._drive()
            if self.wrapped_up and not self.stop_reason:
                reason = await self._wrap_up()
            reason = self.stop_reason or reason
        except SessionEnded:
            reason = "revoked"
        # These two carry sentences written to be shown: what was refused and what to repair.
        except ClaudePolicyError as exc:
            log.error("%s", exc)
            reason, self.detail = "error", str(exc)
        except gitdiff.GitSafetyError as exc:
            reason, self.detail = "error", str(exc)
        except Exception as exc:
            log.error("session failed: %s", type(exc).__name__)
            reason, self.detail = "error", type(exc).__name__
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            for stopping in (self._cap_task, self._shell_task, self._wrap_stop):
                if stopping is not None:
                    await asyncio.gather(stopping, return_exceptions=True)
            if self.drafter is not None:
                await self.drafter.close()
            self.relay.deny_all("The session has ended.", close=True)
            try:
                await self.provider.close()
            except Exception:
                log.warning("provider close failed")
            if self.tools is not None:
                await self.tools.close()
            await self._reap()
            await self._tidy()
        if self.stop_reason == "revoked":
            reason = "revoked"
        if reason == "completed" and self.options.kind == CODING_KIND:
            await self._handoff_check(final=True)
            if (self.tests or {}).get("reason") == testrun.NO_CHANGES and self.result is not None \
                    and self.result.get("tests_claim") is None:
                self.result = {**self.result, "tests_claim": {
                    "command": " ".join(self.tests["command"].split()), "status": "not_run"}}
        if reason == "completed" and self.handoff is not None:
            try:
                gitdiff.verify_repository(self.options.cwd)
                handoff = await asyncio.to_thread(self.handoff, self.result)
            except Exception as exc:
                handoff = {"handoff": "failed", "error": type(exc).__name__}
                if isinstance(exc, handoff_mod.RateLimited):
                    handoff["step"] = exc.step
                    log.error("handoff failed: %s", exc)
                elif isinstance(exc, (gitdiff.GitError, handoff_mod.ForgeError)):
                    handoff["step"] = clip(str(exc), STEP_CAP)
                    log.error("handoff failed: %s (%s)", type(exc).__name__, handoff["step"])
                else:
                    log.error("handoff failed: %s", type(exc).__name__)
            if self.tests is not None and handoff.get("handoff") == "pull request":
                handoff["tests"] = dict(self.tests)
        if self.research is not None and reason != "revoked":
            self.result = await self._research_end(reason)
        outcome = Outcome(reason, self.turns, self.cost, self.result, handoff)
        await self._reap()
        if self.swept.ended and reason != "revoked":
            try:
                await self.poster.post(Event("message", {
                    "role": "user", "text": clip(self.swept.sentence(), PROMPT_PREVIEW), "mode": "queue",
                    "entry_id": None, "author": RUNNER_AUTHOR, "purpose": "cleanup",
                    "ended": self.swept.ended, "names": dict(self.swept.names), "killed": self.swept.killed,
                    "left": self.swept.left}))
            except Exception as exc:
                log.warning("cleanup event not posted: %s", type(exc).__name__)
        if reason != "revoked":
            try:
                await self.poster.post(Event("end", self._end_payload(outcome)))
            except Exception as exc:
                log.error("end event not posted: %s", type(exc).__name__)
        return outcome

    def _end_payload(self, outcome: Outcome) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": outcome.reason, "turns": outcome.turns, "cost_usd": outcome.cost_usd,
                                   "last_turn": self.last_turn or None, "result": outcome.result,
                                   "detail": self.detail}
        if self.options.driven and outcome.reason == "completed":
            payload["ended_by"] = self.ended_by
            payload["limit"] = self.capped
        if outcome.reason == "completed":
            payload["wrapped_up"] = self.wrapped_up
            if (self.tests or {}).get("reason") == testrun.NO_CHANGES:
                payload["tests_skipped"] = testrun.NO_CHANGES
        if self.checks is not None:
            payload["checks"] = self.checks
        if outcome.handoff:
            payload["handoff"] = outcome.handoff
            payload["pr_url"] = outcome.handoff.get("pr_url")
        return {k: v for k, v in payload.items() if v is not None}

    async def _drive(self) -> str:
        while not self._halted():
            if self._turns_used():
                await self._turns_out()
                break
            if not self.urgent and not self.held:
                await self._pull_once()
                if not self.urgent and not self.held and self.timing.linger > 0:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=self.timing.linger)
                    except TimeoutError:
                        pass
                if not self.urgent and not self.held and not await self._await_input():
                    break
                if self._halted():
                    break
                if self._turns_used():
                    await self._turns_out()
                    break
            if not self.options.driven and not self.provider.enforces_max_turns and self.caps.max_turns \
                    and self.turns >= self.caps.max_turns:
                await self._limit("max_turns", turns=self.turns)
                break
            if not self.urgent and not self.held:
                continue
            prompt = self.urgent.popleft() if self.urgent else self.held.popleft()
            if prompt.id:
                self._handed.add(prompt.id)
            await self._turn(prompt)
            spent = self._spent()
            if not self._halted() and self.caps.max_budget_usd is not None and spent is not None \
                    and spent >= self.caps.max_budget_usd:
                detail = {"spent_usd": spent} if self.resuming else {}
                await self._limit("budget", cost_usd=self.cost, max_budget_usd=self.caps.max_budget_usd, **detail)
        return self.stop_reason or "completed"

    def _halted(self) -> bool:
        return bool(self.stop_reason or self.wrapped_up)

    def _spent(self) -> float | None:
        if self.cost is None:
            return None
        before = self.options.cost_before
        if not self.resuming or self.fresh or before is None or not getattr(self.provider, "cost_carries_over", False):
            return self.cost
        return self.cost - before if self.cost >= before else self.cost

    async def _start_resumed(self, ctx: ProviderContext, brief: str) -> None:
        if ctx.resume:
            try:
                await self.provider.start(ctx)
            except Exception as exc:
                refused = getattr(self.provider, "resume_refused", None)
                if refused is None or not refused(exc):
                    raise
                log.warning("the provider holds no conversation for the stored id; starting a new one")
                try:
                    await self.provider.close()
                except Exception as closing:
                    log.warning("closing the refused resume failed: %s", type(closing).__name__)
                ctx.resume = None
                await self.provider.start(ctx)
                self.fresh = True
        else:
            await self.provider.start(ctx)
            self.fresh = True
        await self._conversation_seen()
        text = self.options.resume_text or ""
        if self.fresh:
            await self.poster.post(Event("message", {"role": "runner", "purpose": RESUMED_FRESH_PURPOSE}))
            carry = self.options.carry_over or RESUMED_WITHOUT.format(brief=brief.strip())
            prompt = f"{carry.rstrip()}\n\n---\n\n{text}"
        else:
            self._note = None
            prompt = text
        lines = [line for line in (self.options.base_fetch, self.options.merge_note) if line]
        if lines:
            prompt = f"{prompt.rstrip()}\n\n---\n\n" + "\n\n".join(lines)
        self._first = Prompt(0, prompt, "queue", "launcher")
        self._first_shown = text
        self.held.append(self._first)

    async def _baseline_said(self) -> None:
        if self.options.kind != CODING_KIND:
            return
        found = handoff_mod.instructions_file(self.provider.name, self.options.cwd)
        project = (found[0], len(found[1].encode("utf-8")), found[2]) if found is not None else None
        how = baseline_mod.applied(self.provider.name, self.ctx is not None and self.ctx.baseline is not None,
                                   self.resuming and not self.fresh)
        await self.poster.post(Event("message", baseline_mod.event(how, project)))

    async def _conversation_seen(self) -> None:
        if self.options.kind != CODING_KIND:
            return
        get = getattr(self.provider, "conversation_id", None)
        found = get() if callable(get) else None
        if not found or found == self._posted_id:
            return
        self._posted_id = found
        if not usable_id(self.provider.name, found):
            log.warning("the provider's conversation id is not one the control plane stores; not posted")
            return
        await self.poster.post(Event("message", {"role": "runner", "purpose": PROVIDER_SESSION_PURPOSE,
                                                 "id": found}))

    async def _research_end(self, reason: str) -> dict[str, Any]:
        limits = list(self.limits_hit)
        cap = self.options.max_searches
        if cap and self.searches >= cap:
            limits.append("max_searches")
        note = self._unfinished_note(reason)
        try:
            partial, why = await asyncio.to_thread(self.research.post_outputs, reason, self.detail, note)
        except Exception as exc:
            log.error("research outputs not posted: %s", type(exc).__name__)
            partial, why = True, f"The run's files could not be posted to the control plane ({type(exc).__name__})."
        return self.research.result(self.last_text, limits, partial, why)

    def _track_subagent(self, payload: dict[str, Any]) -> None:
        key = payload.get("tool_use_id")
        phase = payload.get("phase")
        if not key or phase == "progress":
            return
        if phase in ("requested", "started"):
            if subagent_start(payload) is not None and key not in self._reported:
                self._running.add(str(key))
        else:
            self._running.discard(str(key))
            self._reported.add(str(key))

    def _note_turn_end(self) -> None:
        if self._running:
            self._left_running = max(self._left_running, len(self._running))
            log.warning("research turn ended with %d subagent(s) still running", len(self._running))

    async def _research_continue(self) -> str:
        self._note_turn_end()
        follow = getattr(self.provider, "follow", None)
        while not self.stop_reason and self._left_running:
            missing = await asyncio.to_thread(self.research.missing)
            if not missing:
                break
            if self.caps.max_turns and self.turns >= self.caps.max_turns:
                self._no_more = "turns"
                break
            if self._running and follow is not None and self._follows <= len(self._reported):
                self._follows += 1
                await self._follow_turn(missing)
            elif self._continuations < CONTINUATIONS:
                self._continuations += 1
                await self._turn(Prompt(0, self._continue_prompt(missing), "queue", RUNNER_AUTHOR))
            else:
                self._no_more = "continuations"
                break
            self._note_turn_end()
        return self.stop_reason or "completed"

    def _continue_prompt(self, missing: list[str]) -> str:
        n = len(self._running)
        running = (f"{n} subagent{'s' if n != 1 else ''} you started {'have' if n != 1 else 'has'} not reported. "
                   if n else "")
        return CONTINUE_PROMPT.format(files=_and(missing), running=running)

    async def _follow_turn(self, missing: list[str]) -> None:
        n = len(self._running)
        await self.poster.post(Event("message", {
            "role": "user", "mode": "queue", "entry_id": None, "author": RUNNER_AUTHOR, "purpose": CONTINUE_PURPOSE,
            "text": FOLLOW_WORDS.format(n=n, s="s" if n != 1 else "", were="were" if n != 1 else "was",
                                        files=_and(missing))}))
        ends = self._turn_ends
        self._seen_at = asyncio.get_running_loop().time()
        self.turn_task = asyncio.create_task(self._consume(None))
        watch = asyncio.create_task(self._follow_watch(self.turn_task))
        try:
            await self.turn_task
        except asyncio.CancelledError:
            if not self.turn_task.cancelled():
                raise
            log.info("stopped waiting for the subagents' reports")
        finally:
            self.turn_task = None
            watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)
        if self._turn_ends > ends:
            self.turns += 1

    async def _follow_watch(self, task: asyncio.Task) -> None:
        loop = asyncio.get_running_loop()
        step = max(min(self.timing.follow_quiet, self.timing.follow_idle) / 4, 0.01)
        while not task.done():
            await asyncio.sleep(step)
            idle = loop.time() - self._seen_at
            if (not self._running and idle >= self.timing.follow_quiet) or idle >= self.timing.follow_idle:
                log.warning("no turn on the subagents' reports after %.0f s; waiting no longer", idle)
                task.cancel()
                return

    def _unfinished_note(self, reason: str) -> str | None:
        n = self._left_running
        if not n:
            return None
        words = [f"Its turn ended while {n} subagent{'s' if n != 1 else ''} it had started "
                 f"{'were' if n != 1 else 'was'} still running."]
        given = self._follows + self._continuations
        if given:
            words.append(f"The runner waited for their reports and continued the session {given} "
                         f"time{'s' if given != 1 else ''}.")
        if reason == "completed" and self._no_more == "continuations":
            words.append("The continuations ran out before the session wrote its files.")
        elif reason == "completed" and self._no_more == "turns":
            words.append("Its turns ran out before the runner could continue it.")
        if self._running:
            k = len(self._running)
            words.append(f"{k} subagent{'s' if k != 1 else ''} had still not reported when it ended.")
        return " ".join(words)

    def _count_subagent(self, payload: dict[str, Any]) -> None:
        cap = self.options.max_subagents
        key = subagent_start(payload)
        if cap is None or key is None or key in self._subagents:
            return
        self._subagents.add(key)
        if len(self._subagents) > cap and self._cap_task is None:
            log.warning("subagent %d started past the cap of %d; stopping the run", len(self._subagents), cap)
            if self.stop_reason is None:
                self.stop_reason = "max_subagents"
            self._cap_task = asyncio.create_task(self._subagents_out(cap))

    def _shell_seen(self) -> None:
        if self._shell_task is not None:
            return
        log.error("a command ran in a research session whose shell must be off; stopping the run")
        self.stop_reason = "shell_used"
        self._shell_task = asyncio.create_task(self._shell_out())

    async def _shell_out(self) -> None:
        await self._limit("shell_used")
        await self._stop_turn("A command ran in a research session whose shell must be off.")

    async def _subagents_out(self, cap: int) -> None:
        await self._limit("max_subagents", subagents=len(self._subagents), max_subagents=cap)
        await self._stop_turn(f"The run started more subagents than its cap of {cap}.")

    async def _limit(self, which: str, **detail: Any) -> None:
        self.limits_hit.append(which)
        if which in WRAP_UP_LIMITS and self._wraps_up(which):
            self._wrap_asked(which)
        elif self.stop_reason is None:
            self.stop_reason = which
        self.urgent.clear()
        self.held.clear()
        self._wake.set()
        await self.poster.post(Event("limit", {"limit": which, "source": "runner", **detail}))

    def _wraps_up(self, why: str | None = None) -> bool:
        return self.options.kind == CODING_KIND and (not self.options.driven or
                                                     why == "wall_clock" or
                                                     bool(self.watch and self.watch.wrap_requested))

    def _wrap_asked(self, why: str) -> bool:
        if self.wrapped_up is not None or self.stop_reason or not self._wraps_up(why):
            return False
        log.info("wrapping the session up (%s)", why)
        self.wrapped_up = why
        dropped = len(self.urgent) + len(self.held)
        self.urgent.clear()
        self.held.clear()
        if dropped:
            log.info("%d waiting prompt(s) not sent: the session is wrapping up", dropped)
        self._wake.set()
        return True

    # A bounded coding session that reaches a cap, or is asked to finish, still ends with a typed result:
    # the running turn is interrupted and one runner-written closing turn asks for it.
    async def _wrap_up(self) -> str:
        if self._wrap_stop is not None:
            await asyncio.gather(self._wrap_stop, return_exceptions=True)
        self.tests = None
        why = self.wrapped_up or "asked"
        if self._terminated:
            log.warning("the provider was terminated; handing off without a closing turn")
            return self.stop_reason or "completed"
        self._wrapping = True
        try:
            await self._within(self.caps.wrap_up, self._turn(
                Prompt(0, wrap_up_prompt(why, self.caps.wrap_up), "queue", RUNNER_AUTHOR)))
        finally:
            self._wrapping = False
        if self.stop_reason:
            return self.stop_reason
        await self._test_before_handoff(wrapping=True)
        return self.stop_reason or "completed"

    async def _within(self, seconds: float, work: Any,
                      timed_out: str = "The session's wrap-up ran out of time.") -> None:
        fired = False

        async def overrun() -> None:
            nonlocal fired
            await asyncio.sleep(seconds)
            if self.stop_reason or self.turn_task is None or self.turn_task.done():
                return
            fired = True
            log.warning("the closing turn is still running after %g s; stopping it", seconds)
            await self._stop_turn(timed_out, close=False, turn=self.turn_task)

        watch = asyncio.create_task(overrun())
        try:
            await work
        finally:
            if not fired:
                watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)

    async def _await_input(self) -> bool:
        if not self.options.driven or self.ended_by or self.capped:
            return False
        loop = asyncio.get_running_loop()
        # The idle clock runs before the waiting event goes out: a keep_open pressed on seeing it can be pulled
        # while this post is still in flight, and it must find the session waiting.
        self._idle_deadline, self._extensions = loop.time() + self.options.idle_seconds, 0
        try:
            await self.poster.post(Event("turn", {"phase": "waiting"}))
            while True:
                if self.urgent or self.held:
                    return True
                if self.stop_reason or self.ended_by:
                    return False
                remaining = self._idle_deadline - loop.time()
                if remaining <= 0:
                    log.info("no message for %g s (%d extension(s)); finishing the session",
                             self.options.idle_seconds, self._extensions)
                    self.ended_by = "idle"
                    return False
                self._wake.clear()
                if self.urgent or self.held or self.stop_reason or self.ended_by:
                    continue
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                except TimeoutError:
                    pass
        finally:
            self._idle_deadline = None

    def _turns_used(self) -> bool:
        return bool(self.options.driven and self.caps.max_turns and self.turns >= self.caps.max_turns)

    async def _turns_out(self) -> None:
        if self.capped:
            return
        self.capped = "max_turns"
        dropped = len(self.urgent) + len(self.held)
        self.urgent.clear()
        self.held.clear()
        if dropped:
            log.info("max turns reached; %d waiting prompt(s) not sent", dropped)
        await self.poster.post(Event("limit", {"limit": "max_turns", "source": "runner", "turns": self.turns}))

    def _needs_closing(self) -> bool:
        if not self.options.driven or self.turns == 0 or self.turns == self._summarised:
            return False
        return self.turns > 1 or self._schema_skipped

    async def _closing_turn(self) -> None:
        self._summarised = self.turns
        self._closing = True
        try:
            await self._turn(Prompt(0, handoff_mod.SUMMARY_PROMPT, "queue", RUNNER_AUTHOR))
        finally:
            self._closing = False

    async def _handoff_check(self, repair: bool = False, final: bool = False) -> None:
        try:
            status = await asyncio.to_thread(handoff_mod.worktree_status, self.options.cwd, self.options.base_ref)
            ignored = set(placeholders.ignorable(self.options.cwd, self.existed))
            status = handoff_check.Status(tuple(p for p in status.changed if p not in ignored),
                                          tuple(p for p in status.untracked if p not in ignored), status.base_tracked)
        except (gitdiff.GitError, OSError):
            log.warning("hand-off check: worktree status unavailable")
            status = handoff_check.Status()
        brief = self._latest_request if handoff_check.numbered_items(self._latest_request or "") else self._brief
        self.checks = handoff_check.check(brief, self.result, status,
                                         tests=self.tests if final else None, test_cmd=self.options.test_cmd)
        if repair and not self._repaired and not self.wrapped_up and not self.stop_reason and not self._terminated \
                and handoff_check.result_findings(brief, self.result):
            self._repaired, self._repairing = True, True
            try:
                await self._within(handoff_check.REPAIR_SECONDS, self._turn(Prompt(
                    0, handoff_check.repair_prompt(brief, self.result), "queue", RUNNER_AUTHOR)),
                    timed_out=handoff_check.REPAIR_TIMEOUT)
            finally:
                self._repairing = False
            await self._handoff_check()

    def _set_schema(self) -> None:
        if not self.per_turn_schema or self.ctx is None or self._testing:
            return
        self._schema_on = self._closing or self._wrapping or self._repairing
        self.ctx.output_schema = self.options.output_schema if self._schema_on else None
        if not self._schema_on:
            self._schema_skipped = True

    async def _turn_clock(self) -> None:
        await asyncio.sleep(max(self.caps.wall_clock - self._worked, 0.0))
        if self.stop_reason or self.wrapped_up:
            return
        self._clock_fired = True
        running = self.turn_task
        await self._limit("wall_clock", seconds=self.caps.wall_clock)
        if self.wrapped_up == "wall_clock":
            await self._stop_turn("The session's wall clock ran out; it is wrapping up.", close=False, turn=running)
        else:
            await self._stop_turn("The session's wall clock ran out.")

    async def _test_before_handoff(self, *, wrapping: bool = False) -> None:
        command = self.options.test_cmd
        if not command or self._terminated:
            return
        unchanged = False
        if self.options.kind == CODING_KIND and self._test_base:
            try:
                unchanged = not await asyncio.to_thread(handoff_mod.has_changes, self.options.cwd, self._test_base)
            except (gitdiff.GitError, OSError):
                log.warning("test check: worktree status unavailable")
        if self.stop_reason or self._terminated or (self.wrapped_up and not wrapping):
            return
        if unchanged:
            self.tests = {"command": command, "status": "not_run", "exit_code": None, "reason": testrun.NO_CHANGES}
            await self._post_test_verdict(self.tests)
        elif wrapping:
            await self._within(self.caps.wrap_up, self._test_turn(command))
        else:
            await self._test_turn(command)

    # The test turn's own result never replaces the work's: the typed result, the PR body and the end
    # reason stay those of the work (docs/design.md, "How a session runs").
    async def _test_turn(self, command: str) -> None:
        self._testing, self._test_raws = True, []
        try:
            before, after = await self._turn(Prompt(0, testrun.prompt(command), "queue", RUNNER_AUTHOR))
        finally:
            self._testing = False
        denied = {rid for rid, decision in self.relay.decisions if not decision.allow}
        self.tests = testrun.verdict(self.provider.name, self._test_raws, command, denied, before, after,
                                     self.options.cwd,
                                     claude_project_cwd=getattr(self.provider, "bash_starts_in_worktree", False))
        self._test_raws = []
        await self._post_test_verdict(self.tests)

    async def _post_test_verdict(self, tests: dict[str, Any]) -> None:
        await self.poster.post(Event("message", {
            "role": "user", "text": clip(testrun.sentence(tests), PROMPT_PREVIEW), "mode": "queue",
            "entry_id": None, "author": RUNNER_AUTHOR, "purpose": "tests-verdict",
            **({"reason": tests["reason"]} if "reason" in tests else {})}))

    async def _turn(self, prompt: Prompt) -> tuple[str | None, str | None]:
        try:
            async with self._settings_lock:
                self._in_turn = True
                if self._held_settings:
                    held, self._held_settings = self._held_settings, {}
                    control_id, self._held_control = self._held_control, None
                    await self._apply_settings(held, control_id, {})
                self._turn_settings = self.provider.settings_now() if self._settings_shown else None
                self._turn_thinking = self._thinking_now()
            return await self._turn_body(prompt)
        finally:
            self._in_turn = False

    def _thinking_now(self) -> bool:
        try:
            return self.provider.settings_now().get("thinking") == "on"
        except Exception:
            return False

    async def _turn_body(self, prompt: Prompt) -> tuple[str | None, str | None]:
        if self.watch is not None:
            self.watch.begin()
        before = await self._snapshot()
        resumed = prompt is self._first and self._first_shown is not None
        if prompt.author != RUNNER_AUTHOR:
            self._latest_request = self._first_shown if resumed else prompt.text
        message = {"role": "user", "text": clip(self._first_shown if resumed else prompt.text, PROMPT_PREVIEW),
                   "mode": prompt.mode, "entry_id": prompt.id or None, "author": prompt.author}
        if self._testing:
            message["purpose"] = "tests"
        elif self._repairing:
            message["purpose"] = handoff_check.REPAIR_PURPOSE
        elif self._closing:
            message["purpose"] = "summary"
        elif self._wrapping:
            message["purpose"], message["why"] = WRAP_UP_PURPOSE, self.wrapped_up
        elif resumed:
            message["purpose"] = RESUME_PURPOSE
        await self.poster.post(Event("message", message))
        self.last_text = ""
        self._set_schema()
        text = handoff_mod.with_note(prompt.text, self._note) if prompt is self._first else prompt.text
        self.turn_task = asyncio.create_task(self._consume(text))
        loop = asyncio.get_running_loop()
        began = loop.time()
        if self.options.driven and not self.wrapped_up:
            self._clock_fired = False
            self._clock = asyncio.create_task(self._turn_clock())
        try:
            await self.turn_task
        except asyncio.CancelledError:
            if not self.turn_task.cancelled():
                raise
            log.warning("turn cancelled after the grace period")
        finally:
            self.turn_task = None
            if self._clock is not None:
                clock, self._clock = self._clock, None
                if not self._clock_fired:
                    clock.cancel()
                await asyncio.gather(clock, return_exceptions=True)
            self._worked += loop.time() - began
        if not self._testing and not self._closing and not self._wrapping and not self._repairing:
            self.turns += 1
        after = await self._snapshot()
        if before and self._diff_base is None:
            self._diff_base = before
        if before and after:
            try:
                payload = await asyncio.to_thread(gitdiff.diff, self.options.cwd, before, after)
            except gitdiff.GitError as exc:
                log.warning("turn diff failed: %s", exc)
                payload = None
            if payload:
                await self.poster.post(Event("diff", payload))
                if self.options.file_diffs and self._diff_base:
                    await self._session_diff(after)
        return before, after

    async def _session_diff(self, after: str) -> None:
        try:
            files, summary = await asyncio.to_thread(gitdiff.session_diff, self.options.cwd, self._diff_base,
                                                     after, self._files_sent, self._file_sizes)
        except gitdiff.GitError as exc:
            log.warning("session diff failed: %s", exc)
            return
        for payload in files:
            await self.poster.post(Event("diff", payload))
        await self.poster.post(Event("diff", summary))

    async def _reap(self) -> None:
        try:
            self.swept.add(await asyncio.to_thread(self.reaper.sweep))
        except Exception as exc:
            log.warning("leftover processes not ended: %s", type(exc).__name__)

    async def _tidy(self) -> None:
        if self.existed is None:
            return
        try:
            if await asyncio.to_thread(placeholders.settle, self.options.cwd):
                await asyncio.to_thread(placeholders.sweep, self.options.cwd, self.existed)
            else:
                log.warning("a process is still running in the worktree; sandbox placeholders left on disk")
        except Exception as exc:
            log.warning("placeholder sweep failed: %s", type(exc).__name__)

    def _tree(self) -> str:
        with gitdiff.checked(self.options.cwd):
            return gitdiff.snapshot(self.options.cwd, placeholders.ignorable(self.options.cwd, self.existed))

    async def _snapshot(self) -> str | None:
        if not self.options.diff:
            return None
        try:
            return await asyncio.to_thread(self._tree)
        except gitdiff.GitError as exc:
            log.warning("worktree snapshot failed: %s", exc)
            return None

    async def _consume(self, text: str | None) -> None:
        stream = self.provider.turn(text) if text is not None else self.provider.follow()
        loop = asyncio.get_running_loop()
        async for raw in stream:
            self._seen_at = loop.time()
            await self._watch_notices()
            if self._testing:
                self._test_raws.append(raw)
            if self.drafter is not None:
                self.drafter.feed(raw)
            await self._conversation_seen()
            for event in self.provider.normalise(raw):
                if event.kind == "thinking" and not self._turn_thinking:
                    continue
                if event.kind == "turn" and event.payload.get("phase") == "started":
                    shown = {k: v for k, v in (self._turn_settings or {}).items()
                             if k in settings_mod.TURN_FIELDS and v is not None}
                    if self._turn_thinking:
                        shown["thinking"] = "on"
                    if shown:
                        event = Event("turn", {**event.payload, **shown})
                self._observe(event)
                await self.poster.post(event)

    def _observe(self, event: Event) -> None:
        p = event.payload
        if self._testing:
            if event.kind == "usage" and isinstance(p.get("total_cost_usd"), (int, float)):
                self.cost = float(p["total_cost_usd"])
            return
        if event.kind == "message" and p.get("role") == "assistant" and not p.get("parent_tool_use_id"):
            later = isinstance(p.get("part"), int) and p["part"] > 1
            self.last_text = (self.last_text if later else "") + p.get("text", "")
        elif event.kind == "turn" and p.get("phase") == "completed":
            self._turn_ends += 1
            self.last_turn ={k: p[k] for k in ("subtype", "status", "terminal_reason", "interrupted") if k in p}
            if isinstance(p.get("structured_output"), dict):
                self.result = p["structured_output"]
            elif self.provider.name == "codex" and self.last_text and self._schema_on:
                try:
                    parsed = json.loads(self.last_text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    self.result = tidy_result(parsed)
        elif event.kind == "usage" and isinstance(p.get("total_cost_usd"), (int, float)):
            self.cost = float(p["total_cost_usd"])
        elif event.kind == "limit" and p.get("source") == "provider" and p.get("limit") in ("max_turns", "budget"):
            if self._wrapping:
                return
            self.limits_hit.append(p["limit"])
            if self._wraps_up():
                self._wrap_asked(p["limit"])
            elif self.stop_reason is None:
                self.stop_reason = p["limit"]
            self.urgent.clear()
            self.held.clear()
        elif event.kind == "subagent":
            self._count_subagent(p)
            if self.options.kind == RESEARCH_KIND:
                self._track_subagent(p)
        elif event.kind == "tool" and self.options.kind == RESEARCH_KIND and p.get("name") in SHELL_ITEMS:
            self._shell_seen()
        elif event.kind == "tool" and p.get("phase") == "started" and p.get("name") == SEARCH_TOOL:
            self.searches += 1

    async def _pull_loop(self) -> None:
        while True:
            await asyncio.sleep(self.timing.pull_interval)
            try:
                await self._pull_once()
            except SessionEnded:
                log.warning("the control plane refused the runner token; stopping")
                self.stop_reason = "revoked"
                self._wake.set()
                await self._stop_turn("The session's token was revoked.")
                return
            except Exception as exc:
                log.warning("pull refused: %s", type(exc).__name__)

    async def _pull_once(self) -> None:
        async with self._pull_lock:
            try:
                got = await pull_full(self.client)
            except Unreachable as exc:
                log.warning("pull failed (%s); will retry", exc)
                return
        if got.title is not None:
            self.title = got.title
        for answer in got.answers:
            self.relay.deliver(answer)
        for prompt in got.prompts:
            await self._incoming(prompt)
        for control in got.controls:
            await self._control(control)

    async def _control(self, control: Control) -> None:
        if control.kind == "withdraw":
            await self._withdraw(control)
            return
        if control.kind == "settings":
            await self._settings(control)
            return
        if control.kind == "keep_open":
            await self._keep_open(control)
            return
        if not self.options.driven:
            if control.kind == "finish" and self._wraps_up():
                if self._wrap_asked("asked"):
                    log.info("control %d: finish asked; wrapping up", control.id)
                    self._wrap_stop = asyncio.create_task(
                        self._stop_turn("The session's owner asked for it to be wrapped up.", close=False,
                                        turn=self.turn_task))
                else:
                    log.info("control %d: finish asked; the session is already wrapping up or stopping", control.id)
                return
            log.info("control %d (%s) ignored: this session is bounded", control.id, control.kind)
            return
        if control.kind == "finish":
            if self.ended_by is None:
                self.ended_by = "finish"
            log.info("control %d: finish asked", control.id)
        elif self.turn_task is not None and not self.turn_task.done():
            log.info("control %d: interrupting the running turn", control.id)
            self.relay.deny_all("Interrupted by the session's owner.")
            await self.provider.interrupt()
        else:
            log.info("control %d: interrupt with no turn running; nothing to do", control.id)
        self._wake.set()

    async def _refusal_note(self, payload: dict[str, Any]) -> None:
        await self.poster.post(Event("message", {"role": "runner", "purpose": refusals.PURPOSE, **payload}))
        if self.watch is not None and self._watch_working():
            self.watch.repeated("refused", getattr(payload, "command", payload.get("command")))
            await self._watch_notices()

    def _watch_working(self) -> bool:
        return not (self._halted() or self._testing or self._closing or self._wrapping or self._repairing)

    def _raw_frame(self, raw: dict[str, Any]) -> None:
        if self.watch is not None:
            self.watch.frame(raw, count=self._watch_working())

    async def _watch_loop(self) -> None:
        while True:
            await asyncio.sleep(watch_mod.WATCH_INTERVAL)
            if self.watch is not None and self._watch_working() and self.turn_task is not None \
                    and not self.turn_task.done():
                self.watch.tick()
            await self._watch_notices()

    async def _watch_notices(self) -> None:
        if self.watch is None:
            return
        for event, note in self.watch.drain():
            await self.poster.post(event)
            if note is not None and self._watch_working():
                await self._incoming(Prompt(0, note, "steer", RUNNER_AUTHOR))
            elif event.payload.get("action") == "wrapping_up" and self._wrap_asked("looping"):
                if self.turn_task is not None and not self.turn_task.done():
                    self._wrap_stop = asyncio.create_task(self._stop_turn(
                        watch_mod.LOOP_WRAP_REASON, close=False, turn=self.turn_task))

    async def _keep_open(self, control: Control) -> None:
        loop = asyncio.get_running_loop()
        step = None
        if self.stop_reason or self.ended_by or self.capped:
            step = "closed"
        elif self._idle_deadline is None:
            step = "not_waiting"
        elif self._idle_deadline - loop.time() <= 0:
            step = "closed"
        payload: dict[str, Any] = {"role": "runner", "purpose": KEEP_OPEN_PURPOSE, "control_id": control.id or None}
        if step is not None:
            log.info("control %d: keep_open refused (%s)", control.id, step)
            payload.update(outcome="refused", step=step, text=KEEP_OPEN_WORDS[step])
        else:
            self._idle_deadline += self.options.idle_seconds
            self._extensions += 1
            left = max(int(self._idle_deadline - loop.time()), 0)
            log.info("control %d: keep_open; the idle limit is now %d s away (extension %d)", control.id, left,
                     self._extensions)
            payload.update(outcome="extended", idle_left=left, extensions=self._extensions,
                           idle_seconds=int(self.options.idle_seconds),
                           text=KEEP_OPEN_WORDS["extended"].format(span=_span(self.options.idle_seconds),
                                                                   left=_span(left)))
        await self.poster.post(Event("message", payload))
        self._wake.set()

    async def _withdraw(self, control: Control) -> None:
        entry = control.entry_id
        removed = 0
        for waiting in (self.held, self.urgent):
            for old in [p for p in waiting if entry and p.id == entry]:
                waiting.remove(old)
                removed += 1
        outcome = "withdrawn" if removed else "too_late" if entry in self._handed else "unknown"
        log.info("control %d: withdraw prompt %s: %s", control.id, entry, outcome)
        await self.poster.post(Event("message", {
            "role": "runner", "purpose": "withdraw", "text": WITHDRAW_WORDS[outcome], "outcome": outcome,
            "entry_id": entry, "control_id": control.id or None}))
        self._wake.set()

    async def _settings(self, control: Control) -> None:
        asked = control.settings or {}
        good: dict[str, str] = {}
        refused: dict[str, str] = {}
        for name in settings_mod.FIELDS:
            if name in asked:
                reason = settings_mod.check(name, asked[name])
                if reason is None:
                    good[name] = asked[name]
                else:
                    refused[name] = reason
        async with self._settings_lock:
            if self._in_turn:
                self._held_settings.update(good)
                if good:
                    self._held_control = control.id or None
                log.info("control %d: settings %s held for the next turn", control.id, sorted(good))
                await self._settings_event(control.id or None, {}, good, refused)
            else:
                await self._apply_settings(good, control.id or None, refused)

    async def _start_in_mode(self, word: str) -> None:
        self._settings_shown = True
        reason = settings_mod.check("mode", word)
        if reason is not None:
            await self._settings_event(None, {}, {}, {"mode": reason}, start={"mode": str(word)[:40]})
            return
        await self._apply_settings({"mode": word}, None, {}, start=True)

    async def _apply_settings(self, changes: dict[str, str], control_id: int | None,
                              refused: dict[str, str], start: bool = False) -> None:
        done: dict[str, str] = {}
        refused = dict(refused)
        if changes:
            try:
                results = await self.provider.change_settings(dict(changes))
            except Exception as exc:
                log.warning("settings change failed: %s", type(exc).__name__)
                results = {}
                fault = f"the session could not change it ({type(exc).__name__})"
            else:
                fault = "the session could not change it"
            for name in changes:
                result = results.get(name) or Change(False, reason=fault)
                if result.applied:
                    done[name] = result.value or changes[name]
                else:
                    refused[name] = result.reason or fault
        if done:
            self._settings_shown = True
        log.info("settings applied %s, refused %s", sorted(done), sorted(refused))
        await self._settings_event(control_id, done, {}, refused, start=dict(changes) if start else None)

    async def _settings_event(self, control_id: int | None, done: dict[str, str], held: dict[str, str],
                              refused: dict[str, str], start: dict[str, str] | None = None) -> None:
        order = settings_mod.FIELDS
        done, held, refused = ({k: d[k] for k in order if k in d} for d in (done, held, refused))
        said = []
        if done:
            lead = "Started with: " if start is not None else "From the next turn: "
            said.append(lead + ", ".join(f"{k} {v}" for k, v in done.items()) + ".")
        if held:
            said.append("Held until the running turn ends, then applied: "
                        + ", ".join(f"{k} {v}" for k, v in held.items()) + ".")
        if refused:
            said.append("Not changed: " + "; ".join(f"{k} ({v})" for k, v in refused.items()) + ".")
        payload: dict[str, Any] = {
            "role": "runner", "purpose": "settings", "text": " ".join(said) or "Nothing to change.",
            "control_id": control_id, "applied": done, "held": held, "refused": refused}
        if start is not None:
            try:
                now = self.provider.settings_now().get("mode")
            except Exception:
                now = None
            payload.update(start=True, asked=dict(start), **({"mode": now} if isinstance(now, str) else {}))
        await self.poster.post(Event("message", payload))

    async def _incoming(self, prompt: Prompt) -> None:
        if self.stop_reason:
            log.info("prompt %d arrived after the session stopped; not sent", prompt.id)
            return
        if self.capped:
            log.info("prompt %d arrived after the session used its turns; not sent", prompt.id)
            return
        if self.wrapped_up:
            log.info("prompt %d arrived while the session is wrapping up; not sent", prompt.id)
            return
        if prompt.replaces is not None:
            withdrawn = 0
            for waiting in (self.held, self.urgent):
                for old in [p for p in waiting if p.id == prompt.replaces]:
                    waiting.remove(old)
                    withdrawn += 1
            if withdrawn:
                log.info("prompt %d replaces prompt %d, withdrawn before it was sent", prompt.id, prompt.replaces)
        running = self.turn_task is not None and not self.turn_task.done()
        if not running or prompt.mode == "queue":
            self.held.append(prompt)
        elif prompt.mode == "now":
            self.urgent.append(prompt)
            self.relay.deny_all("Interrupted by the session's owner.")
            await self.provider.interrupt()
        elif prompt.mode == "steer":
            if await self.provider.steer(prompt.text):
                if prompt.id:
                    self._handed.add(prompt.id)
                await self.poster.post(Event("message", {"role": "user", "text": clip(prompt.text, PROMPT_PREVIEW),
                                                         "mode": "steer", "entry_id": prompt.id or None,
                                                         "author": prompt.author, "delivered": "next step"}))
            else:
                log.info("steer not accepted; holding prompt %d for after the turn", prompt.id)
                self.held.append(prompt)
        self._wake.set()

    # Caps are never extended: interrupt, a grace period, then terminate.
    async def _wall_clock(self) -> None:
        await asyncio.sleep(self.caps.wall_clock)
        if self.stop_reason or self.wrapped_up:
            return
        running = self.turn_task
        await self._limit("wall_clock", seconds=self.caps.wall_clock)
        if self.wrapped_up == "wall_clock":
            await self._stop_turn("The session's wall clock ran out; it is wrapping up.", close=False, turn=running)
        else:
            await self._stop_turn("The session's wall clock ran out.")

    async def _stop_turn(self, reason: str, close: bool = True, turn: Any = _RUNNING) -> None:
        task = self.turn_task if turn is _RUNNING else turn
        if close:
            self.relay.deny_all(reason, close=True)
        if task is None or task.done():
            return
        if not close:
            self.relay.deny_all(reason)
        try:
            await self.provider.interrupt()
        except Exception:
            log.warning("interrupt failed")
        done, _ = await asyncio.wait({task}, timeout=self.caps.grace)
        if not done:
            log.warning("turn still running after the grace period; terminating")
            self._terminated = True
            try:
                await self.provider.terminate()
            finally:
                task.cancel()

