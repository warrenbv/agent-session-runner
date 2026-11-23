"""The platform baseline: its pinned text, how each provider is given it, the `baseline` event, and the
--no-baseline switch."""

from __future__ import annotations

import asyncio
import hashlib
import types

import pytest

from conftest import fast_timing
from runner import __main__ as main_mod
from runner import baseline
from runner.providers.base import RESEARCH_KIND, RESULT_SCHEMA, ProviderContext
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session

VERSION = 3
SHA256 = "21b3eb34250294b8cbfce9bef5d457e9315e8ed6e742d728f2109dc4a395c23d"
UUID_1 = "0b7c4a52-8f1e-4d3a-9c6b-2e5f7a8d9e10"


def test_the_hash_and_the_version_are_pinned_together():
    found = baseline.load()
    assert (found.version, found.sha) == (VERSION, SHA256)
    assert hashlib.sha256(baseline.PATH.read_bytes()).hexdigest() == SHA256


def test_the_texts_first_line_names_its_version():
    text = baseline.load().text
    assert text.split("\n", 1)[0] == f"Baseline {VERSION}" and not text.endswith("\n")


def test_a_first_line_that_names_no_version_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / "baseline.md"
    bad.write_text("Standing guidance\n\nNo number.\n")
    monkeypatch.setattr(baseline, "PATH", bad)
    baseline.load.cache_clear()
    try:
        with pytest.raises(ValueError):
            baseline.load()
    finally:
        monkeypatch.undo()
        baseline.load.cache_clear()
    assert baseline.load().version == VERSION


async def _never(_payload):
    raise AssertionError("not called")


def _ctx(**kw):
    return ProviderContext(request_approval=_never, cwd="/tmp/wt", output_schema=RESULT_SCHEMA, **kw)


def _cli_args(options) -> list[str]:
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    transport = SubprocessCLITransport("", options)
    transport._cli_path = "/bin/claude"
    return transport._build_command()


def _flag(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


def test_a_claude_coding_session_takes_the_preset_with_the_baseline_appended():
    import json

    from runner.providers.claude import ClaudeProvider

    text = baseline.load().text
    options = ClaudeProvider().options(_ctx(baseline=text))
    assert options.system_prompt == {"type": "preset", "preset": "claude_code", "append": text}
    assert json.loads(options.settings)["includeGitInstructions"] is False
    assert options.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    args = _cli_args(options)
    assert _flag(args, "--append-system-prompt") == text and "--system-prompt" not in args


def test_a_claude_coding_session_with_the_baseline_off_keeps_the_preset_without_an_append():
    import json

    from runner.providers.claude import ClaudeProvider

    options = ClaudeProvider().options(_ctx(baseline=None))
    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert json.loads(options.settings)["includeGitInstructions"] is False
    assert options.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    args = _cli_args(options)
    assert "--append-system-prompt" not in args and "--system-prompt" not in args


def test_a_claude_research_session_is_unchanged_on_the_minimal_prompt():
    from runner.providers.claude import ClaudeProvider

    for text in (None, baseline.load().text):
        options = ClaudeProvider().options(_ctx(kind=RESEARCH_KIND, baseline=text, max_subagents=2,
                                                max_searches=10, max_turns=5))
        assert options.system_prompt is None and options.settings is None
        assert "CLAUDE_CODE_DISABLE_AUTO_MEMORY" not in options.env
        args = _cli_args(options)
        assert _flag(args, "--system-prompt") == "" and "--append-system-prompt" not in args


class FakeCodexClient:

    made: list["FakeCodexClient"] = []

    def __init__(self, config=None, approval_handler=None) -> None:
        self.calls: list[tuple[str, str | None, dict]] = []
        FakeCodexClient.made.append(self)

    def start(self):
        pass

    def initialize(self):
        pass

    def thread_start(self, params):
        self.calls.append(("thread/start", None, dict(params)))
        return types.SimpleNamespace(thread=types.SimpleNamespace(id="thread-new"))

    def thread_resume(self, thread_id, params):
        self.calls.append(("thread/resume", thread_id, dict(params)))
        return types.SimpleNamespace(thread=types.SimpleNamespace(id=thread_id))

    def next_notification(self):
        raise EOFError("closed")

    def close(self):
        pass


def _codex_sent(monkeypatch, tmp_path, **kw) -> tuple[str, str | None, dict]:
    import openai_codex.client

    from runner.providers.codex import CodexProvider

    FakeCodexClient.made.clear()
    monkeypatch.setattr(openai_codex.client, "CodexClient", FakeCodexClient)
    provider = CodexProvider(codex_bin="/bin/false")

    async def go():
        await provider.start(ProviderContext(request_approval=_never, cwd=str(tmp_path), output_schema=RESULT_SCHEMA,
                                             **kw))
        await provider.close()

    asyncio.run(go())
    [client] = FakeCodexClient.made
    [call] = client.calls
    return call


def test_a_codex_coding_thread_takes_the_baseline_as_developer_instructions_on_start_and_resume(monkeypatch,
                                                                                               tmp_path):
    text = baseline.load().text
    method, _, params = _codex_sent(monkeypatch, tmp_path, baseline=text)
    assert method == "thread/start" and params["developerInstructions"] == text
    assert "baseInstructions" not in params
    method, thread, params = _codex_sent(monkeypatch, tmp_path, baseline=text, resume="019a-old-thread")
    assert (method, thread) == ("thread/resume", "019a-old-thread") and params["developerInstructions"] == text


def test_a_codex_coding_thread_with_the_baseline_off_gets_no_developer_instructions(monkeypatch, tmp_path):
    for resume in (None, "019a-old-thread"):
        _, _, params = _codex_sent(monkeypatch, tmp_path, baseline=None, resume=resume)
        assert "developerInstructions" not in params and "baseInstructions" not in params


def test_a_codex_research_thread_gets_no_developer_instructions(tmp_path):
    from runner.providers.codex import CodexProvider

    ctx = ProviderContext(request_approval=_never, cwd=str(tmp_path), kind=RESEARCH_KIND, max_subagents=2,
                          baseline=baseline.load().text)
    params = CodexProvider().thread_params(ctx)
    assert "developerInstructions" not in params and "baseInstructions" not in params


DONE = {"type": "ResultMessage", "subtype": "success", "duration_ms": 1, "duration_api_ms": 1, "is_error": False,
        "num_turns": 1, "session_id": UUID_1, "total_cost_usd": 0.01, "terminal_reason": "completed"}
CODEX_DONE = {"method": "turn/completed", "params": {"threadId": "t", "turn": {"id": "u1", "items": [],
                                                                               "status": "completed"}}}


def claude(**extra):
    return {"provider": "claude-code", "default_turn": [DONE], **extra}


def codex(**extra):
    return {"provider": "codex", "default_turn": [CODEX_DONE], **extra}


def _run(fixture, client, cwd, **options):
    provider = FakeProvider(fixture)
    session = Session(provider, client, Options(cwd=str(cwd), diff=False, **options), Caps(wall_clock=30, grace=1),
                      fast_timing())
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    assert outcome.reason == "completed"
    return provider, session


def _said(orch) -> list[dict]:
    return [p for p in orch.of("message", role="runner") if p.get("purpose") == "baseline"]


def _expected(applied, project_file=None, size=0, cut=False) -> dict:
    found = baseline.load()
    return {"role": "runner", "purpose": "baseline", "version": found.version, "sha": found.sha[:12],
            "applied": applied, "project_file": project_file, "project_file_bytes": size, "cut": cut}


def _before_first_turn(orch) -> bool:
    kinds = [(e["kind"], e["payload"].get("purpose")) for e in orch.events]
    return kinds.index(("message", "baseline")) < next(i for i, (k, _) in enumerate(kinds) if k == "turn")


def test_a_fresh_claude_run_is_given_the_text_and_says_append_once_before_its_first_turn(orch, client, worktree):
    (worktree / "CLAUDE.md").write_text("Run make test.\n")
    (worktree / "AGENTS.md").write_text("Codex's file.\n")
    fake, _ = _run(claude(), client, worktree)
    assert fake.ctx.baseline == baseline.load().text
    assert _said(orch) == [_expected("append", "CLAUDE.md", 15)] and _before_first_turn(orch)


def test_a_fresh_codex_run_says_developer_and_names_its_own_file_first(orch, client, worktree):
    (worktree / "CLAUDE.md").write_text("Claude's file.\n")
    (worktree / "AGENTS.md").write_text("Run make test, Codex.\n")
    fake, _ = _run(codex(), client, worktree)
    assert fake.ctx.baseline == baseline.load().text
    assert _said(orch) == [_expected("developer", "AGENTS.md", 22)] and _before_first_turn(orch)


def test_a_project_file_cut_at_32_kb_is_said_so(orch, client, worktree):
    (worktree / "CLAUDE.md").write_text("a" * (40 * 1024))
    _run(claude(), client, worktree)
    assert _said(orch) == [_expected("append", "CLAUDE.md", 32 * 1024, True)]


@pytest.mark.parametrize("fixture,how", [(claude(), "append"), (codex(), "developer")])
def test_with_the_baseline_off_the_provider_gets_no_text_and_the_event_says_off(orch, client, worktree, fixture, how):
    fake, _ = _run(fixture, client, worktree, baseline=False)
    assert fake.ctx.baseline is None
    assert _said(orch) == [_expected("off")]


@pytest.mark.parametrize("fixture,stored", [(claude(ids=True), UUID_1), (codex(ids=True), "019a-old-thread")])
def test_a_conversation_resumed_by_id_is_given_the_same_text_and_says_recorded(orch, client, worktree, fixture,
                                                                              stored):
    fake, session = _run(fixture, client, worktree, resume=stored, resume_text="Also the restore script.")
    assert fake.starts == [stored] and not session.fresh
    assert fake.ctx.baseline == baseline.load().text
    assert _said(orch) == [_expected("recorded")]


@pytest.mark.parametrize("fixture,stored,how", [(claude(refuse_resume=True), UUID_1, "append"),
                                                (codex(refuse_resume=True), "019a-old-thread", "developer")])
def test_a_resume_whose_id_is_refused_starts_fresh_with_the_text_and_says_how_it_sent_it(orch, client, worktree,
                                                                                        fixture, stored, how):
    fake, session = _run(fixture, client, worktree, resume=stored, resume_text="Also the restore script.",
                         carry_over="The brief and the last result.")
    assert fake.starts == [stored, None] and session.fresh
    assert fake.ctx.baseline == baseline.load().text
    assert _said(orch) == [_expected(how)]
    purposes = [p.get("purpose") for p in orch.of("message")]
    assert purposes.index("resumed_fresh") < purposes.index("baseline") < purposes.index("resume")


def test_a_resumed_run_with_no_stored_id_starts_fresh_and_says_append(orch, client, worktree):
    _run(claude(), client, worktree, resume=None, resume_text="Also the restore script.")
    assert _said(orch) == [_expected("append")]


def test_a_research_session_gets_no_text_and_no_event(orch, client, worktree):
    fake, _ = _run(claude(), client, worktree, kind=RESEARCH_KIND, output_schema=None)
    assert fake.ctx.baseline is None and not _said(orch)


def test_how_it_was_applied_for_every_case():
    assert [baseline.applied(p, on, resumed) for p in ("claude-code", "codex")
            for on in (True, False) for resumed in (False, True)] == [
        "append", "recorded", "off", "recorded", "developer", "recorded", "off", "recorded"]


def test_the_runner_takes_no_baseline_only_from_its_flag():
    base = ["--provider", "claude", "--cwd", "/tmp/wt", "--prompt-file", "p"]
    assert main_mod.parse(base).baseline is True
    assert main_mod.parse([*base, "--no-baseline"]).baseline is False

