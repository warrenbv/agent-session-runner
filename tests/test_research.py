"""Research sessions: caps, subagent counting and continuation, output files checked, split and posted
before the end event, inputs written before the first turn, and each provider's research settings."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import stat

import pytest

from conftest import fast_timing
from helpers import CLAIMS, SYNTHESIS, provider_files, research_result as result, research_run as run, say, write
from runner.stub_control_plane import RESEARCH_PART
from runner import __main__ as main_mod
from runner import research_out
from runner.control_plane import ResearchClient
from runner.providers import codex as codex_mod
from runner.providers.base import RESEARCH_KIND, ProviderContext
from runner.providers.codex import CodexProvider
from runner.providers.fake import FakeProvider
from runner.research_out import ResearchOut, split
from runner.session import Caps, subagent_start


def use(tool_id, name, parent=None, **tool_input):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": parent,
            "content": [{"type": "ToolUseBlock", "id": tool_id, "name": name, "input": tool_input}]}


def agent(tool_id):
    return use(tool_id, "Agent", subagent_type="general-purpose", description="look", prompt="look it up")


def task_started(tool_id, task_type):
    return {"type": "TaskStartedMessage", "task_id": f"task-{tool_id}", "tool_use_id": tool_id,
            "task_type": task_type, "description": "d", "uuid": "u", "session_id": "s"}


BRIEF = {"question": "Q?", "sub_questions": ["a"], "scope": "s", "exclusions": [], "time_window": "2025",
         "source_types": ["papers"], "depth": "quick", "output": "report", "questions_for_owner": ["one?"]}


def joined(orch, name):
    parts = sorted((p for p in orch.research_parts if p["file"] == name), key=lambda p: p["index"])
    return "".join(p["content"] for p in parts), parts


def end_after_done(orch):
    [done] = orch.research_done
    return orch.events[-1]["kind"] == "end" and done["events_before"] == len(orch.events) - 1


def test_a_provider_runs_files_are_checked_split_and_posted_before_the_end_event(orch, client, tmp_path):
    steps = [use("ws1", "WebSearch", query="x"), use("ws2", "WebSearch", query="y"), *provider_files(),
             say("Two claims found."), result()]
    outcome, _, _, _, _ = run(orch, client, tmp_path, steps, max_searches=2)
    assert outcome.reason == "completed" and outcome.exit_code == 0
    text, parts = joined(orch, "synthesis.md")
    assert text == SYNTHESIS and len(parts) == len(split(SYNTHESIS)) >= 3
    assert all(len(p["content"].encode()) <= RESEARCH_PART and p["total"] == len(parts) for p in parts)
    assert [p["index"] for p in parts] == list(range(len(parts))) and all(p["session_id"] == 5 for p in parts)
    [done] = orch.research_done
    assert done["session_id"] == 5 and done["partial"] is False and done["reason"] is None
    assert done["files"] == [
        {"name": "synthesis.md", "parts": len(parts), "sha256": hashlib.sha256(SYNTHESIS.encode()).hexdigest()},
        {"name": "claims.json", "parts": 1, "sha256": hashlib.sha256(json.dumps(CLAIMS).encode()).hexdigest()}]
    assert end_after_done(orch)
    end = orch.of("end")[0]
    assert end["result"] == {"summary": "Two claims found.", "claims": 2, "limits": ["max_searches"], "partial": False,
                             "reason": None, "files": [{"name": "synthesis.md", "status": "posted"},
                                                       {"name": "claims.json", "status": "posted"}]}
    assert not orch.of("diff") and "pr_url" not in end


def test_a_missing_file_and_one_that_does_not_validate_are_reported_not_dropped(orch, client, tmp_path):
    bad = [dict(CLAIMS[0], id="C9")]
    outcome, _, _, _, _ = run(orch, client, tmp_path, [write("claims.json", json.dumps(bad)), say("x"), result()])
    assert outcome.reason == "completed"
    [done] = orch.research_done
    assert done["files"][0] == {"name": "synthesis.md", "parts": 0, "sha256": None}
    assert done["files"][1]["name"] == "claims.json" and done["files"][1]["parts"] == 1
    assert done["partial"] is True
    assert "synthesis.md was not written" in done["reason"]
    result_files = orch.of("end")[0]["result"]["files"]
    assert result_files == [{"name": "synthesis.md", "status": "missing"}, {"name": "claims.json", "status": "posted"}]

    out = ResearchOut(ResearchClient(_Recorder(), 6), "provider", str(tmp_path / "research" / "5"))
    (tmp_path / "research" / "5" / "synthesis.md").write_text(SYNTHESIS)
    partial, why = out.post_outputs("completed")
    assert partial and "ids do not match" in why and out.statuses["claims.json"] == "invalid"


def test_a_hash_or_count_refusal_resends_the_parts_a_bounded_number_of_times(orch, client, tmp_path):
    mismatch = (400, {"error": {"code": "invalid", "message": "synthesis.md: 0 parts held, 3 named"}})
    orch.research_done_answers = [mismatch, mismatch]
    outcome, _, _, _, _ = run(orch, client, tmp_path, [*provider_files(), say("Done."), result()])
    assert outcome.reason == "completed" and outcome.exit_code == 0
    assert len(orch.research_done) == 3 and orch.research_done[-1]["partial"] is False
    sent = [p for p in orch.research_parts if p["file"] == "claims.json"]
    assert len(sent) == 3
    assert orch.of("end")[0]["result"]["files"] == [{"name": "synthesis.md", "status": "posted"},
                                                    {"name": "claims.json", "status": "posted"}]


def test_a_refusal_that_never_clears_ends_the_step_with_the_reason(orch, client, tmp_path):
    mismatch = (400, {"error": {"code": "invalid", "message": "the SHA-256 does not match"}})
    orch.research_done_answers = [mismatch] * research_out.OUTPUT_ATTEMPTS
    scratch = tmp_path / "s"
    scratch.mkdir()
    for step in provider_files():
        (scratch / step["path"]).write_text(step["text"])
    out = ResearchOut(ResearchClient(client, 5, sleep=lambda _s: None), "provider", str(scratch))
    partial, why = out.post_outputs("completed")
    assert len(orch.research_done) == research_out.OUTPUT_ATTEMPTS
    assert partial and "refused the files' part counts or hashes 3 times (400 invalid)" in why
    assert not out.done and set(out.statuses.values()) == {"not_kept"}


def test_a_shape_refusal_is_not_resent_and_its_reason_ends_the_step(orch, client, tmp_path):
    orch.research_done_answers = [(200, {"accepted": False, "reason": "claims.json's ids do not match", "claims": 0,
                                         "redacted": False})]
    outcome, _, _, _, _ = run(orch, client, tmp_path, [*provider_files(), say("Done."), result()])
    assert len(orch.research_done) == 1 and outcome.reason == "completed"
    result_ = orch.of("end")[0]["result"]
    assert result_["partial"] is True
    assert "The control plane did not keep the files: claims.json's ids do not match" in result_["reason"]
    assert result_["files"] == [{"name": "synthesis.md", "status": "not_kept"}, {"name": "claims.json",
                                                                                 "status": "not_kept"}]


class _Recorder:

    def __init__(self):
        self.calls = []

    def call(self, op, body):
        self.calls.append((op, body))
        return {}


@pytest.mark.parametrize("synthesis,problem", [
    ("## C1\nx\n## C2\ny\n", "no closing `## Methods and limits`"),
    ("## Methods and limits\nx\n## C1\ny\n", "not its closing heading"),
    ("## C2\nx\n## C1\ny\n## Methods and limits\nz\n", "out of order"),
    ("# Title\n\nno claims\n## Methods and limits\n", "no `## C<n>` claim headings"),
])
def test_the_synthesis_shape(synthesis, problem):
    _, problems = research_out.synthesis_ids(synthesis)
    assert any(problem in p for p in problems)


def test_the_lead_files_shapes():
    assert research_out.check_brief(BRIEF) == []
    assert research_out.check_brief({**BRIEF, "questions_for_owner": ["1", "2", "3", "4"]})
    assert "lacks" in research_out.check_brief({"question": "q"})[0]
    report = {"findings": [{"text": "t", "cites": [{"provider": "claude", "claim": "C7"}]}], "disagreements": [],
              "gaps": [], "overlap": [{"url": "https://example.com", "providers": ["claude"]}],
              "source_checks": [{"url": "https://example.com", "status": "checked", "note": ""}],
              "providers_reached": ["claude"]}
    assert research_out.check_report(report) == []
    assert research_out.check_report({**report, "source_checks": [{"url": "u", "status": "maybe"}]})


CHALLENGE_MD = ("## Verdicts\n\n### V1 claude:C1 Capacity\nHeld.\n\n### V2 codex:C1 Units work\nWeakened.\n\n"
                "## The case against\n\nDucts.\n\n## Pre-mortem\n\nReasoning, not evidence.\n\n"
                "## Methods and limits\n\n8 searches.\n")
SOURCE = {"url": "https://example.org/study", "title": "Study", "accessed": "2026-11-13"}
VERDICTS = {"verdicts": [{"id": "V1", "claim": "claude:C1", "verdict": "held", "reason": "r", "sources": [SOURCE]},
                         {"id": "V2", "claim": "codex:C1", "verdict": "weakened", "reason": "r", "sources": [SOURCE]}],
            "case_against": [{"id": "K1", "text": "Ducts.", "against": ["claude:C1"], "sources": [SOURCE]}],
            "pre_mortem": [{"id": "P1", "text": "Backup heat.", "rests_on": ["claude:C2"]}], "not_examined": []}


def test_the_challengers_files_shapes():
    problems, examined = research_out.validate("challenger", {"challenge.md": CHALLENGE_MD,
                                                              "verdicts.json": json.dumps(VERDICTS)})
    assert problems == {"challenge.md": [], "verdicts.json": []} and examined == 2
    numbers, said = research_out.challenge_ids(CHALLENGE_MD.replace("## Pre-mortem", "## Risks"))
    assert numbers == [1, 2] and "headings are not" in said[0]
    faults = {
        "not numbered V1": {**VERDICTS, "verdicts": [VERDICTS["verdicts"][1], VERDICTS["verdicts"][0]]},
        "not <provider>:C<n>": {**VERDICTS, "verdicts": [{**VERDICTS["verdicts"][0], "claim": "C1"},
                                                         VERDICTS["verdicts"][1]]},
        "judges a claim twice": {**VERDICTS, "verdicts": [VERDICTS["verdicts"][0], {**VERDICTS["verdicts"][1],
                                                                                   "claim": "claude:C1"}]},
        "no sources": {**VERDICTS, "verdicts": [VERDICTS["verdicts"][0], {**VERDICTS["verdicts"][1], "sources": []}]},
        "has no reason": {**VERDICTS, "verdicts": [VERDICTS["verdicts"][0], {
            "id": "V2", "claim": "codex:C1", "verdict": "could_not_check"}]},
        "case_against_none says nothing": {**VERDICTS, "case_against": []},
        "case_against is malformed": {**VERDICTS, "case_against": [{**VERDICTS["case_against"][0], "sources": []}]},
        "at most 5": {**VERDICTS, "pre_mortem": [VERDICTS["pre_mortem"][0]] * 6},
    }
    for said, data in faults.items():
        assert any(said in p for p in research_out.check_verdicts(data, [1, 2])), said
    assert any("do not match" in p for p in research_out.check_verdicts(VERDICTS, [1]))
    fine = {**VERDICTS, "verdicts": [VERDICTS["verdicts"][0], {"id": "V2", "claim": "codex:C1",
                                                               "verdict": "could_not_check", "reason": "a login"}],
            "case_against": [], "case_against_none": "Searched 12 ways."}
    assert research_out.check_verdicts(fine, [1, 2]) == []
    assert research_out.check_verdicts({**VERDICTS, "verdicts": [
        {**VERDICTS["verdicts"][0], "claim": "other:C99"}, VERDICTS["verdicts"][1]]}, [1, 2]) == []


def test_a_linked_or_oversized_output_is_not_read(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("not for posting")
    scratch = tmp_path / "s"
    scratch.mkdir()
    os.symlink(outside, scratch / "synthesis.md")
    (scratch / "claims.json").write_bytes(b"x" * (research_out.FILE_CAP + 1))
    assert research_out.read_output(str(scratch), "synthesis.md") == (None, "not_regular")
    assert research_out.read_output(str(scratch), "claims.json") == (None, "too_large")
    recorder = _Recorder()
    out = ResearchOut(ResearchClient(recorder, 7), "provider", str(scratch))
    partial, why = out.post_outputs("completed")
    assert [op for op, _ in recorder.calls] == ["research.output_done"]
    assert partial and "is not a regular file" in why and "is larger than" in why


def test_split_never_cuts_a_character():
    text = "aé" * 40000 + "z"
    parts = split(text, size=1001)
    assert "".join(parts) == text and all(len(p.encode()) <= 1001 for p in parts)
    assert split("") == [""]


def test_the_subagent_cap_ends_the_run_as_partial_counting_agent_starts_only(orch, client, tmp_path):
    steps = [*provider_files(), agent("a1"), task_started("a1", "local_agent"), task_started("sh1", "local_bash"),
             agent("a2"), {"fake": "sleep", "seconds": 10}, say("never"), result()]
    outcome, provider, session, _, took = run(orch, client, tmp_path, steps, max_subagents=1)
    assert outcome.reason == "max_subagents" and outcome.exit_code == 2 and took < 8
    assert session._subagents == {"a1", "a2"} and provider.interrupts >= 1
    assert orch.of("limit", limit="max_subagents")[0]["max_subagents"] == 1
    [done] = orch.research_done
    assert done["partial"] is True and "more subagents than its cap" in done["reason"]
    assert all(f["parts"] >= 1 for f in done["files"])
    assert end_after_done(orch) and orch.of("end")[0]["result"]["limits"] == ["max_subagents"]


def test_subagent_starts_are_keyed_once_and_shell_tasks_are_not_counted():
    assert subagent_start({"phase": "requested", "tool_use_id": "a"}) == "a"
    assert subagent_start({"phase": "started", "tool_use_id": "a", "task_type": "local_agent"}) == "a"
    assert subagent_start({"phase": "started", "tool_use_id": "b", "task_type": "local_bash"}) is None
    assert subagent_start({"phase": "progress", "tool_use_id": "a"}) is None
    assert subagent_start({"phase": "started", "id": "x", "agent_thread_id": "th"}) == "th"
    assert subagent_start({"phase": "completed", "id": "w", "tool": "wait"}) is None


def test_the_wall_clock_ends_a_research_run_and_its_files_still_go(orch, client, tmp_path):
    steps = [*provider_files(), {"fake": "sleep", "seconds": 10}, result()]
    outcome, _, _, _, took = run(orch, client, tmp_path, steps, caps=Caps(max_turns=10, wall_clock=0.5, grace=1))
    assert outcome.reason == "wall_clock" and took < 8
    [done] = orch.research_done
    assert done["partial"] is True and "wall clock ran out" in done["reason"]
    assert [f["parts"] >= 1 for f in done["files"]] == [True, True] and end_after_done(orch)


def test_the_turn_cap_ends_a_research_run_as_partial(orch, client, tmp_path):
    outcome, _, _, _, _ = run(orch, client, tmp_path, [*provider_files(), result("error_max_turns")])
    assert outcome.reason == "max_turns"
    [done] = orch.research_done
    assert done["partial"] is True and "used all its turns" in done["reason"]
    assert orch.of("end")[0]["result"]["limits"] == ["max_turns"]


def launched(tool_id):
    return {"type": "UserMessage", "parent_tool_use_id": None, "content": [{
        "type": "ToolResultBlock", "tool_use_id": tool_id, "is_error": None,
        "content": [{"type": "text", "text": "Async agent launched successfully. The agent is working in the "
                                             "background. You will be notified automatically when it completes."}]}]}


def reported(tool_id, status="completed"):
    return {"type": "TaskNotificationMessage", "task_id": f"task-{tool_id}", "tool_use_id": tool_id,
            "status": status, "summary": "found three sources", "uuid": "u", "session_id": "s"}


def background_turn(*ids):
    steps = []
    for tool_id in ids:
        steps += [agent(tool_id), task_started(tool_id, "local_agent"), launched(tool_id)]
    return [*steps, say("The subagents are still running, and I'll continue when they report back."), result()]


def timing(**kw):
    return dataclasses.replace(fast_timing(), **kw)


def continued(orch):
    return [m for m in orch.of("message") if m.get("purpose") == "continue"]


def test_a_turn_that_ends_with_subagents_running_is_continued_on_their_reports(orch, client, tmp_path):
    follow = [reported("a1"), reported("a2"), *provider_files(), say("Two claims found."), result()]
    outcome, provider, session, _, took = run(orch, client, tmp_path, background_turn("a1", "a2"), follows=[follow])
    assert outcome.reason == "completed" and outcome.exit_code == 0 and took < 8
    assert provider.followed == 1 and len(provider.sent) == 1 and session.turns == 2 and not session._running
    [note] = continued(orch)
    assert note["author"] == "runner" and "2 subagents it started were still running" in note["text"]
    [done] = orch.research_done
    assert done["partial"] is False and done["reason"] is None and all(f["parts"] >= 1 for f in done["files"])
    assert orch.of("end")[0]["result"]["summary"] == "Two claims found." and end_after_done(orch)


def test_when_no_turn_follows_the_reports_the_runner_gives_one(orch, client, tmp_path):
    follow = [reported("a1"), {"fake": "sleep", "seconds": 3600}]
    later = [[*provider_files(), say("Written."), result()]]
    outcome, provider, session, _, took = run(orch, client, tmp_path, background_turn("a1"), later=later,
                                              follows=[follow], timing=timing(follow_quiet=0.2))
    assert outcome.reason == "completed" and took < 8
    assert provider.followed == 1 and len(provider.sent) == 2
    assert provider.sent[1].startswith("Your turn ended before you wrote synthesis.md and claims.json")
    [done] = orch.research_done
    assert done["partial"] is False


def test_when_the_continuations_run_out_the_reason_says_what_happened(orch, client, tmp_path):
    later = [[say("Still waiting."), result()]] * 3
    outcome, provider, session, _, took = run(orch, client, tmp_path, background_turn("a1", "a2"), later=later,
                                              timing=timing(follow_idle=0.2))
    assert outcome.reason == "completed" and took < 8
    assert provider.followed == 1 and len(provider.sent) == 3
    assert "2 subagents you started have not reported" in provider.sent[1]
    [done] = orch.research_done
    assert done["partial"] is True
    assert "turn ended while 2 subagents it had started were still running" in done["reason"]
    assert "continued the session 3 times" in done["reason"] and "continuations ran out" in done["reason"]
    assert "2 subagents had still not reported" in done["reason"] and "synthesis.md was not written" in done["reason"]


@pytest.mark.parametrize("cap", ["wall_clock", "max_subagents", "max_turns"])
def test_the_caps_still_end_a_continued_run(orch, client, tmp_path, cap):
    follows, caps, later = [], Caps(max_turns=10, wall_clock=30, grace=1), ()
    if cap == "wall_clock":
        caps = Caps(max_turns=10, wall_clock=0.5, grace=1)
    elif cap == "max_subagents":
        follows = [[reported("a1"), agent("a2"), agent("a3"), {"fake": "sleep", "seconds": 10}, result()]]
    else:
        caps = Caps(max_turns=1, wall_clock=30, grace=1)
    outcome, provider, _, _, took = run(orch, client, tmp_path, background_turn("a1"), caps=caps, later=later,
                                        follows=follows, max_subagents=2)
    assert took < 8
    [done] = orch.research_done
    assert done["partial"] is True and "Its turn ended while" in done["reason"]
    if cap == "max_turns":
        assert outcome.reason == "completed" and provider.followed == 0 and "turns ran out" in done["reason"]
    else:
        assert outcome.reason == cap and provider.followed == 1


def test_a_turn_without_subagents_running_and_codex_are_not_continued(orch, client, tmp_path):
    steps = [agent("a1"), task_started("a1", "local_agent"), reported("a1"), say("Done."), result()]
    outcome, provider, session, _, _ = run(orch, client, tmp_path, steps)
    assert outcome.reason == "completed" and len(provider.sent) == 1 and provider.followed == 0
    assert not continued(orch) and session._left_running == 0
    assert "subagent" not in orch.research_done[0]["reason"]


def test_a_codex_research_turn_with_a_spawn_left_open_is_not_continued(orch, client, tmp_path):
    spawn = _codex_item("item/completed", type="subAgentActivity", id="s1", kind="started", agentThreadId="th1")
    outcome, provider, session, _, _ = run(orch, client, tmp_path, [spawn, CODEX_DONE], name="codex")
    assert outcome.reason == "completed" and len(provider.sent) == 1 and provider.followed == 0
    assert not continued(orch) and session._left_running == 0


def test_a_revoked_token_posts_nothing(orch, client, tmp_path):
    orch.when(lambda e: e["kind"] == "turn", lambda e: orch.revoke())
    outcome, _, _, _, _ = run(orch, client, tmp_path, [*provider_files(), {"fake": "sleep", "seconds": 2}, result()])
    assert outcome.reason == "revoked" and not orch.research_done and not orch.research_parts


class _Peeking(FakeProvider):

    def turn(self, text):
        root = os.path.join(self.ctx.cwd, "inputs")
        self.seen = sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs)
        return super().turn(text)


def test_a_lead_roles_inputs_are_written_before_the_first_turn(orch, client, tmp_path):
    big = "Brief é. " * 12000
    orch.research_input_files = [("brief.md", big), ("claude/synthesis.md", "## C1\n")]
    steps = [write("brief.json", json.dumps(BRIEF)), write("brief.md", "# Brief\n"), say("Drafted."), result()]
    outcome, provider, _, scratch, _ = run(orch, client, tmp_path, steps, role="lead_brief",
                                           provider_cls=_Peeking)
    assert outcome.reason == "completed"
    assert provider.seen == ["brief.md", "claude/synthesis.md", "manifest.md"]
    assert (scratch / "inputs" / "brief.md").read_text() == big
    listed = (scratch / "inputs" / "manifest.md").read_text()
    assert [line.split("`")[1] for line in listed.splitlines() if line.startswith("- ")] == [
        "inputs/brief.md", "inputs/claude/synthesis.md"]
    assert f"({len(big.encode())} bytes)" in listed
    assert stat.S_IMODE((scratch / "inputs" / "manifest.md").stat().st_mode) == 0o600
    assert stat.S_IMODE((scratch / "inputs").stat().st_mode) == 0o700
    assert stat.S_IMODE((scratch / "inputs" / "brief.md").stat().st_mode) == 0o600
    gets = [r["body"] for r in orch.requests if r["op"] == "research.inputs_get"]
    assert gets == [{"session_id": 5, "part": n} for n in range(len(split(big)) + 1)]
    [done] = orch.research_done
    assert [f["name"] for f in done["files"]] == ["brief.json", "brief.md"] and done["partial"] is False


def test_the_challengers_inputs_are_written_first_and_its_challenge_posted(orch, client, tmp_path):
    orch.research_input_files = [("brief.md", "# Brief\n"), ("claude/synthesis.md", "## C1\n"),
                                 ("codex/synthesis.md", "## C1\n"), ("providers.json", "{}")]
    steps = [write("challenge.md", CHALLENGE_MD), write("verdicts.json", json.dumps(VERDICTS)), say("Challenged."),
             result()]
    outcome, provider, _, _, _ = run(orch, client, tmp_path, steps, role="challenger", provider_cls=_Peeking)
    assert outcome.reason == "completed"
    assert provider.seen == ["brief.md", "claude/synthesis.md", "codex/synthesis.md", "manifest.md", "providers.json"]
    [done] = orch.research_done
    assert [f["name"] for f in done["files"]] == ["challenge.md", "verdicts.json"] and done["partial"] is False


def _piece(name, text, **over):
    return {"name": name, "index": 0, "parts": 1, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "content": text, **over}


@pytest.mark.parametrize("answers,words", [
    ([_piece("../escape.md", "x")], "name or content was not usable"),
    ([_piece("manifest.md", "x")], "name or content was not usable"),
    ([_piece("brief.md", "x", sha256="0" * 64)], "does not match its SHA-256"),
    ([_piece("brief.md", "x", parts=2)], "arrived with pieces missing"),
    ([_piece("brief.md", "x", index=1)], "piece number"),
    ([_piece("brief.md", "x", total=2), _piece("brief.md", "x", total=3)], "part count changed"),
])
def test_unusable_inputs_stop_the_start_and_are_reported(orch, client, tmp_path, answers, words):
    orch.research_inputs = answers
    scratch = tmp_path / "s"
    scratch.mkdir()
    out = ResearchOut(ResearchClient(client, 5, sleep=lambda _s: None), "lead_report", str(scratch))
    with pytest.raises(research_out.InputsError, match=words):
        out.fetch_inputs()
    assert not (scratch / "inputs").exists()
    outcome, provider, _, _, _ = run(orch, client, tmp_path, [say("never"), result()], role="lead_report")
    assert outcome.reason == "error" and provider.sent == []
    assert not (tmp_path / "research" / "escape.md").exists()
    [done] = orch.research_done
    assert done["partial"] is True and "session failed (InputsError)" in done["reason"]
    assert "report.md was not written" in done["reason"]


def test_a_lead_step_with_no_inputs_writes_only_the_manifest_saying_so(orch, client, tmp_path):
    out = ResearchOut(ResearchClient(client, 5, sleep=lambda _s: None), "lead_brief", str(tmp_path))
    assert out.fetch_inputs() == 0 and [p.name for p in (tmp_path / "inputs").iterdir()] == ["manifest.md"]
    assert "This step has no input files." in (tmp_path / "inputs" / "manifest.md").read_text()
    assert [r["body"] for r in orch.requests if r["op"] == "research.inputs_get"] == [{"session_id": 5, "part": 0}]


def test_a_provider_role_fetches_no_inputs(orch, client, tmp_path):
    run(orch, client, tmp_path, [*provider_files(), result()])
    assert not [r for r in orch.requests if r["op"] == "research.inputs_get"]


async def _never(_payload):
    raise AssertionError("no card in a research session")


def test_claudes_research_options_are_the_probes_tool_set(tmp_path):
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    from runner.providers.claude import RESEARCH_TOOLS, ClaudeProvider

    scratch = os.path.realpath(tmp_path)
    ctx = ProviderContext(request_approval=_never, cwd=scratch, kind=RESEARCH_KIND, model="claude-opus-5-5",
                          effort="xhigh", max_turns=40, max_subagents=1, max_searches=60, web=False,
                          allowed_tools=["Bash"], disallowed_tools=["WebSearch"], mcp_socket="/tmp/s", drafts=True)
    provider = ClaudeProvider(oauth_token="placeholder-oauth-0000")
    options = provider.options(ctx)
    assert options.tools == list(RESEARCH_TOOLS) == ["Read", "Write", "Edit", "WebSearch", "WebFetch", "Task"]
    assert options.allowed_tools == [f"Read(/{scratch}/**)", f"Edit(/{scratch}/**)", "WebSearch", "WebFetch"]
    assert options.disallowed_tools == [] and options.mcp_servers == {} and options.strict_mcp_config is True
    assert options.env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "1"
    assert options.env["CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION"] == "60"
    assert options.env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert (options.model, options.effort, options.max_turns, options.permission_mode) == (
        "claude-opus-5-5", "xhigh", 40, "default")
    assert options.output_format is None and options.include_partial_messages is False
    fable = provider.options(ProviderContext(request_approval=_never, cwd=scratch, kind=RESEARCH_KIND,
                                             model="claude-fable-5-2", effort="xhigh"))
    assert fable.model == "claude-fable-5-2" and fable.env["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-opus-5-5"

    provider.ctx = ctx
    decide = lambda name: asyncio.run(provider._can_use_tool(name, {}, None))
    assert isinstance(decide("Bash"), PermissionResultDeny)
    assert isinstance(decide("Agent"), PermissionResultAllow)
    assert isinstance(decide("Agent"), PermissionResultDeny)
    assert isinstance(decide("mcp__platform__project_status"), PermissionResultDeny)


def test_a_research_step_with_a_subagent_cap_of_zero_is_not_offered_the_subagent_tool(tmp_path):
    from runner.providers.claude import ClaudeProvider

    scratch = os.path.realpath(tmp_path)
    provider = ClaudeProvider(oauth_token="placeholder-oauth-0000")
    none = provider.options(ProviderContext(request_approval=_never, cwd=scratch, kind=RESEARCH_KIND, max_subagents=0))
    assert none.tools == ["Read", "Write", "Edit", "WebSearch", "WebFetch"]
    assert none.disallowed_tools == ["Task", "Agent"]
    for cap in (None, 3):
        some = provider.options(ProviderContext(request_approval=_never, cwd=scratch, kind=RESEARCH_KIND,
                                                max_subagents=cap))
        assert "Task" in some.tools and some.disallowed_tools == []


def test_a_scratch_path_that_cannot_ride_in_a_rule_is_refused():
    from runner.providers.claude import research_rules

    with pytest.raises(ValueError):
        research_rules("/tmp/with space")


CODEX_ARGS = ["--provider", "codex", "--kind", "research", "--cwd", "/tmp/x", "--prompt-file", "/tmp/p",
              "--session-id", "3", "--max-subagents", "2", "--max-searches", "5", "--max-turns", "10"]


def test_codex_research_is_the_probes_settings(tmp_path):
    scratch = os.path.realpath(tmp_path)
    ctx = ProviderContext(request_approval=_never, cwd=scratch, kind=RESEARCH_KIND, model="gpt-6-astra",
                          effort="high", max_subagents=6, max_searches=200, mcp_socket="/tmp/s", web=True)
    provider = CodexProvider()
    config = provider.config(ctx)
    overrides = list(config.config_overrides)
    for setting in ('web_search="live"', "features.shell_tool=false", "features.unified_exec=false",
                    "features.view_image=false", "agents.max_threads=6", 'sandbox_mode="workspace-write"',
                    "sandbox_workspace_write.network_access=false", 'approval_policy="never"', "features.apps=false"):
        assert setting in overrides
    assert codex_mod.MCP_ENABLE not in overrides and config.env is None
    assert codex_mod.MCP_SERVER not in overrides
    assert not any(o.startswith("mcp_servers") for o in overrides) and 'web_search="disabled"' not in overrides
    assert overrides[-1] == codex_mod.projects_override(scratch)
    params = provider.thread_params(ctx)
    assert (params["approvalPolicy"], params["sandbox"], params["cwd"], params["model"]) == (
        "never", "workspace-write", scratch, "gpt-6-astra")
    assert params["config"]["features.shell_tool"] is False and params["config"]["web_search"] == "live"
    assert params["config"]["projects"] == codex_mod.projects_untrusted(scratch)
    coding = provider.config(ProviderContext(request_approval=_never, cwd=scratch))
    assert "features.shell_tool=false" not in coding.config_overrides
    assert provider.thread_params(ProviderContext(request_approval=_never, cwd=scratch))["approvalPolicy"] == \
        "on-request"


def test_codex_research_refuses_a_lead_and_a_codex_it_was_not_probed_on(monkeypatch):
    pinned = {p: codex_mod.RESEARCH_CODEX_PIN for p in codex_mod.RESEARCH_CODEX_PACKAGES}
    assert codex_mod.installed_codex() == pinned
    assert codex_mod.research_refusal("provider") is None
    for role in ("lead_brief", "lead_report"):
        assert codex_mod.research_refusal(role) == codex_mod.CODEX_NOT_LEAD
    assert codex_mod.research_refusal("challenger") == codex_mod.CODEX_NOT_CHALLENGER == (
        "Codex cannot be the challenger of a research run: with its shell off it has no tool to read the syntheses it"
        " challenges, so the runner refuses it.")
    assert set(research_out.INPUT_ROLES) == set(codex_mod.RESEARCH_LEAD_ROLES)
    later = codex_mod.research_refusal("provider", {**pinned, "openai-codex-cli-bin": "0.168.0"})
    assert "held to Codex 0.167.0" in later and "openai-codex-cli-bin 0.168.0" in later
    assert "RESEARCH_CODEX_PIN" in later

    main_mod.check_research(main_mod.parse([*CODEX_ARGS, "--role", "provider"]))
    with pytest.raises(SystemExit) as exc:
        main_mod.check_research(main_mod.parse([*CODEX_ARGS, "--role", "lead_report"]))
    assert str(exc.value) == codex_mod.CODEX_NOT_LEAD

    monkeypatch.setattr(codex_mod, "installed_codex", lambda: {**pinned, "openai-codex": "0.170.1"})
    with pytest.raises(SystemExit, match="held to Codex 0.167.0"):
        main_mod.check_research(main_mod.parse([*CODEX_ARGS, "--role", "provider"]))
    with pytest.raises(ValueError, match="held to Codex 0.167.0"):
        asyncio.run(CodexProvider().start(ProviderContext(request_approval=_never, cwd="/tmp/x", kind=RESEARCH_KIND)))


class _Status:
    def __init__(self, servers=None, fail=False):
        self.servers, self.fail = servers or {}, fail

    async def __call__(self):
        if self.fail:
            raise RuntimeError("no list")
        return self.servers


def test_codex_research_stops_when_any_mcp_server_is_registered_or_cannot_be_ruled_out():
    provider = CodexProvider()
    provider.mcp_status = _Status({})
    asyncio.run(provider._research_mcp_check())
    provider.mcp_status = _Status({"platform": []})
    with pytest.raises(ValueError, match=r"MCP servers registered \(platform\)"):
        asyncio.run(provider._research_mcp_check())
    provider.mcp_status = _Status({"platform": ["tasks_list"]})
    with pytest.raises(ValueError, match=r"MCP tools offered to it \(platform\)"):
        asyncio.run(provider._research_mcp_check())
    provider.mcp_status = _Status(fail=True)
    with pytest.raises(ValueError, match="cannot confirm"):
        asyncio.run(provider._research_mcp_check())


def _codex_item(method, **item):
    return {"method": method, "params": {"threadId": "t", "turnId": "u", "item": item}}


CODEX_DONE = {"method": "turn/completed", "params": {"threadId": "t", "turn": {"id": "u", "items": [],
                                                                                "status": "completed"}}}


def test_a_command_in_a_codex_research_stream_ends_the_run_at_once_as_failed(orch, client, tmp_path):
    steps = [*provider_files(), _codex_item("item/started", type="commandExecution", id="c1", command="cat /etc/hostname",
                                            cwd="/", status="inProgress"),
             {"fake": "sleep", "seconds": 10}, CODEX_DONE]
    outcome, provider, _, _, took = run(orch, client, tmp_path, steps, name="codex")
    assert outcome.reason == "shell_used" and outcome.exit_code == 1 and took < 8
    assert provider.interrupts >= 1 and orch.of("limit", limit="shell_used")
    [done] = orch.research_done
    assert done["partial"] is True and "whose shell must be off" in done["reason"]
    assert orch.of("end")[0]["reason"] == "shell_used"


def test_codex_subagent_starts_count_once_per_thread(orch, client, tmp_path):
    def activity(method, thread, kind="started"):
        return _codex_item(method, type="subAgentActivity", id=f"a-{thread}-{method}", kind=kind,
                           agentThreadId=thread)
    steps = [*provider_files(), activity("item/started", "th1"), activity("item/completed", "th1"),
             activity("item/completed", "th1", kind="completed"), activity("item/started", "th2"),
             activity("item/completed", "th2"), {"fake": "sleep", "seconds": 10}, CODEX_DONE]
    outcome, _, session, _, took = run(orch, client, tmp_path, steps, name="codex", max_subagents=1)
    assert session._subagents == {"th1", "th2"} and outcome.reason == "max_subagents" and took < 8


def test_the_runners_research_arguments_build_a_research_session(client, tmp_path):
    args = main_mod.parse(["--provider", "claude", "--kind", "research", "--cwd", str(tmp_path), "--prompt-file",
                           "/tmp/p", "--role", "lead_report", "--session-id", "9", "--research-run-id", "4",
                           "--max-subagents", "2", "--max-searches", "5", "--max-turns", "10", "--wall-clock", "600",
                           "--model", "claude-fable-5-2", "--effort", "xhigh"])
    main_mod.check_research(args)
    session = main_mod.research_session(args, FakeProvider({"provider": "claude-code"}), client)
    o = session.options
    assert (o.kind, o.diff, o.output_schema, o.max_subagents, o.max_searches, o.model, o.effort) == (
        RESEARCH_KIND, False, None, 2, 5, "claude-fable-5-2", "xhigh")
    assert session.caps.wall_clock == 600 and session.caps.max_turns == 10 and session.handoff is None
    assert session.research.role == "lead_report" and session.research.client.session_id == 9
    for missing in (["--role", "provider"], ["--max-subagents", "1"]):
        partial = [a for a in ["--provider", "claude", "--kind", "research", "--cwd", "/x", "--prompt-file", "/p",
                               "--session-id", "1", "--max-searches", "5", "--max-turns", "3", "--role", "provider",
                               "--max-subagents", "1"]]
        i = partial.index(missing[0])
        del partial[i:i + 2]
        with pytest.raises(SystemExit):
            main_mod.check_research(main_mod.parse(partial))
    with pytest.raises(SystemExit):
        main_mod.check_research(main_mod.parse(["--provider", "claude", "--kind", "research", "--cwd", "/x",
                                                "--prompt-file", "/p", "--session-id", "1", "--max-searches", "5",
                                                "--max-turns", "3", "--role", "provider", "--max-subagents", "1",
                                                "--slug", "s", "--repo", "o/n"]))

