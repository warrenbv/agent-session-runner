"""The test turn: after work that completed, the session runs the project's test command and the verdict
comes from the provider's record of that call only, checked against recorded exit-code sessions."""

from __future__ import annotations

import asyncio
import json
import shlex

import pytest

from conftest import fast_timing, git, typed_result
from helpers import FakeForge, TEST_CMD as CMD, bash, bash_result, codex_cmd
from runner import handoff as handoff_mod
from runner import testrun
from runner.providers.fake import FakeProvider
from runner.session import RUNNER_AUTHOR, Caps, Options, Session

WORK = typed_result("Exclude the cache directory", files_changed=["a.txt"])
OTHER = typed_result("Ran the final check", tests_run=[CMD], branch="x", open_questions=["should not be the PR body"])


def result(structured=None, subtype="success", **extra):
    return {"type": "ResultMessage", "subtype": subtype, "duration_ms": 1, "duration_api_ms": 1,
            "is_error": subtype != "success", "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01,
            "terminal_reason": "completed", "structured_output": structured, **extra}


def say(text):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "TextBlock", "text": text}]}


def write(path="a.txt", text="work\n"):
    return {"fake": "write", "path": path, "text": text}


WORK_TURN = [write(), say("Done."), result(WORK)]


def claude(*turns, **extra):
    return {"provider": "claude-code", "turns": [list(t) for t in turns],
            "default_turn": [say("Nothing more."), result(OTHER)], **extra}


def codex_turn(*items, status="completed", tid="turn_t"):
    return ([{"method": "turn/started", "params": {"threadId": "thr", "turn": {"id": tid, "items": [],
                                                                                 "status": "inProgress"}}}]
            + list(items)
            + [{"method": "turn/completed", "params": {"threadId": "thr", "turn": {"id": tid, "items": [],
                                                                                     "status": status}}}])


def codex_msg(text):
    return {"method": "item/completed", "params": {"threadId": "thr", "turnId": "t",
                                                   "item": {"type": "agentMessage", "id": "m", "text": text}}}


def codex(*turns, **extra):
    return {"provider": "codex", "turns": [list(t) for t in turns],
            "default_turn": codex_turn(codex_msg(json.dumps(OTHER))), **extra}


CODEX_WORK = codex_turn({"fake": "write", "path": "a.txt", "text": "work\n"}, codex_msg(json.dumps(WORK)),
                        tid="turn_w")


def run(fixture, client, worktree, test_cmd=CMD, caps=None, forge=None):
    provider = FakeProvider(fixture)
    forge = forge or FakeForge()
    session = Session(provider, client, Options(cwd=str(worktree), test_cmd=test_cmd),
                      caps or Caps(wall_clock=30, grace=1), fast_timing(),
                      handoff=lambda res: handoff_mod.hand_off(str(worktree), "fix-backup", "main", res, forge))
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    return outcome, provider, forge


def verdict_of(orch):
    return orch.of("end")[0]["handoff"].get("tests")


@pytest.mark.parametrize("state", ["clean", "ignored", "untracked", "modified", "deleted", "staged",
                                   "staged_but_restored_on_disk", "commit", "empty_commit", "base_advanced"])
def test_git_change_gate_checks_index_worktree_and_branch(worktree, state):
    base = git(worktree, "rev-parse", "HEAD").strip()
    readme = worktree / "README.txt"
    if state == "ignored":
        (worktree / ".git/info/exclude").write_text("scratch.txt\n")
        (worktree / "scratch.txt").touch()
    if state == "untracked":
        (worktree / "new-file").touch()
    if state == "deleted":
        readme.unlink()
    if state in ("modified", "staged", "staged_but_restored_on_disk", "commit"):
        readme.write_text("changed\n")
        if state != "modified":
            git(worktree, "add", "README.txt")
    if state == "staged_but_restored_on_disk":
        readme.write_text("hello\n")
    if state in ("commit", "empty_commit"):
        git(worktree, "commit", "--allow-empty", "-m", "Work")
    if state == "base_advanced":
        git(worktree, "switch", "main")
        git(worktree, "commit", "--allow-empty", "-m", "Unrelated")
        base = git(worktree, "rev-parse", "HEAD").strip()
        git(worktree, "switch", "agent/fix-backup")
    assert handoff_mod.has_changes(str(worktree), base) is (state not in ("clean", "ignored", "base_advanced"))


def test_passed_from_the_provider_record(orch, client, worktree):
    outcome, fake, forge = run(claude(WORK_TURN, [bash(CMD), bash_result(), say("All passed."), result(OTHER)]),
                               client, worktree)
    assert fake.sent[1] == testrun.prompt(CMD) and len(fake.sent) == 2
    assert verdict_of(orch) == {"command": CMD, "status": "passed", "exit_code": None}
    assert outcome.reason == "completed" and outcome.handoff["tests"]["status"] == "passed"
    assert outcome.turns == 1


def test_failed_with_the_cli_s_exit_code(orch, client, worktree):
    run(claude(WORK_TURN, [bash(CMD), bash_result(ok=False, code=3), say("Exit code 0, all good!"),
                           result(OTHER)]), client, worktree)
    assert verdict_of(orch) == {"command": CMD, "status": "failed", "exit_code": 3}


def test_the_command_not_run(orch, client, worktree):
    run(claude(WORK_TURN, [say("I ran the tests and they all passed."), result(OTHER)]), client, worktree)
    assert verdict_of(orch) == {"command": CMD, "status": "not_run", "exit_code": None}


def test_a_different_command_run_instead(orch, client, worktree):
    run(claude(WORK_TURN, [bash("uv run pytest -q -k nothing"), bash_result(), result(OTHER)]), client, worktree)
    assert verdict_of(orch)["status"] == "not_run"


def test_surrounding_whitespace_still_matches(orch, client, worktree):
    run(claude(WORK_TURN, [bash(f"  {CMD}\t "), bash_result(), result(OTHER)]), client, worktree)
    assert verdict_of(orch) == {"command": CMD, "status": "passed", "exit_code": None}


def test_a_subagent_s_call_does_not_count(orch, client, worktree):
    run(claude(WORK_TURN, [bash(CMD, parent="toolu_agent"), bash_result(parent="toolu_agent"), result(OTHER)]),
        client, worktree)
    assert verdict_of(orch)["status"] == "not_run"


def test_the_call_denied_is_not_run(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "approval",
              lambda e: orch.answer(e["payload"]["request_id"], "deny", feedback="Exit code 0"))
    denied = {"type": "UserMessage", "parent_tool_use_id": None, "tool_use_result": "Error: Exit code 0",
              "content": [{"type": "ToolResultBlock", "tool_use_id": "toolu_t1", "content": "$DENY_MESSAGE",
                           "is_error": True}]}
    step = {"fake": "approval", "tool_name": "Bash", "input": {"command": CMD}, "tool_use_id": "toolu_t1",
            "on_allow": [bash_result()], "on_deny": [denied]}
    run(claude(WORK_TURN, [bash(CMD), step, result(OTHER)]), client, worktree)
    assert orch.of("approval")[0]["request_id"] == "toolu_t1"
    assert verdict_of(orch)["status"] == "not_run"


def test_a_denial_in_the_result_message_is_not_run(orch, client, worktree):
    denials = [{"tool_name": "Bash", "tool_use_id": "toolu_t1", "tool_input": {"command": CMD}}]
    run(claude(WORK_TURN, [bash(CMD), bash_result(), result(OTHER, permission_denials=denials)]), client, worktree)
    assert verdict_of(orch)["status"] == "not_run"


@pytest.mark.parametrize("output", [
    {"stdout": "", "stderr": "", "interrupted": False, "backgroundTaskId": "b1"},
    {"stdout": "", "stderr": "", "interrupted": False, "timedOutAfterMs": 120000},
    {"stdout": "", "stderr": "", "interrupted": True},
    None,
])
def test_a_command_that_did_not_finish_in_the_foreground_is_not_run(orch, client, worktree, output):
    tur = bash_result(output=output)
    if output is None:
        tur["tool_use_result"] = None
    run(claude(WORK_TURN, [bash(CMD), tur, result(OTHER)]), client, worktree)
    assert verdict_of(orch)["status"] == "not_run"


def test_a_non_zero_exit_read_as_not_an_error_is_failed(orch, client, worktree):
    out = {"stdout": "", "stderr": "", "interrupted": False, "returnCodeInterpretation": "No matches found"}
    run(claude(WORK_TURN, [bash(CMD), bash_result(output=out), result(OTHER)]), client, worktree)
    assert verdict_of(orch) == {"command": CMD, "status": "failed", "exit_code": None}
    assert verdict_notes(orch) == [f"Tests failed: `{CMD}`."]


def verdict_notes(orch):
    return [m["text"] for m in orch.of("message", role="user") if m.get("purpose") == "tests-verdict"]


@pytest.mark.parametrize("tests,text", [
    ({"command": CMD, "status": "passed", "exit_code": None}, f"Tests passed: `{CMD}`."),
    ({"command": CMD, "status": "passed", "exit_code": 0}, f"Tests passed: `{CMD}`."),
    ({"command": CMD, "status": "failed", "exit_code": 2}, f"Tests failed with exit code 2: `{CMD}`."),
    ({"command": CMD, "status": "failed", "exit_code": None}, f"Tests failed: `{CMD}`."),
    ({"command": CMD, "status": "not_run", "exit_code": None}, f"Tests not run: `{CMD}`."),
    ({"command": CMD, "status": "not_run", "exit_code": None, "reason": "no_changes"},
     "Tests were skipped because no file changed."),
    ({"command": CMD, "status": "tree_changed", "exit_code": None},
     f"The test run changed files, so its result does not count: `{CMD}`."),
])
def test_the_verdict_sentence(tests, text):
    assert testrun.sentence(tests) == text


def test_the_verdict_note_is_posted_after_the_test_turn_and_starts_no_turn(orch, client, worktree):
    outcome, fake, _ = run(claude(WORK_TURN, [bash(CMD), bash_result(ok=False, code=2), result(OTHER)]),
                           client, worktree)
    assert fake.sent == ["Fix the backup job.", testrun.prompt(CMD)]
    assert outcome.turns == 1
    users = orch.of("message", role="user")
    assert [m.get("purpose") for m in users] == [None, "tests", "tests-verdict"]
    assert users[2] == {"role": "user", "text": f"Tests failed with exit code 2: `{CMD}`.", "mode": "queue",
                        "entry_id": None, "author": "runner", "purpose": "tests-verdict"}
    kinds = [e["kind"] for e in orch.events]
    assert kinds.index("end") > max(i for i, e in enumerate(orch.events)
                                    if e["kind"] == "message" and e["payload"].get("purpose") == "tests-verdict")


def test_one_verdict_note_per_test_turn(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("purpose") == "tests",
              lambda e: orch.queue("one more thing", "queue"))
    slow = [bash(CMD), {"fake": "sleep", "seconds": 0.3}, bash_result(), result(OTHER)]
    run(claude(WORK_TURN, slow, [say("ok"), result(WORK)], [bash(CMD), bash_result(), result(OTHER)]),
        client, worktree)
    assert verdict_notes(orch) == [f"Tests passed: `{CMD}`."] * 2


def test_the_turn_ended_without_the_result(orch, client, worktree):
    run(claude(WORK_TURN, [bash(CMD), result(OTHER, subtype="error_during_execution")]), client, worktree)
    assert verdict_of(orch)["status"] == "not_run"


def test_a_tree_changed_by_the_test_turn(orch, client, worktree):
    outcome, _, _ = run(claude(WORK_TURN, [bash(CMD), write("b.txt", "sneaked in\n"), bash_result(),
                                           result(OTHER)]), client, worktree)
    assert verdict_of(orch) == {"command": CMD, "status": "tree_changed", "exit_code": None}
    assert "b.txt" in git(worktree, "show", "--name-only", "--format=", "HEAD")


def test_the_commit_is_the_tree_the_tests_ran_on(orch, client, worktree):
    run(claude(WORK_TURN, [bash(CMD), bash_result(), result(OTHER)]), client, worktree)
    assert verdict_of(orch)["status"] == "passed"
    [work_diff] = orch.of("diff")
    assert git(worktree, "rev-parse", "HEAD^{tree}").strip() == work_diff["to"]
    assert git(worktree, "status", "--porcelain") == ""


def test_no_test_command_no_extra_turn_and_no_tests_key(orch, client, worktree):
    outcome, fake, _ = run(claude(WORK_TURN), client, worktree, test_cmd=None)
    assert fake.sent == ["Fix the backup job."]
    assert "tests" not in orch.of("end")[0]["handoff"] and outcome.handoff["handoff"] == "pull request"
    assert not [m for m in orch.of("message", role="user") if m.get("author") == RUNNER_AUTHOR]


def test_the_work_s_result_is_preserved(orch, client, worktree):
    outcome, _, forge = run(claude(WORK_TURN, [bash(CMD), bash_result(), result(OTHER)]), client, worktree)
    assert outcome.result == WORK and orch.of("end")[0]["result"] == WORK
    assert forge.prs[0]["title"] == WORK["summary"] and "should not be the PR body" not in forge.prs[0]["body"]
    assert orch.of("end")[0]["last_turn"]["subtype"] == "success"


def test_the_prompt_event_is_marked_as_the_runner_s(orch, client, worktree):
    run(claude(WORK_TURN, [bash(CMD), bash_result(), result(OTHER)]), client, worktree)
    prompts = orch.of("message", role="user")
    assert prompts[0]["author"] == "launcher" and "purpose" not in prompts[0]
    assert prompts[1] == {"role": "user", "text": testrun.prompt(CMD), "mode": "queue", "entry_id": None,
                          "author": "runner", "purpose": "tests"}
    assert CMD in prompts[1]["text"]


def test_no_test_turn_when_the_work_did_not_complete(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("role") == "assistant", lambda e: orch.revoke())
    outcome, fake, forge = run(claude([write(), say("a"), {"fake": "sleep", "seconds": 5}, result(WORK)]), client,
                               worktree)
    assert outcome.reason == "revoked" and len(fake.sent) == 1 and forge.pushed == []


def test_a_limit_wraps_up_and_the_test_turn_follows_the_closing_turn(orch, client, worktree):
    from runner.session import wrap_up_prompt

    closing = [say("Wrapped."), result({**WORK, "summary": "Wrapped up"})]
    outcome, fake, forge = run(claude([write(), result(WORK, subtype="error_max_turns")], closing,
                                      [bash(CMD), bash_result(), result(OTHER)]),
                               client, worktree, caps=Caps(max_turns=3, wall_clock=30, grace=1))
    assert outcome.reason == "completed" and fake.sent[1] == wrap_up_prompt("max_turns")
    assert fake.sent[2] == testrun.prompt(CMD) and len(fake.sent) == 3
    assert outcome.result["summary"] == "Wrapped up" and forge.pushed == ["agent/fix-backup"]
    assert verdict_of(orch)["status"] == "passed" and orch.of("end")[0]["wrapped_up"] == "max_turns"


def test_the_test_turn_s_own_cap_does_not_end_the_completed_run(orch, client, worktree):
    outcome, fake, _ = run(claude(WORK_TURN, [bash(CMD), bash_result(), result(OTHER, subtype="error_max_turns")]),
                           client, worktree, caps=Caps(max_turns=1, wall_clock=30, grace=1))
    assert outcome.reason == "completed" and len(fake.sent) == 2
    assert verdict_of(orch)["status"] == "passed"


def test_a_prompt_queued_during_the_test_turn_is_sent_and_tested_again(orch, client, worktree):
    orch.when(lambda e: e["kind"] == "message" and e["payload"].get("purpose") == "tests",
              lambda e: orch.queue("one more thing", "queue"))
    slow = [bash(CMD), {"fake": "sleep", "seconds": 0.3}, bash_result(), result(OTHER)]
    outcome, fake, _ = run(claude(WORK_TURN, slow, [say("ok"), result(WORK)],
                                  [bash(CMD), bash_result(ok=False, code=2), result(OTHER)]), client, worktree)
    assert fake.sent[1:] == [testrun.prompt(CMD), "one more thing", testrun.prompt(CMD)]
    assert verdict_of(orch) == {"command": CMD, "status": "failed", "exit_code": 2}
    assert outcome.turns == 2


def test_codex_wrapped_command_passed(orch, client, worktree):
    outcome, _, _ = run(codex(CODEX_WORK, codex_turn(codex_cmd(CMD, phase="started", status="inProgress", code=None),
                                                     codex_cmd(CMD, cwd=str(worktree)), codex_msg(json.dumps(OTHER)))),
                        client, worktree)
    assert verdict_of(orch) == {"command": CMD, "status": "passed", "exit_code": 0}
    assert outcome.result == WORK


def test_codex_failed_with_its_exit_code(orch, client, worktree):
    item = codex_cmd(CMD, status="failed", code=1, cwd=str(worktree))
    run(codex(CODEX_WORK, codex_turn(item)), client, worktree)
    assert verdict_of(orch) == {"command": CMD, "status": "failed", "exit_code": 1}


def test_codex_quoted_command_unwraps(orch, client, worktree):
    cmd = "pytest -k 'not slow'"
    item = codex_cmd("x", cwd=str(worktree))
    item["params"]["item"]["command"] = "/usr/bin/bash -lc 'pytest -k '\"'\"'not slow'\"'\"''"
    run(codex(CODEX_WORK, codex_turn(item)), client, worktree, test_cmd=cmd)
    assert verdict_of(orch) == {"command": cmd, "status": "passed", "exit_code": 0}


def test_codex_actions_alone_do_not_match(orch, client, worktree):
    item = codex_cmd(CMD, cwd=str(worktree))
    item["params"]["item"]["command"] = f"/usr/bin/bash -lc 'cd sub && {CMD}'"
    run(codex(CODEX_WORK, codex_turn(item)), client, worktree)
    assert verdict_of(orch)["status"] == "not_run"


def test_codex_declined_is_not_run(orch, client, worktree):
    item = codex_cmd(CMD, status="declined", code=None, cwd=str(worktree))
    run(codex(CODEX_WORK, codex_turn(item)), client, worktree)
    assert verdict_of(orch)["status"] == "not_run"


def test_codex_from_another_directory_is_not_run(orch, client, worktree):
    run(codex(CODEX_WORK, codex_turn(codex_cmd(CMD, cwd="/tmp"))), client, worktree)
    assert verdict_of(orch)["status"] == "not_run"


def test_codex_extra_turn_at_max_turns(orch, client, worktree):
    outcome, fake, _ = run(codex(CODEX_WORK, codex_turn(codex_cmd(CMD, cwd=str(worktree)))), client, worktree,
                           caps=Caps(max_turns=1, wall_clock=30, grace=1))
    assert len(fake.sent) == 2 and outcome.reason == "completed" and outcome.turns == 1
    assert verdict_of(orch)["status"] == "passed"
    assert outcome.result == WORK


@pytest.mark.parametrize("recorded,expected", [
    ("printf '%s' 'a  b'", "printf '%s' 'a b'"),
    ("make\ntest", "make test"),
    ("make\rtest", "make test"),
    ("make\ttest", "make test"),
    ("make test\r\n", "make test"),
    ("make test\u2028", "make test"),
])
def test_command_matching_preserves_internal_whitespace_and_refuses_line_breaks(recorded, expected):
    assert not testrun.same(recorded, expected)
    assert testrun.same(f" \t{expected}  ", expected)


def test_bounded_golden_pins_the_bare_command_prompt():
    from conftest import load

    [text] = [payload["text"] for _, _, _, payload in load("bounded_run.golden.json")
              if payload.get("purpose") == "tests"]
    assert text == testrun.prompt("pytest -q")


# Directly on testrun.verdict, so it runs where sockets are refused too.
@pytest.mark.parametrize("provider", ["claude-code", "codex"])
def test_verdict_requires_the_exact_project_command_and_worktree_evidence(tmp_path, provider):
    worktree = tmp_path / "project's worktree"
    worktree.mkdir()
    cwd = str(worktree)
    prompt = testrun.prompt(CMD)
    assert "from the repository root, the directory this session started in," in prompt
    assert prompt.splitlines()[2] == CMD and "cd " not in prompt

    def frames(command, *, code=0, reported_cwd=cwd, after_cd=False):
        prior = "cd /tmp"
        if provider == "codex":
            return ([codex_cmd(prior, cwd=cwd)] if after_cd else []) + [
                codex_cmd(command, status="failed" if code else "completed", code=code, cwd=reported_cwd)]
        return ([bash(prior, "prior"), bash_result("prior")] if after_cd else []) + [
            bash(command), bash_result(ok=not code, code=code)]

    def verdict(raws, after="same", *, claude_project_cwd=True):
        return testrun.verdict(provider, raws, CMD, set(), "same", after, cwd,
                               claude_project_cwd=provider == "claude-code" and claude_project_cwd)

    assert verdict(frames(CMD)) == {
        "command": CMD, "status": "passed", "exit_code": 0 if provider == "codex" else None}
    assert verdict(frames(CMD, code=3)) == {"command": CMD, "status": "failed", "exit_code": 3}
    assert verdict(frames(CMD), "changed")["status"] == "tree_changed"
    assert verdict(frames(f"cd {shlex.quote(cwd)} && {CMD}"))["status"] == "not_run"
    assert verdict(frames(CMD.replace(" run ", "\trun ")))["status"] == "not_run"
    if provider == "codex":
        assert verdict(frames(CMD, after_cd=True, reported_cwd="/tmp"))["status"] == "not_run"
        for bad_cwd in (None, 7, "/tmp", "."):
            assert verdict(frames(CMD, reported_cwd=bad_cwd))["status"] == "not_run"
        alias = tmp_path / "alias"
        alias.symlink_to(worktree)
        assert verdict(frames(CMD, reported_cwd=str(alias)))["status"] == "passed"
    else:
        # The CLI puts the shell back in the worktree after each call, so an earlier cd does not carry over.
        assert verdict(frames(CMD, after_cd=True))["status"] == "passed"
        assert verdict(frames(CMD), claude_project_cwd=False)["status"] == "not_run"


def test_check_test_cmd():
    assert testrun.check_test_cmd(None) is None and testrun.check_test_cmd("") is None
    assert testrun.check_test_cmd("make test") == "make test"
    assert testrun.check_test_cmd("x" * 500) == "x" * 500
    for bad in ("x" * 501, "a\nb", "a\rb", "a\0b", "a b", 3, ["a"]):
        with pytest.raises(ValueError):
            testrun.check_test_cmd(bad)


CLAUDE_INTERPRETED = ["claude_exits_interpreted_recorded", "claude_exits_recorded_2_1_321"]


@pytest.mark.parametrize("provider,stem", [("claude", "claude_exits_recorded"), ("codex", "codex_exits_recorded"),
                                           ("claude", "claude_exits_interpreted_recorded"),
                                           ("codex", "codex_exits_interpreted_recorded"),
                                           ("claude", "claude_exits_recorded_2_1_321")])
def test_recorded_exit_codes(provider, stem):
    from conftest import FIXTURES
    data = json.loads((FIXTURES / f"{stem}.raw.json").read_text(encoding="utf-8"))
    raws = data["messages"]
    cwd = next((m["params"]["thread"]["cwd"] for m in raws if m.get("method") == "thread/started"), None)
    assert (cwd is not None) == (provider == "codex")
    interpreted = ("failed", 1 if provider == "codex" else None)
    expected = [("failed", 3), ("failed", 4), ("passed", 0 if provider == "codex" else None),
                interpreted, interpreted]
    from fixture_rules import EXIT_COMMANDS
    assert len(data["commands"]) in (3, len(expected))
    assert data["commands"] == list(EXIT_COMMANDS[:len(data["commands"])])
    expected = expected[:len(data["commands"])]
    # The recordings come from coding launches, so a Claude replay carries the provider's confirmation.
    name = "claude-code" if provider == "claude" else provider

    def verdict(command, denied, before="same", after="same"):
        return testrun.verdict(name, raws, command, denied, before, after, cwd or "/unused",
                               claude_project_cwd=provider == "claude")

    for command, (status, code) in zip(data["commands"], expected):
        assert verdict(command, set()) == {"command": command, "status": status, "exit_code": code}
    ok = data["commands"][2]
    assert verdict(ok, set(), "a", "b")["status"] == "tree_changed"
    call_id = next(b["id"] for m in raws if m.get("type") == "AssistantMessage" for b in m["content"]
                   if b.get("type") == "ToolUseBlock" and b["input"].get("command") == ok) if provider == "claude" \
        else next(m["params"]["item"]["id"] for m in raws if m.get("method") == "item/completed"
                  and m["params"]["item"].get("type") == "commandExecution" and ok in m["params"]["item"]["command"])
    assert verdict(ok, {call_id})["status"] == "not_run"


@pytest.mark.parametrize("stem", CLAUDE_INTERPRETED)
def test_recorded_interpreted_exits_are_not_errors_to_claude_code(stem):
    from conftest import FIXTURES
    data = json.loads((FIXTURES / f"{stem}.raw.json").read_text(encoding="utf-8"))
    assert len(data["commands"]) == 5
    ids = {b["input"]["command"]: b["id"] for m in data["messages"] if m.get("type") == "AssistantMessage"
           for b in m["content"] if b.get("type") == "ToolUseBlock"}
    results = {b["tool_use_id"]: (b, m["tool_use_result"]) for m in data["messages"] if m.get("type") == "UserMessage"
               for b in m["content"] if b.get("type") == "ToolResultBlock"}
    for command, reading in zip(data["commands"][3:], ("Condition is false", "No matches found")):
        block, output = results[ids[command]]
        assert block["is_error"] is False
        assert output["interrupted"] is False and output["returnCodeInterpretation"] == reading
    assert "returnCodeInterpretation" not in results[ids[data["commands"][2]]][1]


def test_the_runner_never_runs_the_command(monkeypatch, orch, client, worktree):
    import subprocess

    real = subprocess.run
    seen = []

    def watch(argv, *a, **kw):
        seen.append(argv)
        return real(argv, *a, **kw)
    monkeypatch.setattr(subprocess, "run", watch)
    run(claude(WORK_TURN, [bash(CMD), bash_result(), result(OTHER)]), client, worktree)
    assert seen and all(argv[0] == "git" for argv in seen)

