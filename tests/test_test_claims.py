"""Typed test claims compared with the recorded verdicts, including the driver's final check."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from conftest import typed_result
from runner import handoff, handoff_check as hc, testrun
from runner.events import Event
from runner.providers.base import (RESULT_SCHEMA, TEST_CLAIM_DESCRIPTION, ProviderContext, tidy_result,
                                   tidy_test_claim)
from runner.session import Options, Outcome, Session, wrap_up_prompt
from helpers import CodexStub, bash, bash_result, codex_cmd, codex_go as _codex_go, codex_on_stub as _codex

CMD = "pytest -q"
FAILED = "The session reports that tests passed, but the runner recorded a failed final test check."
NOT_RUN = "The session reports that tests passed, but the runner recorded no verified final test run."


def claimed(**fields):
    return typed_result(tests_claim={"command": CMD, "status": "passed"}, **fields)


def test_schema_and_both_closing_prompts_describe_the_required_claim_in_one_sentence():
    assert TEST_CLAIM_DESCRIPTION == (
        "Set tests_claim to an object containing command and status for your own check of the exact configured "
        "final test command before the runner's independent test turn, using passed, failed, not_run or "
        "unverified, with an empty command and status none when no command is configured "
        "or you make no claim about it.")
    assert TEST_CLAIM_DESCRIPTION.count(".") == 1
    assert TEST_CLAIM_DESCRIPTION in handoff.SUMMARY_PROMPT
    assert TEST_CLAIM_DESCRIPTION in wrap_up_prompt("asked")
    assert set(RESULT_SCHEMA["required"]) == set(RESULT_SCHEMA["properties"])
    field = RESULT_SCHEMA["properties"]["tests_claim"]
    assert field["description"] == TEST_CLAIM_DESCRIPTION
    obj = field
    assert obj["type"] == "object"
    assert obj["required"] == ["command", "status"] and obj["additionalProperties"] is False
    assert obj["properties"]["status"]["enum"] == ["passed", "failed", "not_run", "unverified", "none"]
    assert obj["properties"]["command"] == {"type": "string"}


def test_result_schema_uses_only_the_established_provider_keywords():
    allowed = {"type", "properties", "required", "additionalProperties", "enum", "items", "description"}
    pending = [RESULT_SCHEMA]
    while pending:
        node = pending.pop()
        assert set(node) <= allowed
        assert node["type"] in ("object", "array", "string")
        pending.extend(node.get("properties", {}).values())
        if "items" in node:
            pending.append(node["items"])


@pytest.mark.parametrize("value", [
    {"command": "", "status": "none"}, {"command": CMD, "status": "none"},
    *[{"command": "", "status": status} for status in ("passed", "failed", "not_run", "unverified")]])
def test_none_or_an_empty_command_is_no_claim_without_changing_intake(value):
    assert tidy_test_claim(value) is None
    res = typed_result(tests_claim=value)
    assert tidy_result(res)["tests_claim"] is None
    assert not hc.test_claim_findings(res, {"command": CMD, "status": "failed", "exit_code": 1}, CMD)


@pytest.mark.parametrize("value", [None, True, "passed", [], {}, {"status": "passed"},
    {"command": CMD}, {"command": CMD, "status": "maybe"}, {"command": CMD, "status": []},
    {"command": CMD, "status": "passed", "extra": 1},
    *[{"command": c, "status": "passed"} for c in (
        None, 1, "", "  ", "a" * 501, "pytest\n-q", "pytest\r-q", "pytest\0-q", "pytest\u2028-q")]])
def test_invalid_claims_are_unknown_and_never_generate_a_disagreement(value):
    assert tidy_test_claim(value) is None
    result = typed_result(tests_claim=value, summary="Tests passed", tests_run=[CMD + " passed"])
    assert tidy_result(result)["tests_claim"] is None
    assert not hc.test_claim_findings(result, {"command": CMD, "status": "failed", "exit_code": 1}, CMD)


def test_claim_bounds_whitespace_and_older_results():
    for status in ("passed", "failed", "not_run", "unverified"):
        assert tidy_test_claim({"command": " pytest\t -q ", "status": status}) == {
            "command": CMD, "status": status}
    assert tidy_test_claim({"command": "a" * 500, "status": "passed"})["command"] == "a" * 500
    assert tidy_result({"summary": "Tests passed"}) == {"summary": "Tests passed"}


def test_both_provider_requests_carry_the_claim_schema(tmp_path):
    from runner.providers.claude import ClaudeProvider

    async def deny(_payload):
        raise AssertionError("no approval expected")

    ctx = ProviderContext(request_approval=deny, cwd=str(tmp_path), output_schema=RESULT_SCHEMA)
    options = ClaudeProvider().options(ctx)
    assert options.output_format == {"type": "json_schema", "schema": RESULT_SCHEMA}
    stub = CodexStub()
    provider = _codex(stub)
    provider.ctx.output_schema = RESULT_SCHEMA
    _codex_go(provider, "turn")
    assert stub.started == [{"outputSchema": RESULT_SCHEMA}]


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("status,key,text", [
    ("failed", "tests_claim_failed", FAILED), ("not_run", "tests_claim_not_run", NOT_RUN),
    ("passed", None, None), ("tree_changed", None, None)])
def test_provider_records_generate_only_the_two_fixed_findings(tmp_path, provider, status, key, text):
    cwd = str(tmp_path)
    raws = [codex_cmd(CMD, code=1 if status == "failed" else 0, cwd=cwd)] if provider == "codex" else [
        bash(CMD), bash_result(ok=status != "failed")]
    if status == "not_run":
        raws = []
    tests = testrun.verdict(provider, raws, CMD, set(), "before",
                            "after" if status == "tree_changed" else "before", cwd,
                            claude_project_cwd=provider == "claude-code")
    assert tests["status"] == status
    expected = [{"key": key, "text": text}] if key else []
    assert hc.TESTS_CLAIM_FAILED == FAILED and hc.TESTS_CLAIM_NOT_RUN == NOT_RUN
    checks = hc.check("Work", claimed(), hc.Status(), tests=tests, test_cmd=CMD)
    assert checks["findings"] == expected
    body = handoff.render(claimed(), checks=checks)
    assert (text in body) if text else "## What the runner checked\nNothing found." in body


@pytest.mark.parametrize("claim_status", ["failed", "not_run", "unverified"])
def test_other_claim_statuses_say_nothing_new(claim_status):
    res = typed_result(tests_claim={"command": CMD, "status": claim_status}, summary="Passed all tests")
    assert not hc.test_claim_findings(res, {"command": CMD, "status": "failed", "exit_code": 1}, CMD)


def test_claim_command_matching_is_exact_after_whitespace_normalisation():
    tests = {"command": "pytest\t-q", "status": "failed", "exit_code": 1}
    for command in ("pytest", CMD + " -k focused"):
        res = typed_result(tests_claim={"command": command, "status": "passed"})
        assert not hc.test_claim_findings(res, tests, CMD)
    assert hc.test_claim_findings(claimed(), tests, " pytest  -q ") == [
        {"key": "tests_claim_failed", "text": FAILED}]


@pytest.mark.parametrize("tests,command", [
    (None, CMD), ({}, CMD), ({"command": CMD, "status": "failed"}, CMD),
    ({"command": CMD, "status": "failed", "exit_code": True}, CMD),
    ({"command": CMD, "status": "failed", "exit_code": "1"}, CMD),
    ({"command": CMD, "status": "failed", "exit_code": 65536}, CMD),
    ({"command": CMD, "status": "unknown", "exit_code": None}, CMD),
    ({"command": CMD + " -k focused", "status": "failed", "exit_code": 1}, CMD),
    ({"command": "pytest\n-q", "status": "failed", "exit_code": 1}, CMD),
    ({"command": CMD, "status": "failed", "exit_code": 1}, None),
    ({"command": CMD, "status": "failed", "exit_code": 1}, "pytest")])
def test_unknown_verdicts_and_different_commands_say_nothing_new(tests, command):
    assert not hc.test_claim_findings(claimed(), tests, command)
    no_claim = typed_result(summary="All tests passed", tests_run=[CMD + " passed"])
    assert not hc.test_claim_findings(no_claim, tests, command)


@pytest.mark.parametrize("baseline", [True, False])
def test_final_driver_check_keeps_work_result_and_never_repairs_a_disagreement(tmp_path, monkeypatch, baseline):
    session = Session(SimpleNamespace(name="codex"), SimpleNamespace(),
                      Options(cwd=str(tmp_path), test_cmd=CMD, baseline=baseline))
    monkeypatch.setattr(handoff, "worktree_status", lambda *args: hc.Status())
    monkeypatch.setattr("runner.session.placeholders.ignorable", lambda *args: ())
    session.result = claimed()
    session._brief = "Work"

    async def check_sequence():
        session.tests = {"command": CMD, "status": "failed", "exit_code": 1}
        await session._handoff_check(repair=True)
        assert session.checks["findings"] == [] and not session._repaired

        async def recorded_turn(prompt):
            assert prompt.text == testrun.prompt(CMD)
            session._test_raws = [codex_cmd(CMD, code=1, cwd=str(tmp_path))]
            session._observe(Event("turn", {"phase": "completed", "structured_output": typed_result()}))
            return "same", "same"

        async def posted(event):
            assert event.payload["purpose"] == "tests-verdict"

        monkeypatch.setattr(session, "_turn", recorded_turn)
        monkeypatch.setattr(session.poster, "post", posted)
        await session._test_turn(CMD)
        await session._handoff_check(final=True)

    asyncio.run(check_sequence())
    assert session.result == claimed() and not session._repaired
    assert session.checks["findings"] == [{"key": "tests_claim_failed", "text": FAILED}]
    ending = session._end_payload(Outcome("completed", 1, None, session.result))
    assert ending["checks"] == session.checks and ending["result"] == claimed()
    assert FAILED in handoff.render(session.result, checks=session.checks)

