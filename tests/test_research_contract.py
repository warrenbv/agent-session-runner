"""The research contract: the fixtures pin the exact inputs and outputs messages, which the runner must
parse and build exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from runner import research_out
from runner.control_plane import PART_BYTES, Rejected, ResearchClient
from runner.research_out import ResearchOut, split

FIXTURES = Path(__file__).parent / "fixtures" / "research_contract"
CONTRACT_DIGEST = "93ff919eadc61e2cdb8e83f6e35890acfefb5010abde351545807b3a452bfeab"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def digest(root: Path) -> str:
    lines = "".join(f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in sorted(root.glob("*.json")))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


class Scripted:

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def call(self, op, body):
        self.calls.append((op, json.loads(json.dumps(body))))
        queue = self.replies[op]
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if "status" in reply:
            raise Rejected(reply["status"], reply["body"]["error"]["code"])
        return reply


def scratch_with(tmp_path, *names):
    root = tmp_path / "scratch"
    root.mkdir()
    for name in names:
        part = load(f"output_part.{name.split('.')[0]}.request.json")
        (root / name).write_text(part["content"], encoding="utf-8")
    return root


def test_the_fixtures_are_the_pinned_set():
    assert digest(FIXTURES) == CONTRACT_DIGEST


def test_the_limits_are_the_runners():
    limits = load("limits.json")
    assert (PART_BYTES, research_out.MAX_PARTS, research_out.FILE_CAP, research_out.REASON_CAP) == (
        limits["part_bytes"], limits["max_parts"], limits["file_bytes"], limits["reason_chars"])
    assert len(split("€" * (research_out.FILE_CAP // 3))) <= research_out.MAX_PARTS
    assert research_out.STATUS_WORDS["too_large"] == "is larger than 512 KB"


def test_the_inputs_reply_is_joined_checked_and_written(tmp_path):
    client = Scripted({"research.inputs_get": [load("inputs_get.reply.json")]})
    out = ResearchOut(ResearchClient(client, 41), "lead_brief", str(tmp_path))
    assert out.fetch_inputs() == 1
    reply = load("inputs_get.reply.json")
    assert (tmp_path / "inputs" / reply["name"]).read_text(encoding="utf-8") == reply["content"]
    assert client.calls == [("research.inputs_get", load("inputs_get.request.json"))]


def test_a_step_with_no_inputs_is_total_zero(tmp_path):
    client = Scripted({"research.inputs_get": [load("inputs_get.none.reply.json")]})
    out = ResearchOut(ResearchClient(client, 41), "lead_report", str(tmp_path))
    assert out.fetch_inputs() == 0 and [p.name for p in (tmp_path / "inputs").iterdir()] == ["manifest.md"]


def test_the_output_requests_are_built_exactly_and_the_acceptance_read(tmp_path):
    client = Scripted({"research.output_part": [load("output_part.reply.json")],
                       "research.output_done": [load("output_done.reply.json")]})
    out = ResearchOut(ResearchClient(client, 41), "provider", str(scratch_with(tmp_path, "synthesis.md", "claims.json")))
    assert out.post_outputs("completed") == (False, None) and out.done and out.claims == 2
    assert client.calls == [("research.output_part", load("output_part.synthesis.request.json")),
                            ("research.output_part", load("output_part.claims.request.json")),
                            ("research.output_done", load("output_done.request.json"))]


def test_a_missing_file_is_named_and_its_refusal_ends_the_step_without_a_resend(tmp_path):
    client = Scripted({"research.output_part": [load("output_part.reply.json")],
                       "research.output_done": [load("output_done.missing.reply.json")]})
    out = ResearchOut(ResearchClient(client, 41), "provider", str(scratch_with(tmp_path, "claims.json")))
    partial, why = out.post_outputs("completed")
    assert [c for c in client.calls if c[0] == "research.output_done"] == [
        ("research.output_done", load("output_done.missing.request.json"))]
    assert partial and why.endswith(research_out.NOT_KEPT.format(reason=load("output_done.missing.reply.json")["reason"]))
    assert out.statuses == {"synthesis.md": "missing", "claims.json": "not_kept"}


def test_a_mismatch_is_resent_and_then_accepted(tmp_path):
    client = Scripted({"research.output_part": [load("output_part.reply.json")],
                       "research.output_done": [load("output_done.mismatch.reply.json"), load("output_done.reply.json")]})
    out = ResearchOut(ResearchClient(client, 41), "provider", str(scratch_with(tmp_path, "synthesis.md", "claims.json")))
    assert out.post_outputs("completed") == (False, None) and out.done
    ops = [op for op, _ in client.calls]
    assert ops == ["research.output_part", "research.output_part", "research.output_done"] * 2


@pytest.mark.parametrize("name", ["output_done.request.json", "output_done.missing.request.json",
                                  "output_done.mismatch.request.json", "output_done.challenge.request.json"])
def test_every_output_done_names_each_file_once_with_parts_and_hash_together(name):
    body = load(name)
    assert set(body) == {"session_id", "files", "partial", "reason"}
    role = "challenger" if "challenge" in name else "provider"
    assert [f["name"] for f in body["files"]] == list(research_out.OUTPUTS[role])
    assert all((f["parts"] == 0) == (f["sha256"] is None) for f in body["files"])


def test_the_challengers_inputs_two_syntheses_are_joined_checked_and_written(tmp_path):
    replies = load("challenger.inputs_get.replies.json")
    client = Scripted({"research.inputs_get": list(replies)})
    out = ResearchOut(ResearchClient(client, 41), "challenger", str(tmp_path))
    assert out.fetch_inputs() == len(replies) == 7
    for reply in replies:
        assert (tmp_path / "inputs" / reply["name"]).read_text(encoding="utf-8") == reply["content"]
    assert (tmp_path / "inputs" / "codex" / "synthesis.md").is_file()
    assert [c[1]["part"] for c in client.calls] == list(range(7))
    assert "`inputs/claude/claims.json`" in (tmp_path / "inputs" / "manifest.md").read_text()


def test_the_challengers_output_requests_are_built_exactly(tmp_path):
    client = Scripted({"research.output_part": [load("output_part.reply.json")],
                       "research.output_done": [load("output_done.challenge.reply.json")]})
    out = ResearchOut(ResearchClient(client, 41), "challenger",
                      str(scratch_with(tmp_path, "challenge.md", "verdicts.json")))
    assert out.post_outputs("completed") == (False, None) and out.done and out.claims == 2
    assert client.calls == [("research.output_part", load("output_part.challenge.request.json")),
                            ("research.output_part", load("output_part.verdicts.request.json")),
                            ("research.output_done", load("output_done.challenge.request.json"))]
    assert out.statuses == {"challenge.md": "posted", "verdicts.json": "posted"}

