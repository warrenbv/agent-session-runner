"""The normaliser against golden files, and the payload cap. After a deliberate change, regenerate with
`RUNNER_UPDATE_GOLDEN=1 uv run pytest tests/test_normalise.py` and read the golden diff before keeping it."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from runner import normalise
from runner.events import KINDS, MARK, PAYLOAD_CAP, Event, cap_payload, size, stub

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _changes(messages: list, item_id: str):
    for m in messages:
        item = (m.get("params") or {}).get("item") or {}
        if m.get("method") == "item/started" and item.get("type") == "fileChange" and item.get("id") == item_id:
            return item.get("changes")
    return None


def _approval(provider: str, request: dict, messages: list) -> dict:
    if provider == "claude-code":
        ctx = request["context"]
        return normalise.claude_approval(request["tool_name"], request["input"], ctx["tool_use_id"],
                                         ctx.get("agent_id"), ctx.get("title"))
    changes = _changes(messages, request["params"].get("itemId")) \
        if request["method"] == normalise.APPROVAL_METHODS[1] else None
    return normalise.codex_approval(request["method"], request["params"], changes)


def _render(provider: str, raw: dict) -> dict:
    fn = normalise.NORMALISERS[provider]
    return {"messages": [[{"kind": e.kind, "payload": e.payload} for e in fn(m)] for m in raw["messages"]],
            "approvals": [_approval(provider, a, raw["messages"]) for a in raw["approvals"]]}


STEMS = [("claude-code", "claude_turn"), ("codex", "codex_turn"),
         ("claude-code", "claude_recorded"), ("codex", "codex_recorded"),
         ("claude-code", "claude_recorded_2_1_321")]
CLAUDE_RECORDED = [("claude_recorded", 0.5, "claude-sonnet-5-5"),
                   ("claude_recorded_2_1_321", 0.5, "claude-opus-5-5")]


def _load(stem: str) -> dict:
    return json.loads((FIXTURES / f"{stem}.raw.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("provider,stem", STEMS)
def test_golden(provider, stem):
    raw = json.loads((FIXTURES / f"{stem}.raw.json").read_text(encoding="utf-8"))
    got = _render(provider, raw)
    golden = FIXTURES / f"{stem}.golden.json"
    # A missing golden fails. Only RUNNER_UPDATE_GOLDEN=1 writes or rewrites it.
    if os.environ.get("RUNNER_UPDATE_GOLDEN") == "1":
        golden.write_text(json.dumps(got, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    assert golden.exists(), f"{golden.name} is missing: run with RUNNER_UPDATE_GOLDEN=1 to write it"
    assert got == json.loads(golden.read_text(encoding="utf-8"))


@pytest.mark.parametrize("provider,stem", STEMS)
def test_golden_covers_the_provider_kinds(provider, stem):
    raw = json.loads((FIXTURES / f"{stem}.raw.json").read_text(encoding="utf-8"))
    kinds = {e["kind"] for events in _render(provider, raw)["messages"] for e in events}
    expected = set(KINDS) - {"diff", "end", "approval"} - ({"thinking"} if stem != "claude_turn" else set())
    assert kinds == expected


def test_claude_mapping_details():
    raw = json.loads((FIXTURES / "claude_turn.raw.json").read_text(encoding="utf-8"))
    got = _render("claude-code", raw)["messages"]
    assert got[1] == [] and got[13] == []
    assert [e["kind"] for e in got[2]] == ["thinking", "message", "tool"]
    assert got[2][0]["payload"] == {"text": "Look at the script first.", "id": "claude:0"}
    assert got[4][1] == {"kind": "subagent", "payload": {"phase": "requested", "tool_use_id": "toolu_02",
                                                          "subagent_type": "test-writer",
                                                          "description": "Write a test for the backup script"}}
    assert got[6][0]["payload"]["parent_tool_use_id"] == "toolu_02"
    assert got[7][0]["payload"]["output_preview"] == "3 passed in 0.41s"
    result = got[14]
    assert [e["kind"] for e in result] == ["turn", "usage"]
    assert result[0]["payload"]["structured_output"]["branch"] == "agent/fix-backup"
    assert result[1]["payload"]["total_cost_usd"] == 0.4215
    assert [e["kind"] for e in got[15]] == ["turn", "usage", "limit"]
    assert got[15][2]["payload"] == {"limit": "max_turns", "source": "provider", "subtype": "error_max_turns"}
    assert [e["kind"] for e in got[16]] == ["turn"] and got[16][0]["payload"]["interrupted"] is True


def _frame(windows, named="seven_day", utilization=None):
    return {"type": "RateLimitEvent", "uuid": "u", "session_id": "s",
            "rate_limit_info": {"status": "allowed", "resets_at": 1794567600, "rate_limit_type": named,
                                "utilization": utilization, "overage_status": "rejected",
                                "raw": {"unifiedWindows": windows}}}


def test_every_window_in_a_claude_frame_is_recorded():
    windows = {"five_hour": {"utilization": 0.25, "resetsAt": 1794268800, "status": "allowed"},
               "seven_day": {"utilization": 0.5, "resetsAt": 1794567600, "status": "allowed"},
               "seven_day_fable": {"utilization": 0.75, "resetsAt": 1794567600, "status": "allowed_warning"},
               "Not A Name!": {"utilization": 0.5, "resetsAt": 1},
               "seven_day_overage_included": "not a window"}
    events = normalise.claude(_frame(windows))
    assert [e.kind for e in events] == ["limit"] * 3
    payloads = [e.payload for e in events]
    assert payloads[0] == {"limit": "rate_window", "source": "provider", "window": "seven_day", "status": "allowed",
                           "utilization": 0.5, "resets_at": 1794567600, "overage_status": "rejected"}
    assert payloads[1:] == [
        {"limit": "rate_window", "source": "provider", "window": "five_hour", "status": "allowed",
         "utilization": 0.25, "resets_at": 1794268800},
        {"limit": "rate_window", "source": "provider", "window": "seven_day_fable", "status": "allowed_warning",
         "utilization": 0.75, "resets_at": 1794567600}]


def test_a_claude_frame_without_a_map_still_names_its_window():
    events = normalise.claude(_frame({}, named="five_hour", utilization=0.81))
    assert [e.payload["window"] for e in events] == ["five_hour"] and events[0].payload["utilization"] == 0.81
    frame = _frame(None)
    frame["rate_limit_info"]["raw"] = "odd"
    assert [e.payload["window"] for e in normalise.claude(frame)] == ["seven_day"]


def _flat(rendered: dict) -> list:
    return [e for events in rendered["messages"] for e in events]


@pytest.mark.parametrize("stem,utilization,model", CLAUDE_RECORDED)
def test_recorded_claude_session(stem, utilization, model):
    raw = _load(stem)
    got = _render("claude-code", raw)
    events = _flat(got)
    agent = next(e for e in events if e["kind"] == "tool" and e["payload"].get("name") == "Agent")
    spawn = agent["payload"]["id"]
    requested = next(e for e in events if e["kind"] == "subagent" and e["payload"]["phase"] == "requested")
    assert requested["payload"]["tool_use_id"] == spawn and requested["payload"]["subagent_type"] == "general-purpose"
    phases = [e["payload"]["phase"] for e in events if e["kind"] == "subagent"]
    assert phases == ["requested", "started", "progress", "completed"]
    inner = [e for e in events if e["kind"] == "tool" and e["payload"].get("parent_tool_use_id") == spawn]
    assert {e["payload"]["phase"] for e in inner} == {"started", "completed"}
    limits = [e["payload"] for e in events if e["kind"] == "limit" and e["payload"]["limit"] == "rate_window"]
    assert limits and all(p["utilization"] == utilization and p["window"] == "seven_day" for p in limits)
    ends = [e["payload"] for e in events if e["kind"] == "turn" and e["payload"]["phase"] == "completed"]
    assert [p["subtype"] for p in ends] == ["success", "error_max_turns"]
    assert ends[1]["terminal_reason"] == "max_turns" and ends[1]["errors"] == ["Reached maximum number of turns (1)"]
    assert any(e["kind"] == "limit" and e["payload"]["limit"] == "max_turns" for e in events)
    usage = [e["payload"] for e in events if e["kind"] == "usage"]
    assert usage[0]["total_cost_usd"] > 0 and model in usage[0]["model_usage"]
    inits = [m["data"]["model"] for m in raw["messages"] if m.get("type") == "SystemMessage"
             and m.get("subtype") == "init"]
    assert inits == [model, model]
    write, network = got["approvals"]
    started_ids = {e["payload"]["id"] for e in events if e["kind"] == "tool" and e["payload"]["phase"] == "started"}
    assert write["tool"] == "Write" and write["request_id"] in started_ids
    assert network["tool"] == "SandboxNetworkAccess" and json.loads(network["input_preview"]) == {"host": "example.com"}


def test_recorded_codex_session():
    raw = _load("codex_recorded")
    got = _render("codex", raw)
    events = _flat(got)
    subagents = [e["payload"] for e in events if e["kind"] == "subagent"]
    activity = [p for p in subagents if "agent_thread_id" in p]
    assert [p["phase"] for p in activity] == ["started", "completed"]
    assert len({p["agent_thread_id"] for p in activity}) == 1 and activity[0]["agent_path"]
    assert any(p.get("tool") == "wait" for p in subagents)
    messages = [e["payload"] for e in events if e["kind"] == "message"]
    assert messages[-1]["text"] == "DONE" and messages[-1]["phase"] == "final_answer"
    tools = [e["payload"] for e in events if e["kind"] == "tool" and e["payload"]["phase"] == "completed"]
    assert [t["status"] for t in tools if t["name"] == "fileChange"] == ["completed", "declined"]
    assert json.loads(tools[1]["input_preview"])[0]["kind"] == "add"
    declined = [t for t in tools if t["status"] == "declined"]
    assert len(declined) == 2 and all(t["is_error"] for t in declined)
    command, patch = got["approvals"]
    ids = {t["id"] for t in tools}
    assert command["tool"] == "command" and command["request_id"] in ids
    assert patch["tool"] == "fileChange" and patch["request_id"] in ids
    assert json.loads(patch["input_preview"])["changes"][0]["path"].endswith("by-patch.txt")
    assert json.loads(command["input_preview"])["reason"]
    usage = [e["payload"] for e in events if e["kind"] == "usage"]
    assert usage and all(p["total"]["totalTokens"] > 0 and p["model_context_window"] for p in usage)
    limits = [e["payload"] for e in events if e["kind"] == "limit"]
    assert limits and all(p["limit_id"] == "codex" and p["primary"]["usedPercent"] >= 0 for p in limits)
    end = [e["payload"] for e in events if e["kind"] == "turn" and e["payload"]["phase"] == "completed"]
    assert len(end) == 1 and end[0]["status"] == "completed" and end[0]["duration_ms"] > 0


def test_the_codex_provider_hands_a_file_approval_its_changes():
    from runner.providers.codex import CodexProvider
    provider = CodexProvider()
    item = {"type": "fileChange", "id": "exec-1", "changes": [{"path": "/w/a.txt", "kind": {"type": "add"},
                                                                 "diff": "hi\n"}]}
    provider.seen({"method": "item/started", "params": {"item": item}})
    assert provider.changes_for("exec-1") == item["changes"]
    assert provider.changes_for("exec-1", wait=0.01) is None
    assert provider.changes_for("exec-unknown", wait=0.01) is None


@pytest.mark.parametrize("stem", ["claude_recorded", "claude_recorded_2_1_321", "claude_turn"])
def test_the_claude_provider_names_completed_tools_and_changes_nothing_else(stem):
    from runner.providers.claude import ClaudeProvider
    provider = ClaudeProvider()
    messages = _load(stem)["messages"]
    started = {}
    for m in messages:
        plain, named = normalise.claude(m), provider.normalise(m)
        assert len(plain) == len(named)
        for a, b in zip(plain, named):
            if a.kind == "tool" and a.payload["phase"] == "started":
                started[a.payload["id"]] = a.payload["name"]
            if a.kind == "tool" and a.payload["phase"] == "completed" and a.payload["id"] in started:
                assert "name" not in a.payload
                assert b.payload == {**a.payload, "name": started[a.payload["id"]]}
            elif a.kind == "turn" and a.payload["phase"] == "completed":
                extra = {k: v for k, v in b.payload.items() if k not in a.payload}
                assert set(extra) <= {"context_tokens", "context_window"}
                assert {k: b.payload[k] for k in a.payload} == a.payload
            else:
                assert b == a
    assert started


def test_tool_names_are_forgotten_once_used_and_bounded():
    from runner.providers.base import ToolNames
    names = ToolNames()
    tool = lambda phase, i, **kw: Event("tool", {"phase": phase, "id": i, **kw})
    names.annotate([tool("started", "a", name="Read")])
    assert names.annotate([tool("completed", "a")])[0].payload["name"] == "Read"
    assert "name" not in names.annotate([tool("completed", "a")])[0].payload
    assert names.annotate([tool("completed", "z", name="command")])[0].payload["name"] == "command"
    names.annotate([tool("started", str(i), name="Bash") for i in range(ToolNames.LIMIT + 5)])
    assert len(names._names) == ToolNames.LIMIT


def test_the_codex_provider_drops_repeated_rate_limit_frames():
    from runner.providers.codex import CodexProvider
    provider = CodexProvider()
    frames = [m for m in _load("codex_recorded")["messages"] if m["method"] == "account/rateLimits/updated"]
    assert len(frames) > 1
    kept = [f for f in frames if not provider.repeated(f)]
    assert kept == frames[:1]
    changed = {"method": frames[0]["method"], "params": {"rateLimits": {"limitId": "codex", "primary": {}}}}
    assert not provider.repeated(changed) and not provider.repeated({"method": "turn/started"})


def test_recorded_fixtures_hold_no_secret_shapes():
    from fixture_rules import HOME_DETAIL, SECRET_VALUES
    for stem in ("claude_recorded", "codex_recorded", "claude_recorded_2_1_321", "claude_exits_recorded_2_1_321"):
        text = (FIXTURES / f"{stem}.raw.json").read_text(encoding="utf-8")
        assert not SECRET_VALUES.search(text) and not HOME_DETAIL.search(text)


def test_approval_payloads_fit_the_control_plane():
    for provider, stem in STEMS:
        raw = json.loads((FIXTURES / f"{stem}.raw.json").read_text(encoding="utf-8"))
        for payload in _render(provider, raw)["approvals"]:
            assert isinstance(payload["request_id"], str) and 0 < len(payload["request_id"]) <= 100
            assert isinstance(payload["tool"], str) and 0 < len(payload["tool"]) <= 200
            assert len(payload.get("input_preview", "")) <= 2000
            assert "requester" not in payload or 0 < len(payload["requester"]) <= 100


def test_payload_cap_truncates_and_marks():
    big = {"tool": "Bash", "output": "x" * 40000, "lines": ["y" * 9000, "short"], "n": 3}
    capped = cap_payload(big)
    assert size(capped) <= PAYLOAD_CAP and capped["truncated"] is True
    assert capped["n"] == 3 and capped["tool"] == "Bash" and capped["lines"][1] == "short"
    assert capped["output"].endswith(MARK)
    assert cap_payload(big) == capped
    small = {"text": "fine"}
    assert cap_payload(small) is small
    wide = cap_payload({"text": "é" * 12000})
    assert size(wide) <= PAYLOAD_CAP and wide["truncated"] is True


def test_payload_cap_falls_back_to_a_stub_for_many_small_strings():
    many = {f"k{i}": "z" * 60 for i in range(600)}
    capped = cap_payload(many)
    assert size(capped) <= PAYLOAD_CAP and capped["truncated"] is True
    assert size(stub(many)) <= PAYLOAD_CAP


def test_normalised_payloads_never_exceed_the_cap():
    huge_input = {"command": "echo " + "a" * 50000}
    events = normalise.claude({"type": "AssistantMessage", "content": [
        {"type": "ToolUseBlock", "id": "t", "name": "Bash", "input": huge_input},
        {"type": "TextBlock", "text": "b" * 50000}]})
    events += normalise.claude({"type": "ResultMessage", "subtype": "success", "structured_output": {
        "summary": "c" * 30000, "files_changed": ["d" * 9000] * 3}})
    for event in events:
        assert size(event.payload) <= PAYLOAD_CAP
    assert any(e.payload.get("truncated") for e in events)


def test_unknown_messages_are_dropped_and_kinds_are_checked():
    assert normalise.claude({"type": "SomethingNew"}) == []
    assert normalise.codex({"method": "thread/name/updated", "params": {}}) == []
    with pytest.raises(ValueError):
        Event("stdout", {})

