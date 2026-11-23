"""The tool names a web search or a page fetch takes on the tool events the runner posts."""

from __future__ import annotations

import pathlib
import re

import pytest

from runner import normalise

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
WEB_TOOLS = ("WebFetch", "WebSearch", "webSearch")


def test_claude_code_offers_its_web_tools_under_these_names():
    assert {"WebFetch", "WebSearch"} <= set(re.findall(r'"(Web[A-Z][a-z]+)"', (
        FIXTURES / "claude_recorded_2_1_321.raw.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize("name", ["WebSearch", "WebFetch"])
def test_a_claude_web_tool_use_is_a_started_tool_event_with_its_name(name):
    raw = {"type": "AssistantMessage", "content": [{"type": "ToolUseBlock", "id": "toolu_1", "name": name,
                                                    "input": {"query": "kiln glaze"}}]}
    [event] = [e for e in normalise.claude(raw) if e.kind == "tool"]
    assert (event.payload["phase"], event.payload["name"]) == ("started", name)


@pytest.mark.parametrize("phase", ["started", "completed"])
@pytest.mark.parametrize("action", [{"type": "search", "query": "kiln glaze"},
                                    {"type": "openPage", "url": "https://example.org/glaze"}])
def test_a_codex_web_search_item_is_a_tool_event_named_webSearch(phase, action):
    raw = {"method": f"item/{phase}", "params": {"item": {"type": "webSearch", "id": "ws_1", "query": "kiln glaze",
                                                          "action": action, "status": "completed"}}}
    [event] = normalise.codex(raw)
    assert (event.kind, event.payload["name"]) == ("tool", "webSearch")


def test_the_names_are_the_control_planes_list():
    assert sorted(WEB_TOOLS) == list(WEB_TOOLS)
    assert set(WEB_TOOLS) == {"WebFetch", "WebSearch", "webSearch"}

