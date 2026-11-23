"""The web tools in coding sessions: allowed outright only for a session given the web, refused
otherwise, on both providers."""

from __future__ import annotations

import asyncio

from runner.providers.base import CODING_KIND, RESEARCH_KIND, ProviderContext
from runner.providers.fake import FakeProvider


async def _never(_payload):
    raise AssertionError("not called")


def _pctx(**kw):
    return ProviderContext(request_approval=_never, cwd="/tmp/wt", **kw)


MODES = ("default", "acceptEdits", "plan", "auto")


def test_a_coding_session_given_the_web_has_both_tools_allowed_outright_in_every_mode():
    from runner.providers.claude import WEB_TOOLS, ClaudeProvider

    assert WEB_TOOLS == ("WebSearch", "WebFetch") and _pctx().kind == CODING_KIND
    for mode in MODES:
        options = ClaudeProvider().options(_pctx(permission_mode=mode, web=True, allowed_tools=["Read"],
                                                 disallowed_tools=["Bash(git push:*)"]))
        assert options.allowed_tools == ["Read", "WebSearch", "WebFetch"]
        assert options.disallowed_tools == ["Bash(git push:*)"]
    with_tools = ClaudeProvider().options(_pctx(web=True, mcp_socket="/tmp/s"))
    lead_tools = ["mcp__platform__workers_start", "mcp__platform__worker_changes", "mcp__platform__worker_send_back",
                  "mcp__platform__worker_merge"]
    assert with_tools.allowed_tools[-2:] == ["WebSearch", "WebFetch"]
    assert with_tools.disallowed_tools == ["mcp__platform__sessions_propose", *lead_tools]
    proposing = ClaudeProvider().options(_pctx(web=True, mcp_socket="/tmp/s", propose=True))
    assert proposing.allowed_tools[-2:] == ["WebSearch", "WebFetch"] and proposing.disallowed_tools == lead_tools


def test_a_session_not_given_the_web_or_without_the_field_is_refused_both_in_every_mode():
    from runner.providers.claude import ClaudeProvider

    for given in ({"web": False}, {}):
        for mode in MODES:
            options = ClaudeProvider().options(_pctx(permission_mode=mode, disallowed_tools=["Bash(git push:*)"], **given))
            assert options.disallowed_tools == ["Bash(git push:*)", "WebSearch", "WebFetch"]
            assert "WebSearch" not in options.allowed_tools and "WebFetch" not in options.allowed_tools
    assert ProviderContext(request_approval=_never, cwd="/tmp/wt").web is False
    assert "WebSearch" in ClaudeProvider().options(_pctx(web="true")).disallowed_tools


def test_a_web_tool_already_named_is_not_repeated_and_the_research_kind_has_its_own_tool_set():
    from runner.providers.claude import RESEARCH_TOOLS, ClaudeProvider, allowed_web_tools, refused_web_tools, research_rules

    assert ClaudeProvider().options(_pctx(disallowed_tools=["WebFetch"])).disallowed_tools == ["WebFetch", "WebSearch"]
    assert ClaudeProvider().options(_pctx(web=True, allowed_tools=["WebSearch"])).allowed_tools == ["WebSearch", "WebFetch"]
    denied = ClaudeProvider().options(_pctx(web=True, disallowed_tools=["WebFetch"]))
    assert denied.allowed_tools == ["WebSearch"] and denied.disallowed_tools == ["WebFetch"]
    for web_field in (True, False):
        assert refused_web_tools(_pctx(kind=RESEARCH_KIND, web=web_field)) == []
        assert allowed_web_tools(_pctx(kind=RESEARCH_KIND, web=web_field)) == []
        built = ClaudeProvider().options(_pctx(kind=RESEARCH_KIND, web=web_field))
        assert built.disallowed_tools == [] and built.allowed_tools == research_rules("/tmp/wt")
        assert built.tools == list(RESEARCH_TOOLS)


def test_codex_web_search_is_live_only_for_a_coding_session_given_the_web_and_a_research_session():
    from runner.providers.codex import CONFIG_OVERRIDES, CodexProvider

    assert 'web_search="disabled"' in CONFIG_OVERRIDES
    for socket in (None, "/tmp/s"):
        live = CodexProvider().config(_pctx(web=True, mcp_socket=socket)).config_overrides
        assert 'web_search="live"' in live and 'web_search="disabled"' not in live
        assert "sandbox_workspace_write.network_access=false" in live
        for given in ({"web": False}, {}):
            off = CodexProvider().config(_pctx(mcp_socket=socket, **given)).config_overrides
            assert 'web_search="disabled"' in off and 'web_search="live"' not in off
        for web in (True, False):
            research = CodexProvider().config(_pctx(mcp_socket=socket, web=web, kind=RESEARCH_KIND)).config_overrides
            assert 'web_search="live"' in research and "features.shell_tool=false" in research


def test_the_runner_takes_web_from_its_command_line_into_the_provider_context(tmp_path):
    from runner.__main__ import parse
    from runner.session import Options, Session

    base = ["--provider", "fake", "--cwd", "/tmp", "--prompt-file", "/tmp/p"]
    assert parse(base + ["--web"]).web is True and parse(base).web is False
    assert Options(cwd="/tmp").web is False
    seen = {}

    class Grab(FakeProvider):
        async def start(self, ctx):
            seen["web"] = ctx.web
            raise RuntimeError("enough")

    for flag in (True, False):
        session = Session(Grab({"provider": "claude-code"}), client=None, options=Options(cwd=str(tmp_path), web=flag))
        asyncio.run(session.run("hi"))
        assert seen.pop("web") is flag

