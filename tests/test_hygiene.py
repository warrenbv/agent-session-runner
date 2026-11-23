"""What the runner never does, checked on its source: third-party imports outside the two providers, the
environment outside config.py, a shell, subprocess outside gitdiff.py. Also the real providers by import
and construction only."""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "runner"
SDK_MODULES = {"claude_agent_sdk", "openai_codex"}


def _sources():
    return sorted(PACKAGE.rglob("*.py"))


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield node, [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node, [node.module or ""]


def test_standard_library_only_except_the_sdks_inside_their_providers():
    for path in _sources():
        tree = ast.parse(path.read_text())
        top_level = {id(n) for n in tree.body}
        for node, names in _imports(tree):
            for name in names:
                top = name.split(".")[0]
                if top in SDK_MODULES:
                    assert path.name in ("claude.py", "codex.py"), (path.name, name)
                    assert id(node) not in top_level, f"{path.name} imports {name} at module level"
                else:
                    assert top in sys.stdlib_module_names or top in ("runner", "__future__"), (path.name, name)


def test_environment_only_in_config_and_no_shell():
    for path in _sources():
        text = path.read_text()
        if path.name == "mcp_shim.py":
            assert text.count("os.environ") == 1 and "os.environ.get(SOCKET_ENV)" in text and "getenv" not in text
        elif path.name != "config.py":
            assert "os.environ" not in text and "getenv" not in text, path.name
        assert "shell=True" not in text and "os.system" not in text and "os.popen" not in text, path.name
        if path.name != "gitdiff.py":
            assert "subprocess" not in text, path.name


LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def _names(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            yield sub.id
        elif isinstance(sub, ast.Attribute):
            yield sub.attr
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def test_no_log_or_print_line_names_a_token():
    for path in _sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_log = isinstance(func, ast.Attribute) and func.attr in LOG_METHODS and \
                isinstance(func.value, ast.Name) and func.value.id in ("log", "logging")
            is_print = isinstance(func, ast.Name) and func.id == "print"
            if not (is_log or is_print):
                continue
            args = node.args[1:] if is_log else node.args
            for arg in [*args, *[k.value for k in node.keywords]]:
                for name in _names(arg):
                    assert "token" not in name.lower() and "brief" not in name.lower() \
                        and "secret" not in name.lower(), f"{path.name}:{node.lineno} logs {name}"


def test_the_fake_path_loads_no_sdk():
    import subprocess

    code = ("import sys, runner.__main__, runner.session, runner.providers.fake, runner.stub_control_plane,"
            " runner.demo;"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('claude_agent_sdk', 'openai_codex')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_the_sdk_pins_match_the_lock_file():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == ["claude-agent-sdk==0.2.180", "openai-codex==0.167.0"]
    lock = (ROOT / "uv.lock").read_text()
    assert 'name = "claude-agent-sdk"\nversion = "0.2.180"' in lock
    assert 'name = "openai-codex"\nversion = "0.167.0"' in lock


def _ctx(**kw):
    from runner.providers.base import RESULT_SCHEMA, ProviderContext

    async def never(_payload):
        raise AssertionError("not called")

    return ProviderContext(request_approval=never, cwd="/tmp/wt", output_schema=RESULT_SCHEMA, **kw)


def test_claude_options_are_built_from_the_plan():
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, ToolUseBlock
    from runner.providers.claude import ClaudeProvider, check_resume, steer_message, to_raw

    provider = ClaudeProvider()
    options = provider.options(_ctx(max_turns=20, max_budget_usd=2.0, allowed_tools=["Read", "Bash(pytest:*)"],
                                    disallowed_tools=["Bash(git push:*)"],
                                    resume="6F1D2C1E-0000-4000-8000-000000000001"))
    assert isinstance(options, ClaudeAgentOptions)
    assert (options.permission_mode, options.max_turns, options.max_budget_usd) == ("default", 20, 2.0)
    assert options.allowed_tools == ["Read", "Bash(pytest:*)"] and options.disallowed_tools == ["Bash(git push:*)", "WebSearch", "WebFetch"]
    assert options.resume == "6f1d2c1e-0000-4000-8000-000000000001"
    assert options.output_format["type"] == "json_schema" and options.setting_sources == ["user"]
    schema = options.output_format["schema"]
    assert "next_steps" in schema["required"]
    assert "tests_claim" in schema["required"]
    assert schema["properties"]["next_steps"] == {"type": "array", "items": {"type": "string"}}
    assert options.can_use_tool is not None
    with pytest.raises(ValueError):
        check_resume("--mcp-config=/tmp/x")
    with pytest.raises(ValueError):
        provider.options(_ctx(permission_mode="bypassPermissions"))
    assert steer_message("x")["priority"] == "next"
    raw = to_raw(AssistantMessage(content=[TextBlock("hi"), ToolUseBlock("t1", "Read", {"file_path": "a"})],
                                  model="m"))
    assert raw["type"] == "AssistantMessage"
    assert [b["type"] for b in raw["content"]] == ["TextBlock", "ToolUseBlock"]


@pytest.mark.parametrize("model,fable", [("claude-fable-5-2", True), ("claude-fable-5-2[1m]", True),
                                         ("claude-opus-5-5", False), ("claude-sonnet-4-6", False), (None, False)])
def test_a_fable_session_s_subagents_default_to_opus(model, fable):
    from runner.providers.claude import ClaudeProvider, is_fable

    assert is_fable(model) is fable
    options = ClaudeProvider().options(_ctx(model=model))
    assert options.model == (model or "claude-opus-5-5")
    env = options.env
    temp = options.add_dirs[0]
    pin = {"ANTHROPIC_DEFAULT_FABLE_MODEL": "claude-fable-5-2",
           "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1",
           "CLAUDE_CODE_TMPDIR": temp, "TMPDIR": temp,
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    assert env == ({**pin, "CLAUDE_CODE_SUBAGENT_MODEL": "claude-opus-5-5"} if fable else pin)
    assert "CLAUDE_CODE_SUBAGENT_MODEL_FORCE" not in env
    with_token = ClaudeProvider(oauth_token="tok-placeholder").options(_ctx(model=model))
    other = with_token.add_dirs[0]
    assert with_token.env == {"CLAUDE_CODE_OAUTH_TOKEN": "tok-placeholder", **env, "CLAUDE_CODE_TMPDIR": other,
                              "TMPDIR": other} and other != temp


@pytest.mark.parametrize("model", ["fable", "fable[1m]", "opus", "sonnet", "best", "default", "claude-fable-5-1",
                                   "claude-fable-5-1[1m]", "claude-fable-5", "fable-5-1", "claude-fable-5-3",
                                   "Claude-Opus-5-5", "claude-opus", 3])
def test_a_claude_start_on_fable_5_1_or_an_alias_is_refused(model):
    from runner.providers.claude import ClaudeProvider, model_refusal

    reason = model_refusal(model)
    assert reason and "was not started" in reason
    assert "Pick a model from the start form's Model list: Fable 5.2," in reason
    with pytest.raises(ValueError, match="was not started"):
        ClaudeProvider().options(_ctx(model=model))


def test_fable_5_2_and_other_full_ids_pass():
    from runner.providers.claude import model_refusal

    for model in (None, "claude-fable-5-2", "claude-fable-5-2[1m]", "claude-opus-5-5", "claude-haiku-5-5",
                  "claude-sonnet-4-5-20250929"):
        assert model_refusal(model) is None


def test_claude_takes_the_effort_and_the_cli_accepts_only_its_levels():
    from runner.__main__ import parse
    from runner.providers.claude import EFFORTS, ClaudeProvider

    assert ClaudeProvider().options(_ctx(effort="xhigh")).effort == "xhigh"
    assert ClaudeProvider().options(_ctx()).effort is None
    with pytest.raises(ValueError):
        ClaudeProvider().options(_ctx(effort="turbo"))
    base = ["--provider", "claude", "--cwd", "/tmp/wt", "--prompt-file", "/tmp/p"]
    assert [parse([*base, "--effort", e]).effort for e in EFFORTS] == list(EFFORTS)
    assert parse(base).effort is None and parse(base).base_ref is None
    commit = "0123456789abcdef0123456789abcdef01234567"
    assert parse([*base, "--base-ref", commit]).base_ref == commit
    for bad in (["--effort", "turbo"], ["--base-ref", "main"], ["--base-ref", "--upload-pack=x"]):
        with pytest.raises(SystemExit):
            parse([*base, *bad])


def test_codex_client_config_and_thread_params():
    from openai_codex import CodexConfig
    from runner.providers.codex import (CONFIG_OVERRIDES, MCP_SERVER, CodexProvider, projects_override,
                                        read_overrides)

    provider = CodexProvider()
    ctx = _ctx(model="gpt-5.5-codex")
    config = provider.config(ctx)
    assert isinstance(config, CodexConfig) and config.cwd == "/tmp/wt"
    assert config.config_overrides == (*CONFIG_OVERRIDES, projects_override("/tmp/wt"), MCP_SERVER,
                                       *read_overrides(provider.read_scope(ctx)))
    assert "analytics.enabled=false" in CONFIG_OVERRIDES
    assert "features.apps=false" in CONFIG_OVERRIDES
    params = provider.thread_params(ctx)
    assert params["approvalPolicy"] == "on-request" and params["approvalsReviewer"] == "user"
    assert "sandbox" not in params and params["model"] == "gpt-5.5-codex"
    assert params["config"]["default_permissions"] == "session_write"
    assert provider._on_server_request("item/tool/requestUserInput", {}) == {"answers": {}}

