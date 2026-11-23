"""The token from the environment or a 0600 file, the environment the CLIs inherit trimmed to a list,
the control-plane URL rules, and the command line end to end on the fake provider."""

from __future__ import annotations

import os

import pytest

from conftest import FIXTURES
from runner.stub_control_plane import TOKEN
from runner import __main__ as cli
from runner import config


def test_token_from_the_environment_and_secrets_stripped():
    env = {"RUNNER_TOKEN": "tok-1", "RUNNER_GITHUB_TOKEN": "gh-1", "ANTHROPIC_API_KEY": "sk-placeholder",
           "PATH": "/usr/bin"}
    settings = config.take_secrets(env)
    assert (settings.token, settings.github_token) == ("tok-1", "gh-1")
    assert settings.control_url == "https://control.example.internal"
    assert env == {"PATH": "/usr/bin"}
    assert "tok-1" not in repr(settings) and "gh-1" not in repr(settings)


def test_token_file_must_be_0600(tmp_path):
    path = tmp_path / "token"
    path.write_text("tok-file\n")
    path.chmod(0o600)
    assert config.take_secrets({"RUNNER_TOKEN_FILE": str(path)}).token == "tok-file"
    path.chmod(0o640)
    with pytest.raises(ValueError, match="0600") as err:
        config.take_secrets({"RUNNER_TOKEN_FILE": str(path)})
    assert "tok-file" not in str(err.value)
    link = tmp_path / "link"
    path.chmod(0o600)
    link.symlink_to(path)
    with pytest.raises(OSError):
        config.read_token_file(str(link))


def test_a_token_is_required_and_one_line(tmp_path):
    with pytest.raises(ValueError):
        config.take_secrets({})
    path = tmp_path / "token"
    path.write_text("two\nlines\n")
    path.chmod(0o600)
    with pytest.raises(ValueError):
        config.take_secrets({"RUNNER_TOKEN_FILE": str(path)})


@pytest.mark.parametrize("url", ["http://control.example.internal", "ftp://x", "https://u:p@control.example.internal",
                                 "https://control.example.internal/?a=1", "http://192.0.2.5:8080"])
def test_control_url_rules(url):
    with pytest.raises(ValueError):
        config.check_url(url)


def test_loopback_http_and_https_are_accepted():
    assert config.check_url("https://control.example.internal/") == "https://control.example.internal"
    for url in ("http://127.0.0.1:8080", "http://127.0.0.2:8080", "http://localhost:8080", "http://[::1]:8080"):
        assert config.check_url(url) == url


def _refusal(host):
    return ("RUNNER_CONTROL_URL may use http only to the loopback, by the name localhost or a loopback address, "
            f"and {host} is neither, so use https")


# An address of one of this machine's own interfaces is refused too, whatever it is now, because it can
# later belong to another machine. The check makes no probe of this machine's interfaces at all.
def test_http_to_this_machines_own_interface_address_is_refused(monkeypatch):
    import socket

    def no_probe(*args, **kwargs):
        raise AssertionError("check_url probed a socket")

    monkeypatch.setattr(socket, "socket", no_probe)
    for host in ("192.0.2.10", "198.51.100.7", "2001:db8::7", "fe80::1", "0.0.0.0", "::"):
        url = f"http://[{host}]:8080" if ":" in host else f"http://{host}:8080"
        with pytest.raises(ValueError) as refused:
            config.check_url(url)
        assert str(refused.value) == _refusal(host)
    assert not hasattr(config, "is_local_address")


def test_http_to_any_other_machine_is_refused_and_https_to_it_is_accepted():
    from urllib.parse import urlsplit

    for url in ("http://192.0.2.10:8080", "http://192.0.2.10:8081", "http://192.0.2.10", "http://192.0.2.11:8080",
                "http://control.example.internal:8080", "http://localhost.example.internal:8080",
                "http://203.0.113.254:8080"):
        with pytest.raises(ValueError) as refused:
            config.check_url(url)
        assert str(refused.value) == _refusal(urlsplit(url).hostname)
    assert not hasattr(config, "CONTROL_PLANE_LAN")
    assert config.check_url("https://192.0.2.10:8080/") == "https://192.0.2.10:8080"


def test_the_host_header_is_sent_when_configured_and_absent_otherwise(orch, monkeypatch):
    from runner.control_plane import Client

    Client(orch.url, TOKEN, timeout=5, host="control.example.internal").call("sessions.pull", {})
    Client(orch.url, TOKEN, timeout=5).call("sessions.pull", {})
    assert orch.hosts[0] == "control.example.internal"
    assert orch.hosts[1] == orch.url.removeprefix("http://")
    env = {"RUNNER_TOKEN": "t", "RUNNER_CONTROL_HOST": "control.example.internal"}
    assert config.take_secrets(env).control_host == "control.example.internal"
    assert config.take_secrets({"RUNNER_TOKEN": "t"}).control_host is None
    with pytest.raises(ValueError, match="RUNNER_CONTROL_HOST"):
        config.take_secrets({"RUNNER_TOKEN": "t", "RUNNER_CONTROL_HOST": "control.example.internal\r\nX: y"})


def test_the_command_line_runs_a_fake_session(orch, worktree, tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Fix the backup job.")
    orch.when(lambda e: e["kind"] == "approval", lambda e: orch.answer(e["payload"]["request_id"], "allow"))
    monkeypatch.setenv("RUNNER_CONTROL_URL", orch.url)
    monkeypatch.setenv("RUNNER_TOKEN", TOKEN)
    code = cli.main(["--provider", "fake", "--fixture", str(FIXTURES / "fake_claude_full.json"),
                     "--cwd", str(worktree), "--prompt-file", str(prompt), "--pull-interval", "0.02"])
    assert code == 0 and orch.kinds()[-1] == "end"
    assert "RUNNER_TOKEN" not in os.environ
    assert TOKEN not in caplog.text


# A linked worktree's git paths come from the command line. Without them the runner does not let git find
# the repository through the worktree's own .git file, and the session ends before any turn.
@pytest.mark.parametrize("given", [True, False])
def test_the_command_line_takes_a_linked_worktree_s_git_paths(orch, worktree, tmp_path, monkeypatch, given):
    from conftest import git
    from runner import gitdiff

    linked = tmp_path / "linked"
    git(worktree, "worktree", "add", "--quiet", "-b", "agent/linked", str(linked))
    paths = gitdiff.Worktree.registered(str(worktree), str(linked))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Fix the backup job.")
    orch.when(lambda e: e["kind"] == "approval", lambda e: orch.answer(e["payload"]["request_id"], "allow"))
    monkeypatch.setenv("RUNNER_CONTROL_URL", orch.url)
    monkeypatch.setenv("RUNNER_TOKEN", TOKEN)
    extra = ["--git-dir", paths.git_dir, "--git-common-dir", paths.common_dir] if given else []
    code = cli.main(["--provider", "fake", "--fixture", str(FIXTURES / "fake_claude_full.json"),
                     "--cwd", str(linked), *extra, "--prompt-file", str(prompt), "--pull-interval", "0.02"])
    [end] = orch.of("end")
    if given:
        assert code == 0 and end["reason"] == "completed" and orch.of("diff")
    else:
        assert code == 1 and end["reason"] == "error" and end["detail"] == gitdiff.BAD_REGISTRATION
        assert "turn" not in orch.kinds()


CLAUDE_SECRET = "claude-oauth-test-token-0000"


def _claude_token_file(tmp_path):
    path = tmp_path / "claude-token"
    path.write_text(CLAUDE_SECRET + "\n")
    path.chmod(0o600)
    return path


def test_the_claude_token_comes_from_its_file_and_never_from_the_environment(tmp_path):
    path = _claude_token_file(tmp_path)
    env = {"RUNNER_TOKEN": "t", "CLAUDE_CODE_OAUTH_TOKEN_FILE": str(path), "CLAUDE_CODE_OAUTH_TOKEN": "stray"}
    settings = config.take_secrets(env)
    assert settings.claude_token == CLAUDE_SECRET and CLAUDE_SECRET not in repr(settings)
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env and CLAUDE_SECRET not in env.values()
    assert config.take_secrets({"RUNNER_TOKEN": "t", "CLAUDE_CODE_OAUTH_TOKEN": "stray"}).claude_token == ""
    path.chmod(0o640)
    with pytest.raises(ValueError, match="0600") as err:
        config.take_secrets({"RUNNER_TOKEN": "t", "CLAUDE_CODE_OAUTH_TOKEN_FILE": str(path)})
    assert CLAUDE_SECRET not in str(err.value)


def test_the_claude_provider_hands_the_token_to_the_cli_subprocess_only(tmp_path):
    from runner.providers.base import RESULT_SCHEMA, ProviderContext
    from runner.providers.claude import ClaudeProvider

    async def never(_payload):
        raise AssertionError("not called")

    settings = config.take_secrets({"RUNNER_TOKEN": "t",
                                    "CLAUDE_CODE_OAUTH_TOKEN_FILE": str(_claude_token_file(tmp_path))})
    provider = cli.provider_for(cli.parse(["--provider", "claude", "--cwd", "/tmp/wt", "--prompt-file", "p"]),
                                settings)
    assert isinstance(provider, ClaudeProvider)
    ctx = ProviderContext(request_approval=never, cwd="/tmp/wt", output_schema=RESULT_SCHEMA)
    assert "next_steps" in RESULT_SCHEMA["required"]
    assert "tests_claim" in RESULT_SCHEMA["required"]
    assert RESULT_SCHEMA["properties"]["next_steps"] == {"type": "array", "items": {"type": "string"}}
    pin = {"ANTHROPIC_DEFAULT_FABLE_MODEL": "claude-fable-5-2",
           "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1",
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    temp = lambda p: {"CLAUDE_CODE_TMPDIR": p.temp.path, "TMPDIR": p.temp.path}
    assert provider.options(ctx).env == {"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_SECRET, **pin, **temp(provider)}
    other = ClaudeProvider()
    assert other.options(ctx).env == {**pin, **temp(other)}
    assert CLAUDE_SECRET not in os.environ.values()


RUNNER_VARIABLES = ("RUNNER_TOKEN", "RUNNER_TOKEN_FILE", "RUNNER_GITHUB_TOKEN", "RUNNER_GITHUB_TOKEN_FILE",
                    "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN_FILE", "ANTHROPIC_API_KEY",
                    "RUNNER_CONTROL_URL", "RUNNER_CONTROL_HOST", "RUNNER_CA_FILE", "RUNNER_WORK_ROOT")


def test_the_environment_each_provider_inherits_holds_none_of_the_runners_variables(tmp_path, monkeypatch):
    from runner.providers.base import ProviderContext
    from runner.providers.claude import ClaudeProvider
    from runner.providers.codex import CodexProvider

    async def never(_payload):
        raise AssertionError("not called")

    token_file = tmp_path / "runner-token"
    token_file.write_text("tok\n")
    token_file.chmod(0o600)
    gh = tmp_path / "gh"
    gh.write_text("gh-placeholder\n")
    gh.chmod(0o600)
    for name in RUNNER_VARIABLES:
        monkeypatch.setenv(name, "x")
    monkeypatch.delenv("RUNNER_TOKEN")
    monkeypatch.delenv("RUNNER_GITHUB_TOKEN")
    monkeypatch.setenv("RUNNER_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("RUNNER_GITHUB_TOKEN_FILE", str(gh))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN_FILE", str(_claude_token_file(tmp_path)))
    monkeypatch.setenv("RUNNER_CONTROL_URL", "https://control.example.internal")
    monkeypatch.setenv("RUNNER_CONTROL_HOST", "control.example.internal")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    settings = config.take_secrets()
    assert (settings.token, settings.github_token, settings.claude_token) == ("tok", "gh-placeholder", CLAUDE_SECRET)
    assert settings.control_host == "control.example.internal"
    ctx = ProviderContext(request_approval=never, cwd="/tmp/wt", mcp_socket="/run/user/1000/mcp-x/s")
    claude_child = {**os.environ, **ClaudeProvider(oauth_token=settings.claude_token).options(ctx).env}
    codex_child = {**os.environ, **(CodexProvider().config(ctx).env or {})}
    for child in (claude_child, codex_child):
        assert not [n for n in RUNNER_VARIABLES if n in child and n != "CLAUDE_CODE_OAUTH_TOKEN"]
        assert not [n for n in child if n.startswith("RUNNER_")]
    assert claude_child["CLAUDE_CODE_OAUTH_TOKEN"] == CLAUDE_SECRET
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in codex_child
    assert codex_child["CODEX_HOME"] == str(tmp_path / "codex-home")


def test_the_command_line_keeps_the_claude_token_out_of_environment_logs_and_requests(
        orch, worktree, tmp_path, monkeypatch, caplog):
    caplog.set_level("DEBUG")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Fix the backup job.")
    orch.when(lambda e: e["kind"] == "approval", lambda e: orch.answer(e["payload"]["request_id"], "allow"))
    monkeypatch.setenv("RUNNER_CONTROL_URL", orch.url)
    monkeypatch.setenv("RUNNER_TOKEN", TOKEN)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN_FILE", str(_claude_token_file(tmp_path)))
    code = cli.main(["--provider", "fake", "--fixture", str(FIXTURES / "fake_claude_full.json"),
                     "--cwd", str(worktree), "--prompt-file", str(prompt), "--pull-interval", "0.02"])
    assert code == 0 and orch.kinds()[-1] == "end"
    assert CLAUDE_SECRET not in os.environ.values() and "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ
    assert CLAUDE_SECRET not in caplog.text
    assert CLAUDE_SECRET not in repr(orch.requests)



# Credentials and settings that a launching shell may hold for other work. None is on the runner's list.
FOREIGN = {"AWS_SECRET_ACCESS_KEY": "aws-secret-placeholder",
           "DATABASE_URL": "postgres://app:db-password-placeholder@db.example.internal/app",
           "GITHUB_TOKEN": "github-token-placeholder", "SOME_UNKNOWN_SETTING": "unknown-placeholder"}
# Listed variables, each set to a value the test can recognise.
LISTED = {"LANG": "en_US.UTF-8", "LC_CTYPE": "en_US.UTF-8", "TERM": "xterm-256color", "TZ": "UTC",
          "SHELL": "/bin/sh", "USER": "agent", "LOGNAME": "agent", "SSL_CERT_FILE": "/etc/ssl/lab-ca.pem",
          "SSL_CERT_DIR": "/etc/ssl/certs", "NODE_EXTRA_CA_CERTS": "/etc/ssl/lab-ca.pem",
          "CODEX_CA_CERTIFICATE": "/etc/ssl/lab-ca.pem", "CLAUDE_CONFIG_DIR": "/home/agent/.claude",
          "CODEX_HOME": "/home/agent/.codex"}


@pytest.fixture(autouse=True)
def _restore_the_process_environment():
    # take_secrets() with no argument trims the test process's own environment to the list, so each
    # test gets it back whole.
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _provider_children(settings):
    from runner.providers.base import ProviderContext
    from runner.providers.claude import ClaudeProvider
    from runner.providers.codex import CodexProvider

    async def never(_payload):
        raise AssertionError("not called")

    ctx = ProviderContext(request_approval=never, cwd="/tmp/wt", mcp_socket="/run/user/1000/mcp-x/s")
    claude_child = {**os.environ, **ClaudeProvider(oauth_token=settings.claude_token).options(ctx).env}
    codex_child = {**os.environ, **(CodexProvider().config(ctx).env or {})}
    return claude_child, codex_child


def test_other_credentials_in_the_launching_shell_never_reach_either_provider(monkeypatch):
    for name, value in {**FOREIGN, **LISTED}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RUNNER_TOKEN", "tok")
    kept = {name: os.environ[name] for name in ("PATH", "HOME") if name in os.environ}
    settings = config.take_secrets()
    for child in _provider_children(settings):
        assert not [name for name in FOREIGN if name in child]
        assert not [value for value in FOREIGN.values() if value in child.values()]
    assert {name: os.environ[name] for name in LISTED} == LISTED
    assert {name: os.environ[name] for name in kept} == kept
    assert not [name for name in os.environ if name not in config.PASSED]


def test_only_listed_names_pass():
    env = {"RUNNER_TOKEN": "t", "PATH": "/usr/bin", "HOME": "/home/agent", "TMPDIR": "/tmp/x", "LC_ALL": "C",
           "LCX": "x", "lang": "x", "PATHX": "x", "HTTPS_PROXY": "http://user:pw@proxy.example.internal:3128",
           "NO_PROXY": "localhost", **FOREIGN}
    config.take_secrets(env)
    assert env == {"PATH": "/usr/bin", "HOME": "/home/agent", "TMPDIR": "/tmp/x", "LC_ALL": "C"}


def test_locale_variables_pass_by_category_name_and_not_by_prefix():
    categories = {"LC_ALL": "C", "LC_TIME": "en_GB.UTF-8", "LC_PAPER": "en_GB.UTF-8"}
    env = {"RUNNER_TOKEN": "t", **categories, "LC_SOMETHING": "carried-over-ssh-placeholder", "LC_": "x",
           "LC_ALLX": "x", "lc_all": "x"}
    config.take_secrets(env)
    assert env == categories


def test_an_operator_passes_a_named_variable_through_runner_pass_env():
    env = {"RUNNER_TOKEN": "t", "PATH": "/usr/bin", "RUNNER_PASS_ENV": " CLAUDE_CODE_USE_BEDROCK, AWS_REGION ,",
           "CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "eu-west-1", **FOREIGN}
    config.take_secrets(env)
    assert env == {"PATH": "/usr/bin", "CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "eu-west-1"}


@pytest.mark.parametrize("name", ["RUNNER_TOKEN", "RUNNER_GITHUB_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
                                  "CLAUDE_CODE_OAUTH_TOKEN_FILE", "RUNNER_TOKEN_FILE", "RUNNER_PASS_ENV"])
def test_runner_pass_env_refuses_a_stripped_secret_or_a_runner_variable_by_name(name):
    env = {"RUNNER_TOKEN": "tok-secret", "RUNNER_PASS_ENV": f"AWS_REGION,{name}", "AWS_REGION": "eu-west-1"}
    with pytest.raises(ValueError, match=f"RUNNER_PASS_ENV may not name {name}") as err:
        config.take_secrets(env)
    assert "tok-secret" not in str(err.value) and "RUNNER_TOKEN" not in env


def test_runner_pass_env_refuses_any_other_runner_name_by_position_without_quoting_it():
    env = {"RUNNER_TOKEN": "t", "RUNNER_PASS_ENV": "AWS_REGION,RUNNER_WORK_ROOT", "AWS_REGION": "eu-west-1"}
    with pytest.raises(ValueError, match="RUNNER_PASS_ENV entry 2 starts with RUNNER_") as err:
        config.take_secrets(env)
    assert "WORK_ROOT" not in str(err.value)


@pytest.mark.parametrize("value", ["A-B", "1ABC", "A=B", "LC_*", "A B", "x" * 129])
def test_runner_pass_env_refuses_a_name_that_is_not_a_variable_name(value):
    with pytest.raises(ValueError, match="RUNNER_PASS_ENV entry 1 is not a variable name") as err:
        config.take_secrets({"RUNNER_TOKEN": "t", "RUNNER_PASS_ENV": value})
    assert value not in str(err.value)


def test_a_refused_runner_pass_env_entry_never_echoes_any_of_its_text():
    entry = "SOME_KEY=s3cr3t-placeholder-0123456789abcdef"
    env = {"RUNNER_TOKEN": "t", "RUNNER_PASS_ENV": f" ,AWS_REGION,,{entry}"}
    with pytest.raises(ValueError, match="RUNNER_PASS_ENV entry 4 is not a variable name") as err:
        config.take_secrets(env)
    message = f"{err.value!s} {err.value!r}"
    assert not [entry[i:i + 5] for i in range(len(entry) - 4) if entry[i:i + 5] in message]
