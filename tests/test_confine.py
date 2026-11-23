"""Read confinement: a coding session's reads held to its worktree, git directory and own temp folder in
the settings each provider builds, with research sessions unchanged."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import tomllib

import pytest

from runner.providers import confine
from runner.providers.base import RESEARCH_KIND, ProviderContext

GIT = ("git", "-c", "user.name=t", "-c", "user.email=t@localhost", "-c", "commit.gpgsign=false")


async def _never(_payload):
    raise AssertionError("not called")


def _git(cwd, *args):
    subprocess.run([*GIT, *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / "work" / "proj" / "repo"
    repo.mkdir(parents=True)
    (repo / "a.txt").write_text("a\n")
    _git(repo, "init", "--quiet")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "first")
    worktree = repo.parent / "wt" / "7"
    _git(repo, "worktree", "add", "--quiet", "-b", "agent/session-7", str(worktree))
    return os.path.realpath(repo), os.path.realpath(worktree)


def _ctx(cwd, **kw):
    return ProviderContext(request_approval=_never, cwd=cwd, **kw)


def test_a_worktrees_git_directory_is_its_main_checkouts_dot_git(project, tmp_path):
    repo, worktree = project
    assert confine.git_common_dir(worktree) == os.path.join(repo, ".git")
    assert confine.git_common_dir(repo) == os.path.join(repo, ".git")
    assert confine.git_common_dir(str(tmp_path)) is None
    scope = confine.read_scope(worktree, str(tmp_path / "t"), user_home=str(tmp_path))
    assert scope.readable == (worktree, os.path.join(repo, ".git"), os.path.realpath(tmp_path / "t"))
    assert repo not in scope.readable


def test_a_worktree_that_is_the_home_or_a_temp_root_is_refused(tmp_path):
    for cwd in (str(tmp_path), "/tmp", "/"):
        with pytest.raises(ValueError):
            confine.read_scope(cwd, "/tmp/rs-x", user_home=str(tmp_path))


def test_a_session_temp_folder_is_private_and_goes_on_close(tmp_path):
    temp = confine.SessionTemp(str(tmp_path))
    assert os.path.basename(temp.path).startswith(confine.TEMP_PREFIX)
    assert stat.S_IMODE(os.stat(temp.path).st_mode) == 0o700
    temp.close()
    assert not os.path.exists(temp.path)


MODES = ("default", "acceptEdits", "plan", "dontAsk")


def test_claudes_coding_session_blocks_reads_outside_its_worktree_and_temp_folder(project):
    from runner.providers.claude import ClaudeProvider, read_settings

    repo, worktree = project
    provider = ClaudeProvider()
    for mode in MODES:
        for web in (True, False):
            options = provider.options(_ctx(worktree, permission_mode=mode, web=web, mcp_socket="/tmp/s"))
            temp = provider.temp.path
            settings = json.loads(options.settings)
            assert settings == {"permissions": {"blockReadsOutsideWorkingDirectories": True},
                                "sandbox": {"enabled": True, "failIfUnavailable": True,
                                            "allowUnsandboxedCommands": False,
                                            "filesystem": {"denyRead": ["/tmp", "/var/tmp"], "allowRead": [temp]}},
                                "includeGitInstructions": False}
            assert settings == read_settings(confine.read_scope(worktree, temp))
            assert options.add_dirs == [temp] and options.cwd == worktree
            assert options.env["CLAUDE_CODE_TMPDIR"] == temp and options.env["TMPDIR"] == temp
            assert options.setting_sources == ["user"]
    assert provider.options(_ctx(worktree)).add_dirs == [temp]
    assert stat.S_IMODE(os.stat(temp).st_mode) == 0o700
    asyncio.run(provider.close())
    assert provider.temp is None and not os.path.exists(temp)


def test_claudes_temp_folder_must_fit_the_clis_limit_or_the_session_does_not_start():
    from runner.providers.claude import CLAUDE_TMPDIR_MAX, session_temp_env

    assert session_temp_env("/tmp/rs-abcdefgh") == {"CLAUDE_CODE_TMPDIR": "/tmp/rs-abcdefgh", "TMPDIR": "/tmp/rs-abcdefgh"}
    assert len(f"/tmp/rs-abcdefgh/claude-{os.getuid()}") <= CLAUDE_TMPDIR_MAX
    with pytest.raises(ValueError):
        session_temp_env("/tmp/" + "x" * 40)


def test_claudes_research_session_is_not_changed(tmp_path):
    from runner.providers.claude import PROJECT_CWD_ENV, ClaudeProvider

    scratch = os.path.realpath(tmp_path)
    provider = ClaudeProvider()
    options = provider.options(_ctx(scratch, kind=RESEARCH_KIND))
    assert options.settings is None and options.add_dirs == []
    assert "CLAUDE_CODE_TMPDIR" not in options.env and "TMPDIR" not in options.env
    assert PROJECT_CWD_ENV not in options.env
    assert options.allowed_tools[:2] == [f"Read(/{scratch}/**)", f"Edit(/{scratch}/**)"]
    assert provider.temp is None


# The test verdict's evidence for Claude: the launch options of the client actually connected.
def test_claude_records_the_bash_directory_setting_at_connect_and_reconnect(project, monkeypatch):
    import claude_agent_sdk

    from runner.providers.claude import ClaudeProvider

    _, worktree = project
    provider = ClaudeProvider()
    launched = []

    class Client:
        def __init__(self, options):
            self.options = options

        async def connect(self):
            launched.append(self.options)

        async def disconnect(self):
            pass

    async def go():
        await provider.start(_ctx(worktree))
        assert provider.bash_starts_in_worktree is True
        await provider._reconnect()
        assert provider.bash_starts_in_worktree is True
        await provider.close()
        await provider.start(_ctx(worktree, kind=RESEARCH_KIND))
        assert provider.bash_starts_in_worktree is False
        await provider.close()

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    assert provider.bash_starts_in_worktree is False
    asyncio.run(go())
    assert [options.env.get("CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR") for options in launched] == ["1", "1", None]
    assert all(options.cwd == worktree for options in launched)


def _override(overrides, key):
    [found] = [o for o in overrides if o.startswith(f"{key}=")]
    return tomllib.loads(found)[key]


def test_codexs_coding_session_runs_on_a_profile_naming_only_what_it_may_read(project, monkeypatch):
    from runner.providers import codex as codex_mod
    from runner.providers.codex import PROFILE_READ, PROFILE_WRITE, CodexProvider

    monkeypatch.setattr(codex_mod, "tool_dirs", lambda **kw: ())
    repo, worktree = project
    provider = CodexProvider()
    ctx = _ctx(worktree, mcp_socket="/run/user/1/s", web=True)
    config_ = provider.config(ctx)
    temp = provider.temp.path
    overrides = config_.config_overrides
    git_dir = os.path.join(repo, ".git")
    assert _override(overrides, "default_permissions") == PROFILE_WRITE
    profiles = _override(overrides, "permissions")
    assert profiles[PROFILE_WRITE] == {"filesystem": {":minimal": "read", "/opt/agent-runner": "read",
                                                      "/etc/agent-runner/secrets": "deny", worktree: "write",
                                                      git_dir: "read", temp: "write"},
                                       "network": {"enabled": False}}
    assert profiles[PROFILE_READ]["filesystem"] == {":minimal": "read", "/opt/agent-runner": "read",
                                                    "/etc/agent-runner/secrets": "deny", worktree: "read",
                                                    git_dir: "read", temp: "read"}
    for profile in profiles.values():
        assert not {":root", ":tmpdir", ":slash_tmp", ":workspace_roots", os.path.expanduser("~"), "/tmp", repo} \
            & set(profile["filesystem"])
    assert _override(overrides, "shell_environment_policy") == {"set": {"TMPDIR": temp}}
    assert config_.env == {"TMPDIR": temp, "AGENT_MCP_SOCKET": "/run/user/1/s"}
    assert codex_mod.MCP_ENABLE in overrides and 'web_search="live"' in overrides
    params = provider.thread_params(ctx)
    assert "sandbox" not in params
    assert params["config"]["default_permissions"] == PROFILE_WRITE
    assert params["config"]["permissions"] == profiles
    assert params["config"]["projects"] == codex_mod.projects_untrusted(worktree)
    asyncio.run(provider.close())
    assert not os.path.exists(temp)


def test_codexs_mode_changes_switch_profile_and_never_send_a_sandbox_policy(project):
    from runner import settings as words
    from runner.providers.codex import PROFILE_READ, PROFILE_WRITE, CodexProvider

    _repo, worktree = project
    provider = CodexProvider()
    provider.ctx = _ctx(worktree)
    expected = {"manual": ("untrusted", PROFILE_WRITE), "accept-edits": ("on-request", PROFILE_WRITE),
                "plan": ("on-request", PROFILE_READ), "auto": ("never", PROFILE_WRITE)}
    assert set(expected) == set(words.CODEX_MODE)
    for word, (policy, profile) in expected.items():
        provider.overrides = {}
        asyncio.run(provider.change_settings({"mode": word}))
        assert provider.overrides == {"approvalPolicy": policy, "permissions": profile}


@pytest.fixture
def boxlike(tmp_path, monkeypatch):
    from runner.providers import codex as codex_mod

    root = tmp_path / "box"
    for part in ("usr/local/bin", "opt", "home/agent/.local/bin"):
        (root / part).mkdir(parents=True)
    root = os.path.realpath(root)
    monkeypatch.setattr(codex_mod, "MINIMAL_ROOTS", (os.path.join(root, "usr"),))
    monkeypatch.setattr(codex_mod, "TOOLCHAINS", os.path.join(root, "opt/agent-runner"))
    monkeypatch.setattr(codex_mod, "TOOL_PARENT", os.path.join(root, "opt"))

    def install(rel):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write("#!/bin/sh\n")
        os.chmod(path, 0o755)
        return path
    return root, install


def test_codex_reads_uvs_own_directory_when_the_box_links_it_from_opt(boxlike, project, monkeypatch, caplog):
    from runner.providers import codex as codex_mod
    from runner.providers.codex import PROFILE_READ, PROFILE_WRITE, CodexProvider

    root, install = boxlike
    real = install("opt/pipx/venvs/uv/bin/uv")
    bindir = os.path.join(root, "usr/local/bin")
    os.symlink(real, os.path.join(bindir, "uv"))
    nowhere = {"uv": os.path.join(root, "absent")}
    assert codex_mod.tool_dirs(path=bindir, fallback=nowhere) == (os.path.dirname(real),)
    assert codex_mod.tool_dirs(path="", fallback={"uv": os.path.join(bindir, "uv")}) == (os.path.dirname(real),)

    monkeypatch.setenv("PATH", bindir)
    monkeypatch.setattr(confine, "SHARED_TEMP", ())
    _repo, worktree = project
    provider = CodexProvider()
    ctx = _ctx(worktree)
    profiles = _override(provider.config(ctx).config_overrides, "permissions")
    for name in (PROFILE_WRITE, PROFILE_READ):
        entries = profiles[name]["filesystem"]
        assert entries[os.path.dirname(real)] == "read"
        under_opt = [p for p in entries if p.startswith(os.path.join(root, "opt") + "/") and p != codex_mod.TOOLCHAINS]
        assert under_opt == [os.path.dirname(real)]
        assert profiles[name]["network"] == {"enabled": False}
    assert provider.thread_params(ctx)["config"]["permissions"] == profiles
    asyncio.run(provider.close())


def test_codex_adds_nothing_for_uv_where_it_is_read_already_or_may_not_be(boxlike, monkeypatch, caplog):
    from runner.providers import codex as codex_mod

    root, install = boxlike
    nowhere = {"uv": os.path.join(root, "absent")}
    home = os.path.join(root, "home/agent")
    bindir = os.path.join(root, "usr/local/bin")
    install("usr/local/bin/uv")
    assert codex_mod.tool_dirs(path=bindir, fallback=nowhere) == ()
    os.remove(os.path.join(bindir, "uv"))
    assert codex_mod.tool_dirs(path=bindir, fallback=nowhere) == ()
    os.symlink(os.path.join(root, "opt/gone/uv"), os.path.join(bindir, "uv"))
    assert codex_mod.tool_dirs(path=bindir, fallback=nowhere) == ()
    os.remove(os.path.join(bindir, "uv"))
    for rel, denied in (("home/agent/.local/bin/uv", (home,)), ("opt/uv", ()), ("opt/agent/.local/bin/uv",
                                                                                    (os.path.join(root, "opt/agent"),))):
        real = install(rel)
        os.symlink(real, os.path.join(bindir, "uv"))
        with caplog.at_level("WARNING", logger="runner.codex"):
            assert codex_mod.tool_dirs(path=bindir, fallback=nowhere, denied=denied) == (), rel
        assert f"uv resolves to {real}" in caplog.text
        os.remove(os.path.join(bindir, "uv"))
        caplog.clear()
    real = install("opt/pipx/venvs/uv/bin/uv")
    os.symlink(real, os.path.join(bindir, "uv"))
    monkeypatch.setenv("PATH", bindir)
    scope = confine.ReadScope(worktree=os.path.join(root, "wt"), git_dir=None, temp=os.path.join(root, "t"),
                              denied=(os.path.join(root, "opt/pipx"),))
    assert os.path.dirname(real) not in codex_mod.read_profiles(scope)[codex_mod.PROFILE_WRITE]["filesystem"]


def test_codexs_research_session_is_not_changed(tmp_path):
    from runner.providers.codex import CodexProvider

    scratch = os.path.realpath(tmp_path)
    provider = CodexProvider()
    ctx = _ctx(scratch, kind=RESEARCH_KIND, max_subagents=1)
    overrides = provider.config(ctx).config_overrides
    assert 'sandbox_mode="workspace-write"' in overrides and "features.shell_tool=false" in overrides
    assert not any(o.startswith(("default_permissions", "permissions", "shell_environment_policy")) for o in overrides)
    params = provider.thread_params(ctx)
    assert params["sandbox"] == "workspace-write" and "default_permissions" not in params["config"]
    assert provider.temp is None

