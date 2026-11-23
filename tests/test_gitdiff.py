"""The turn diff: snapshots through a private index, the capped patch, and the repository's own git
configuration never running code: it is refused before git reads it, in a plain checkout and in a
linked worktree whose paths are checked first."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from conftest import git
from runner import gitdiff, handoff


def _trap(tmp_path, name):
    marker = tmp_path / f"{name}.ran"
    script = tmp_path / f"{name}.sh"
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, marker


def test_snapshot_and_diff_cover_tracked_and_untracked_files(worktree):
    before = gitdiff.snapshot(str(worktree))
    index_before = (worktree / ".git" / "index").read_bytes()
    (worktree / "README.txt").write_text("hello\nworld\n")
    (worktree / "new.txt").write_text("new\n")
    (worktree / ".gitignore").write_text("ignored.txt\n")
    (worktree / "ignored.txt").write_text("secret-ish\n")
    after = gitdiff.snapshot(str(worktree))
    payload = gitdiff.diff(str(worktree), before, after)
    assert sorted(payload["paths"]) == [".gitignore", "README.txt", "new.txt"]
    assert "+world" in payload["patch"] and "a/README.txt" in payload["patch"]
    assert payload["files"] == 3 and "3 files changed" in payload["stat"]
    assert (worktree / ".git" / "index").read_bytes() == index_before
    assert git(worktree, "status", "--porcelain") == " M README.txt\n?? .gitignore\n?? new.txt\n"
    assert gitdiff.diff(str(worktree), after, after) is None


def test_the_patch_is_capped(worktree):
    before = gitdiff.snapshot(str(worktree))
    (worktree / "big.txt").write_text("line\n" * 20000)
    payload = gitdiff.diff(str(worktree), before, gitdiff.snapshot(str(worktree)))
    assert payload["patch_truncated"] is True and len(payload["patch"]) <= gitdiff.PATCH_CAP


@pytest.fixture
def linked(worktree, tmp_path):
    path = tmp_path / "linked"
    git(worktree, "worktree", "add", "--quiet", "-b", "agent/fix", str(path))
    return path, gitdiff.Worktree.registered(str(worktree), str(path))


def _refusal(key):
    return (f"Git refused {key.lower()!r}. Ask the machine's administrator to remove that key from the project's "
            "Git config before restarting the session.")


# The repository's config is refused before git reads attributes, so no driver runs, even one that FIXED
# would also have overridden.
@pytest.mark.parametrize("layout", ["plain", "linked"])
@pytest.mark.parametrize("key", ["core.fsmonitor", "diff.external", "diff.trap.command", "diff.trap.textconv",
                                 "filter.trap.clean", "filter.trap.process", "merge.trap.driver"])
def test_the_worktree_s_configuration_never_runs_code(worktree, tmp_path, layout, key):
    if layout == "linked":
        path = tmp_path / "linked"
        git(worktree, "worktree", "add", "--quiet", "-b", "agent/fix", str(path))
        paths = gitdiff.Worktree.registered(str(worktree), str(path))
    else:
        path, paths = worktree, None
    script, marker = _trap(tmp_path, "driver")
    (path / ".gitattributes").write_text("*.txt filter=trap merge=trap diff=trap\n")
    (path / "README.txt").write_text("changed\n")
    git(worktree, "config", key, str(script))
    with gitdiff.use_worktree(paths), pytest.raises(gitdiff.GitSafetyError) as refused:
        gitdiff.snapshot(str(path))
    assert str(refused.value) == _refusal(key)
    assert not marker.exists()


def test_a_filter_is_refused_before_git_add_runs(worktree, tmp_path, monkeypatch):
    script, marker = _trap(tmp_path, "clean")
    git(worktree, "config", "filter.trap.clean", str(script))
    (worktree / ".gitattributes").write_text("*.txt filter=trap\n")
    (worktree / "README.txt").write_text("changed\n")
    calls = []
    real_run = gitdiff.subprocess.run

    def recording(argv, *args, **kwargs):
        calls.append(argv)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(gitdiff.subprocess, "run", recording)
    with pytest.raises(gitdiff.GitSafetyError, match="filter.trap.clean"):
        gitdiff.snapshot(str(worktree))
    assert [argv[1:3] for argv in calls] == [["--no-pager", "config"]]
    assert not marker.exists()


def test_a_linked_worktree_without_its_paths_is_refused(linked):
    path, _ = linked
    with pytest.raises(gitdiff.GitSafetyError) as refused:
        gitdiff.snapshot(str(path))
    assert str(refused.value) == gitdiff.BAD_REGISTRATION


@pytest.mark.parametrize(("field", "sentence"), [
    (".git", "Git could not verify the worktree's .git as a regular file at the expected path naming the "
             "registered Git directory. "),
    ("commondir", "The commondir file is missing, is not a regular file, or names a different common directory. "),
])
def test_the_registration_cannot_be_redirected(linked, tmp_path, field, sentence):
    path, paths = linked
    decoy = path / "decoy"
    decoy.mkdir()
    git(decoy, "init", "--quiet")
    script, marker = _trap(tmp_path, "driver")
    git(decoy, "config", "filter.trap.clean", str(script))
    (path / ".gitattributes").write_text("*.txt filter=trap\n")
    (path / "README.txt").write_text("changed\n")
    if field == ".git":
        (path / ".git").write_text(f"gitdir: {decoy / '.git'}\n")
    else:
        Path(paths.git_dir, "commondir").write_text(f"{decoy / '.git'}\n")
    with gitdiff.use_worktree(paths):
        for operation in (lambda: gitdiff.snapshot(str(path)),
                          lambda: handoff.hand_off(str(path), "fix", "main", None, None)):
            with pytest.raises(gitdiff.GitSafetyError) as refused:
                operation()
            assert str(refused.value) == (
                sentence + "Ask the machine's administrator to restore it before restarting the session.")
    assert not marker.exists()


# Once checked, git is handed the verified paths, so a pointer changed afterwards is never read.
def test_git_commands_use_the_explicit_paths_after_the_check(linked):
    path, paths = linked
    with gitdiff.use_worktree(paths), gitdiff.checked(str(path)):
        (path / ".git").write_text("invalid Git pointer\n")
        Path(paths.git_dir, "commondir").write_text("invalid common directory\n")
        assert gitdiff.git(str(path), "rev-parse", "--absolute-git-dir", "--git-common-dir", "--show-toplevel") \
            .splitlines() == [paths.git_dir, paths.common_dir, str(path)]


class LocalForge(handoff.GitHub):
    def open_pr(self, branch, base, title, body):
        assert (base, title) == ("main", "Change text") and "Change text" in body
        return handoff.Opened("https://example.invalid/pr/1", 1)


# An attributes file cannot define a driver by itself, and the allowed values of the checked keys pass,
# so the snapshot, the diff and a real commit and push work as before.
@pytest.mark.parametrize("layout", ["plain", "linked"])
def test_undefined_attribute_drivers_and_disabled_hooks_allow_normal_handoff(worktree, tmp_path, layout):
    if layout == "linked":
        path = tmp_path / "linked"
        git(worktree, "worktree", "add", "--quiet", "-b", "agent/fix", str(path))
        paths, slug = gitdiff.Worktree.registered(str(worktree), str(path)), "fix"
    else:
        path, paths, slug = worktree, None, "fix-backup"
    git(worktree, "config", "diff.noprefix", "true")
    git(worktree, "config", "core.hooksPath", "/dev/null")
    git(worktree, "config", "core.fsmonitor", "false")
    hooks = worktree / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    hook, hook_marker = _trap(tmp_path, "hook")
    os.symlink(hook, hooks / "pre-commit")
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--quiet", "--bare", str(remote))
    (path / ".gitattributes").write_text("*.txt filter=trap merge=trap diff=trap\n")
    with gitdiff.use_worktree(paths):
        before = gitdiff.snapshot(str(path))
        (path / "README.txt").write_text("changed\n")
        payload = gitdiff.diff(str(path), before, gitdiff.snapshot(str(path)))
        forge = LocalForge("owner/repo", "placeholder")
        forge.push_url, forge.push_protocol = remote.as_uri(), "file"
        result = handoff.hand_off(str(path), slug, "main", {"summary": "Change text"}, forge, title="Change text")
    assert "-hello" in payload["patch"] and "a/README.txt" in payload["patch"]   # noprefix overridden
    assert result["head_sha"] == git(remote, "rev-parse", f"agent/{slug}").strip()
    assert git(path, "log", "-1", "--format=%an <%ae>|%s").strip() == (
        f"{handoff.AGENT_NAME} <{handoff.AGENT_EMAIL}>|Change text")
    assert not hook_marker.exists()


def test_the_config_read_never_follows_includes_or_reads_global(worktree, linked, tmp_path, monkeypatch):
    path, paths = linked
    global_config = tmp_path / ".gitconfig"
    global_config.write_text("[include]\npath = nonsense\n[alias]\nfoo = !false\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    decoy = path / "not-a-config"
    decoy.write_text("not valid git configuration")
    env = gitdiff.git_environment({"GIT_CONFIG_COUNT": "2"})
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_CONFIG_NOSYSTEM"] == env["GIT_ATTR_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_COUNT"] == "2"
    with gitdiff.use_worktree(paths):
        assert gitdiff.snapshot(str(path))
        git(worktree, "config", "--local", "include.path", str(decoy))
        with pytest.raises(gitdiff.GitSafetyError, match="include.path"):
            gitdiff.snapshot(str(path))


def test_a_failure_names_the_subcommand_only(tmp_path):
    try:
        gitdiff.git(str(tmp_path), "rev-parse", "HEAD")
    except gitdiff.GitError as exc:
        assert str(exc) == "git rev-parse exited 128"
    else:
        raise AssertionError("expected GitError")



# A caller may narrow git's transports for one call but never widen them.
def test_a_caller_narrows_the_transports_but_never_widens_them(worktree):
    def allowed(extra=None):
        return gitdiff._environment(str(worktree), extra)["GIT_ALLOW_PROTOCOL"]

    assert allowed() == "file:git:ssh:http:https:ftp:ftps"
    assert allowed({"GIT_ALLOW_PROTOCOL": "https"}) == "https"
    assert allowed({"GIT_ALLOW_PROTOCOL": "https:ext:fd"}) == "https"
    assert allowed({"GIT_ALLOW_PROTOCOL": "ext"}) == "none"


# The repository's config may set no http. key, general or keyed to a URL, so it cannot weaken TLS, add a
# certificate, cookie or header, or follow a redirect on the push that carries the GitHub token. The refusal
# comes from the same config listing as every other key, before any git call reads the repository.
@pytest.mark.parametrize("key,value", [
    ("http.sslVerify", "false"), ("http.https://github.com/.sslVerify", "false"),
    ("http.sslCAInfo", "/tmp/ca.pem"), ("http.sslCAPath", "/tmp/certs"), ("http.sslCert", "/tmp/cert.pem"),
    ("http.sslKey", "/tmp/key.pem"), ("http.cookieFile", "/tmp/cookies"),
    ("http.https://github.com/.extraHeader", "X-Extra: 1"), ("http.followRedirects", "true"),
    ("http.proxy", "http://proxy.example.internal:3128"), ("http.https://github.com/.curloptResolve", "x"),
])
def test_any_http_key_in_the_repository_config_is_refused_before_git_runs(worktree, tmp_path, monkeypatch,
                                                                           key, value):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--quiet", "--bare", str(remote))
    git(worktree, "config", key, value)
    (worktree / "README.txt").write_text("changed\n")
    calls = []
    real_run = gitdiff.subprocess.run

    def recording(argv, *args, **kwargs):
        calls.append(argv)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(gitdiff.subprocess, "run", recording)
    with pytest.raises(gitdiff.GitSafetyError) as refused:
        gitdiff.snapshot(str(worktree))
    assert str(refused.value) == _refusal(key)
    forge = handoff.GitHub("owner/repo", "t0k")
    forge.push_url, forge.push_protocol = remote.as_uri(), "file"
    with pytest.raises(gitdiff.GitSafetyError) as refused:
        handoff.hand_off(str(worktree), "fix-backup", "main", {"summary": "Work"}, forge)
    assert str(refused.value) == _refusal(key)
    assert [argv[1:3] for argv in calls] == [["--no-pager", "config"]] * 2
    assert git(remote, "for-each-ref", "refs/heads/") == ""
