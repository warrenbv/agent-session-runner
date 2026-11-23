"""The hand-off's statistics on real repositories, and the GitHub REST calls' request shapes and rate-limit
handling against a stub opener (no network)."""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request

import pytest

from conftest import git
from email.message import Message
from urllib.error import HTTPError

from runner import handoff

ID = ("-c", "user.name=Other", "-c", "user.email=other@example.invalid", "-c", "commit.gpgsign=false")


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "--quiet", "--initial-branch=main")
    for key, value in (("user.name", "Runner Test"), ("user.email", "runner-test@example.invalid"),
                       ("commit.gpgsign", "false")):
        git(path, "config", key, value)
    (path / "a.txt").write_text("one\ntwo\nthree\n")
    (path / "gone.txt").write_text("bye\n")
    (path / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00")
    git(path, "add", "--all")
    git(path, "commit", "--quiet", "-m", "initial")
    git(path, "switch", "--quiet", "-c", "agent/fix-backup")
    return path


class FakeForge:
    def __init__(self) -> None:
        self.pushed: list[str] = []
        self.prs: list[dict] = []

    def push(self, cwd, branch):
        self.pushed.append(branch)

    def open_pr(self, branch, base, title, body):
        self.prs.append({"branch": branch, "base": base})
        return handoff.Opened("https://github.com/example-owner/example/pull/1", 1)


def _hand_off(repo, since=None):
    forge = FakeForge()
    return handoff.hand_off(str(repo), "fix-backup", "main", {"summary": "Work"}, forge, since=since), forge


def test_the_hand_off_reports_head_number_base_and_stats(repo):
    since = git(repo, "rev-parse", "main").strip()
    (repo / "a.txt").write_text("one\nTWO\nthree\n")
    (repo / "new.txt").write_text("first\nsecond\n")
    (repo / "gone.txt").unlink()
    (repo / "blob.bin").write_bytes(b"\x00\x09binary changed\x00")
    out, _ = _hand_off(repo, since=since)
    tip = git(repo, "rev-parse", "agent/fix-backup").strip()
    assert out["head_sha"] == tip and len(tip) in (40, 64)
    assert out["pr_number"] == 1 and out["base"] == "main" and out["commits"] == 1
    assert out["stats"] == {"files": 4, "insertions": 3, "deletions": 2,
                            "paths": ["a.txt", "blob.bin", "gone.txt", "new.txt"], "paths_truncated": False}


def test_stats_cover_every_commit_since_the_cut_and_cap_the_paths(repo):
    since = git(repo, "rev-parse", "main").strip()
    (repo / "early.txt").write_text("committed by the agent itself\n")
    git(repo, "add", "early.txt")
    git(repo, "commit", "--quiet", "-m", "an earlier commit on the branch")
    deep = repo / ("d" * 200)
    deep.mkdir()
    (deep / ("e" * 200)).write_text("long\n")
    for i in range(105):
        (repo / f"f{i:03}.txt").write_text(f"{i}\n")
    out, _ = _hand_off(repo, since=since)
    stats = out["stats"]
    assert out["commits"] == 2
    assert stats["files"] == 107 and stats["insertions"] == 107 and stats["deletions"] == 0
    assert len(stats["paths"]) == 100 and stats["paths_truncated"] is True
    assert stats["paths"][0] == "d" * 200 + "/" + "e" * 99 and len(stats["paths"][0]) == 300
    assert stats["paths"][1] == "early.txt" and stats["paths"][2] == "f000.txt"


def test_a_rename_names_both_paths(repo):
    git(repo, "mv", "a.txt", "b.txt")
    out, _ = _hand_off(repo)
    assert out["stats"] == {"files": 2, "insertions": 3, "deletions": 3, "paths": ["a.txt", "b.txt"],
                            "paths_truncated": False}


class StubOpener:
    def __init__(self, answers):
        self.answers, self.requests = list(answers), []

    def open(self, request, timeout):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(json.dumps(answer).encode() if answer is not None else b"")


def _http(status):
    return urllib.error.HTTPError("https://api.github.com/x?t0k", status, "t0k", {}, io.BytesIO(b"t0k"))


@pytest.fixture
def stub(monkeypatch):
    holder = {}

    def install(*answers):
        holder["opener"] = StubOpener(answers)
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: holder["opener"])
        return holder["opener"]
    return install


SHA = "0123456789abcdef0123456789abcdef01234567"
MERGED = "89abcdef0123456789abcdef0123456789abcdef"


def test_github_merge_is_a_pinned_put_and_reports_405_and_409(stub):
    gh = handoff.GitHub("owner/repo", "t0k")
    opener = stub({"sha": MERGED, "merged": True})
    assert gh.merge(12, SHA) == MERGED
    [request] = opener.requests
    assert (request.get_method(), request.full_url) == ("PUT", "https://api.github.com/repos/owner/repo/pulls/12/merge")
    assert json.loads(request.data) == {"merge_method": "merge", "sha": SHA}
    assert request.get_header("Authorization") == "Bearer t0k"
    for status in (405, 409):
        stub(_http(status))
        with pytest.raises(handoff.ForgeError) as caught:
            gh.merge(12, SHA)
        assert str(caught.value) == f"merge: HTTP {status}"
    opener = stub()
    for bad in (None, "", "abc", SHA.upper()):
        with pytest.raises(handoff.ForgeError, match="merge: no head_sha"):
            gh.merge(12, bad)
    assert opener.requests == []


def test_github_close_delete_and_open_pr_shapes(stub):
    gh = handoff.GitHub("owner/repo", "t0k")
    opener = stub({"state": "closed"}, None, {"html_url": "https://github.com/owner/repo/pull/9", "number": 9})
    gh.close(12)
    gh.delete_branch("agent/session-5")
    assert gh.open_pr("agent/session-5", "main", "t", "b") == handoff.Opened("https://github.com/owner/repo/pull/9", 9)
    close, delete, create = opener.requests
    assert (close.get_method(), close.full_url) == ("PATCH", "https://api.github.com/repos/owner/repo/pulls/12")
    assert json.loads(close.data) == {"state": "closed"}
    assert (delete.get_method(), delete.full_url) == (
        "DELETE", "https://api.github.com/repos/owner/repo/git/refs/heads/agent/session-5")
    assert create.get_method() == "POST" and create.full_url.endswith("/repos/owner/repo/pulls")
    for bad in ("main", "agent/../main", "agent/Session", "refs/heads/agent/x"):
        with pytest.raises(handoff.ForgeError, match="delete_branch: not an agent branch"):
            gh.delete_branch(bad)
    assert len(opener.requests) == 3
    stub(_http(422))
    with pytest.raises(handoff.ForgeError) as caught:
        gh.close(12)
    assert str(caught.value) == "close: HTTP 422" and "t0k" not in str(caught.value)


REPO = "example-owner/example"


@pytest.mark.parametrize("status, headers, delay", [
    (403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "100120"}, 120),
    (429, {"retry-after": "90", "x-ratelimit-remaining": "12"}, 90),
    (429, {"retry-after": "150", "x-ratelimit-remaining": "0", "x-ratelimit-reset": "100120"}, 150),
    (403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "bad"}, 60),
    (403, {"retry-after": "Fri, 02 Jan 1970 03:48:00 GMT"}, 80),
])
def test_github_rate_limits_get_the_safe_deadline_on_reads_and_writes(stub, monkeypatch, status, headers, delay):
    monkeypatch.setattr(handoff.time, "time", lambda: 100000)
    message = Message()
    for key, value in headers.items():
        message[key] = value
    error = HTTPError("https://example.invalid/private", status, "private", message, None)
    opener = stub(error, error, error)
    gh = handoff.GitHub(REPO, "token")
    for method, args in ((gh.pr_state, ("agent/session-7",)), (gh.open_pr, ("agent/session-7", "main", "T", "B")),
                         (gh.merge, (7, SHA))):
        with pytest.raises(handoff.RateLimited) as caught:
            method(*args)
        assert caught.value.until == 100000 + delay
        assert caught.value.step == f"github_rate_limit {100000 + delay}"
        assert str(caught.value).startswith("GitHub API rate limit reached. Retry after Fri 2 Jan at ")
        assert str(caught.value).endswith("UTC.")
    assert [request.get_method() for request in opener.requests] == ["GET", "POST", "PUT"]



# The push and the pull request name the same repository. A checkout's remotes, origin included, are never
# where the push goes. Tests push to a local stand-in for that repository, so the transport is widened to
# file for them alone.
class RecordingGitHub(handoff.GitHub):
    def __init__(self, target, protocol="file"):
        super().__init__(REPO, "t0k")
        self.opened: list[str | None] = []
        self.target = target
        if target is not None:
            self.push_url, self.push_protocol = target.as_uri(), protocol

    def open_pr(self, branch, base, title, body):
        found = git(self.target, "for-each-ref", "--format=%(objectname)", f"refs/heads/{branch}").strip() \
            if self.target is not None else None
        self.opened.append(found or None)
        return handoff.Opened("https://github.com/example-owner/example/pull/1", 1)


def _bare(path):
    path.mkdir(parents=True)
    git(path, "init", "--quiet", "--bare")
    return path


def _heads(bare):
    return git(bare, "for-each-ref", "--format=%(refname)", "refs/heads/").split()


@pytest.mark.parametrize("origin", [True, False], ids=["origin-accepts-pushes", "no-origin"])
def test_the_push_goes_to_the_trusted_repository_and_never_to_origin(repo, tmp_path, origin):
    elsewhere = _bare(tmp_path / "elsewhere.git")
    if origin:
        git(repo, "remote", "add", "origin", str(elsewhere))
    trusted = _bare(tmp_path / "github" / "example-owner" / "example.git")
    (repo / "a.txt").write_text("changed\n")
    forge = RecordingGitHub(trusted)
    out = handoff.hand_off(str(repo), "fix-backup", "main", {"summary": "Work"}, forge)
    tip = git(repo, "rev-parse", "agent/fix-backup").strip()
    assert _heads(elsewhere) == []
    assert _heads(trusted) == ["refs/heads/agent/fix-backup"]
    assert git(trusted, "rev-parse", "agent/fix-backup").strip() == tip
    assert forge.opened == [tip] and out["pr_number"] == 1 and out["head_sha"] == tip


def test_the_push_address_is_built_from_the_trusted_name_with_https_only_and_no_redirect():
    forge = handoff.GitHub(REPO, "t0k")
    assert (forge.push_url, forge.push_protocol) == ("https://github.com/example-owner/example.git", "https")
    env = handoff.push_env(handoff.auth_env("t0k"), forge.push_url, forge.push_protocol)
    assert {k: v for k, v in env.items() if k != "GIT_CONFIG_VALUE_0"} == {
        "GIT_ALLOW_PROTOCOL": "https", "GIT_CONFIG_COUNT": "7",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_KEY_1": "http.followRedirects", "GIT_CONFIG_VALUE_1": "false",
        "GIT_CONFIG_KEY_2": "http.https://github.com/example-owner/example.git.followRedirects",
        "GIT_CONFIG_VALUE_2": "false",
        "GIT_CONFIG_KEY_3": "http.https://github.com/example-owner/example.git.sslVerify",
        "GIT_CONFIG_VALUE_3": "true",
        "GIT_CONFIG_KEY_4": "http.https://github.com/example-owner/example.git.proxy", "GIT_CONFIG_VALUE_4": "",
        "GIT_CONFIG_KEY_5": "http.https://github.com/example-owner/example.git.curloptResolve",
        "GIT_CONFIG_VALUE_5": "",
        "GIT_CONFIG_KEY_6": "push.pushOption", "GIT_CONFIG_VALUE_6": ""}
    assert env["GIT_CONFIG_VALUE_0"] == handoff.auth_env("t0k")["GIT_CONFIG_VALUE_0"]


# With the real transport rule, a push address that is not https is refused by git itself: nothing lands
# anywhere and no pull request is opened.
def test_the_push_uses_https_only(repo, tmp_path):
    elsewhere = _bare(tmp_path / "elsewhere.git")
    git(repo, "remote", "add", "origin", str(elsewhere))
    trusted = _bare(tmp_path / "github" / "example-owner" / "example.git")
    (repo / "a.txt").write_text("changed\n")
    forge = RecordingGitHub(trusted, protocol="https")
    with pytest.raises(handoff.GitError, match="^git push exited 128$"):
        handoff.hand_off(str(repo), "fix-backup", "main", {"summary": "Work"}, forge)
    assert _heads(elsewhere) == [] and _heads(trusted) == [] and forge.opened == []


# Git looks a push address up as a remote name first, so a remote named by that address would send the
# push wherever its url says. The configuration check refuses such a remote before any git call.
def test_a_remote_named_by_the_push_address_is_refused(repo, tmp_path):
    elsewhere = _bare(tmp_path / "elsewhere.git")
    trusted = _bare(tmp_path / "github" / "example-owner" / "example.git")
    forge = RecordingGitHub(trusted)
    git(repo, "config", f"remote.{forge.push_url}.url", str(elsewhere))
    (repo / "a.txt").write_text("changed\n")
    with pytest.raises(handoff.GitSafetyError) as refused:
        handoff.hand_off(str(repo), "fix-backup", "main", {"summary": "Work"}, forge)
    assert str(refused.value).startswith("Git refused 'remote.file://")
    assert _heads(elsewhere) == [] and _heads(trusted) == [] and forge.opened == []


# Two loopback servers: the push address answers with a redirect to the second, which must receive nothing.
# Git's own default would follow it. The repository's config cannot ask for it either, because every http.
# key there is refused (test_gitdiff.py).
@pytest.mark.needs_sockets
def test_the_push_follows_no_redirect(repo, tmp_path):
    from contextlib import ExitStack
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    source, target = [], []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            target.append(self.path)
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    class Redirect(Target):
        def do_GET(self):
            source.append(self.headers.get("Authorization"))
            self.send_response(302)
            self.send_header("Location", f"{target_url}{self.path}")
            self.end_headers()

    with ExitStack() as stack:
        urls = []
        for handler in (Redirect, Target):
            server = stack.enter_context(HTTPServer(("127.0.0.1", 0), handler))
            thread = Thread(target=server.serve_forever)
            thread.start()
            stack.callback(thread.join)
            stack.callback(server.shutdown)
            urls.append(f"http://127.0.0.1:{server.server_port}")
        source_url, target_url = urls
        (repo / "a.txt").write_text("changed\n")
        forge = RecordingGitHub(None)
        forge.push_url, forge.push_protocol = f"{source_url}/example-owner/example.git", "http"
        with pytest.raises(handoff.GitError, match="^git push exited 128$"):
            handoff.hand_off(str(repo), "fix-backup", "main", {"summary": "Work"}, forge)
    assert source == [None] and target == [] and forge.opened == []


# The suite's guard (conftest.py, no_outbound) stops a push to the real host before git starts, so the real
# client's default address can be inspected but never used.
def test_the_suites_guard_stops_a_push_to_the_real_host_before_git_runs(repo, monkeypatch, no_outbound):
    from runner import gitdiff

    (repo / "a.txt").write_text("changed\n")
    started = []
    real_run = gitdiff.subprocess.run

    def recording(argv, *args, **kwargs):
        started.append(argv)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(gitdiff.subprocess, "run", recording)
    with pytest.raises(pytest.fail.Exception) as stopped:
        handoff.hand_off(str(repo), "fix-backup", "main", {"summary": "Work"}, handoff.GitHub(REPO, "t0k"))
    assert str(stopped.value) == ("a test tried to reach GitHub with git push to "
                                  "https://github.com/example-owner/example.git, and the suite never leaves "
                                  "this machine")
    assert started and not [argv for argv in started if "push" in argv]
    no_outbound.clear()


def test_the_suites_guard_stops_an_api_call_and_a_connection_off_this_machine(no_outbound):
    import socket

    with pytest.raises(pytest.fail.Exception, match="the GitHub API at https://api.github.com/repos/"):
        handoff.GitHub(REPO, "t0k").open_pr("agent/x", "main", "title", "body")
    with socket.socket() as sock, pytest.raises(pytest.fail.Exception, match="the network at"):
        sock.connect(("192.0.2.1", 443))
    assert len(no_outbound) == 2
    no_outbound.clear()


TAG = ("-c", "tag.gpgsign=false", "-c", "user.name=Runner Test", "-c", "user.email=runner-test@example.invalid")


# push.followTags in the repository's config would also create every annotated tag reachable from the branch
# in the trusted repository. Only the agent branch may land there.
def test_the_push_carries_no_tags_even_when_the_repository_asks_for_them(repo, tmp_path):
    git(repo, *TAG, "tag", "-a", "v-main", "-m", "on main", "main")
    (repo / "a.txt").write_text("changed\n")
    git(repo, "commit", "--quiet", "-am", "on the branch")
    git(repo, *TAG, "tag", "-a", "v-branch", "-m", "on the branch")
    git(repo, "config", "push.followTags", "true")
    trusted = _bare(tmp_path / "github" / "example-owner" / "example.git")
    forge = RecordingGitHub(trusted)
    handoff.hand_off(str(repo), "fix-backup", "main", {"summary": "Work"}, forge, since="main")
    assert git(trusted, "for-each-ref", "--format=%(refname)").split() == ["refs/heads/agent/fix-backup"]


# push.gpgSign makes git run a signing program in the runner when the receiving side asks for a signed push.
@pytest.mark.parametrize("value", ["true", "if-asked"])
def test_the_push_is_never_signed_so_no_signing_program_runs(repo, tmp_path, monkeypatch, value):
    marker = tmp_path / "signer-ran"
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "gpg").write_text(f"#!/bin/sh\necho ran > '{marker}'\nexit 1\n")
    (tools / "gpg").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    git(repo, "config", "push.gpgSign", value)
    trusted = _bare(tmp_path / "github" / "example-owner" / "example.git")
    git(trusted, "config", "receive.certNonceSeed", "test-seed")
    (repo / "a.txt").write_text("changed\n")
    forge = RecordingGitHub(trusted)
    handoff.hand_off(str(repo), "fix-backup", "main", {"summary": "Work"}, forge)
    assert _heads(trusted) == ["refs/heads/agent/fix-backup"] and not marker.exists()


# The repository's config may hold no http. key at the check, but git reads the file again afterwards. A proxy
# written into it in between is overridden by the push's own setting for its exact address. Git is run here
# as GitHub.push runs it, with the proxy already in the config, which the check would have refused.
@pytest.mark.needs_sockets
def test_a_proxy_written_after_the_check_is_not_used_for_the_push(repo, tmp_path):
    import subprocess
    from contextlib import ExitStack
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    from runner import gitdiff

    seen = {"proxy": [], "push": []}

    def handler(name):
        class Answer(BaseHTTPRequestHandler):
            def do_GET(self):
                seen[name].append(self.path)
                self.send_response(404)
                self.end_headers()

            do_CONNECT = do_GET

            def log_message(self, *args):
                pass
        return Answer

    with ExitStack() as stack:
        ports = {}
        for name in ("proxy", "push"):
            server = stack.enter_context(HTTPServer(("127.0.0.1", 0), handler(name)))
            thread = Thread(target=server.serve_forever)
            thread.start()
            stack.callback(thread.join)
            stack.callback(server.shutdown)
            ports[name] = server.server_port
        url = f"http://127.0.0.1:{ports['push']}/example-owner/example.git"
        git(repo, "config", "http.proxy", f"http://127.0.0.1:{ports['proxy']}")
        env = gitdiff._environment(str(repo), handoff.push_env(handoff.auth_env("t0k"), url, "http"))
        done = subprocess.run(["git", *gitdiff.FIXED, "push", "--no-follow-tags", "--no-signed", url,
                               "refs/heads/agent/fix-backup:refs/heads/agent/fix-backup"],
                              cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode != 0
    assert seen["proxy"] == [] and seen["push"]
