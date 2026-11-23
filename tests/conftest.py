"""Shared fixtures: the stub control plane, a client for it, a throwaway git worktree on an agent branch,
a passing managed settings file for Claude starts, and `run_session`, which drives the fake provider
through the real Session with fast timings."""

from __future__ import annotations

import asyncio
import json
import pathlib
import socket
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from runner.stub_control_plane import TOKEN, StubControlPlane
from runner.control_plane import Client, Retry
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session, Timing

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

SOCKET_SKIP_REASON = "needs sockets, which this sandbox refuses"


def _sockets_refused() -> bool:
    try:
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with socket.socket() as client:
                client.connect(server.getsockname())
        return False
    except OSError:
        return True


SOCKETS_REFUSED = _sockets_refused()


def pytest_collection_modifyitems(items):
    if not SOCKETS_REFUSED:
        return
    for item in items:
        if "orch" in item.fixturenames or any(item.iter_markers("needs_sockets")):
            item.add_marker(pytest.mark.skip(reason=SOCKET_SKIP_REASON))


def _send_refused() -> bool:
    import socket

    a, b = socket.socketpair()
    try:
        a.send(b"\0")
        return False
    except PermissionError:
        return True
    finally:
        a.close()
        b.close()


def _wake_loops_with_write() -> None:
    import logging
    import os
    from asyncio import selector_events

    def _write_to_self(self) -> None:
        csock = self._csock
        if csock is None:
            return
        try:
            os.write(csock.fileno(), b"\0")
        except OSError:
            if self._debug:
                logging.getLogger("asyncio").debug("Fail to write a null byte into the self-pipe socket",
                                                   exc_info=True)

    selector_events.BaseSelectorEventLoop._write_to_self = _write_to_self


if _send_refused():
    _wake_loops_with_write()


def _short_temp_parent() -> str:
    import os
    import tempfile

    from runner.providers.claude import CLAUDE_TMPDIR_MAX
    from runner.providers.confine import TEMP_PREFIX

    uid = "9" * max(len(str(os.getuid())), 4)
    for parent in ("/tmp", "/tmp/claude", "/private/tmp/claude"):
        real = os.path.realpath(parent)
        if len(f"{real}/{TEMP_PREFIX}XXXXXXXX/claude-{uid}".encode()) > CLAUDE_TMPDIR_MAX:
            continue
        try:
            os.makedirs(real, mode=0o700, exist_ok=True)
            os.rmdir(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=real))
        except OSError:
            continue
        return real
    raise RuntimeError("no short writable parent for the session temp folders (tried /tmp and /tmp/claude)")


def pytest_configure(config):
    from runner.providers import confine

    config.addinivalue_line("markers", "needs_sockets: opens a real socket")
    confine.TEMP_PARENT = _short_temp_parent()


def pytest_sessionfinish(session, exitstatus):
    import shutil

    from runner.providers.confine import SessionTemp

    for path in SessionTemp.made:
        shutil.rmtree(path, ignore_errors=True)


class _NoGroup:

    def members(self) -> None:
        return None

    def name(self, pid: int) -> str:
        return "?"

    def signal(self, pid: int, sig: int) -> None:
        raise AssertionError("the suite's reaper signalled a process")


@pytest.fixture(autouse=True)
def _no_real_sweep(monkeypatch):
    from runner import reaper

    real_init = reaper.Reaper.__init__

    def init(self, group=None, *args, **kwargs):
        real_init(self, _NoGroup() if group is None else group, *args, **kwargs)

    monkeypatch.setattr(reaper.Reaper, "__init__", init)


def _github_host(address: str) -> bool:
    from urllib.parse import urlsplit

    if "://" in address:
        host = urlsplit(address).hostname or ""
    else:
        host = address.split(":", 1)[0].rpartition("@")[2] if ":" in address else ""
    return host == "github.com" or host.endswith(".github.com")


def _loopback_or_unix(sock: socket.socket, address: Any) -> bool:
    import ipaddress

    if sock.family == getattr(socket, "AF_UNIX", None):
        return True
    host = address[0] if isinstance(address, tuple) and address else ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


# No test may leave this machine. Several tests build the real GitHub client, whose default push address and
# API are on the real host, so a push or an API request there fails the test before git or urllib starts,
# and any Python socket connection to an address other than the loopback or a unix socket fails it too.
# Each stop is also recorded and fails the test at teardown, in case the code under test caught it.
@pytest.fixture(autouse=True)
def no_outbound(monkeypatch):
    import urllib.request

    from runner import handoff

    stopped: list[str] = []

    def stop(what: str) -> None:
        stopped.append(what)
        pytest.fail(f"a test tried to reach {what}, and the suite never leaves this machine", pytrace=False)

    real_git = handoff.git

    def guarded_git(cwd, *args, **kwargs):
        if args and args[0] == "push":
            for arg in args[1:]:
                if _github_host(arg):
                    stop(f"GitHub with git push to {arg}")
        return real_git(cwd, *args, **kwargs)

    real_open = urllib.request.OpenerDirector.open

    def guarded_open(self, fullurl, *args, **kwargs):
        url = fullurl if isinstance(fullurl, str) else fullurl.full_url
        if _github_host(url):
            stop(f"the GitHub API at {url}")
        return real_open(self, fullurl, *args, **kwargs)

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded_connect(self, address):
        if not _loopback_or_unix(self, address):
            stop(f"the network at {address!r}")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not _loopback_or_unix(self, address):
            stop(f"the network at {address!r}")
        return real_connect_ex(self, address)

    monkeypatch.setattr(handoff, "git", guarded_git)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", guarded_open)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield stopped
    if stopped:
        pytest.fail(f"a test tried to leave this machine: {stopped}", pytrace=False)


# Every Claude start checks the managed settings file, which on a real machine is root-owned. This writes a
# passing file to a temp path and reports root as the owner of whatever the check opens, keeping the real
# mode bits, so the suite needs no root.
@pytest.fixture(autouse=True)
def managed_policy(tmp_path_factory, monkeypatch):
    from os import fstat

    from runner import claude_policy

    path = tmp_path_factory.mktemp("claude-policy") / "managed-settings.json"
    path.write_text(json.dumps({
        "sandbox": {"enabled": True, "failIfUnavailable": True, "allowUnsandboxedCommands": False,
                    "network": {"allowedDomains": ["example.invalid"], "allowManagedDomainsOnly": True}},
    }), encoding="utf-8")
    path.chmod(0o644)
    monkeypatch.setattr(claude_policy, "MANAGED_SETTINGS", str(path))

    def metadata(fd):
        info = fstat(fd)
        return SimpleNamespace(st_mode=info.st_mode, st_uid=0)

    monkeypatch.setattr(claude_policy, "fstat", metadata)
    return path


async def _no_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


def fast_timing(**kw: Any) -> Timing:
    return Timing(pull_interval=kw.pop("pull_interval", 0.02), approval_ceiling=kw.pop("approval_ceiling", 5.0),
                  linger=kw.pop("linger", 0.0), retry=Retry(attempts=5, first=0, sleep=_no_sleep))


@pytest.fixture
def orch():
    server = StubControlPlane()
    yield server
    server.close()


@pytest.fixture
def client(orch):
    return Client(orch.url, TOKEN, timeout=5)


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet", "--initial-branch=main")
    git(repo, "config", "user.name", "Runner Test")
    git(repo, "config", "user.email", "runner-test@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.txt").write_text("hello\n")
    git(repo, "add", "README.txt")
    git(repo, "commit", "--quiet", "-m", "initial")
    git(repo, "switch", "--quiet", "-c", "agent/fix-backup")
    return repo


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def typed_result(summary="Work", **fields):
    return {"summary": summary, "files_changed": [], "tests_run": [], "tests_claim": None,
            "branch": "agent/fix-backup",
            "open_questions": [], "next_steps": [], "items": [
                {"label": "Work", "status": "unverified", "where": ""}], **fields}


def run_session(fixture: dict[str, Any] | str, client: Client, cwd, prompt: str = "Fix the backup job.",
                caps: Caps | None = None, timing: Timing | None = None, handoff=None, diff: bool = True):
    provider = FakeProvider(load(fixture) if isinstance(fixture, str) else fixture)
    session = Session(provider, client, Options(cwd=str(cwd), diff=diff), caps or Caps(wall_clock=30, grace=1),
                      timing or fast_timing(), handoff=handoff)
    outcome = asyncio.run(asyncio.wait_for(session.run(prompt), timeout=30))
    return outcome, provider, session

