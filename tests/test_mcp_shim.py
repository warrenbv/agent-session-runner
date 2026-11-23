"""The MCP bridge: the pinned catalogue, the stdio shim run as a real subprocess through bin/runner-mcp,
the session's tools socket against the stub control plane, and how each provider attaches the server."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from runner.stub_control_plane import TOKEN
from runner import mcp_shim, mcp_tools, session_socket
from runner.control_plane import Client
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session

from conftest import fast_timing

RUNNER = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = RUNNER / "examples"
WRAPPER = RUNNER / "bin" / "runner-mcp"
CATALOGUE_SHA256 = "809380c6d41ce7be2e245fe0e7a5a1d32b1c5e81445d75c8b1bb422e00c3a1bc"
LEAD_TOOLS = ["workers_start", "worker_changes", "worker_send_back", "worker_merge"]


def _can_bind_unix() -> bool:
    directory = tempfile.mkdtemp(prefix="m", dir=tempfile.gettempdir())
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(os.path.join(directory, "b"))
        return True
    except OSError:
        return False
    finally:
        probe.close()
        shutil.rmtree(directory, ignore_errors=True)


needs_unix = pytest.mark.skipif(not _can_bind_unix(), reason="this sandbox refuses binding a unix socket")


@pytest.fixture
def short_dir():
    path = tempfile.mkdtemp(prefix="m", dir=tempfile.gettempdir())
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_the_catalogue_is_pinned():
    assert mcp_tools.catalogue_hash() == CATALOGUE_SHA256


def test_every_tool_is_described_for_a_model_with_a_closed_schema():
    for tool in mcp_tools.TOOLS:
        assert set(tool) == {"name", "description", "inputSchema", "annotations"}
        assert "this session's own project" in tool["description"]
        if tool["name"] != mcp_tools.PROPOSE and tool["name"] not in mcp_tools.LEAD_TOOLS:
            assert len(re.findall(r"[.!?](?:\s|$)", tool["description"])) <= 3
        schema = tool["inputSchema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema.get("required", [])) <= set(schema["properties"])
        if not tool["annotations"]["readOnlyHint"]:
            assert "written by this session" in tool["description"]


def test_the_catalogues_names_are_pinned():
    assert sorted(mcp_tools.NAMES) == [
        "notes_search",
        "project_get",
        "sessions_propose",
        "task_comment_add",
        "task_complete",
        "task_create",
        "task_get",
        "task_update",
        "tasks_list",
        "worker_changes",
        "worker_merge",
        "worker_send_back",
        "workers_start",
    ]
    assert list(mcp_tools.NAMES) == [*BASE_TOOLS, "sessions_propose", *LEAD_TOOLS]
    assert len(set(mcp_tools.NAMES)) == len(mcp_tools.NAMES) == 13
    assert list(mcp_tools.BASE) == BASE_TOOLS and list(mcp_tools.LEAD_TOOLS) == LEAD_TOOLS


def test_the_catalogue_offers_sessions_propose_only_when_enabled():
    assert mcp_tools.PROPOSE == "sessions_propose"
    assert mcp_tools.offered(True) == (*BASE_TOOLS, "sessions_propose")
    assert mcp_tools.offered(False) == tuple(BASE_TOOLS) and mcp_tools.PROPOSE not in mcp_tools.offered(False)
    assert mcp_tools.claude_names(False) == [f"mcp__platform__{n}" for n in BASE_TOOLS]
    assert mcp_tools.claude_names(True)[-1] == "mcp__platform__sessions_propose"
    assert mcp_tools.withheld(False) == ("sessions_propose", *LEAD_TOOLS)
    assert mcp_tools.withheld(True) == tuple(LEAD_TOOLS)


@pytest.mark.parametrize("propose", [False, True])
@pytest.mark.parametrize("lead", [False, None, "true", 1, "yes"])
def test_only_a_literal_true_lead_offers_any_of_the_four(propose, lead):
    shown = mcp_tools.offered(propose, lead)
    assert not set(shown) & set(LEAD_TOOLS)
    assert shown == mcp_tools.offered(propose)
    assert set(mcp_tools.withheld(propose, lead)) >= set(LEAD_TOOLS)


@pytest.mark.parametrize("propose", [False, True])
def test_a_lead_is_offered_the_four_and_not_sessions_propose(propose):
    assert mcp_tools.offered(propose, True) == (*BASE_TOOLS, *LEAD_TOOLS)
    assert mcp_tools.withheld(propose, True) == ("sessions_propose",)
    assert mcp_tools.claude_names(propose, True) == [f"mcp__platform__{n}" for n in [*BASE_TOOLS, *LEAD_TOOLS]]


def _tool_named(name):
    return next(t for t in mcp_tools.TOOLS if t["name"] == name)


def test_the_lead_tools_have_the_plans_closed_schemas():
    start = _tool_named("workers_start")["inputSchema"]
    assert start["additionalProperties"] is False and start["required"] == ["workers"]
    assert set(start["properties"]) == {"workers"}
    workers = start["properties"]["workers"]
    assert (workers["type"], workers["minItems"], workers["maxItems"]) == ("array", 1, 4)
    item = workers["items"]
    assert item["type"] == "object" and item["additionalProperties"] is False
    assert set(item["properties"]) == {"title", "brief", "provider", "model", "effort", "wall_clock_minutes"}
    assert sorted(item["required"]) == ["brief", "title"]
    props = item["properties"]
    assert (props["title"]["minLength"], props["title"]["maxLength"]) == (1, 120)
    assert (props["brief"]["minLength"], props["brief"]["maxLength"]) == (1, 16000)
    assert props["provider"]["enum"] == ["codex", "claude"]
    assert "enum" not in props["model"]
    assert (props["model"]["type"], props["model"]["minLength"], props["model"]["maxLength"],
            props["model"]["pattern"]) == ("string", 1, 40, "^[a-z0-9.-]{1,40}$")
    assert props["effort"]["enum"] == ["medium", "high", "xhigh"]
    assert (props["wall_clock_minutes"]["type"], props["wall_clock_minutes"]["minimum"],
            props["wall_clock_minutes"]["maximum"]) == ("integer", 5, 120)

    changes = _tool_named("worker_changes")["inputSchema"]
    assert changes["additionalProperties"] is False and set(changes["properties"]) == {"worker", "page"}
    assert changes["required"] == ["worker"]
    assert (changes["properties"]["worker"]["type"], changes["properties"]["worker"]["minimum"]) == ("integer", 1)
    assert (changes["properties"]["page"]["type"], changes["properties"]["page"]["minimum"]) == ("integer", 1)

    back = _tool_named("worker_send_back")["inputSchema"]
    assert back["additionalProperties"] is False and set(back["properties"]) == {"worker", "notes"}
    assert sorted(back["required"]) == ["notes", "worker"]
    assert (back["properties"]["notes"]["minLength"], back["properties"]["notes"]["maxLength"]) == (1, 10000)

    merge = _tool_named("worker_merge")["inputSchema"]
    assert merge["additionalProperties"] is False and set(merge["properties"]) == {"worker"}
    assert merge["required"] == ["worker"]


def test_only_worker_changes_reads_and_none_of_the_four_is_open_world():
    for name in LEAD_TOOLS:
        assert _tool_named(name)["annotations"] == {"readOnlyHint": name == "worker_changes", "destructiveHint": False,
                                                    "openWorldHint": False}, name


def test_the_lead_tools_say_what_they_do_what_comes_back_and_their_limits():
    text = {name: _tool_named(name)["description"] for name in LEAD_TOOLS}
    start = text["workers_start"]
    for words in ("one to four", "default branch", "cannot start or propose", "Returns each worker's number",
                  "runs now or waits", "you are told when each worker ends", "one slot on each machine", "30 starts",
                  "5 to 120 minutes, default 30", "default high", "You choose each worker's model and effort",
                  "the project's default model family", "takes that provider's default family",
                  "A lighter model family may be named", "Some model families are never offered to a worker",
                  "refused with the families that are",
                  "medium, high or xhigh", "medium only for a small mechanical piece",
                  "xhigh for a hard one"):
        assert words in start, words
    assert "low" not in start.replace("allow", "")
    changes = text["worker_changes"]
    for words in ("statistics and patch", "page", "`whole`", "Only your own workers"):
        assert words in changes, words
    back = text["worker_send_back"]
    for words in ("once it has ended", "same branch and pull request", "at most three", "Only your own ended"):
        assert words in back, words
    merge = text["worker_merge"]
    for words in ("Code decides whether the merge goes through", "held", "Merge and Close", "tell your owner why",
                  "read the web", "Only your own", "cannot revert"):
        assert words in merge, words
    for name, description in text.items():
        assert "Fable" not in description and "Astra" not in description, name


def test_sessions_propose_is_the_plans_description_and_closed_schema():
    tool = next(t for t in mcp_tools.TOOLS if t["name"] == "sessions_propose")
    schema = tool["inputSchema"]
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"pieces", "why"} and sorted(schema["required"]) == ["pieces", "why"]
    pieces = schema["properties"]["pieces"]
    assert (pieces["type"], pieces["minItems"], pieces["maxItems"]) == ("array", 1, 4)
    item = pieces["items"]
    assert item["type"] == "object" and item["additionalProperties"] is False
    assert set(item["properties"]) == {"title", "brief"} and sorted(item["required"]) == ["brief", "title"]
    assert (item["properties"]["title"]["minLength"], item["properties"]["title"]["maxLength"]) == (1, 120)
    assert (item["properties"]["brief"]["minLength"], item["properties"]["brief"]["maxLength"]) == (1, 4000)
    why = schema["properties"]["why"]
    assert (why["type"], why["minLength"], why["maxLength"]) == ("string", 1, 300)
    assert tool["annotations"] == {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}


def _wrapper_command() -> list[str]:
    if pathlib.Path(sys.prefix).resolve().parent == RUNNER:
        return [sys.executable, "-I", str(WRAPPER)]
    code = ("import runpy, sys; sys.prefix = sys.argv[1]; sys.argv = sys.argv[2:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')")
    return [sys.executable, "-I", "-c", code, str(RUNNER / ".venv"), str(WRAPPER)]


class Shim:
    def __init__(self, socket_path: str | None, extra_env: dict[str, str] | None = None) -> None:
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **(extra_env or {})}
        if socket_path is not None:
            env[mcp_tools.SOCKET_ENV] = socket_path
        self.proc = subprocess.Popen(_wrapper_command(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    def send(self, message) -> None:
        raw = message if isinstance(message, bytes) else json.dumps(message).encode()
        self.proc.stdin.write(raw + b"\n")
        self.proc.stdin.flush()

    def ask(self, message):
        self.send(message)
        return json.loads(self.proc.stdout.readline())

    def close(self) -> tuple[bytes, bytes]:
        if not self.proc.stdin.closed:
            self.proc.stdin.close()
        out, err = self.proc.stdout.read(), self.proc.stderr.read()
        self.proc.wait(timeout=10)
        return out, err


@pytest.fixture
def shim_factory():
    made = []

    def make(socket_path, **kw):
        shim = Shim(socket_path, **kw)
        made.append(shim)
        return shim
    yield make
    for shim in made:
        if shim.proc.poll() is None:
            shim.proc.kill()
            shim.proc.wait()


class StubServer:

    def __init__(self, path: str, reply) -> None:
        self.path, self.reply, self.requests = path, reply, []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(8)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn):
        with conn:
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            request = json.loads(data.split(b"\n")[0])
            self.requests.append(request)
            answer = self.reply(request)
            if isinstance(answer, tuple):
                time.sleep(answer[0])
                answer = answer[1]
            if answer is not None:
                try:
                    conn.sendall(answer)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    def close(self):
        self.sock.close()


def _ok(request):
    return json.dumps({"ok": True, "result": {"items": [], "echo": request}, "error": None}).encode() + b"\n"


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}


@pytest.mark.parametrize("revision", ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05", "2026-07-28", None])
def test_the_handshake_offers_the_clients_revision_or_the_latest(shim_factory, short_dir, revision):
    shim = shim_factory(os.path.join(short_dir, "none"))
    params = {**INIT["params"], "protocolVersion": revision}
    answer = shim.ask({**INIT, "params": params})
    expected = revision if revision in mcp_shim.SUPPORTED else "2025-11-25"
    assert answer == {"jsonrpc": "2.0", "id": 1, "result": {
        "protocolVersion": expected, "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "platform", "version": mcp_tools.SERVER_VERSION}}}
    shim.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    shim.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 9}})
    assert shim.ask({"jsonrpc": "2.0", "id": "p", "method": "ping"}) == {"jsonrpc": "2.0", "id": "p", "result": {}}
    shim.proc.stdin.close()
    out, _ = shim.close()
    assert out == b"" and shim.proc.returncode == 0


@needs_unix
def test_tools_list_is_the_catalogue_and_a_call_relays(shim_factory, short_dir):
    path = os.path.join(short_dir, "s")
    server = StubServer(path, _ok)
    try:
        shim = shim_factory(path)
        shim.ask(INIT)
        listed = shim.ask({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert listed["result"] == {"tools": list(mcp_tools.TOOLS)}
        answer = shim.ask({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                           "params": {"name": "task_get", "arguments": {"task_id": 4}}})
        assert server.requests[-1] == {"tool": "task_get", "arguments": {"task_id": 4}}
        result = answer["result"]
        assert result["isError"] is False
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
        assert result["structuredContent"]["echo"] == {"tool": "task_get", "arguments": {"task_id": 4}}
        shim.ask({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "project_get"}})
        assert server.requests[-1] == {"tool": "project_get", "arguments": {}}
    finally:
        server.close()


@needs_unix
def test_a_list_result_is_text_only(shim_factory, short_dir):
    path = os.path.join(short_dir, "s")
    server = StubServer(path, lambda r: b'{"ok": true, "result": [1, 2], "error": null}\n')
    try:
        shim = shim_factory(path)
        answer = shim.ask({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "tasks_list"}})
        assert answer["result"] == {"content": [{"type": "text", "text": "[1, 2]"}], "isError": False}
    finally:
        server.close()


@needs_unix
def test_a_refusal_is_its_sentence_with_is_error(shim_factory, short_dir):
    path = os.path.join(short_dir, "s")
    sentence = "There is no task 7 in this project; tasks_list shows the open tasks and their numbers."
    server = StubServer(path, lambda r: json.dumps({"ok": False, "result": None, "error": sentence}).encode() + b"\n")
    try:
        shim = shim_factory(path)
        answer = shim.ask({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "task_get", "arguments": {"task_id": 7}}})
        assert answer["result"] == {"content": [{"type": "text", "text": sentence}], "isError": True}
    finally:
        server.close()


@needs_unix
@pytest.mark.parametrize("reply", [None, b"not json\n", b'{"ok": false}\n', b'{"ok": true'])
def test_a_broken_socket_answer_is_one_plain_sentence(shim_factory, short_dir, reply):
    path = os.path.join(short_dir, "s")
    server = StubServer(path, lambda r: reply)
    try:
        shim = shim_factory(path)
        answer = shim.ask({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "tasks_list"}})
        assert answer["result"]["isError"] is True
        assert answer["result"]["content"][0]["text"] in (mcp_shim.NO_ANSWER, mcp_shim.BAD_REPLY)
    finally:
        server.close()


@needs_unix
def test_a_slow_socket_times_out_with_a_sentence(short_dir):
    path = os.path.join(short_dir, "s")
    server = StubServer(path, lambda r: (1.0, _ok(r)))
    try:
        reply = mcp_shim.Shim(path, timeout=0.2).relay("tasks_list", {})
        assert reply == {"ok": False, "result": None, "error": mcp_shim.NO_ANSWER}
    finally:
        server.close()


def test_unknown_methods_malformed_lines_and_requests(shim_factory, short_dir):
    shim = shim_factory(os.path.join(short_dir, "none"))
    assert shim.ask({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})["error"]["code"] == -32601
    assert shim.ask({"jsonrpc": "2.0", "id": 2, "method": "server/discover"})["error"]["code"] == -32601
    assert shim.ask(b"{not json")["error"] == {"code": -32700, "message": "Parse error"}
    assert shim.ask(b"[1, 2]")["error"]["code"] == -32600
    assert shim.ask({"jsonrpc": "2.0", "id": True, "method": "ping"})["error"]["code"] == -32600
    shim.send({"jsonrpc": "2.0", "id": 5, "result": {}})
    assert shim.ask({"jsonrpc": "2.0", "id": 6, "method": "ping"})["id"] == 6
    shim.proc.stdin.close()
    out, err = shim.close()
    assert out == b""
    assert b"runner-mcp:" in err or err == b""


@pytest.mark.parametrize("configured", [False, True])
def test_with_no_socket_it_starts_lists_nothing_and_answers_a_call(shim_factory, short_dir, configured):
    shim = shim_factory(os.path.join(short_dir, "missing") if configured else None)
    assert shim.ask(INIT)["result"]["protocolVersion"] == "2025-11-25"
    assert shim.ask({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"] == {"tools": []}
    answer = shim.ask({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "tasks_list"}})
    assert answer["result"] == {"content": [{"type": "text", "text": mcp_shim.NO_TOOLS}], "isError": True}


def test_importing_the_shim_loads_no_sdk_and_no_other_runner_module():
    code = (f"import sys; sys.path.insert(0, {str(RUNNER)!r})\n"
            "import runner.mcp_shim\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('runner', 'mcp', 'claude_agent_sdk', 'openai_codex', 'pydantic', 'anyio')))")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.replace("'", '"')) == ["runner", "runner.mcp_shim", "runner.mcp_tools"]


def test_the_shim_reads_no_environment_variable_but_its_socket(monkeypatch):
    read: list[str] = []

    class Watched(dict):
        def __getitem__(self, key):
            read.append(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            read.append(key)
            return super().get(key, default)

        def __contains__(self, key):
            read.append(key)
            return super().__contains__(key)

    monkeypatch.setattr(os, "environ", Watched({mcp_tools.SOCKET_ENV: "/nonexistent/s", "SECRET_TOKEN": "x"}))

    class Pipe:
        def __init__(self, data=b""):
            self.data, self.out = data, b""

        def readline(self, _n=-1):
            line, _, self.data = self.data.partition(b"\n")
            return line + b"\n" if line or self.data else b""

        def write(self, b):
            self.out += b

        def flush(self):
            pass

    stdin = Pipe(json.dumps(INIT).encode() + b"\n" + b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')
    stdout = Pipe()
    monkeypatch.setattr(sys, "stdin", type("S", (), {"buffer": stdin})())
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": stdout})())
    assert mcp_shim.main() == 0
    assert set(read) == {mcp_tools.SOCKET_ENV}
    assert len(stdout.out.splitlines()) == 2
    source = (RUNNER / "runner" / "mcp_shim.py").read_text(encoding="utf-8")
    assert source.count("os.environ") == 1 and "getenv" not in source


async def _ask(path: str, *lines: bytes) -> list[dict]:
    reader, writer = await asyncio.open_unix_connection(path)
    out = []
    for line in lines:
        writer.write(line + b"\n")
        await writer.drain()
        out.append(json.loads(await reader.readline()))
    writer.close()
    return out


@needs_unix
def test_the_socket_forwards_to_the_control_plane_and_cleans_up(orch, short_dir):
    client = Client(orch.url, TOKEN, timeout=5)

    async def go():
        tools = session_socket.ToolSocket(client)
        path = await tools.open(short_dir)
        assert path is not None and os.path.dirname(path).startswith(short_dir)
        assert stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600 and stat.S_ISSOCK(os.stat(path).st_mode)
        one, two = await asyncio.gather(
            _ask(path, b'{"tool": "tasks_list", "arguments": {"status": "done"}}', b'{"tool": "project_get"}'),
            _ask(path, b'{"tool": "task_get", "arguments": {"task_id": 3}}'))
        await tools.close()
        return path, one, two

    path, one, two = asyncio.run(go())
    assert one[0] == {"ok": True, "result": {"tool": "tasks_list", "arguments": {"status": "done"}}, "error": None}
    assert one[1]["result"] == {"tool": "project_get", "arguments": {}}
    assert two[0]["result"]["arguments"] == {"task_id": 3}
    assert sorted(c["tool"] for c in orch.tool_calls) == ["project_get", "task_get", "tasks_list"]
    assert all(r["auth"] == f"Bearer {TOKEN}" for r in orch.requests if r["op"] == "sessions.tool_call")
    assert not os.path.exists(path) and not os.path.exists(os.path.dirname(path))
    assert os.listdir(short_dir) == []


@needs_unix
def test_the_socket_answers_failures_with_sentences(orch, short_dir):
    client = Client(orch.url, TOKEN, timeout=5)
    refusal = "There is no tool called 'x'; the tools you can use are tasks_list."

    async def go():
        tools = session_socket.ToolSocket(client)
        path = await tools.open(short_dir)
        out = {}
        orch.tool_reply = lambda body: (200, {"ok": False, "result": None, "error": refusal})
        out["refused"] = (await _ask(path, b'{"tool": "x", "arguments": {}}'))[0]
        orch.tool_reply = lambda body: (500, {"error": {"code": "internal", "message": "boom"}})
        out["5xx"] = (await _ask(path, b'{"tool": "tasks_list"}'))[0]
        orch.tool_reply = lambda body: (400, {"error": {"code": "invalid", "message": "no"}})
        out["400"] = (await _ask(path, b'{"tool": "tasks_list"}'))[0]
        orch.tool_reply = lambda body: (200, {"unexpected": True})
        out["malformed"] = (await _ask(path, b'{"tool": "tasks_list"}'))[0]
        out["unreadable"] = (await _ask(path, b'{"tool": 5}', b"not json"))
        orch.revoke()
        out["ended"] = (await _ask(path, b'{"tool": "tasks_list"}'))[0]
        await tools.close()
        return out

    out = asyncio.run(go())
    assert out["refused"] == {"ok": False, "result": None, "error": refusal}
    assert out["5xx"]["error"] == session_socket.UNREACHABLE
    assert out["400"]["error"] == session_socket.REFUSED
    assert out["malformed"]["error"] == session_socket.BROKEN
    assert [a["error"] for a in out["unreadable"]] == [session_socket.UNREADABLE] * 2
    assert out["ended"]["error"] == session_socket.ENDED
    for answer in (out["5xx"], out["400"], out["malformed"], out["ended"]):
        assert answer["ok"] is False and answer["result"] is None and "boom" not in answer["error"]


@needs_unix
def test_the_socket_times_out_a_slow_control_plane(short_dir):
    class Slow:
        def call(self, op, payload):
            time.sleep(1.0)
            return {"ok": True, "result": {}, "error": None}

    async def go():
        tools = session_socket.ToolSocket(Slow(), timeout=0.1)
        path = await tools.open(short_dir)
        answer = (await _ask(path, b'{"tool": "tasks_list"}'))[0]
        await tools.close()
        return answer

    assert asyncio.run(go())["error"] == session_socket.SLOW


def test_the_shims_answers_from_a_relayed_reply(monkeypatch):
    shim = mcp_shim.Shim("/nonexistent/s")
    replies = iter([{"ok": True, "result": {"id": 1}, "error": None}, {"ok": True, "result": [1], "error": None},
                    {"ok": False, "result": None, "error": "No."}])
    seen = []
    monkeypatch.setattr(shim, "relay", lambda tool, args: seen.append((tool, args)) or next(replies))
    call = {"name": "task_get", "arguments": {"task_id": 1}}
    assert shim.call(1, call)["result"] == {"content": [{"type": "text", "text": '{"id": 1}'}], "isError": False,
                                            "structuredContent": {"id": 1}}
    assert shim.call(2, {"name": "tasks_list", "arguments": "junk"})["result"] == {
        "content": [{"type": "text", "text": "[1]"}], "isError": False}
    assert shim.call(3, call)["result"] == {"content": [{"type": "text", "text": "No."}], "isError": True}
    assert seen == [("task_get", {"task_id": 1}), ("tasks_list", {}), ("task_get", {"task_id": 1})]
    assert shim.call(4, {})["result"]["isError"] is True
    assert len(seen) == 3


def test_the_socket_handler_forwards_and_answers_failures_bind_free(orch):
    client = Client(orch.url, TOKEN, timeout=5)
    tools = session_socket.ToolSocket(client)

    async def go():
        out = [await tools.handle(b'{"tool": "tasks_list", "arguments": {"status": "done"}}'),
               await tools.handle(b'{"tool": "project_get"}'),
               await tools.handle(b'{"tool": 5}'), await tools.handle(b"not json"),
               await tools.handle(b'{"tool": "x", "arguments": []}')]
        orch.tool_reply = lambda body: (200, {"ok": False, "result": None, "error": "Refused for a reason."})
        out.append(await tools.handle(b'{"tool": "tasks_list"}'))
        orch.tool_reply = lambda body: (500, {"error": {"code": "internal", "message": "boom"}})
        out.append(await tools.handle(b'{"tool": "tasks_list"}'))
        orch.tool_reply = lambda body: (400, {"error": {"code": "invalid", "message": "boom"}})
        out.append(await tools.handle(b'{"tool": "tasks_list"}'))
        orch.tool_reply = lambda body: (200, {"unexpected": True})
        out.append(await tools.handle(b'{"tool": "tasks_list"}'))
        orch.revoke()
        out.append(await tools.handle(b'{"tool": "tasks_list"}'))
        return out

    out = asyncio.run(go())
    assert out[0] == {"ok": True, "result": {"tool": "tasks_list", "arguments": {"status": "done"}}, "error": None}
    assert out[1]["result"] == {"tool": "project_get", "arguments": {}}
    assert [a["error"] for a in out[2:5]] == [session_socket.UNREADABLE] * 3
    assert [a["error"] for a in out[5:]] == ["Refused for a reason.", session_socket.UNREACHABLE,
                                             session_socket.REFUSED, session_socket.BROKEN, session_socket.ENDED]
    assert all(a["ok"] is False and a["result"] is None for a in out[2:])
    assert orch.tool_calls[:2] == [{"tool": "tasks_list", "arguments": {"status": "done"}},
                                   {"tool": "project_get", "arguments": {}}]
    assert all(r["auth"] == f"Bearer {TOKEN}" for r in orch.requests if r["op"] == "sessions.tool_call")


def test_the_socket_handler_times_out_bind_free():
    class Slow:
        def call(self, op, payload):
            time.sleep(0.5)
            return {"ok": True, "result": {}, "error": None}

    answer = asyncio.run(session_socket.ToolSocket(Slow(), timeout=0.05).handle(b'{"tool": "tasks_list"}'))
    assert answer == {"ok": False, "result": None, "error": session_socket.SLOW}


def test_a_socket_that_cannot_be_opened_leaves_nothing_behind(monkeypatch, short_dir, caplog):
    async def refuse(*_a, **_kw):
        raise PermissionError("no")
    monkeypatch.setattr(asyncio, "start_unix_server", refuse)

    async def go():
        tools = session_socket.ToolSocket(None)
        return await tools.open(short_dir), tools
    path, tools = asyncio.run(go())
    assert path is None and tools.path is None and os.listdir(short_dir) == []
    assert "without its project tools" in caplog.text


def test_no_runtime_directory_refuses_tools(caplog):
    async def go():
        return await session_socket.ToolSocket(None).open(None)
    assert asyncio.run(go()) is None
    assert "without its project tools" in caplog.text


def test_runtime_dir_is_the_users_own_under_run_user(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 987654)
    assert session_socket.runtime_dir() is None


RESULT = {"type": "ResultMessage", "subtype": "success", "is_error": False, "num_turns": 1, "duration_ms": 1,
          "duration_api_ms": 1, "session_id": "fake", "result": "done"}


class ToolCallingProvider(FakeProvider):

    answers: list = []

    async def start(self, ctx):
        await super().start(ctx)
        if ctx.mcp_socket:
            self.answers.extend(await _ask(ctx.mcp_socket, b'{"tool": "tasks_list", "arguments": {}}'))


@pytest.mark.parametrize("tools", [False, pytest.param(True, marks=needs_unix)])
def test_a_session_opens_its_socket_only_with_tools_and_removes_it(client, orch, worktree, short_dir, tools):
    ToolCallingProvider.answers = []
    provider = ToolCallingProvider({"provider": "claude-code", "turns": [[RESULT]]})
    session = Session(provider, client, Options(cwd=str(worktree), diff=False, tools=tools),
                      Caps(wall_clock=30, grace=1), fast_timing())
    session.tools_parent = lambda: short_dir
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix it."), timeout=30))
    assert outcome.reason == "completed"
    if tools:
        assert provider.ctx.mcp_socket is not None and provider.ctx.mcp_socket.startswith(short_dir)
        assert ToolCallingProvider.answers[0]["ok"] is True and orch.tool_calls == [{"tool": "tasks_list",
                                                                                     "arguments": {}}]
    else:
        assert provider.ctx.mcp_socket is None and session.tools is None and orch.tool_calls == []
    assert os.listdir(short_dir) == []


def test_a_session_without_a_runtime_directory_runs_without_tools(client, worktree):
    provider = ToolCallingProvider({"provider": "claude-code", "turns": [[RESULT]]})
    session = Session(provider, client, Options(cwd=str(worktree), diff=False, tools=True),
                      Caps(wall_clock=30, grace=1), fast_timing())
    session.tools_parent = lambda: None
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix it."), timeout=30))
    assert outcome.reason == "completed" and provider.ctx.mcp_socket is None


def test_the_tools_flag_reaches_the_runner_only_from_a_literal_true():
    from runner.__main__ import parse
    base = ["--provider", "fake", "--cwd", "/tmp/wt", "--prompt-file", "/tmp/p"]
    assert parse(base).tools is False and parse([*base, "--tools"]).tools is True
    assert parse(base).propose is False and parse([*base, "--propose"]).propose is True
    assert parse(base).lead is False and parse([*base, "--lead"]).lead is True


def _ctx(**kw):
    from runner.providers.base import ProviderContext

    async def never(_payload):
        raise AssertionError("not called")
    return ProviderContext(request_approval=never, cwd="/tmp/wt", **kw)


def test_claude_attaches_nothing_without_tools():
    from runner.providers.claude import ClaudeProvider

    options = ClaudeProvider().options(_ctx(allowed_tools=["Read"]))
    assert options.mcp_servers == {} and options.strict_mcp_config is False and options.allowed_tools == ["Read"]


BASE_TOOLS = ["tasks_list", "task_get", "project_get", "notes_search", "task_create", "task_update", "task_complete",
        "task_comment_add"]


def test_claude_gets_the_shim_as_its_only_server_with_base_tools_allowed():
    from runner.providers.claude import ClaudeProvider

    options = ClaudeProvider().options(_ctx(allowed_tools=["Read"], mcp_socket="/run/user/1000/mcp-x/s"))
    assert options.mcp_servers == {"platform": {"type": "stdio", "command": os.path.join(sys.prefix, "bin", "runner-mcp"),
                                               "args": [], "env": {"AGENT_MCP_SOCKET": "/run/user/1000/mcp-x/s"}}}
    assert options.strict_mcp_config is True
    assert options.allowed_tools == ["Read", *[f"mcp__platform__{n}" for n in BASE_TOOLS]]
    assert "mcp__platform__sessions_propose" in options.disallowed_tools
    assert "mcp__platform__sessions_propose" not in options.allowed_tools
    assert not any(t in ("mcp__platform", "mcp__*") for t in options.allowed_tools)
    assert [t for t in options.disallowed_tools if t.startswith("mcp__")] == [
        f"mcp__platform__{n}" for n in ["sessions_propose", *LEAD_TOOLS]]


def test_claude_with_proposals_allows_the_base_tools_and_sessions_propose_only():
    from runner.providers.claude import ClaudeProvider

    options = ClaudeProvider().options(_ctx(allowed_tools=["Read"], mcp_socket="/run/user/1000/mcp-x/s", propose=True))
    assert options.allowed_tools == ["Read", *[f"mcp__platform__{n}" for n in [*BASE_TOOLS, "sessions_propose"]]]
    assert [t for t in options.disallowed_tools if t.startswith("mcp__")] == [f"mcp__platform__{n}" for n in LEAD_TOOLS]


@pytest.mark.parametrize("propose", [False, True])
@pytest.mark.parametrize("lead", [False, None, "true", 1])
def test_claude_without_a_literal_true_lead_offers_none_of_the_four(propose, lead):
    from runner.providers.claude import ClaudeProvider

    options = ClaudeProvider().options(_ctx(mcp_socket="/run/user/1000/mcp-x/s", propose=propose, lead=lead))
    for name in LEAD_TOOLS:
        assert f"mcp__platform__{name}" not in options.allowed_tools
        assert f"mcp__platform__{name}" in options.disallowed_tools


@pytest.mark.parametrize("propose", [False, True])
def test_claude_as_a_lead_allows_the_four_and_refuses_sessions_propose(propose):
    from claude_agent_sdk import PermissionResultDeny
    from runner.providers.claude import ClaudeProvider

    ctx = _ctx(allowed_tools=["Read"], mcp_socket="/run/user/1000/mcp-x/s", propose=propose, lead=True)
    options = ClaudeProvider().options(ctx)
    assert options.allowed_tools == ["Read", *[f"mcp__platform__{n}" for n in [*BASE_TOOLS, *LEAD_TOOLS]]]
    assert [t for t in options.disallowed_tools if t.startswith("mcp__")] == ["mcp__platform__sessions_propose"]
    provider = ClaudeProvider()
    provider.ctx = ctx
    assert isinstance(asyncio.run(provider._can_use_tool("mcp__platform__sessions_propose", {}, None)),
                      PermissionResultDeny)


def test_claude_refuses_a_lead_tool_without_asking_when_not_a_lead():
    from claude_agent_sdk import PermissionResultDeny
    from runner.providers.claude import ClaudeProvider

    provider = ClaudeProvider()
    provider.ctx = _ctx(mcp_socket="/run/user/1000/mcp-x/s", propose=True)
    for name in LEAD_TOOLS:
        assert isinstance(asyncio.run(provider._can_use_tool(f"mcp__platform__{name}", {}, None)), PermissionResultDeny)


def test_claude_without_tools_names_no_mcp_tool_either_way():
    from runner.providers.claude import ClaudeProvider

    for propose in (False, True):
        options = ClaudeProvider().options(_ctx(allowed_tools=["Read"], propose=propose))
        assert not any(t.startswith("mcp__") for t in [*options.allowed_tools, *options.disallowed_tools])


def test_claude_refuses_sessions_propose_without_asking_when_not_offered():
    from claude_agent_sdk import PermissionResultDeny
    from runner.providers.claude import ClaudeProvider

    provider = ClaudeProvider()
    provider.ctx = _ctx(mcp_socket="/run/user/1000/mcp-x/s")
    denied = asyncio.run(provider._can_use_tool("mcp__platform__sessions_propose", {}, None))
    assert isinstance(denied, PermissionResultDeny)


def test_claude_refuses_another_mcp_tool_without_asking_when_tools_are_on():
    from claude_agent_sdk import PermissionResultDeny
    from runner.providers.claude import ClaudeProvider

    provider = ClaudeProvider()
    provider.ctx = _ctx(mcp_socket="/run/user/1000/mcp-x/s")
    denied = asyncio.run(provider._can_use_tool("mcp__decoy__steal", {}, None))
    assert isinstance(denied, PermissionResultDeny)


def test_codex_registers_the_server_off_without_tools():
    from runner.providers.codex import (CONFIG_OVERRIDES, MCP_SERVER, CodexProvider, projects_override,
                                        read_overrides)

    provider = CodexProvider()
    config = provider.config(_ctx())
    assert config.config_overrides == (*CONFIG_OVERRIDES, projects_override("/tmp/wt"), MCP_SERVER,
                                       *read_overrides(provider.read_scope(_ctx())))
    assert config.env == {"TMPDIR": provider.temp.path}


def test_codex_switches_the_server_on_and_passes_the_socket():
    from runner.providers.codex import (CONFIG_OVERRIDES, MCP_SERVER, CodexProvider, projects_override,
                                        read_overrides)

    provider = CodexProvider()
    config = provider.config(_ctx(mcp_socket="/run/user/1000/mcp-x/s"))
    assert config.config_overrides == (*CONFIG_OVERRIDES, projects_override("/tmp/wt"), MCP_SERVER,
                                       *read_overrides(provider.read_scope(_ctx())), "mcp_servers.platform.enabled=true")
    assert config.env == {"AGENT_MCP_SOCKET": "/run/user/1000/mcp-x/s", "TMPDIR": provider.temp.path}
    assert not any("sessions_propose" in o for o in config.config_overrides)


def test_codex_with_proposals_widens_enabled_tools_to_the_full_catalogue():
    import tomllib
    from runner.providers.codex import CONFIG_OVERRIDES, MCP_PROPOSE, MCP_SERVER, CodexProvider, projects_override, \
        read_overrides

    provider = CodexProvider()
    config = provider.config(_ctx(mcp_socket="/run/user/1000/mcp-x/s", propose=True))
    assert config.config_overrides == (*CONFIG_OVERRIDES, projects_override("/tmp/wt"), MCP_SERVER,
                                       *read_overrides(provider.read_scope(_ctx())), "mcp_servers.platform.enabled=true",
                                       MCP_PROPOSE)
    key, _, value = MCP_PROPOSE.partition("=")
    assert key == "mcp_servers.platform.enabled_tools"
    assert tomllib.loads(f"x = {value}")["x"] == [*BASE_TOOLS, "sessions_propose"]
    assert MCP_PROPOSE not in CodexProvider().config(_ctx(propose=True)).config_overrides


@pytest.mark.parametrize("propose", [False, True])
def test_codex_as_a_lead_enables_the_four_and_not_sessions_propose(propose):
    import tomllib
    from runner.providers.codex import (CONFIG_OVERRIDES, MCP_LEAD, MCP_PROPOSE, MCP_SERVER, CodexProvider,
                                        projects_override, read_overrides)

    provider = CodexProvider()
    config = provider.config(_ctx(mcp_socket="/run/user/1000/mcp-x/s", propose=propose, lead=True))
    assert config.config_overrides == (*CONFIG_OVERRIDES, projects_override("/tmp/wt"), MCP_SERVER,
                                       *read_overrides(provider.read_scope(_ctx())), "mcp_servers.platform.enabled=true",
                                       MCP_LEAD)
    assert MCP_PROPOSE not in config.config_overrides
    key, _, value = MCP_LEAD.partition("=")
    assert key == "mcp_servers.platform.enabled_tools"
    assert tomllib.loads(f"x = {value}")["x"] == [*BASE_TOOLS, *LEAD_TOOLS]
    assert MCP_LEAD not in CodexProvider().config(_ctx(lead=True)).config_overrides


@pytest.mark.parametrize("propose", [False, True])
@pytest.mark.parametrize("lead", [False, None, "true", 1])
def test_codex_without_a_literal_true_lead_enables_none_of_the_four(propose, lead):
    import tomllib
    from runner.providers.codex import MCP_SERVER, CodexProvider

    config = CodexProvider().config(_ctx(mcp_socket="/run/user/1000/mcp-x/s", propose=propose, lead=lead))
    assert not any(name in o for o in config.config_overrides if o != MCP_SERVER for name in LEAD_TOOLS)
    table = tomllib.loads(f"x = {MCP_SERVER.partition('=')[2]}")["x"]
    assert not set(table["enabled_tools"]) & set(LEAD_TOOLS)


def test_the_server_override_is_the_table_the_box_file_held():
    import tomllib
    from runner.providers.codex import MCP_SERVER

    key, _, value = MCP_SERVER.partition("=")
    assert key == "mcp_servers.platform"
    server = tomllib.loads(f"x = {value}")["x"]
    assert server == {"command": BOX_WRAPPER, "args": [], "env_vars": ["AGENT_MCP_SOCKET"], "enabled": False,
                      "required": False, "startup_timeout_sec": 10, "tool_timeout_sec": 60,
                      "enabled_tools": BASE_TOOLS, "default_tools_approval_mode": "approve"}
    assert mcp_tools.wrapper_path("/opt/agent-runner/.venv") == BOX_WRAPPER
    assert server["tool_timeout_sec"] > mcp_shim.CALL_TIMEOUT > session_socket.FORWARD_TIMEOUT


def _worktree_layout(tmp_path, name="wt.v1"):
    main = tmp_path / "proj.example" / "repo"
    admin = main / ".git" / "worktrees" / "s1"
    admin.mkdir(parents=True)
    (admin / "commondir").write_text("../..\n")
    worktree = tmp_path / "proj.example" / "wt" / name
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {admin}\n")
    return main, worktree


def test_codex_marks_the_worktree_and_its_main_checkout_untrusted_on_every_start(tmp_path):
    import tomllib
    from runner.providers.codex import CodexProvider, git_root_for_trust, projects_override

    main, worktree = _worktree_layout(tmp_path)
    (worktree / "sub").mkdir()
    assert git_root_for_trust(str(worktree)) == os.path.realpath(main)
    assert git_root_for_trust(str(worktree / "sub")) == os.path.realpath(main)
    assert git_root_for_trust(str(main)) == str(main)
    provider = CodexProvider()
    for socket_path in (None, "/run/user/1000/mcp-x/s"):
        ctx = _ctx(mcp_socket=socket_path)
        ctx.cwd = str(worktree)
        overrides = provider.config(ctx).config_overrides
        assert overrides[:3] == ("analytics.enabled=false", "sandbox_workspace_write.network_access=false",
                                 "features.apps=false")
        projects = [o for o in overrides if o.startswith("projects=")]
        assert projects == [projects_override(str(worktree))]
        table = tomllib.loads(projects[0])["projects"]
        assert str(worktree) in table and os.path.realpath(main) in table
        assert all(v == {"trust_level": "untrusted"} for v in table.values())
        params = provider.thread_params(ctx)
        assert params["config"]["projects"] == table
        assert "sandbox" not in params and params["approvalPolicy"] == "on-request"


def test_codex_project_override_survives_awkward_paths(tmp_path):
    import tomllib
    from runner.providers.codex import projects_override

    (tmp_path / ".git").mkdir()
    odd = tmp_path / 'a.b "q" \\ é ✓'
    odd.mkdir()
    table = tomllib.loads(projects_override(str(odd)))["projects"]
    expected = {str(odd), os.path.realpath(odd), str(tmp_path), os.path.realpath(tmp_path)}
    assert table == {path: {"trust_level": "untrusted"} for path in expected}
    outside = projects_override("/nonexistent/wt")
    assert tomllib.loads(outside)["projects"] == {"/nonexistent/wt": {"trust_level": "untrusted"}}


def test_codex_declines_an_mcp_elicitation():
    from runner.providers.codex import CodexProvider

    assert CodexProvider()._on_server_request("mcpServer/elicitation/request", {}) == {"action": "decline"}


BOX_WRAPPER = "/opt/agent-runner/.venv/bin/runner-mcp"


def test_the_example_managed_settings_lock_names_the_wrappers_exact_command():
    data = json.loads((EXAMPLES / "managed-settings.json").read_text(encoding="utf-8"))
    assert data["allowManagedMcpServersOnly"] is True
    assert data["allowedMcpServers"] == [{"serverCommand": [BOX_WRAPPER]}]
    assert mcp_tools.wrapper_path("/opt/agent-runner/.venv") == BOX_WRAPPER
    assert os.access(WRAPPER, os.X_OK) and WRAPPER.read_text().startswith("#!/usr/bin/env -S python3 -I\n")


def test_the_wrapper_runs_from_a_clone_through_its_own_first_line(tmp_path):
    env = {"PATH": os.pathsep.join([os.path.dirname(sys.executable), "/usr/bin", "/bin"])}
    proc = subprocess.run([str(WRAPPER)], input=json.dumps({"jsonrpc": "2.0", "id": "p", "method": "ping"}) + "\n",
                          capture_output=True, text=True, env=env, cwd=tmp_path, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.splitlines()[0]) == {"jsonrpc": "2.0", "id": "p", "result": {}}


def test_the_codex_template_defines_no_server_so_a_research_session_has_none():
    import tomllib

    data = tomllib.loads((EXAMPLES / "codex-config.toml").read_text(encoding="utf-8"))
    assert "mcp_servers" not in data
    assert data["features"]["apps"] is False

