"""`runner-mcp`: the stdio MCP server that Claude Code or Codex starts for a session.

Standard library only, and it imports nothing of the runner but the catalogue (mcp_tools.py). It holds
no credential and reads one environment variable, the path of the unix socket its session's runner
serves (session_socket.py). It is a bridge: an MCP server towards the coding agent, and a client of
the runner's socket behind it, which forwards each call to the control plane with the session's own
token.

One JSON-RPC message per line on stdin and stdout, and logs go to stderr. Supported: `initialize` (the
client's protocol revision when it is one of SUPPORTED, else LATEST), `ping`, `tools/list` and
`tools/call`. With no socket, or one that cannot be reached, it still starts and lists no tools, so
a client configured with it always starts. See docs/security-model.md, "The MCP server and its scope".
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any, BinaryIO

from runner.mcp_tools import NAMES, SERVER_NAME, SERVER_VERSION, SOCKET_ENV, TOOLS

LATEST = "2025-11-25"
SUPPORTED = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
CALL_TIMEOUT = 50.0
CONNECT_TIMEOUT = 2.0
REPLY_CAP = 1024 * 1024
LINE_CAP = 4 * 1024 * 1024

NO_TOOLS = "The project tools are not available in this session; carry on with the work without them."
NO_ANSWER = "The project tools did not answer; carry on with the work without them."
BAD_REPLY = "The project tools gave an answer that could not be read; carry on with the work without them."
NO_NAME = "A tool call needs the name of one of the project tools."


def log(message: str) -> None:
    print(f"runner-mcp: {message}", file=sys.stderr, flush=True)


def _error(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _result(rid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


class Shim:
    def __init__(self, socket_path: str | None, timeout: float = CALL_TIMEOUT) -> None:
        self.socket_path = socket_path or None
        self.timeout = timeout

    def _connect(self, timeout: float) -> socket.socket:
        if not self.socket_path:
            raise OSError("no socket")
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(timeout)
        try:
            conn.connect(self.socket_path)
        except OSError:
            conn.close()
            raise
        return conn

    def reachable(self) -> bool:
        try:
            self._connect(CONNECT_TIMEOUT).close()
        except OSError:
            return False
        return True

    def relay(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.socket_path:
            return {"ok": False, "result": None, "error": NO_TOOLS}
        line = json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))
        try:
            conn = self._connect(CONNECT_TIMEOUT)
        except OSError as exc:
            log(f"socket unreachable ({type(exc).__name__})")
            return {"ok": False, "result": None, "error": NO_TOOLS}
        try:
            conn.settimeout(self.timeout)
            conn.sendall(line.encode("utf-8") + b"\n")
            data = b""
            while b"\n" not in data and len(data) <= REPLY_CAP:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
        except OSError as exc:
            log(f"no answer from the socket ({type(exc).__name__})")
            return {"ok": False, "result": None, "error": NO_ANSWER}
        finally:
            conn.close()
        if b"\n" not in data:
            log("the socket closed without a whole answer")
            return {"ok": False, "result": None, "error": NO_ANSWER}
        try:
            reply = json.loads(data.split(b"\n", 1)[0])
        except (UnicodeDecodeError, json.JSONDecodeError):
            reply = None
        if not isinstance(reply, dict) or not isinstance(reply.get("ok"), bool):
            log("an unreadable answer from the socket")
            return {"ok": False, "result": None, "error": BAD_REPLY}
        if not reply["ok"] and not isinstance(reply.get("error"), str):
            return {"ok": False, "result": None, "error": BAD_REPLY}
        return reply

    def handle_line(self, line: bytes) -> dict[str, Any] | None:
        if not line.strip():
            return None
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _error(None, -32700, "Parse error")
        if not isinstance(message, dict):
            return _error(None, -32600, "Invalid Request")
        method = message.get("method")
        if "id" not in message:
            return None
        rid = message.get("id")
        if method is None and ("result" in message or "error" in message):
            return None
        if not isinstance(method, str) or not (isinstance(rid, (str, int)) and not isinstance(rid, bool)):
            return _error(rid if isinstance(rid, (str, int)) else None, -32600, "Invalid Request")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if method == "initialize":
            return _result(rid, self.initialize(params))
        if method == "ping":
            return _result(rid, {})
        if method == "tools/list":
            return _result(rid, {"tools": list(TOOLS) if self.reachable() else []})
        if method == "tools/call":
            return self.call(rid, params)
        return _error(rid, -32601, "Method not found")

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        asked = params.get("protocolVersion")
        return {"protocolVersion": asked if asked in SUPPORTED else LATEST,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}}

    def call(self, rid: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(name, str) or not name:
            return _result(rid, _text(NO_NAME, error=True))
        if name not in NAMES:
            log("a call for a tool not in the catalogue; relayed for the control plane to refuse")
        reply = self.relay(name, arguments if isinstance(arguments, dict) else {})
        if not reply["ok"]:
            return _result(rid, _text(reply["error"], error=True))
        result = reply.get("result")
        out = _text(json.dumps(result, ensure_ascii=False), error=False)
        if isinstance(result, dict):
            out["structuredContent"] = result
        return _result(rid, out)

    def serve(self, stdin: BinaryIO, stdout: BinaryIO) -> None:
        while True:
            line = stdin.readline(LINE_CAP + 1)
            if not line:
                return
            if len(line) > LINE_CAP and not line.endswith(b"\n"):
                while line and not line.endswith(b"\n"):
                    line = stdin.readline(LINE_CAP)
                answer: dict[str, Any] | None = _error(None, -32700, "Parse error")
            else:
                try:
                    answer = self.handle_line(line)
                except Exception as exc:
                    log(f"internal error ({type(exc).__name__})")
                    answer = _error(None, -32603, "Internal error")
            if answer is not None:
                stdout.write(json.dumps(answer, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
                stdout.flush()


def _text(text: str, error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


# The only environment read in this module: the socket's path. No token ever reaches the shim.
def main() -> int:
    path = os.environ.get(SOCKET_ENV)
    if not path:
        log("no socket configured; serving no tools")
    try:
        Shim(path).serve(sys.stdin.buffer, sys.stdout.buffer)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

