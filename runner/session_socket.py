"""The session's end of the MCP bridge: a unix socket that the shim (mcp_shim.py) relays tool calls to.

Opened only when the session is given tools. The socket lives in a 0700 directory of its own under the
user's runtime directory (/run/user/<uid>), which the example managed settings deny to sandboxed
commands, and the socket itself is 0600. One JSON line in, one JSON line out. Each call goes to the control
plane's `sessions.tool_call` with the session's own token and a timeout, and any failure is answered
with one plain sentence, never an exception's text. The socket and its directory are removed when the
provider has closed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import tempfile
from typing import Any

from runner.control_plane import Client, Rejected, SessionEnded, TooLarge, Unreachable

log = logging.getLogger("runner.tools")

OP = "sessions.tool_call"
SOCKET_LEAF = "s"
FORWARD_TIMEOUT = 40.0
LINE_LIMIT = 1024 * 1024
TOOL_CAP = 200

ENDED = "This session has ended, so its project tools no longer work."
UNREACHABLE = "The project's tracker could not be reached just now; carry on with the work without it."
SLOW = "The project's tracker took too long to answer; carry on with the work without it."
REFUSED = "The project's tracker refused that call; carry on with the work without it."
UNREADABLE = "That request could not be read; call one of the project tools by name with its arguments."
BROKEN = "Something went wrong passing that call to the project's tracker; carry on with the work without it."


# Only the user's own runtime directory: the example managed settings deny it to sandboxed commands,
# so the model's shell cannot reach the socket directly.
def runtime_dir() -> str | None:
    path = f"/run/user/{os.getuid()}"
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        return None
    return path


def _answer(ok: bool, result: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"ok": ok, "result": result if ok else None, "error": None if ok else error}


class ToolSocket:
    def __init__(self, client: Client, timeout: float = FORWARD_TIMEOUT) -> None:
        self.client, self.timeout = client, timeout
        self.directory: str | None = None
        self.path: str | None = None
        self.calls = 0
        self._server: asyncio.base_events.Server | None = None

    async def open(self, parent: str | None) -> str | None:
        if parent is None:
            log.warning("no runtime directory (/run/user/<uid>); this session runs without its project tools")
            return None
        try:
            self.directory = tempfile.mkdtemp(prefix="mcp-", dir=parent)
            os.chmod(self.directory, 0o700)
            path = os.path.join(self.directory, SOCKET_LEAF)
            self._server = await asyncio.start_unix_server(self._serve, path=path, limit=LINE_LIMIT)
            os.chmod(path, 0o600)
        except OSError as exc:
            log.warning("the tools socket could not be opened (%s); this session runs without its project tools",
                        type(exc).__name__)
            await self.close()
            return None
        self.path = path
        log.info("project tools socket open")
        return path

    async def close(self) -> None:
        if self._server is not None:
            server, self._server = self._server, None
            server.close()
            try:
                await asyncio.wait_for(server.wait_closed(), timeout=5)
            except (TimeoutError, OSError):
                pass
        if self.directory is not None:
            directory, self.directory, self.path = self.directory, None, None
            try:
                os.unlink(os.path.join(directory, SOCKET_LEAF))
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("the tools socket could not be removed (%s)", type(exc).__name__)
            try:
                os.rmdir(directory)
            except OSError as exc:
                log.warning("the tools socket's directory could not be removed (%s)", type(exc).__name__)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    writer.write(self._line(_answer(False, error=UNREADABLE)))
                    await writer.drain()
                    return
                if not line:
                    return
                if not line.strip():
                    continue
                answer = await self.handle(line)
                writer.write(self._line(answer))
                await writer.drain()
        except (ConnectionError, OSError):
            return
        finally:
            writer.close()

    @staticmethod
    def _line(answer: dict[str, Any]) -> bytes:
        return json.dumps(answer, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"

    async def handle(self, line: bytes) -> dict[str, Any]:
        try:
            request = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            request = None
        if not isinstance(request, dict) or not isinstance(request.get("tool"), str) \
                or not isinstance(request.get("arguments", {}), dict):
            return _answer(False, error=UNREADABLE)
        tool = request["tool"][:TOOL_CAP]
        return await self.forward(tool, request.get("arguments") or {})

    async def forward(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        try:
            out = await asyncio.wait_for(asyncio.to_thread(self.client.call, OP, {"tool": tool, "arguments": arguments}),
                                         timeout=self.timeout)
        except TimeoutError:
            log.warning("tool call %d: the control plane did not answer in time", self.calls)
            return _answer(False, error=SLOW)
        except SessionEnded as exc:
            log.warning("tool call %d: refused, the session's token has ended (%s)", self.calls, exc)
            return _answer(False, error=ENDED)
        except Unreachable as exc:
            log.warning("tool call %d: control plane unreachable (%s)", self.calls, exc)
            return _answer(False, error=UNREACHABLE)
        except (Rejected, TooLarge) as exc:
            log.warning("tool call %d: refused (%s)", self.calls, type(exc).__name__)
            return _answer(False, error=REFUSED)
        except Exception as exc:
            log.error("tool call %d failed: %s", self.calls, type(exc).__name__)
            return _answer(False, error=BROKEN)
        if not isinstance(out, dict) or not isinstance(out.get("ok"), bool):
            log.warning("tool call %d: a malformed answer from the control plane", self.calls)
            return _answer(False, error=BROKEN)
        if out["ok"]:
            return _answer(True, out.get("result"))
        error = out.get("error")
        return _answer(False, error=error if isinstance(error, str) and error else REFUSED)

