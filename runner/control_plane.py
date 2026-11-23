"""The client for the control plane: the service that owns sessions, queues the owner's messages,
shows approval cards and stores the event stream.

Every call is `POST <base>/api/v1/call/<operation>` with `Authorization: Bearer <session token>` and
JSON both ways. Errors are `{"error": {"code", "message"}}`. Proxies from the environment are ignored,
and a redirect is never followed: it stops the call as `Rejected(status, "redirect")`.

The Poster numbers events 1, 2, 3 with no gaps. A network error or a 5xx resends the same event with
the same number, and the control plane answers a repeat as a duplicate, so a lost response costs
nothing. A 413 resends the number with a stub payload. A 409 (a gap) or a 400 is a bug and stops the
run. A 401 or 403 means the session has ended and its token is revoked. The token is never logged and
errors carry the status and error code, never a response body. See docs/design.md, "The control-plane
protocol".
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from runner.events import Event, cap_payload, stub
from runner.http_transport import NoRedirects

log = logging.getLogger("runner.control_plane")

MAX_BODY = 1024 * 1024
TIMEOUT = 15


class Unreachable(Exception):
    ...


class SessionEnded(Exception):

    def __init__(self, status: int, code: str = "", message: str = "") -> None:
        super().__init__(f"{status} {code}".strip())
        self.status, self.code, self.message = status, code, message


class Rejected(Exception):

    def __init__(self, status: int, code: str) -> None:
        super().__init__(f"control plane refused the request: {status} {code}")
        self.status, self.code = status, code


class TooLarge(Exception):
    ...


class Client:

    def __init__(self, base_url: str, token: str, ca_file: str | None = None, timeout: float = TIMEOUT,
                 host: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.host = host
        self.timeout = timeout
        handlers: list[Any] = [urllib.request.ProxyHandler({}), NoRedirects()]
        if self.base_url.startswith("https://"):
            context = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()
            handlers.append(urllib.request.HTTPSHandler(context=context))
        self._opener = urllib.request.build_opener(*handlers)

    def __repr__(self) -> str:
        return f"Client({self.base_url})"

    def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/api/v1/call/{op}", data=body, method="POST", headers={
            "Authorization": f"Bearer {self._token}", "Content-Type": "application/json",
            "Accept": "application/json"})
        if self.host:
            request.add_header("Host", self.host)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                status, data = response.status, response.read(MAX_BODY + 1)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                exc.close()
                raise Rejected(exc.code, "redirect") from None
            status, data = exc.code, exc.read(MAX_BODY + 1) if exc.fp else b""
        except (OSError, ValueError) as exc:
            raise Unreachable(type(exc).__name__) from None
        if len(data) > MAX_BODY:
            raise Unreachable("oversize body")
        try:
            parsed = json.loads(data) if data else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        if status == 200 and isinstance(parsed, dict):
            return parsed
        code = message = ""
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            code = str(parsed["error"].get("code", ""))[:40]
            message = " ".join(str(parsed["error"].get("message", "")).split())[:200]
        if status in (401, 403):
            raise SessionEnded(status, code, message)
        if status == 413:
            raise TooLarge()
        if status >= 500 or status == 200:
            raise Unreachable(f"{status}")
        raise Rejected(status, code)


@dataclass
class Retry:
    attempts: int = 30
    first: float = 0.5
    ceiling: float = 10.0
    sleep: Callable[[float], Any] = field(default=asyncio.sleep)


# Numbers events without gaps and resends the same number on a network error or 5xx, so the control
# plane can store each sequence once (docs/design.md, "The control-plane protocol").
class Poster:

    def __init__(self, client: Client, provider: str, retry: Retry | None = None) -> None:
        self.client, self.provider = client, provider
        self.retry = retry or Retry()
        self.next_sequence = 1
        self.posted: list[tuple[int, Event]] = []
        self._lock = asyncio.Lock()

    async def post(self, event: Event) -> dict[str, Any]:
        return await asyncio.shield(self._post(event))

    async def _post(self, event: Event) -> dict[str, Any]:
        async with self._lock:
            payload = cap_payload(event.payload)
            sequence = self.next_sequence
            delay = self.retry.first
            for attempt in range(1, self.retry.attempts + 1):
                body = {"sequence": sequence, "kind": event.kind, "provider": self.provider, "payload": payload}
                try:
                    out = await asyncio.to_thread(self.client.call, "sessions.event_post", body)
                except TooLarge:
                    log.warning("event %d (%s) refused as too large; sending a stub", sequence, event.kind)
                    payload = stub(payload, cap=8192)
                    continue
                except Unreachable as exc:
                    log.warning("event %d not acknowledged (%s), attempt %d; resending", sequence, exc, attempt)
                    await self.retry.sleep(delay)
                    delay = min(delay * 2, self.retry.ceiling)
                    continue
                self.next_sequence = sequence + 1
                self.posted.append((sequence, Event(event.kind, payload)))
                return out
            raise Unreachable(f"event {sequence} not acknowledged after {self.retry.attempts} attempts")


INPUTS_GET = "research.inputs_get"
OUTPUT_PART = "research.output_part"
OUTPUT_DONE = "research.output_done"
PART_BYTES = 48 * 1024


class ResearchClient:

    def __init__(self, client: Client, session_id: int, attempts: int = 5, first: float = 1.0,
                 sleep: Callable[[float], Any] | None = None) -> None:
        self.client, self.session_id = client, session_id
        self.attempts, self.first = attempts, first
        self.sleep = sleep or time.sleep

    def __repr__(self) -> str:
        return f"ResearchClient({self.client!r}, session {self.session_id})"

    def _call(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        delay = self.first
        for attempt in range(1, self.attempts + 1):
            try:
                return self.client.call(op, body)
            except Unreachable as exc:
                if attempt == self.attempts:
                    raise
                log.warning("%s not answered (%s), attempt %d; trying again", op, exc, attempt)
                self.sleep(delay)
                delay = min(delay * 2, 10.0)
        raise Unreachable(op)

    def inputs_get(self, part: int) -> dict[str, Any]:
        return self._call(INPUTS_GET, {"session_id": self.session_id, "part": part})

    def output_part(self, file: str, index: int, total: int, content: str) -> dict[str, Any]:
        return self._call(OUTPUT_PART, {"session_id": self.session_id, "file": file, "index": index,
                                        "total": total, "content": content})

    def output_done(self, files: list[dict[str, Any]], partial: bool, reason: str | None) -> dict[str, Any]:
        return self._call(OUTPUT_DONE, {"session_id": self.session_id, "files": files, "partial": partial,
                                        "reason": reason})


@dataclass(frozen=True)
class Prompt:
    id: int
    text: str
    mode: str
    author: str
    replaces: int | None = None


def _replaces(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


@dataclass(frozen=True)
class Answer:
    request_id: str
    decision: str
    expired: bool
    feedback: str | None


@dataclass(frozen=True)
class Control:
    id: int
    kind: str
    entry_id: int | None = None
    settings: dict[str, Any] | None = None


CONTROL_KINDS = ("interrupt", "finish", "withdraw", "settings", "keep_open")
SETTINGS_FIELDS = ("model", "effort", "mode", "thinking")


def _settings(c: dict[str, Any]) -> dict[str, Any] | None:
    found = {k: c[k] for k in SETTINGS_FIELDS if k in c and c[k] is not None}
    return found or None


async def pull(client: Client) -> tuple[list[Prompt], list[Answer]]:
    prompts, answers, _ = await pull_all(client)
    return prompts, answers


@dataclass(frozen=True)
class Pulled:
    prompts: list[Prompt]
    answers: list[Answer]
    controls: list[Control]
    title: str | None = None


async def pull_all(client: Client) -> tuple[list[Prompt], list[Answer], list[Control]]:
    got = await pull_full(client)
    return got.prompts, got.answers, got.controls


async def pull_full(client: Client) -> Pulled:
    out = await asyncio.to_thread(client.call, "sessions.pull", {})
    title = out.get("title")
    controls = []
    for c in out.get("controls") or []:
        if isinstance(c, dict) and c.get("kind") in CONTROL_KINDS \
                and (c["kind"] != "withdraw" or _replaces(c.get("entry_id")) is not None) \
                and (c["kind"] != "settings" or _settings(c) is not None):
            cid = c.get("id")
            controls.append(Control(cid if isinstance(cid, int) and not isinstance(cid, bool) else 0, c["kind"],
                                    _replaces(c.get("entry_id")) if c["kind"] == "withdraw" else None,
                                    _settings(c) if c["kind"] == "settings" else None))
        else:
            log.warning("dropped a malformed control from pull")
    prompts, answers = [], []
    for p in out.get("prompts") or []:
        if isinstance(p, dict) and isinstance(p.get("text"), str) and p.get("mode") in ("queue", "steer", "now"):
            prompts.append(Prompt(int(p.get("id") or 0), p["text"], p["mode"], str(p.get("author") or ""),
                                  _replaces(p.get("replaces"))))
        else:
            log.warning("dropped a malformed prompt from pull")
    for a in out.get("answers") or []:
        if isinstance(a, dict) and isinstance(a.get("request_id"), str) and a.get("decision") in ("allow", "deny"):
            answers.append(Answer(a["request_id"], a["decision"], bool(a.get("expired")),
                                  a["feedback"] if isinstance(a.get("feedback"), str) else None))
        else:
            log.warning("dropped a malformed answer from pull")
    return Pulled(prompts, answers, controls, title if isinstance(title, str) else None)

