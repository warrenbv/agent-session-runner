"""Replies as they are written: best-effort `draft` posts of the text a provider is producing,
outside the numbered event stream.

Only when the session asks for drafts. At most one post per INTERVAL, one in flight at a time, never
retried and never awaited by the turn. Each draft carries the reply's text so far (or its tail with an
offset), so it stands alone: a lost draft costs nothing and the next one replaces it. The finished
`message` event names the same message id and supersedes the drafts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable

from runner.events import DRAFT_OP
from runner.control_plane import Unreachable

log = logging.getLogger("runner.drafts")

INTERVAL = 1.0
DRAFT_BYTES = 12000
STATS_CAP = 50


def why(exc: BaseException) -> str:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return f"{type(exc).__name__} {status}"
    if isinstance(exc, Unreachable):
        return f"Unreachable {str(exc)[:40]}".strip()
    return type(exc).__name__

def _json_bytes(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def window(text: str, cap: int = DRAFT_BYTES) -> tuple[int, str]:
    if _json_bytes(text) <= cap:
        return 0, text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high) // 2
        if _json_bytes(text[mid:]) <= cap:
            high = mid
        else:
            low = mid + 1
    return low, text[low:]


class Drafter:

    def __init__(self, provider: str, send: Callable[[dict[str, Any]], Awaitable[Any]],
                 interval: float | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.provider, self.send, self.clock = provider, send, clock
        self.interval = INTERVAL if interval is None else interval
        self.current: str | None = None
        self.text = ""
        self.since = 0.0
        self.done: dict[str, None] = {}
        self._message: str | None = None
        self._task: asyncio.Task | None = None
        self.posted: list[dict[str, Any]] = []
        self.frames = 0
        self.stats: dict[str, list[Any]] = {}
        self.deltas = 0
        self.first_delta: float | None = None
        self.last_delta: float | None = None
        self.unnamed = 0

    def feed(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        try:
            if self.provider == "claude-code":
                self._claude(raw)
            else:
                self._codex(raw)
        except Exception as exc:
            log.debug("draft skipped: %s", type(exc).__name__)

    def _claude(self, raw: dict[str, Any]) -> None:
        kind = raw.get("type")
        if kind == "AssistantMessage" and raw.get("parent_tool_use_id") is None:
            mid = raw.get("message_id") or raw.get("uuid")
            texts = [i for i, b in enumerate(raw.get("content") or [])
                     if isinstance(b, dict) and b.get("type") == "TextBlock"]
            if mid and texts:
                self._close(f"{mid}:{texts[0]}", also=[f"{mid}:{i}" for i in texts[1:]] + [f"{mid}:0"])
            return
        if kind != "StreamEvent" or raw.get("parent_tool_use_id") is not None:
            return
        self.frames += 1
        event = raw.get("event") or {}
        etype = event.get("type")
        if etype == "message_start":
            self._message = (event.get("message") or {}).get("id") or None
        elif etype == "message_stop":
            self._message = None
        elif etype == "content_block_start" and (event.get("content_block") or {}).get("type") == "text":
            if self._message:
                self._open(f"{self._message}:0")
                initial = (event.get("content_block") or {}).get("text")
                if isinstance(initial, str) and initial:
                    self._grow(initial)
            else:
                self.unnamed += 1
        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                self._delta()
                if self.current is None and self._message:
                    self._open(f"{self._message}:0")
                self._grow(delta["text"])
        elif etype == "content_block_stop":
            self._post_due()

    def _codex(self, raw: dict[str, Any]) -> None:
        method, params = raw.get("method"), raw.get("params") or {}
        item = params.get("item") or {}
        if method == "item/started" and item.get("type") == "agentMessage" and item.get("id"):
            self._open(str(item["id"]))
        elif method == "item/agentMessage/delta" and isinstance(params.get("delta"), str):
            self.frames += 1
            self._delta()
            item_id = params.get("itemId")
            if item_id and str(item_id) != self.current:
                self._open(str(item_id))
            self._grow(params["delta"])
        elif method == "item/completed" and item.get("type") == "agentMessage" and item.get("id"):
            self._close(str(item["id"]))

    def _open(self, message_id: str) -> None:
        if message_id in self.done:
            self.current = None
            return
        if message_id != self.current:
            self.current, self.text, self.since = message_id, "", self.clock()

    def _grow(self, piece: str) -> None:
        if self.current is None:
            return
        self.text += piece
        self._post_due()

    def _delta(self) -> None:
        now = self.clock()
        self.deltas += 1
        self.last_delta = now
        if self.first_delta is None:
            self.first_delta = now

    def _close(self, message_id: str, also: list[str] | None = None) -> None:
        ids = list(dict.fromkeys([message_id, *(also or [])]))
        drafted = self.current if self.current in ids else None
        if drafted is None:
            drafted = next((i for i in ids if i in self.stats), None)
        for one in ids:
            self.done[one] = None
        while len(self.done) > 1000:
            self.done.pop(next(iter(self.done)))
        if self.current in ids:
            self.current, self.text = None, ""
        self._log_reply(message_id, drafted)
        for one in ids:
            self.stats.pop(one, None)

    def _log_reply(self, message_id: str, drafted: str | None = None) -> None:
        sent, failed, last = self.stats.get(drafted or message_id) or (0, 0, None)
        in_flight = self._task is not None and not self._task.done()
        span = (self.last_delta - self.first_delta) if self.first_delta is not None and self.last_delta is not None else 0.0
        extra = f", drafted as {drafted[:80]}" if drafted and drafted != message_id else ""
        unnamed = f", {self.unnamed} text blocks with no message_start" if self.unnamed else ""
        log.info("reply finished: %d drafts sent, %d failed%s, %d partial frames, %d text deltas over %.1f s%s, "
                 "post in flight %s (message %s%s)",
                 sent, failed, f" (last: {last})" if last else "", self.frames, self.deltas, span, unnamed,
                 "yes" if in_flight else "no", message_id[:80], extra)
        self.frames = self.deltas = self.unnamed = 0
        self.first_delta = self.last_delta = None

    def _post_due(self) -> None:
        if self.current is None or not self.text or self.clock() - self.since < self.interval:
            return
        if self._task is not None and not self._task.done():
            return
        offset, text = window(self.text)
        body = {"message_id": self.current, "offset": offset, "text": text}
        self.since = self.clock()
        self.posted.append(body)
        if self.current not in self.stats:
            while len(self.stats) >= STATS_CAP:
                self.stats.pop(next(iter(self.stats)))
            self.stats[self.current] = [0, 0, None]
        self.stats[self.current][0] += 1
        try:
            self._task = asyncio.get_running_loop().create_task(self._send(body))
        except RuntimeError:
            self._task = None

    async def _send(self, body: dict[str, Any]) -> None:
        try:
            await self.send(body)
        except Exception as exc:
            log.debug("draft not posted: %s", type(exc).__name__)
            counts = self.stats.get(body.get("message_id"))
            if counts is not None:
                counts[1] += 1
                counts[2] = why(exc)

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def poster(client: Any) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    async def send(body: dict[str, Any]) -> Any:
        return await asyncio.to_thread(client.call, DRAFT_OP, body)
    return send

