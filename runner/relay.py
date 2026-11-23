"""The approval relay: a provider's approval request becomes an `approval` event, and the runner
waits for the answer that the control plane's pull delivers.

Silence denies. The runner's own ceiling is ten minutes, the same as the control plane's expiry, so a
lost answer still ends in a denial. A denial carries the owner's feedback as its reason, and stopping
the session denies whatever is still waiting.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from runner.events import Event
from runner.control_plane import Answer

log = logging.getLogger("runner.relay")

# The same ten minutes as the control plane's approval expiry, so a lost answer still ends in a denial.
CEILING = 600.0
NO_ANSWER = "No answer within ten minutes; denied."
DENIED = "Denied by the session's owner."
EXPIRED = "The approval expired unanswered; denied."


@dataclass(frozen=True)
class Decision:
    allow: bool
    message: str = ""


class Relay:
    def __init__(self, post: Callable[[Event], Awaitable[Any]], ceiling: float = CEILING) -> None:
        self._post = post
        self.ceiling = ceiling
        self._waiting: dict[str, asyncio.Future[Decision]] = {}
        self._closed: str | None = None
        self.decisions: list[tuple[str, Decision]] = []

    @property
    def pending(self) -> int:
        return len(self._waiting)

    async def request(self, payload: dict[str, Any]) -> Decision:
        request_id = payload["request_id"]
        if self._closed is not None:
            return self._note(request_id, Decision(False, self._closed))
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        self._waiting[request_id] = future
        try:
            await self._post(Event("approval", payload))
            decision = await asyncio.wait_for(asyncio.shield(future), timeout=self.ceiling)
        except TimeoutError:
            log.info("approval %s unanswered at the ceiling; denying", request_id)
            decision = Decision(False, NO_ANSWER)
        finally:
            self._waiting.pop(request_id, None)
        return self._note(request_id, decision)

    def _note(self, request_id: str, decision: Decision) -> Decision:
        self.decisions.append((request_id, decision))
        return decision

    def deliver(self, answer: Answer) -> None:
        if answer.decision == "allow" and not answer.expired:
            decision = Decision(True)
        else:
            decision = Decision(False, answer.feedback or (EXPIRED if answer.expired else DENIED))
        future = self._waiting.get(answer.request_id)
        if future is not None and not future.done():
            future.set_result(decision)
        elif future is None:
            log.info("answer for %s with no request waiting; dropped", answer.request_id)

    def deny_all(self, reason: str, close: bool = False) -> None:
        if close:
            self._closed = reason
        for future in list(self._waiting.values()):
            if not future.done():
                future.set_result(Decision(False, reason))

