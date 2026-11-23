"""The provider interface the session driver uses, whatever runs underneath.

A provider runs one conversation. `turn(text)` sends a prompt and yields the provider's raw messages
until the turn ends (the ResultMessage for Claude, `turn/completed` for Codex), inclusive. While a turn
runs the driver may call `steer`, `interrupt` and `set_permission_mode` from another task.
`change_settings` applies between turns and answers per field, applied or refused with a reason.
Raw messages go through `normalise` into events.

ProviderContext carries everything a provider needs from the session, including the approval
callback, and defaults to the safe value for every option (for example no web unless `web` is a
literal true). The typed result schema every coding turn returns is defined here too, and its field
descriptions tell the model that "unverified" is the honest default. See docs/design.md,
"The provider interface".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

from runner.events import Event
from runner.relay import Decision
from runner.testrun import check_test_cmd

ITEM_STATUSES = ("done", "not_done", "unverified")
ITEMS_MAX = 30
ITEM_LABEL_CAP = 200
ITEM_WHERE_CAP = 300
ITEMS_DESCRIPTION = (
    "What you were asked to do, one entry per distinct request, in the owner's words, in the order asked. "
    "Leave the list empty when you were only asked a question or asked for one small thing. Do not add "
    "entries for work you chose to do yourself unless the owner asked for it.")
ITEM_PROPERTIES: dict[str, Any] = {
    "label": {"type": "string", "description": "A few words naming the request as the owner put it."},
    "status": {"type": "string", "enum": list(ITEM_STATUSES), "description": (
        "done: finished and you saw it work (a test or command you ran passed, or you read the result back). "
        "not_done: not finished, not attempted, or it failed. unverified: you did the work but did not see it "
        "work. When in doubt, say unverified; it is an honest answer, not a failure.")},
    "where": {"type": "string", "description": (
        "Where the work or its evidence is: a file path, a test name, or the step. Empty if not done.")},
}

TEST_CLAIM_STATUSES = ("passed", "failed", "not_run", "unverified")
# The model's own test claim is kept as evidence to compare with the runner's verdict, never as the verdict.
TEST_CLAIM_DESCRIPTION = (
    "Set tests_claim to an object containing command and status for your own check of the exact configured "
    "final test command before the runner's independent test turn, using passed, failed, not_run or "
    "unverified, with an empty command and status none when no command is configured or you make no claim about it.")

RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "tests_claim": {"type": "object", "description": TEST_CLAIM_DESCRIPTION, "properties": {
            "command": {"type": "string"},
            "status": {"type": "string", "enum": [*TEST_CLAIM_STATUSES, "none"]}},
            "required": ["command", "status"], "additionalProperties": False},
        "branch": {"type": "string"},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "items": {"type": "array", "description": ITEMS_DESCRIPTION, "items": {
            "type": "object", "properties": ITEM_PROPERTIES, "required": ["label", "status", "where"],
            "additionalProperties": False}},
    },
    "required": ["summary", "files_changed", "tests_run", "tests_claim", "branch", "open_questions",
                 "next_steps", "items"],
    "additionalProperties": False,
}


def tidy_test_claim(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict) or set(value) != {"command", "status"} \
            or value.get("status") not in TEST_CLAIM_STATUSES:
        return None
    try:
        command = check_test_cmd(value.get("command"))
    except ValueError:
        return None
    return {"command": " ".join(command.split()), "status": value["status"]} if command else None


def _short(value: Any, cap: int) -> str:
    text = " ".join(str(value).split()) if value is not None else ""
    return text if len(text) <= cap else text[: cap - 1] + "…"


def tidy_items(found: Any) -> tuple[list[dict[str, str]], int]:
    if not isinstance(found, list):
        return [], 0
    out: list[dict[str, str]] = []
    for entry in found:
        if not isinstance(entry, dict) or entry.get("status") not in ITEM_STATUSES:
            continue
        label = _short(entry.get("label"), ITEM_LABEL_CAP)
        if not label:
            continue
        out.append({"label": label, "status": entry["status"], "where": _short(entry.get("where"), ITEM_WHERE_CAP)})
    kept = out[:ITEMS_MAX]
    return kept, len(found) - len(kept)


def tidy_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "tests_claim" in result:
        result = {**result, "tests_claim": tidy_test_claim(result["tests_claim"])}
    if "items" not in result:
        return result
    items, more = tidy_items(result.get("items"))
    out = {**result, "items": items}
    out.pop("items_more", None)
    if more:
        out["items_more"] = more
    return out


CODING_KIND = "coding"
RESEARCH_KIND = "research"


CONVERSATION_ID = {"claude-code": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"),
                   "codex": re.compile(r"^[A-Za-z0-9_-]{1,100}\Z")}


def usable_id(provider: str, value: Any) -> bool:
    pattern = CONVERSATION_ID.get(provider)
    return pattern is not None and isinstance(value, str) and bool(pattern.match(value))


@dataclass
class ProviderContext:

    request_approval: Callable[[dict[str, Any]], Awaitable[Decision]]
    cwd: str
    kind: str = CODING_KIND
    model: str | None = None
    effort: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    permission_mode: str = "default"
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    resume: str | None = None
    output_schema: dict[str, Any] | None = None
    mcp_socket: str | None = None
    propose: bool = False
    lead: bool = False
    drafts: bool = False
    thinking: bool = False
    web: bool = False
    max_subagents: int | None = None
    max_searches: int | None = None
    refusal_note: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    baseline: str | None = None
    raw_frame: Callable[[dict[str, Any]], None] | None = None


# Fail closed: only a literal true gives the web, so an order without the field gets none.
def web_allowed(ctx: ProviderContext) -> bool:
    return ctx.kind == CODING_KIND and ctx.web is True


@dataclass
class Change:
    applied: bool
    value: str | None = None
    reason: str | None = None


def applied(value: str) -> Change:
    return Change(True, value=value)


def refused(reason: str) -> Change:
    return Change(False, reason=reason)


class ToolNames:

    LIMIT = 1000

    def __init__(self) -> None:
        self._names: dict[str, str] = {}

    def annotate(self, events: list[Event]) -> list[Event]:
        out = []
        for event in events:
            p = event.payload
            if event.kind == "tool" and isinstance(p.get("id"), str):
                if p.get("phase") == "started" and p.get("name"):
                    self._names[p["id"]] = p["name"]
                    if len(self._names) > self.LIMIT:
                        self._names.pop(next(iter(self._names)))
                elif p.get("phase") == "completed":
                    name = self._names.pop(p["id"], None)
                    if name and not p.get("name"):
                        event = Event(event.kind, {**p, "name": name})
            out.append(event)
        return out


class Provider(Protocol):
    name: str
    enforces_max_turns: bool

    async def start(self, ctx: ProviderContext) -> None: ...

    def turn(self, text: str) -> AsyncIterator[dict[str, Any]]: ...

    async def steer(self, text: str) -> bool:
        ...

    async def interrupt(self) -> None: ...

    async def set_permission_mode(self, mode: str) -> None: ...

    async def change_settings(self, changes: dict[str, str]) -> dict[str, Change]:
        ...

    def settings_now(self) -> dict[str, str | None]:
        ...

    async def terminate(self) -> None:
        ...

    async def close(self) -> None: ...

    def normalise(self, raw: dict[str, Any]) -> list[Event]: ...

