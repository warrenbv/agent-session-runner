"""The MCP tool catalogue: the tools a coding session may be offered, each with a fixed
description and a closed JSON Schema for its arguments.

Eight tools work on the session's own project in the tracker, `sessions_propose` lets a session
propose separate sessions for the owner to start, and four lead tools let a lead session start,
review, send back and ask to merge worker sessions. Which are offered depends on the session's flags
(`offered`). This file only describes: the control plane enforces scope and argument rules on every
call, whatever was offered. A test pins a hash of the whole catalogue, so a change here is deliberate.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any

SERVER_NAME = "platform"
SOCKET_ENV = "AGENT_MCP_SOCKET"
WRAPPER = "runner-mcp"
SERVER_VERSION = "1"

_SCOPE = "Works only within this session's own project."
_WRITTEN = "The owner sees the change marked as written by this session."
_STATUSES = ["inbox", "planned", "running", "waiting", "blocked", "done", "dropped"]
_TASK_ID = {"type": "integer", "minimum": 1, "description": "The task's number, as tasks_list shows it."}
_DUE = {"type": "string", "maxLength": 16, "description": "YYYY-MM-DD, or YYYY-MM-DD HH:MM"}
_PRIORITY = {"type": "integer", "minimum": 1, "maximum": 4, "description": "1 is the highest, 4 the lowest."}
_WORKER = {"type": "integer", "minimum": 1, "description": "The worker's session number, as workers_start returned it."}


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        out["required"] = required
    return out


def _tool(name: str, description: str, schema: dict[str, Any], read_only: bool) -> dict[str, Any]:
    return {"name": name, "description": description, "inputSchema": schema,
            "annotations": {"readOnlyHint": read_only, "destructiveHint": False, "openWorldHint": False}}


TOOLS: tuple[dict[str, Any], ...] = (
    _tool("tasks_list",
          f"Lists the open tasks of this project in its tracker, with their numbers, titles, status, priority and"
          f" due dates; pass a status to list only those. {_SCOPE}",
          _schema({"status": {"type": "string", "enum": _STATUSES}}), True),
    _tool("task_get",
          f"Reads one task of this project: its notes, subtasks and comments, and who wrote it. {_SCOPE}",
          _schema({"task_id": _TASK_ID}, ["task_id"]), True),
    _tool("project_get",
          f"Reads this project's tracker entry: its name, status, description, next steps, goals and test"
          f" command. {_SCOPE}",
          _schema({}), True),
    _tool("notes_search",
          f"Searches this project's notes and memory for a few words and returns the matching titles with short"
          f" snippets. {_SCOPE}",
          _schema({"query": {"type": "string", "minLength": 1, "maxLength": 200}}, ["query"]), True),
    _tool("task_create",
          f"Creates a task in this project's tracker, with optional notes, due date and priority. {_SCOPE}"
          f" {_WRITTEN}",
          _schema({"title": {"type": "string", "minLength": 1, "maxLength": 300},
                   "notes": {"type": "string", "maxLength": 20000}, "due": _DUE, "priority": _PRIORITY},
                  ["title"]), False),
    _tool("task_update",
          f"Changes a task of this project: any of its title, notes, status, priority and due date. {_SCOPE}"
          f" {_WRITTEN}",
          _schema({"task_id": _TASK_ID, "title": {"type": "string", "minLength": 1, "maxLength": 300},
                   "notes": {"type": "string", "maxLength": 20000}, "status": {"type": "string", "enum": _STATUSES},
                   "priority": _PRIORITY, "due": _DUE}, ["task_id"]), False),
    _tool("task_complete",
          f"Marks a task of this project done. {_SCOPE} {_WRITTEN}",
          _schema({"task_id": _TASK_ID}, ["task_id"]), False),
    _tool("task_comment_add",
          f"Adds a comment to a task of this project, for example what this session found or did. {_SCOPE}"
          f" {_WRITTEN}",
          _schema({"task_id": _TASK_ID, "text": {"type": "string", "minLength": 1, "maxLength": 10000}},
                  ["task_id", "text"]), False),
    _tool("sessions_propose",
          "Proposes splitting off one to four separate pieces of work as new coding sessions of this project. It"
          " puts one card in front of the owner showing each piece's title and whole brief; nothing starts unless"
          " the owner presses Start, so do not wait for an answer: carry on with your own work. Each piece starts"
          " as a fresh session from the project's default branch, so it cannot see or build on your changes, which"
          " are not merged yet; write each brief so that a new session with no other context can do the piece on"
          " its own. Pieces run with your provider and model and no more than your effort and limits. Propose only"
          " pieces that do not depend on each other or on your work and are each more than a few minutes of work."
          " One card may wait at a time, and a session may make three. Works only within this session's own"
          " project. The owner sees the card marked as written by this session.",
          _schema({"pieces": {"type": "array", "minItems": 1, "maxItems": 4,
                              "description": "Each piece of work to propose as its own new session.",
                              "items": _schema({"title": {"type": "string", "minLength": 1, "maxLength": 120,
                                                          "description": "A short title for the piece."},
                                                "brief": {"type": "string", "minLength": 1, "maxLength": 4000,
                                                          "description": "The new session's whole brief."}},
                                               ["title", "brief"])},
                   "why": {"type": "string", "minLength": 1, "maxLength": 300,
                           "description": "One sentence on why splitting this work off helps."}},
                  ["pieces", "why"]), False),
    _tool("workers_start",
          "Starts one to four worker sessions of this project at once, each a fresh coding session on a machine"
          " with the title and whole brief you give. Each worker starts from the project's default branch, so it"
          " cannot see or build on your changes; write each brief so that a new session with no other context can"
          " do the piece on its own. A worker runs on its own, bounded by its wall clock (5 to 120 minutes, default"
          " 30), cannot start or propose sessions itself, and is merged only through worker_merge or by the owner."
          " You choose each worker's model and effort. With neither model nor provider it takes the project's"
          " default model family, and provider alone (codex or claude) takes that provider's default family. A"
          " lighter model family may be named for a small or mechanical piece. Some model families are never"
          " offered to a worker, and a model that is not offered is refused with the families that are. Effort is"
          " medium, high or xhigh, default high: medium only for a small mechanical piece, xhigh for a hard one."
          " Returns each worker's number and whether it runs now or waits for a machine; you are told when each"
          " worker ends, so do not wait for it: carry on with your own work. A start is refused when the machines"
          " have no room (one slot on each machine stays free for the owner) and after 30 starts in this session,"
          " until the owner allows more. Works only within this session's own project. The owner sees each worker"
          " marked as written by this session.",
          _schema({"workers": {"type": "array", "minItems": 1, "maxItems": 4,
                               "description": "Each worker session to start.",
                               "items": _schema({"title": {"type": "string", "minLength": 1, "maxLength": 120,
                                                           "description": "A short title for the worker."},
                                                 "brief": {"type": "string", "minLength": 1, "maxLength": 16000,
                                                           "description": "The worker's whole brief."},
                                                 "provider": {"type": "string", "enum": ["codex", "claude"],
                                                              "description": "codex or claude; a model names its"
                                                                             " own."},
                                                 "model": {"type": "string", "minLength": 1, "maxLength": 40,
                                                           "pattern": "^[a-z0-9.-]{1,40}$",
                                                           "description": "The worker's model family, as one"
                                                                          " lowercase word; it names its own"
                                                                          " provider."},
                                                 "effort": {"type": "string", "enum": ["medium", "high", "xhigh"],
                                                            "description": "The worker's effort; default high."},
                                                 "wall_clock_minutes": {"type": "integer", "minimum": 5,
                                                                        "maximum": 120,
                                                                        "description": "Its time limit in minutes;"
                                                                                       " default 30."}},
                                                ["title", "brief"])}},
                  ["workers"]), False),
    _tool("worker_changes",
          "Reads the change of one of the workers this session started, as the control plane stores it: each file"
          " with its statistics and patch, one page at a time (pass page 2 and on for the rest), and `whole`, which"
          " is false when the change was too large to be kept whole; a merge you ask for of such a change is held"
          " for the owner. Read a worker's change before you ask for it to be merged. Only your own workers'"
          " changes can be read; it does not check anything out. Works only within this session's own project.",
          _schema({"worker": _WORKER,
                   "page": {"type": "integer", "minimum": 1, "description": "The page to read; default 1."}},
                  ["worker"]), True),
    _tool("worker_send_back",
          "Picks one of the workers this session started back up once it has ended, with your notes under a fixed"
          " first line: it carries on in the same branch and pull request, and you are told again when it ends."
          " Write notes that say what is wrong and what you want instead. A worker may be sent back at most three"
          " times. Only your own ended workers. Works only within this session's own project. The owner sees the"
          " notes marked as written by this session.",
          _schema({"worker": _WORKER,
                   "notes": {"type": "string", "minLength": 1, "maxLength": 10000,
                             "description": "What the worker is to change, for its next run."}},
                  ["worker", "notes"]), False),
    _tool("worker_merge",
          "Asks for the pull request of one of the workers this session started to be merged. Code decides whether"
          " the merge goes through, not you; among its checks: the project lets its lead merge, the worker"
          " completed with a pull request, its own tests passed against the project's test command, it left no"
          " open questions, its change is stored whole and touches no protected path, you were given no untrusted"
          " input, neither you nor the worker read the web, and the merged tree passes its tests. Otherwise the"
          " merge is held: the owner is shown the reasons with Merge and Close, and they are returned to you, so"
          " tell your owner why and carry on. You are told when a merge you asked for is done, fails or is held."
          " Only your own"
          " workers; you cannot revert a merge, deploy, or change the project's settings. Works only within this"
          " session's own project. The owner sees the request marked as written by this session.",
          _schema({"worker": _WORKER}, ["worker"]), False),
)

NAMES: tuple[str, ...] = tuple(t["name"] for t in TOOLS)
PROPOSE = "sessions_propose"
LEAD_TOOLS: tuple[str, ...] = ("workers_start", "worker_changes", "worker_send_back", "worker_merge")
BASE: tuple[str, ...] = tuple(n for n in NAMES if n != PROPOSE and n not in LEAD_TOOLS)


# Description, not control: the control plane refuses whatever is out of scope, whatever was offered.
def offered(propose: bool, lead: bool = False) -> tuple[str, ...]:
    if lead is True:
        return tuple(n for n in NAMES if n != PROPOSE)
    return tuple(n for n in NAMES if n not in LEAD_TOOLS and (n != PROPOSE or propose is True))


def withheld(propose: bool, lead: bool = False) -> tuple[str, ...]:
    shown = offered(propose, lead)
    return tuple(n for n in NAMES if n not in shown)


def catalogue_hash() -> str:
    text = json.dumps(list(TOOLS), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def claude_name(name: str) -> str:
    return f"mcp__{SERVER_NAME}__{name}"


def claude_names(propose: bool, lead: bool = False) -> list[str]:
    return [claude_name(name) for name in offered(propose, lead)]


def wrapper_path(prefix: str | None = None) -> str:
    return os.path.join(prefix if prefix is not None else sys.prefix, "bin", WRAPPER)

