"""Both providers' raw messages into one set of event kinds, with payloads capped at 16 KB.

Raw shapes as the providers hand them over:
- Claude Code: each SDK message as a dict tagged with its class name (providers.claude.to_raw),
  plus the runner's own `{"type": "QuerySent"}` when it sends a prompt.
- Codex: each app-server notification as JSON-RPC, params in the wire's camelCase.

Approval payloads and question cards are built here too, so the relay posts one shape for both
providers, and the owner's reply to a question is split back into each provider's answer format.

The mapping was checked against real recorded sessions (tests/fixtures/*_recorded*.raw.json) and
the golden files beside them pin it. Where the recordings differed from the documented shapes, the
recorded shape wins, and the comments below name those places. See docs/design.md, "The event schema".
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from runner.events import MARK, Event, cap_payload, clip, size
from runner.providers.base import tidy_result

TEXT_CAP = 8000
PART_BYTES = 12000
PART_NEAR = 1500
PREVIEW_CAP = 2000
REQUESTER_CAP = 100
# Recorded sessions name the subagent tool `Agent`, although the init frame's tool list still says `Task`.
SUBAGENT_TOOLS = ("Task", "Agent")
LIMIT_SUBTYPES = {"error_max_turns": "max_turns", "error_max_budget_usd": "budget"}


def _event(kind: str, payload: dict[str, Any]) -> Event:
    return Event(kind, cap_payload({k: v for k, v in payload.items() if v is not None}))


def _preview(value: Any) -> str:
    if isinstance(value, str):
        return clip(value, PREVIEW_CAP)
    return clip(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str), PREVIEW_CAP)


def _json_cost(ch: str) -> int:
    if ch in '"\\\n\r\t\b\f':
        return 2
    if ch < " " or "\ud800" <= ch <= "\udfff":
        return 6
    return len(ch.encode("utf-8"))


def _fit(text: str, start: int) -> int:
    end, used, last = start, 0, min(len(text), start + TEXT_CAP)
    while end < last:
        used += _json_cost(text[end])
        if used > PART_BYTES:
            break
        end += 1
    return end


def _joins(ch: str) -> bool:
    return bool(unicodedata.combining(ch)) or ch in "\u200d\ufe0f\ufe0e" or "\U000e0100" <= ch <= "\U000e01ef"


def split_text(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    while True:
        end = _fit(text, start)
        if end >= len(text):
            parts.append(text[start:])
            return parts
        floor = max(start + 1, end - PART_NEAR)
        cut = end
        for sep in ("\n\n", "\n", " "):
            found = text.rfind(sep, floor, end)
            if found != -1:
                cut = found + len(sep)
                break
        while cut > start + 1 and _joins(text[cut]):
            cut -= 1
        parts.append(text[start:cut])
        start = cut


def _messages(payload: dict[str, Any], text: str, message_id: Any, ids: bool = False) -> list[Event]:
    parts = split_text(text)
    if len(parts) == 1:
        return [_event("message", {**payload, "text": text, **({"message_id": str(message_id)} if ids else {})})]
    return [_event("message", {**payload, "text": part, "message_id": str(message_id), "part": i,
                               "parts": len(parts)}) for i, part in enumerate(parts, 1)]


CODEX_QUESTION_TOOL = "request_user_input"
QUESTION_TOOLS = frozenset({"AskUserQuestion", CODEX_QUESTION_TOOL})
QUESTIONS_BYTES = 7000
QUESTIONS_MAX, OPTIONS_MAX = 8, 12
FIELD_CAPS = {"question": 2000, "header": 100, "label": 300, "description": 1000, "preview": 2000}
SHRINK = (("preview", len(MARK)), ("description", 120), ("question", 200), ("header", 40), ("label", 40))


def question_input(tool_input: Any) -> dict[str, Any] | None:
    found = tool_input.get("questions") if isinstance(tool_input, dict) else None
    if not isinstance(found, list) or not found:
        return None
    out: list[dict[str, Any]] = []
    for q in found[:QUESTIONS_MAX]:
        if not isinstance(q, dict):
            continue
        item: dict[str, Any] = {"question": clip(str(q.get("question") or ""), FIELD_CAPS["question"])}
        if q.get("header"):
            item["header"] = clip(str(q["header"]), FIELD_CAPS["header"])
        if "multiSelect" in q:
            item["multiSelect"] = bool(q.get("multiSelect"))
        options = []
        for o in (q.get("options") if isinstance(q.get("options"), list) else [])[:OPTIONS_MAX]:
            o = o if isinstance(o, dict) else {"label": o}
            option = {"label": clip(str(o.get("label") or ""), FIELD_CAPS["label"])}
            for key in ("description", "preview"):
                if isinstance(o.get(key), str) and o[key]:
                    option[key] = clip(o[key], FIELD_CAPS[key])
            options.append(option)
        item["options"] = options
        more = len(q.get("options") or []) - len(options) if isinstance(q.get("options"), list) else 0
        if more > 0:
            item["options_more"] = more
        out.append(item)
    if not out:
        return None
    doc: dict[str, Any] = {"questions": out}
    if len(found) > QUESTIONS_MAX:
        doc["questions_more"] = len(found) - QUESTIONS_MAX
    return doc


def _fields(doc: dict[str, Any], name: str) -> list[tuple[dict[str, Any], str]]:
    places = []
    for q in doc["questions"]:
        if name in q:
            places.append((q, name))
        for o in q.get("options") or []:
            if name in o:
                places.append((o, name))
    return places


def _shrink(doc: dict[str, Any], budget: int, measure: Any, stages: tuple) -> None:
    for name, floor in stages:
        if measure(doc) <= budget:
            return
        places = _fields(doc, name)
        if not places:
            continue
        texts = [holder[key] for holder, key in places]

        def apply(cap: int) -> None:
            for (holder, key), text in zip(places, texts):
                holder[key] = clip(text, cap)

        low, high = floor, max(len(t) for t in texts)
        while low < high:
            mid = (low + high + 1) // 2
            apply(mid)
            if measure(doc) <= budget:
                low = mid
            else:
                high = mid - 1
        apply(low)


def _drop(doc: dict[str, Any], budget: int, measure: Any, name: str) -> None:
    while measure(doc) > budget:
        places = _fields(doc, name)
        if not places:
            return
        holder, key = max(places, key=lambda p: len(p[0][p[1]]))
        del holder[key]


HARD = (("description", 40), ("question", 100), ("header", 20), ("label", 30))


def fit_questions(doc: dict[str, Any], budget: int, measure: Any) -> dict[str, Any]:
    doc = json.loads(json.dumps(doc, ensure_ascii=False))
    _shrink(doc, budget, measure, SHRINK)
    if measure(doc) <= budget:
        return doc
    doc["cut"] = True
    _drop(doc, budget, measure, "preview")
    _shrink(doc, budget, measure, HARD)
    _drop(doc, budget, measure, "description")
    questions = doc["questions"]
    while measure(doc) > budget and questions:
        q = max(questions, key=lambda q: len(q["options"]))
        if len(q["options"]) > 1:
            q["options"].pop()
            q["options_more"] = q.get("options_more", 0) + 1
        else:
            questions.pop()
            doc["questions_more"] = doc.get("questions_more", 0) + 1
    return doc


def _preview_json(doc: dict[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False, sort_keys=True)


def question_payload(tool_input: Any) -> dict[str, Any] | None:
    doc = question_input(tool_input)
    if doc is None:
        return None
    preview = _preview_json(fit_questions(doc, PREVIEW_CAP, lambda d: len(_preview_json(d))))
    full = fit_questions(doc, QUESTIONS_BYTES, size)
    return {"input_preview": preview, "questions": full["questions"],
            **({"questions_more": full["questions_more"]} if "questions_more" in full else {}),
            **({"questions_cut": True} if full.get("cut") else {})}


REPLY_PREFIX = "The session's owner answered: "
CODEX_NOTE = "user_note: "


def owner_reply(message: Any) -> str | None:
    if isinstance(message, str) and message.startswith(REPLY_PREFIX):
        return message[len(REPLY_PREFIX):]
    return None


def _option_labels(question: dict[str, Any]) -> list[str]:
    options = question.get("options") if isinstance(question.get("options"), list) else []
    return [str(o.get("label") if isinstance(o, dict) else o) for o in options]


def reply_answers(reply: str, questions: list[dict[str, Any]]) -> list[tuple[str | None, str | None]]:
    heads = [[h for h in (q.get("header"), q.get("question")) if isinstance(h, str) and h.strip()]
             for q in questions]
    picked: list[str | None] = [None] * len(questions)
    rest: list[str] = []
    for text in reply.split("\n"):
        for i, names in enumerate(heads):
            head = next((h for h in names if text.startswith(h + ": ")), None)
            if picked[i] is None and head is not None and text[len(head) + 2:].strip():
                picked[i] = text[len(head) + 2:].strip()
                break
        else:
            rest.append(text)
    written = "\n".join(rest).strip() or None
    if not any(picked) and written is not None:
        for i, q in enumerate(questions):
            if written in _option_labels(q):
                picked[i], written = written, None
                break
    return [(p, written) for p in picked]


def claude_answers(reply: str, tool_input: Any) -> dict[str, Any] | None:
    found = tool_input.get("questions") if isinstance(tool_input, dict) else None
    questions = [q for q in found if isinstance(q, dict) and isinstance(q.get("question"), str)
                 and q["question"].strip()] if isinstance(found, list) else []
    answers: dict[str, str] = {}
    notes: dict[str, dict[str, str]] = {}
    for q, (picked, written) in zip(questions, reply_answers(reply, questions)):
        if picked is not None:
            answers[q["question"]] = picked
            if written:
                notes[q["question"]] = {"notes": written}
        elif written:
            answers[q["question"]] = written
    if not answers:
        return None
    return {**tool_input, "answers": answers, **({"annotations": notes} if notes else {})}


def codex_answers(reply: str, questions: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for q, (picked, written) in zip(questions, reply_answers(reply, questions)):
        if not isinstance(q.get("id"), str):
            continue
        said: list[str] = []
        if picked is not None:
            parts = picked.split(", ")
            labels = _option_labels(q)
            said.extend(parts if len(parts) > 1 and all(p in labels for p in parts) else [picked])
        if written:
            said.append(written if picked is None else CODEX_NOTE + written)
        if said:
            out[q["id"]] = {"answers": said}
    return out


def _tool_input_fields(name: str, tool_input: Any) -> dict[str, Any]:
    if name in QUESTION_TOOLS:
        found = question_payload(tool_input)
        if found is not None:
            return found
    return {"input_preview": _preview(tool_input)}


THINKING_CAP = 8000


def _thinking(text: str, item_id: Any, parent: Any = None) -> Event:
    cut = len(text) > THINKING_CAP
    return _event("thinking", {"text": clip(text, THINKING_CAP), "id": clip(str(item_id), 200) if item_id else None,
                               "parent_tool_use_id": parent, "truncated": True if cut else None})


class ContextMeter:

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.tokens: int | None = None
        self.window: int | None = None
        self.model: str | None = None

    def _reset(self) -> None:
        self.tokens = self.window = self.model = None

    def see(self, raw: dict[str, Any]) -> None:
        if self.provider == "claude-code":
            kind = raw.get("type")
            if kind == "QuerySent":
                self._reset()
            elif kind == "AssistantMessage" and raw.get("parent_tool_use_id") is None:
                usage = raw.get("usage")
                parts = [usage.get(k) for k in ("input_tokens", "cache_creation_input_tokens",
                                                "cache_read_input_tokens")] if isinstance(usage, dict) else []
                if parts and all(_count(p) for p in parts):
                    self.tokens = sum(parts)
                    self.model = raw.get("model") if isinstance(raw.get("model"), str) else None
            elif kind == "ResultMessage":
                self.window = claude_window(raw.get("model_usage"), self.model)
        else:
            method = raw.get("method")
            if method == "turn/started":
                self._reset()
            elif method == "thread/tokenUsage/updated":
                usage = (raw.get("params") or {}).get("tokenUsage") or {}
                last = usage.get("last") if isinstance(usage.get("last"), dict) else {}
                if _count(last.get("totalTokens")):
                    self.tokens = last["totalTokens"]
                if _count(usage.get("modelContextWindow")) and usage["modelContextWindow"] > 0:
                    self.window = usage["modelContextWindow"]

    def annotate(self, events: list[Event]) -> list[Event]:
        out = []
        for event in events:
            if event.kind == "turn" and event.payload.get("phase") == "completed" \
                    and (self.tokens is not None or self.window is not None):
                extra = {k: v for k, v in (("context_tokens", self.tokens), ("context_window", self.window))
                         if v is not None}
                event = Event("turn", {**event.payload, **extra})
            out.append(event)
        return out


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def claude_window(model_usage: Any, model: str | None) -> int | None:
    if not isinstance(model_usage, dict) or not model_usage:
        return None
    names = {model, (model or "").removesuffix("[1m]")} - {None, ""}
    found = [v for k, v in model_usage.items() if isinstance(v, dict)
             and (k in names or k.removesuffix("[1m]") in names or v.get("canonicalModel") in names)]
    if not found and model is None and len(model_usage) == 1:
        found = [v for v in model_usage.values() if isinstance(v, dict)]
    windows = {v.get("contextWindow") for v in found if _count(v.get("contextWindow")) and v["contextWindow"] > 0}
    return windows.pop() if len(windows) == 1 else None


def _tool_result_text(content: Any) -> Any:
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(parts) if parts else content
    return content


def claude(raw: dict[str, Any], ids: bool = False) -> list[Event]:
    kind = raw.get("type")
    parent = raw.get("parent_tool_use_id")
    out: list[Event] = []
    if kind == "QuerySent":
        out.append(_event("turn", {"phase": "started"}))
    elif kind == "AssistantMessage":
        for index, block in enumerate(raw.get("content") or []):
            btype = block.get("type")
            if btype == "TextBlock":
                text = block.get("text", "")
                text = text if isinstance(text, str) else str(text)
                message_id = f"{raw.get('message_id') or raw.get('uuid') or 'claude'}:{index}"
                out.extend(_messages({"role": "assistant", "text": text, "parent_tool_use_id": parent}, text,
                                     message_id, ids))
            elif btype == "ThinkingBlock":
                thought = block.get("thinking")
                if isinstance(thought, str) and thought.strip():
                    out.append(_thinking(thought, f"{raw.get('message_id') or raw.get('uuid') or 'claude'}:{index}",
                                         parent))
            elif btype == "ToolUseBlock":
                name, tool_input = block.get("name", ""), block.get("input") or {}
                out.append(_event("tool", {"phase": "started", "id": block.get("id"), "name": name,
                                           **_tool_input_fields(name, tool_input), "parent_tool_use_id": parent}))
                if name in SUBAGENT_TOOLS:
                    out.append(_event("subagent", {"phase": "requested", "tool_use_id": block.get("id"),
                                                   "subagent_type": tool_input.get("subagent_type"),
                                                   "description": _preview(tool_input.get("description", ""))}))
        if raw.get("error"):
            out.append(_event("message", {"role": "error", "text": clip(str(raw["error"]), TEXT_CAP)}))
    elif kind == "UserMessage":
        content = raw.get("content")
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "ToolResultBlock":
                out.append(_event("tool", {"phase": "completed", "id": block.get("tool_use_id"),
                                           "is_error": bool(block.get("is_error")),
                                           "output_preview": _preview(_tool_result_text(block.get("content"))),
                                           "parent_tool_use_id": parent}))
    elif kind == "TaskStartedMessage":
        out.append(_event("subagent", {"phase": "started", "task_id": raw.get("task_id"),
                                       "tool_use_id": raw.get("tool_use_id"), "task_type": raw.get("task_type"),
                                       "description": _preview(raw.get("description", ""))}))
    elif kind == "TaskProgressMessage":
        out.append(_event("subagent", {"phase": "progress", "task_id": raw.get("task_id"),
                                       "tool_use_id": raw.get("tool_use_id"),
                                       "last_tool_name": raw.get("last_tool_name"), "usage": raw.get("usage")}))
    elif kind == "TaskNotificationMessage":
        out.append(_event("subagent", {"phase": raw.get("status") or "ended", "task_id": raw.get("task_id"),
                                       "tool_use_id": raw.get("tool_use_id"),
                                       "summary": _preview(raw.get("summary", ""))}))
    elif kind == "ResultMessage":
        subtype = raw.get("subtype")
        out.append(_event("turn", {"phase": "completed", "subtype": subtype, "is_error": raw.get("is_error"),
                                   "num_turns": raw.get("num_turns"), "duration_ms": raw.get("duration_ms"),
                                   "stop_reason": raw.get("stop_reason"), "terminal_reason": raw.get("terminal_reason"),
                                   "interrupted": raw.get("terminal_reason") in ("aborted_streaming", "aborted_tools"),
                                   "session_id": raw.get("session_id"),
                                   "structured_output": tidy_result(raw.get("structured_output")),
                                   "permission_denials": len(raw.get("permission_denials") or []) or None,
                                   "errors": [clip(str(e), 500) for e in raw["errors"][:5]]
                                   if raw.get("errors") else None,
                                   "api_error_status": raw.get("api_error_status")}))
        usage = {"total_cost_usd": raw.get("total_cost_usd"), "usage": raw.get("usage"),
                 "model_usage": raw.get("model_usage")}
        if any(v is not None for v in usage.values()):
            out.append(_event("usage", usage))
        if subtype in LIMIT_SUBTYPES:
            out.append(_event("limit", {"limit": LIMIT_SUBTYPES[subtype], "source": "provider", "subtype": subtype}))
    elif kind == "RateLimitEvent":
        out.extend(claude_windows(raw.get("rate_limit_info") or {}))
    return out


WINDOWS_CAP = 12
WINDOW_NAME_CAP = 40


def _window_name(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= WINDOW_NAME_CAP and value.isascii() \
        and all(c.islower() or c.isdigit() or c == "_" for c in value) and value[0].isalpha()


# On a subscription the top-level utilization is null. Each window's figure is under
# `raw.unifiedWindows`, and every window becomes a `limit` event of its own.
def claude_windows(info: dict[str, Any]) -> list[Event]:
    window = info.get("rate_limit_type")
    windows = (info.get("raw") or {}).get("unifiedWindows") if isinstance(info.get("raw"), dict) else None
    windows = windows if isinstance(windows, dict) else {}
    utilization = info.get("utilization")
    if utilization is None:
        utilization = (windows.get(window) or {}).get("utilization") if isinstance(windows.get(window), dict) else None
    out = [_event("limit", {"limit": "rate_window", "source": "provider", "window": window,
                            "status": info.get("status"), "utilization": utilization,
                            "resets_at": info.get("resets_at"), "overage_status": info.get("overage_status")})]
    for name, found in list(windows.items())[:WINDOWS_CAP]:
        if name == window or not _window_name(name) or not isinstance(found, dict):
            continue
        out.append(_event("limit", {"limit": "rate_window", "source": "provider", "window": name,
                                    "status": found.get("status"), "utilization": found.get("utilization"),
                                    "resets_at": found.get("resetsAt")}))
    return out


def claude_approval(tool_name: str, tool_input: dict[str, Any], tool_use_id: str | None,
                    agent_id: str | None = None, title: str | None = None) -> dict[str, Any]:
    fields = _tool_input_fields(tool_name, tool_input)
    return {k: v for k, v in {
        "request_id": clip(tool_use_id or "", 100), "tool": clip(tool_name or "unknown", 200),
        "input_preview": fields.pop("input_preview") or None,
        "requester": clip(agent_id, REQUESTER_CAP) if agent_id else None,
        "title": clip(title, 300) if title else None,
        **fields,
    }.items() if v is not None}


def _change_kind(kind: Any) -> Any:
    return kind.get("type") if isinstance(kind, dict) else kind


def file_changes(changes: Any) -> list[dict[str, Any]]:
    return [{"path": c.get("path"), "kind": _change_kind(c.get("kind"))}
            for c in (changes if isinstance(changes, list) else []) if isinstance(c, dict)]


def _codex_item(phase: str, item: dict[str, Any], ids: bool = False) -> list[Event]:
    itype = item.get("type")
    if itype == "agentMessage":
        if phase == "completed":
            text = item.get("text", "")
            text = text if isinstance(text, str) else str(text)
            return _messages({"role": "assistant", "text": text, "item_id": item.get("id"),
                              "phase": item.get("phase")}, text, item.get("id") or "codex", ids)
        return []
    if itype == "reasoning":
        summary = item.get("summary")
        parts = [s for s in summary if isinstance(s, str) and s.strip()] if isinstance(summary, list) else []
        if phase != "completed" or not parts:
            return []
        return [_thinking("\n\n".join(parts), item.get("id"))]
    if itype == "commandExecution":
        payload = {"phase": phase, "id": item.get("id"), "name": "command",
                   "input_preview": _preview(item.get("command", "")), "cwd": item.get("cwd"),
                   "status": item.get("status")}
        if phase == "completed":
            payload |= {"exit_code": item.get("exitCode"), "is_error": item.get("status") != "completed",
                        "output_preview": _preview(item.get("aggregatedOutput") or "")}
        return [_event("tool", payload)]
    if itype == "fileChange":
        return [_event("tool", {"phase": phase, "id": item.get("id"), "name": "fileChange",
                                "input_preview": _preview(file_changes(item.get("changes"))),
                                "status": item.get("status"),
                                "is_error": item.get("status") not in ("completed", "inProgress", None)
                                if phase == "completed" else None})]
    if itype == "subAgentActivity":
        if phase != "completed":
            return []
        return [_event("subagent", {"phase": item.get("kind"), "id": item.get("id"),
                                    "agent_thread_id": item.get("agentThreadId"), "agent_path": item.get("agentPath"),
                                    "model": item.get("model")})]
    if itype == "collabAgentToolCall":
        return [_event("subagent", {"phase": phase, "id": item.get("id"), "tool": item.get("tool"),
                                    "sender_thread_id": item.get("senderThreadId"),
                                    "receiver_thread_ids": item.get("receiverThreadIds") or None,
                                    "status": item.get("status"),
                                    "description": _preview(item.get("prompt")) if item.get("prompt") else None})]
    if itype in ("mcpToolCall", "dynamicToolCall", "webSearch"):
        payload = {"phase": phase, "id": item.get("id"), "name": item.get("tool") or itype,
                   "status": item.get("status")}
        if itype == "mcpToolCall":
            payload |= {"server": item.get("server"), "duration_ms": item.get("durationMs")}
        return [_event("tool", payload)]
    return []


def codex(raw: dict[str, Any], ids: bool = False) -> list[Event]:
    method, params = raw.get("method"), raw.get("params") or {}
    if method == "turn/started":
        return [_event("turn", {"phase": "started", "turn_id": (params.get("turn") or {}).get("id")})]
    if method == "turn/completed":
        turn = params.get("turn") or {}
        error = turn.get("error") or {}
        return [_event("turn", {"phase": "completed", "turn_id": turn.get("id"), "status": turn.get("status"),
                                "interrupted": turn.get("status") == "interrupted",
                                "is_error": turn.get("status") == "failed",
                                "duration_ms": turn.get("durationMs"),
                                "error": clip(error.get("message"), 2000) if error.get("message") else None})]
    if method in ("item/started", "item/completed"):
        return _codex_item(method.split("/")[1], params.get("item") or {}, ids)
    if method == "thread/tokenUsage/updated":
        usage = params.get("tokenUsage") or {}
        return [_event("usage", {"turn_id": params.get("turnId"), "total": usage.get("total"),
                                 "last": usage.get("last"), "model_context_window": usage.get("modelContextWindow")})]
    if method == "account/rateLimits/updated":
        limits = params.get("rateLimits") or {}
        return [_event("limit", {"limit": "rate_window", "source": "provider", "limit_id": limits.get("limitId"),
                                 "primary": limits.get("primary"), "secondary": limits.get("secondary"),
                                 "reached": limits.get("rateLimitReachedType")})]
    if method == "error":
        error = params.get("error") or {}
        return [_event("message", {"role": "error", "text": clip(error.get("message") or "error", TEXT_CAP),
                                   "will_retry": params.get("willRetry")})]
    return []


APPROVAL_METHODS = ("item/commandExecution/requestApproval", "item/fileChange/requestApproval")


def codex_approval(method: str, params: dict[str, Any], changes: Any = None) -> dict[str, Any]:
    tool = "command" if method == APPROVAL_METHODS[0] else "fileChange"
    detail = {k: params.get(k) for k in ("command", "cwd", "reason", "grantRoot") if params.get(k) is not None}
    if tool == "fileChange" and changes:
        detail["changes"] = [{"path": c.get("path"), "kind": _change_kind(c.get("kind")),
                              "diff": clip(c["diff"], 600) if isinstance(c.get("diff"), str) else None}
                             for c in changes if isinstance(c, dict)][:20]
    request_id = params.get("itemId") or ""
    return {k: v for k, v in {
        "request_id": clip(str(request_id), 100), "tool": tool,
        "input_preview": _preview(detail) if detail else None,
    }.items() if v is not None}


CODEX_QUESTION_METHOD = "item/tool/requestUserInput"


def codex_questions(params: dict[str, Any]) -> list[dict[str, Any]]:
    found = params.get("questions") if isinstance(params, dict) else None
    return [q for q in found if isinstance(q, dict) and isinstance(q.get("id"), str)
            and isinstance(q.get("question"), str) and q["question"].strip()] if isinstance(found, list) else []


def codex_question(params: dict[str, Any]) -> dict[str, Any]:
    doc = {"questions": [{"question": q["question"], "header": q.get("header"), "multiSelect": False,
                          "options": [o for o in q.get("options") or [] if isinstance(o, dict)]}
                         for q in codex_questions(params)]}
    fields = _tool_input_fields(CODEX_QUESTION_TOOL, doc)
    return {k: v for k, v in {
        "request_id": clip(str(params.get("itemId") or ""), 100), "tool": CODEX_QUESTION_TOOL,
        "input_preview": fields.pop("input_preview") or None, **fields,
    }.items() if v is not None}


NORMALISERS = {"claude-code": claude, "codex": codex}

