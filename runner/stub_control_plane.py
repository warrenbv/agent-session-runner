"""An in-process control plane on 127.0.0.1, for the tests and the demo.

It implements the operations the runner calls with the same shapes and rules a real control plane
enforces: `sessions.event_post` (numbered events, the 16 KB cap, duplicates, gaps, approval events),
`sessions.pull` (prompts, approval answers and controls, each delivered once), `sessions.tool_call`,
`sessions.draft_post` and the three research operations. A wrong or revoked token is 401.

Test controls: queue(), control(), answer(), expire(), revoke(), fault injection (`faults`), and
`when()` hooks that run as an event is stored, so a test can act at an exact point in the stream.
It keeps everything in memory and is not a server to deploy.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

TOKEN = "runner-test-token-0000"
CAP = 16384
RESEARCH_PART = 48 * 1024
OLD_KINDS = ("turn", "message", "tool", "approval", "subagent", "diff", "usage", "limit", "end")
KINDS = OLD_KINDS + ("thinking",)
PROVIDERS = ("claude-code", "codex")
MODES = ("queue", "steer", "now")


class StubControlPlane:
    def __init__(self, token: str = TOKEN, *, serve: bool = True) -> None:
        self.token = token
        self.known_kinds = KINDS
        self.events: list[dict[str, Any]] = []
        self.sequence_base = 0
        self.queue_entries: list[dict[str, Any]] = []
        self.approvals: dict[str, dict[str, Any]] = {}
        self.faults: deque[str] = deque()
        self.requests: list[dict[str, Any]] = []
        self.hosts: list[str | None] = []
        self.hooks: list[tuple[Callable[[dict], bool], Callable[[dict], None], list[bool]]] = []
        self.revoked = False
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_reply: Callable[[dict[str, Any]], tuple[int, Any]] = \
            lambda body: (200, {"ok": True, "result": {"tool": body["tool"], "arguments": body["arguments"]},
                                "error": None})
        self.controls: list[dict[str, Any]] = []
        self.pull_extra: dict[str, Any] = {}
        self.draft_reply: tuple[int, Any] | None = None
        self.drafts: list[dict[str, Any]] = []
        self._next_control = 1
        self.research_input_files: list[tuple[str, str]] = []
        self.research_inputs: list[dict[str, Any]] = []
        self.research_parts: list[dict[str, Any]] = []
        self.research_done: list[dict[str, Any]] = []
        self.research_refuse: set[str] = set()
        self.research_done_answers: list[tuple[int, Any]] = []
        self.lock = threading.RLock()
        self._next_id = 1
        if not serve:
            return
        handler = type("Handler", (_Handler,), {"owner": self})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def queue(self, text: str, mode: str = "queue", author: str = "principal:owner", **extra: Any) -> int:
        assert mode in MODES
        with self.lock:
            entry = {"id": self._next_id, "text": text, "mode": mode, "author": author, "status": "queued",
                     "extra": extra}
            self._next_id += 1
            self.queue_entries.append(entry)
            return entry["id"]

    def control(self, kind: str, entry_id: int | None = None, **fields: Any) -> int:
        assert kind in ("interrupt", "finish", "withdraw", "settings", "keep_open")
        with self.lock:
            cid = self._next_control
            self._next_control += 1
            row: dict[str, Any] = {"id": cid, "kind": kind, "delivered": False, **fields}
            if kind == "withdraw":
                row["entry_id"] = entry_id
            self.controls.append(row)
            return cid

    def answer(self, request_id: str, decision: str, feedback: str | None = None) -> None:
        with self.lock:
            row = self.approvals[request_id]
            assert row["status"] == "pending", "first answer wins"
            row.update(status="allowed" if decision == "allow" else "denied", feedback=feedback)

    def expire(self, request_id: str) -> None:
        with self.lock:
            self.approvals[request_id]["status"] = "expired"

    def revoke(self) -> None:
        with self.lock:
            self.revoked = True

    def when(self, predicate: Callable[[dict], bool], action: Callable[[dict], None]) -> None:
        self.hooks.append((predicate, action, [False]))

    def kinds(self) -> list[str]:
        return [e["kind"] for e in self.events]

    def of(self, kind: str, **match: Any) -> list[dict[str, Any]]:
        return [e["payload"] for e in self.events if e["kind"] == kind
                and all(e["payload"].get(k) == v for k, v in match.items())]

    def handle(self, op: str, auth: str | None, body: Any) -> tuple[int, Any]:
        with self.lock:
            self.requests.append({"op": op, "auth": auth, "body": body})
            if auth != f"Bearer {self.token}" or self.revoked:
                return 401, _error("unauthenticated", "authentication required")
            if not isinstance(body, dict):
                return 400, _error("invalid", "body must be a JSON object")
            if op == "sessions.event_post":
                return self._event_post(body)
            if op == "sessions.pull":
                if body:
                    return 400, _error("invalid", "extra fields")
                return 200, self._pull()
            if op == "sessions.tool_call":
                if set(body) - {"tool", "arguments"} or not isinstance(body.get("tool"), str) \
                        or not isinstance(body.get("arguments", {}), dict):
                    return 400, _error("invalid", "tool and arguments only")
                body = {"tool": body["tool"], "arguments": body.get("arguments", {})}
                self.tool_calls.append(body)
                return self.tool_reply(body)
            if op == "sessions.draft_post" and self.draft_reply is not None:
                if set(body) != {"message_id", "offset", "text"}:
                    return 400, _error("invalid", "fields are message_id, offset, text")
                self.drafts.append({**body, "after": len(self.events)})
                return self.draft_reply
            if op == "sessions.draft_post":
                return 404, _error("not_found", f"unknown operation {op!r}")
            if op in ("research.inputs_get", "research.output_part", "research.output_done"):
                return self._research(op, body)
            return 403, _error("forbidden", "not with this credential")

    def _research(self, op: str, body: dict[str, Any]) -> tuple[int, Any]:
        if op == "research.inputs_get":
            if set(body) != {"session_id", "part"} or not isinstance(body["part"], int):
                return 400, _error("invalid", "fields are session_id, part")
            chunks = list(self.research_inputs) or _input_chunks(self.research_input_files)
            if not chunks:
                if body["part"] != 0:
                    return 400, _error("invalid", "this step has no inputs")
                return 200, {"part": 0, "total": 0, "name": None, "index": 0, "parts": 0, "sha256": None,
                             "content": ""}
            if not 0 <= body["part"] < len(chunks):
                return 400, _error("invalid", f"a part is 0 to {len(chunks) - 1}")
            return 200, {"part": body["part"], "total": len(chunks), **chunks[body["part"]]}
        if op == "research.output_part":
            if set(body) != {"session_id", "file", "index", "total", "content"} or not isinstance(body["content"], str) \
                    or not isinstance(body["index"], int) or not isinstance(body["total"], int):
                return 400, _error("invalid", "fields are session_id, file, index, total, content")
            if len(body["content"].encode("utf-8")) > RESEARCH_PART:
                return 413, _error("too_large", "a part is at most 48 KB")
            if body["file"] in self.research_refuse:
                return 400, _error("invalid", "refused")
            self.research_parts.append(body)
            return 200, {"stored": True}
        if set(body) != {"session_id", "files", "partial", "reason"} or not isinstance(body["files"], list) \
                or not all(isinstance(f, dict) and set(f) == {"name", "parts", "sha256"} for f in body["files"]) \
                or not isinstance(body["partial"], bool):
            return 400, _error("invalid", "fields are session_id, files, partial, reason")
        self.research_done.append({**body, "events_before": len(self.events)})
        if self.research_done_answers:
            return self.research_done_answers.pop(0)
        return 200, {"accepted": True, "reason": None, "claims": 0, "redacted": False}

    def _event_post(self, body: dict[str, Any]) -> tuple[int, Any]:
        if set(body) != {"sequence", "kind", "provider", "payload"}:
            return 400, _error("invalid", "fields are sequence, kind, provider, payload")
        seq, kind, provider, payload = body["sequence"], body["kind"], body["provider"], body["payload"]
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1 or kind not in self.known_kinds \
                or provider not in PROVIDERS or not isinstance(payload, dict):
            return 400, _error("invalid", "bad event")
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        fault = self.faults.popleft() if self.faults else None
        if fault == "413" or size > CAP:
            return 413, _error("too_large", f"an event payload is at most {CAP} bytes of JSON")
        if fault in ("drop_before", "503"):
            return (0 if fault == "drop_before" else 503), _error("internal", "unavailable")
        last = self.events[-1]["sequence"] if self.events else self.sequence_base
        if seq <= last:
            return 200, {"sequence": seq, "duplicate": True, "approval_id": None, "redacted": False}
        if seq != last + 1:
            return 409, _error("conflict", f"expected sequence {last + 1}")
        approval_id = None
        if kind == "approval":
            rid, tool = payload.get("request_id"), payload.get("tool")
            if not (isinstance(rid, str) and rid.strip() and isinstance(tool, str) and tool.strip()):
                return 400, _error("invalid", "an approval event needs request_id and tool as text")
            if rid in self.approvals:
                return 409, _error("conflict", "this request id already has an approval")
            self.approvals[rid] = {"status": "pending", "feedback": None, "delivered": False, "tool": tool}
            approval_id = len(self.approvals)
        event = {"sequence": seq, "kind": kind, "provider": provider, "payload": payload}
        self.events.append(event)
        for predicate, action, fired in self.hooks:
            if not fired[0] and predicate(event):
                fired[0] = True
                action(event)
        if fault == "drop_after":
            return 0, None
        return 200, {"sequence": seq, "duplicate": False, "approval_id": approval_id, "redacted": False}

    def _pull(self) -> dict[str, Any]:
        prompts = [{**{k: e[k] for k in ("id", "text", "mode", "author")}, **e.get("extra", {})}
                   for e in self.queue_entries if e["status"] == "queued"]
        for e in self.queue_entries:
            e["status"] = "sent"
        answers = []
        for rid, row in self.approvals.items():
            if row["status"] != "pending" and not row["delivered"]:
                row["delivered"] = True
                answers.append({"request_id": rid, "decision": "allow" if row["status"] == "allowed" else "deny",
                                "expired": row["status"] == "expired", "feedback": row["feedback"]})
        controls = [{k: v for k, v in c.items() if k != "delivered"} for c in self.controls
                    if not c["delivered"]]
        for c in self.controls:
            c["delivered"] = True
        return {"prompts": prompts, "answers": answers, "controls": controls, **self.pull_extra}


def _error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _input_chunks(files: list[tuple[str, str]]) -> list[dict[str, Any]]:
    import hashlib

    chunks = []
    for name, text in files:
        data, pieces, start = text.encode("utf-8"), [], 0
        while start < len(data):
            end = min(start + RESEARCH_PART, len(data))
            while end < len(data) and (data[end] & 0xC0) == 0x80:
                end -= 1
            pieces.append(data[start:end].decode("utf-8"))
            start = end
        pieces = pieces or [""]
        digest = hashlib.sha256(data).hexdigest()
        chunks += [{"name": name, "index": i, "parts": len(pieces), "sha256": digest, "content": piece}
                   for i, piece in enumerate(pieces)]
    return chunks


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    owner: StubControlPlane

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = None
        prefix = "/api/v1/call/"
        if not self.path.startswith(prefix):
            self._answer(404, _error("not_found", "no such route"))
            return
        with self.owner.lock:
            self.owner.hosts.append(self.headers.get("Host"))
        status, obj = self.owner.handle(self.path[len(prefix):], self.headers.get("Authorization"), body)
        if status == 0:
            self.close_connection = True
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self._answer(status, obj)

    def _answer(self, status: int, obj: Any) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

