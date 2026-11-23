"""One normalised event, and the 16 KB payload cap.

The kinds are the same for both providers (KINDS, described in docs/design.md, "The event schema"). The cap is
measured as the control plane measures it: compact JSON in UTF-8 bytes. A payload over the cap is cut
here, longest strings first, and marked `"truncated": true`, so a 413 should never happen. If it does,
the poster falls back to `stub()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PAYLOAD_CAP = 16384
KINDS = ("turn", "message", "tool", "approval", "subagent", "diff", "usage", "limit", "end", "thinking")
DRAFT_OP = "sessions.draft_post"
MARK = "…[truncated]"


@dataclass(frozen=True)
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")


# Measured exactly as the control plane measures it, so the cap here and the 413 there agree.
def size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def clip(text: Any, limit: int) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: max(0, limit - len(MARK))] + MARK


def _strings(value: Any, path: tuple = ()) -> list[tuple[tuple, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _strings(v, path + (k,))]
    if isinstance(value, list):
        return [s for i, v in enumerate(value) for s in _strings(v, path + (i,))]
    return []


def _set(root: Any, path: tuple, value: Any) -> None:
    for key in path[:-1]:
        root = root[key]
    root[path[-1]] = value


def stub(payload: dict[str, Any], cap: int = PAYLOAD_CAP) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    out = {"truncated": True, "keys": [str(k)[:100] for k in list(payload)[:50]]}
    budget = cap - size(out) - 100
    out["preview"] = clip(text, max(0, budget // 4))
    return out


def cap_payload(payload: dict[str, Any], cap: int = PAYLOAD_CAP) -> dict[str, Any]:
    if size(payload) <= cap:
        return payload
    out = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    out["truncated"] = True
    for _ in range(200):
        if size(out) <= cap:
            return out
        strings = sorted(_strings(out), key=lambda s: len(s[1]), reverse=True)
        if not strings or len(strings[0][1]) <= 64:
            break
        path, longest = strings[0]
        excess = size(out) - cap
        keep = max(32, min(len(longest) // 2, len(longest) - excess - len(MARK)))
        _set(out, path, clip(longest, keep))
    return out if size(out) <= cap else stub(payload, cap)

