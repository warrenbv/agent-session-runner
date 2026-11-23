"""A running session's settings: one vocabulary for both providers, its checks, and how each word maps
onto each provider. Applied from the next turn (session.py).

    model     a model id (Claude also needs a full `claude-...` id in providers/claude.py)
    effort    low, medium, high, xhigh, max
    mode      manual, accept-edits, plan, auto
    thinking  on, off

Claude Code: effort is the SDK's effort level and mode a permission mode, never bypassPermissions.
Codex: effort is a reasoning effort (`max` becomes the highest the model lists), and mode maps to an
approval policy and a permission profile.
"""

from __future__ import annotations

import re
from typing import Any

FIELDS = ("model", "effort", "mode", "thinking")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODES = ("manual", "accept-edits", "plan", "auto")
THINKING = ("on", "off")
TURN_FIELDS = ("model", "effort", "mode")
MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\[\]-]{0,99}$")
REASON_CAP = 300

CLAUDE_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}
CLAUDE_MODE = {"manual": "default", "accept-edits": "acceptEdits", "plan": "plan", "auto": "auto"}
CLAUDE_MODE_WORD = {v: k for k, v in CLAUDE_MODE.items()}

CODEX_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
CODEX_EFFORT_FALLBACK = "xhigh"
CODEX_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}
_WORKSPACE = {"type": "workspaceWrite", "networkAccess": False}
_READ_ONLY = {"type": "readOnly", "networkAccess": False}
CODEX_MODE = {
    "manual": ("untrusted", _WORKSPACE),
    "accept-edits": ("on-request", _WORKSPACE),
    "plan": ("on-request", _READ_ONLY),
    "auto": ("never", _WORKSPACE),
}
CODEX_START_MODE = "accept-edits"

CLAUDE_THINKING_ON = {"type": "adaptive", "display": "summarized"}
CODEX_SUMMARY = {"on": "auto", "off": "none"}


def codex_mode_params(word: str) -> dict[str, Any]:
    policy, sandbox = CODEX_MODE[word]
    return {"approvalPolicy": policy, "sandboxPolicy": dict(sandbox)}


def codex_mode_value(word: str) -> str:
    policy, sandbox = CODEX_MODE[word]
    box = "read-only" if sandbox["type"] == "readOnly" else "workspace-write"
    return f"approval {policy}, sandbox {box}, network off"


def codex_max_effort(supported: list[str] | None) -> str:
    ranked = [e for e in CODEX_EFFORT_ORDER if supported and e in supported]
    return ranked[-1] if ranked else CODEX_EFFORT_FALLBACK


def one_line(text: Any, cap: int = REASON_CAP) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


def check(field: str, value: Any) -> str | None:
    if field == "model":
        if not isinstance(value, str) or not MODEL.match(value):
            return "not a well-formed model id"
        return None
    words = EFFORTS if field == "effort" else THINKING if field == "thinking" else MODES
    if not isinstance(value, str) or value not in words:
        return f"{field} must be one of {', '.join(words)}"
    return None

