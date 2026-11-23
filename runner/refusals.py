"""Permission requests the runner answers itself, with a refusal the model can act on, instead of an
approval card that pages the owner.

Two rules, coding sessions only:
- Claude Code in auto mode: a shell request that still reaches the approval callback is refused at
  once. Auto mode sends what the sandbox does not auto-allow to its classifier and hands the callback
  only what it falls back to asking about, so no command pattern is needed to decide it.
- Codex, every mode: a command approval that asks to run outside the sandbox is declined at once, and
  the reason is steered back into the same turn. Approving it would run the command without the read
  confinement.

Each refusal leaves one `message` line on the stream (purpose `refused`). It is not an approval, so
nothing waits on the owner. The session's turn, budget and time limits still apply. See
docs/security-model.md, "Refusals".
"""

from __future__ import annotations

import logging
from typing import Any

from runner.events import clip
from runner.providers.base import CODING_KIND, ProviderContext

log = logging.getLogger("runner.refusals")

PURPOSE = "refused"
CLAUDE_SHELL_TOOLS = frozenset({"Bash"})
CLAUDE_AUTO = "auto"
CODEX_COMMAND_APPROVAL = "item/commandExecution/requestApproval"
COMMAND_CAP = 300

CLAUDE_SHELL = (
    "Refused without asking anyone: this command needed an approval, and in Auto mode nobody is asked. "
    "Go on without the command, or say in your summary what you could not do without it.")
CLAUDE_UNSANDBOXED = (
    "Refused without asking anyone: this session never runs a command outside its sandbox. Go on without "
    "the command, or say in your summary what you could not do without it.")
CODEX_ESCALATION = (
    "Declined without asking anyone: this session works inside its sandbox, and a request to run a command "
    "outside it is always declined. Proceed without that command, or say in your summary what you could not "
    "do without it.")

CLAUDE_LINE = ("Refused without asking (Auto mode): a shell command that needed approval. The session was "
               "told to go on without it.")
CLAUDE_UNSANDBOXED_LINE = ("Refused without asking (Auto mode): a shell command asked to run outside the sandbox. "
                           "The session was told to go on without it.")
CODEX_LINE = ("Declined without asking: a request to run a command outside the sandbox, which this session is "
              "never given. The session was told to go on without it.")
CODEX_DECLINED_LINE = ("Declined without asking: Codex sent a request of type {method}, which this runner does "
                       "not put on a card.")
CODEX_QUESTION_LINE = ("Answered without asking: a question from Codex that this session does not put on a card "
                       "(in a research session, asking for a secret, or with no question in it). The session "
                       "was given no answer.")


# Only in auto mode and only for the shell tool: every other tool, and every other mode, keeps its card.
def claude_refuses(ctx: ProviderContext, tool_name: str) -> bool:
    return ctx.kind == CODING_KIND and ctx.permission_mode == CLAUDE_AUTO and tool_name in CLAUDE_SHELL_TOOLS


def claude_answer(tool_input: Any) -> tuple[str, str]:
    if isinstance(tool_input, dict) and tool_input.get("dangerouslyDisableSandbox") is True:
        return CLAUDE_UNSANDBOXED, CLAUDE_UNSANDBOXED_LINE
    return CLAUDE_SHELL, CLAUDE_LINE


# The protocol has no field that says "outside the sandbox". With the runner's permission profile an
# on-request approval carries a reason or additional permissions only when the model asked to escalate,
# so either one marks the request as an escalation (docs/security-model.md, "Refusals").
def codex_escalation(method: str, params: dict[str, Any] | None) -> bool:
    if method != CODEX_COMMAND_APPROVAL or not isinstance(params, dict):
        return False
    if params.get("kind", "command") not in (None, "command") or params.get("networkApprovalContext"):
        return False
    reason = params.get("reason")
    return bool(params.get("additionalPermissions")) or (isinstance(reason, str) and bool(reason.strip()))


def codex_refuses(ctx: ProviderContext, method: str, params: dict[str, Any] | None) -> bool:
    return ctx.kind == CODING_KIND and codex_escalation(method, params)


class RefusalLine(dict):

    def __init__(self, text: str, tool: str, command: Any) -> None:
        super().__init__(text=text, tool=tool)
        self.command = command


def line(text: str, tool: str, command: Any = None, reason: Any = None) -> dict[str, Any]:
    out = RefusalLine(text, tool, command)
    if isinstance(command, str) and command.strip():
        out["command"] = clip(command, COMMAND_CAP)
    if isinstance(reason, str) and reason.strip():
        out["reason"] = clip(reason, COMMAND_CAP)
    return out


async def note(ctx: ProviderContext, payload: dict[str, Any]) -> None:
    if ctx.refusal_note is None:
        return
    try:
        await ctx.refusal_note(payload)
    except Exception as exc:
        log.warning("refusal line not posted: %s", type(exc).__name__)

