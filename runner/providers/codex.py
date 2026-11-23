"""Codex through `codex app-server`, driven by the `openai-codex` 0.167.0 SDK's low-level client.

Not exercised by the offline tests beyond import and construction: a live session needs a Codex login.

Why the low-level client: only `CodexClient(config, approval_handler=...)` takes an approval handler,
and its default accepts every command and file change, while the high-level client offers only deny-all or
a model reviewer. The runner always passes its own handler and starts threads with
`approvalPolicy: "on-request"` and `approvalsReviewer: "user"`. The SDK calls the handler on its one
reader thread and blocks, so the handler hands the request to the event loop and waits for the relay.
The driver denies waiting approvals before it interrupts, so an interrupt is never stuck behind one.

Per-session configuration: a named permission profile that limits reads to the worktree, its git
directory and a private temp folder, network off, analytics off, the built-in apps server off, web
search off unless the session is given the web, and the worktree and its main checkout marked
untrusted so a repository's `.codex/` layers never load. See docs/security-model.md, "Confinement" and
"Repository content is untrusted".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
from typing import Any, AsyncIterator

from runner import mcp_tools, normalise
from runner.providers import confine
from runner import refusals
from runner import settings as words
from runner.events import Event, clip
from runner.providers.base import (CODING_KIND, RESEARCH_KIND, Change, ProviderContext, ToolNames, applied,
                                  refused, web_allowed)

log = logging.getLogger("runner.codex")

QUESTION_FEATURE = "features.default_mode_request_user_input=true"

CONFIG_OVERRIDES = (
    "analytics.enabled=false",
    "sandbox_workspace_write.network_access=false",
    "features.apps=false",
    QUESTION_FEATURE,
    'web_search="disabled"',
)
WEB_SEARCH_OFF = CONFIG_OVERRIDES[-1]
WEB_SEARCH_LIVE = 'web_search="live"'


PROFILE_WRITE = "session_write"
PROFILE_READ = "session_read"
TOOLCHAINS = "/opt/agent-runner"
SECRETS = "/etc/agent-runner/secrets"

TOOLS = ("uv",)
TOOL_FALLBACK = {"uv": "/usr/local/bin/uv"}
MINIMAL_ROOTS = ("/bin", "/sbin", "/usr", "/etc", "/lib", "/lib64")
TOOL_PARENT = "/opt"


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def tool_dirs(path: str | None = None, fallback: dict[str, str] | None = None,
              denied: tuple[str, ...] = ()) -> tuple[str, ...]:
    out: list[str] = []
    for name in TOOLS:
        found = shutil.which(name, path=path) or (fallback if fallback is not None else TOOL_FALLBACK).get(name)
        if not found:
            continue
        real = os.path.realpath(found)
        if not (os.path.isfile(real) and os.access(real, os.X_OK)):
            continue
        folder = os.path.dirname(real)
        if any(_within(folder, root) for root in (*MINIMAL_ROOTS, TOOLCHAINS)):
            continue
        if (not folder.startswith(TOOL_PARENT + "/")
                or any(_within(folder, d) or _within(d, folder) for d in (*denied, SECRETS))):
            log.warning("%s resolves to %s, outside what a Codex session may be given to read; not added", name, real)
            continue
        if folder not in out:
            out.append(folder)
    return tuple(out)


# Named permission profiles rather than a sandbox policy: the legacy workspace-write sandbox has no
# read limit (docs/security-model.md, "Confinement").
def read_profiles(scope: confine.ReadScope, tools: tuple[str, ...] | None = None) -> dict[str, Any]:
    if tools is None:
        tools = tool_dirs(denied=scope.denied)

    def profile(write: bool) -> dict[str, Any]:
        access = "write" if write else "read"
        filesystem: dict[str, str] = {":minimal": "read", TOOLCHAINS: "read", SECRETS: "deny",
                                      scope.worktree: access}
        if scope.git_dir:
            filesystem[scope.git_dir] = "read"
        filesystem[scope.temp] = access
        for folder in tools:
            filesystem.setdefault(folder, "read")
        return {"filesystem": filesystem, "network": {"enabled": False}}
    return {PROFILE_WRITE: profile(True), PROFILE_READ: profile(False)}


def read_config(scope: confine.ReadScope, tools: tuple[str, ...] | None = None) -> dict[str, Any]:
    return {"default_permissions": PROFILE_WRITE, "permissions": read_profiles(scope, tools),
            "shell_environment_policy": {"set": {"TMPDIR": scope.temp}}}


def _toml_key(key: str) -> str:
    return key if key and all(c.isalnum() or c in "_-" for c in key) and key.isascii() else json.dumps(key, ensure_ascii=False)


def _toml_table(value: Any) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_table(v)}" for k, v in value.items()) + " }"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return _toml(value)


def read_overrides(scope: confine.ReadScope, tools: tuple[str, ...] | None = None) -> tuple[str, ...]:
    return tuple(f"{key}={_toml_table(value)}" for key, value in read_config(scope, tools).items())


def mode_params(word: str) -> dict[str, Any]:
    policy, sandbox = words.CODEX_MODE[word]
    return {"approvalPolicy": policy, "permissions": PROFILE_READ if sandbox["type"] == "readOnly" else PROFILE_WRITE}


RESEARCH_CODEX_PIN = "0.167.0"
RESEARCH_CODEX_PACKAGES = ("openai-codex", "openai-codex-cli-bin")
RESEARCH_LEAD_ROLES = ("lead_brief", "lead_report", "challenger")
CODEX_NOT_CHALLENGER = ("Codex cannot be the challenger of a research run: with its shell off it has no tool to read"
                        " the syntheses it challenges, so the runner refuses it.")
CODEX_NOT_LEAD = ("Codex cannot be the lead of a research run: with its shell off it has no tool to read the"
                  " input files the lead works from, so the runner refuses it. Next: keep the lead on Claude"
                  " and start the run again.")
CODEX_NOT_PINNED = ("Codex research is held to Codex {pin}, the version it was checked against, and this machine"
                    " has {found}, so the runner refuses it. Next: check research sessions against the installed"
                    " version; if they behave as before, set RESEARCH_CODEX_PIN in runner/providers/codex.py to it."
                    " Until then the research run goes on with the providers it has.")
CODEX_MCP_SEEN = ("The Codex research session found MCP tools offered to it ({names}), and a research session"
                  " must have none, so it was stopped before its first turn. Next: check this user's"
                  " ~/.codex/config.toml (see examples/codex-config.toml) for a server switched on.")
CODEX_MCP_LISTED = ("The Codex research session has MCP servers registered ({names}), and a research session must"
                    " have none, so it was stopped before its first turn. Next: remove the [mcp_servers] table"
                    " from this user's ~/.codex/config.toml (see examples/codex-config.toml); the runner gives"
                    " coding sessions theirs.")
CODEX_MCP_UNREAD = ("The Codex research session could not read which MCP servers it has, so it cannot confirm"
                    " that none offers it a tool, and it was stopped before its first turn. Next: check that"
                    " `codex app-server` starts for this user and answers mcpServerStatus/list.")


NO_ROLLOUT = "no rollout found for thread id"


def resume_refused(exc: BaseException) -> bool:
    return NO_ROLLOUT in str(exc)


def installed_codex() -> dict[str, str | None]:
    from importlib import metadata

    found: dict[str, str | None] = {}
    for name in RESEARCH_CODEX_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def research_refusal(role: str | None = None, versions: dict[str, str | None] | None = None) -> str | None:
    if role in RESEARCH_LEAD_ROLES:
        return CODEX_NOT_CHALLENGER if role == "challenger" else CODEX_NOT_LEAD
    found = installed_codex() if versions is None else versions
    wrong = [f"{name} {version or 'not installed'}" for name, version in found.items() if version != RESEARCH_CODEX_PIN]
    if wrong:
        return CODEX_NOT_PINNED.format(pin=RESEARCH_CODEX_PIN, found=", ".join(wrong))
    return None


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value))


def research_settings(max_subagents: int | None) -> dict[str, Any]:
    return {"analytics.enabled": False, "web_search": "live", "sandbox_mode": "workspace-write",
            "sandbox_workspace_write.network_access": False, "approval_policy": "never",
            "features.apps": False, "features.shell_tool": False, "features.unified_exec": False,
            "features.view_image": False, "agents.max_threads": max(1, max_subagents or 0)}


def research_overrides(max_subagents: int | None) -> tuple[str, ...]:
    return tuple(f"{key}={_toml(value)}" for key, value in research_settings(max_subagents).items())


def session_overrides(ctx: ProviderContext) -> tuple[str, ...]:
    if not web_allowed(ctx):
        return CONFIG_OVERRIDES
    return tuple(WEB_SEARCH_LIVE if o == WEB_SEARCH_OFF else o for o in CONFIG_OVERRIDES)


MCP_WRAPPER = "/opt/agent-runner/.venv/bin/runner-mcp"
# The one MCP server, defined per session and off unless the session has tools. It is not in the
# user's config.toml because an override cannot remove a table that file defines, and a research
# session must have no MCP server at all.
MCP_TABLE: dict[str, Any] = {
    "command": MCP_WRAPPER, "args": [], "env_vars": [mcp_tools.SOCKET_ENV], "enabled": False, "required": False,
    "startup_timeout_sec": 10, "tool_timeout_sec": 60, "enabled_tools": list(mcp_tools.offered(False)),
    "default_tools_approval_mode": "approve"}


def _toml_inline(value: Any) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml_inline(v)}" for key, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_inline(v) for v in value) + "]"
    return _toml(value)


MCP_SERVER = f"mcp_servers.{mcp_tools.SERVER_NAME}={_toml_inline(MCP_TABLE)}"
MCP_ENABLE = f"mcp_servers.{mcp_tools.SERVER_NAME}.enabled=true"


def enabled_tools(names: tuple[str, ...]) -> str:
    return f"mcp_servers.{mcp_tools.SERVER_NAME}.enabled_tools={_toml_inline(list(names))}"


MCP_PROPOSE = enabled_tools(mcp_tools.offered(True))
MCP_LEAD = enabled_tools(mcp_tools.offered(False, True))
MCP_ELICITATION = "mcpServer/elicitation/request"
UNTRUSTED = "untrusted"


# Codex trusts a project on its own when it has no trust level and the thread may write to its cwd,
# and the project it trusts is the worktree's main checkout. Marking both untrusted on every start turns
# that off, so a repository's `.codex/` config, hooks and MCP servers never load.
def git_root_for_trust(cwd: str) -> str | None:
    here = os.path.abspath(cwd)
    while True:
        dot = os.path.join(here, ".git")
        if os.path.isdir(dot):
            return here
        if os.path.isfile(dot):
            try:
                with open(dot, encoding="utf-8") as handle:
                    line = handle.read(4096).strip()
            except (OSError, UnicodeDecodeError):
                return here
            if not line.startswith("gitdir:"):
                return here
            gitdir = os.path.join(here, line.removeprefix("gitdir:").strip())
            common = gitdir
            try:
                with open(os.path.join(gitdir, "commondir"), encoding="utf-8") as handle:
                    common = os.path.join(gitdir, handle.read(4096).strip())
            except (OSError, UnicodeDecodeError):
                pass
            common = os.path.realpath(common)
            return os.path.dirname(common) if os.path.basename(common) == ".git" else here
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def untrusted_paths(cwd: str) -> list[str]:
    out: list[str] = []
    root = git_root_for_trust(cwd)
    for path in (cwd, root):
        if not path:
            continue
        for form in (os.path.abspath(path), os.path.realpath(path)):
            if form not in out:
                out.append(form)
    return out


def projects_untrusted(cwd: str) -> dict[str, dict[str, str]]:
    return {path: {"trust_level": UNTRUSTED} for path in untrusted_paths(cwd)}


# An inline table, so the dots in a path are never read as TOML key separators.
def projects_override(cwd: str) -> str:
    entries = ", ".join(f'{json.dumps(path, ensure_ascii=False)} ={{ trust_level = "{UNTRUSTED}" }}' for path in untrusted_paths(cwd))
    return f"projects={{ {entries} }}"


def to_raw(notification: Any) -> dict[str, Any]:
    payload = notification.payload
    if hasattr(payload, "model_dump"):
        params = payload.model_dump(mode="json", by_alias=True, exclude_none=True)
    else:
        params = getattr(payload, "params", {}) or {}
    return {"method": notification.method, "params": params}


class CodexProvider:
    name = "codex"
    enforces_max_turns = False
    schema_per_turn = True

    def __init__(self, codex_bin: str | None = None, temp_parent: str | None = None) -> None:
        self.codex_bin = codex_bin
        self.temp_parent = temp_parent
        self.temp: confine.SessionTemp | None = None
        self.ctx: ProviderContext | None = None
        self.client: Any = None
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._global: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_feedback: list[str] = []
        self._changes: dict[str, Any] = {}
        self._changes_seen = threading.Condition()
        self._last_limits: Any = None
        self.tool_names = ToolNames()
        self.context = normalise.ContextMeter(self.name)
        self.thinking = False
        self.overrides: dict[str, Any] = {}
        self.effort: str | None = None
        self.start_refused: dict[str, str] = {}
        self.mode = words.CODEX_START_MODE

    def normalise(self, raw: dict[str, Any]) -> list[Event]:
        self.context.see(raw)
        return self.context.annotate(self.tool_names.annotate(normalise.codex(raw, ids=bool(self.ctx and self.ctx.drafts))))

    def conversation_id(self) -> str | None:
        return self.thread_id

    def resume_refused(self, exc: BaseException) -> bool:
        return resume_refused(exc)

    def settings_now(self) -> dict[str, str | None]:
        return {"model": self.ctx.model if self.ctx else None, "effort": self.effort, "mode": self.mode,
                "thinking": "on" if self.thinking else "off"}

    async def _models(self) -> list[Any] | None:
        try:
            response = await asyncio.to_thread(self.client.model_list, True)
        except Exception as exc:
            log.info("model/list failed (%s)", type(exc).__name__)
            return None
        return list(getattr(response, "data", None) or [])

    async def _supported_efforts(self, model: str | None) -> list[str] | None:
        for entry in await self._models() or []:
            if (model and model in (getattr(entry, "model", None), getattr(entry, "id", None))) \
                    or (not model and getattr(entry, "is_default", False)):
                found = []
                for option in getattr(entry, "supported_reasoning_efforts", None) or []:
                    effort = getattr(option, "reasoning_effort", None)
                    found.append(str(getattr(effort, "value", effort)))
                return found
        return None

    async def change_settings(self, changes: dict[str, str]) -> dict[str, Change]:
        assert self.ctx is not None
        out: dict[str, Change] = {}
        if "model" in changes:
            model = changes["model"]
            listed = await self._models()
            if listed is not None and not any(model in (getattr(e, "model", None), getattr(e, "id", None))
                                              for e in listed):
                out["model"] = refused("Codex does not offer this model here (model/list does not name it)")
            else:
                self.ctx.model = model
                self.overrides["model"] = model
                out["model"] = applied(model)
        if "effort" in changes:
            word = changes["effort"]
            effort = words.CODEX_EFFORT.get(word) or words.codex_max_effort(await self._supported_efforts(self.ctx.model))
            self.effort = effort
            self.overrides["effort"] = effort
            out["effort"] = applied(effort)
        if "mode" in changes:
            word = changes["mode"]
            self.mode = word
            self.overrides.update(mode_params(word) if self.ctx.kind == CODING_KIND else words.codex_mode_params(word))
            out["mode"] = applied(words.codex_mode_value(word))
        if "thinking" in changes:
            on = changes["thinking"] == "on"
            if on != self.thinking:
                self.thinking = on
                self.overrides["summary"] = words.CODEX_SUMMARY[changes["thinking"]]
            out["thinking"] = applied(f"reasoning summary {words.CODEX_SUMMARY[changes['thinking']]}")
        return out

    def read_scope(self, ctx: ProviderContext) -> confine.ReadScope:
        if self.temp is None:
            self.temp = confine.SessionTemp(self.temp_parent)
        return confine.read_scope(ctx.cwd, self.temp.path)

    def config(self, ctx: ProviderContext) -> Any:
        from openai_codex import CodexConfig

        if ctx.kind == RESEARCH_KIND:
            return CodexConfig(codex_bin=self.codex_bin, cwd=ctx.cwd, env=None,
                               config_overrides=(*research_overrides(ctx.max_subagents), projects_override(ctx.cwd)))
        scope = self.read_scope(ctx)
        overrides = (*session_overrides(ctx), projects_override(ctx.cwd), MCP_SERVER, *read_overrides(scope))
        env = {"TMPDIR": scope.temp}
        if ctx.mcp_socket:
            overrides = (*overrides, MCP_ENABLE)
            if ctx.lead is True:
                overrides = (*overrides, MCP_LEAD)
            elif ctx.propose is True:
                overrides = (*overrides, MCP_PROPOSE)
            env[mcp_tools.SOCKET_ENV] = ctx.mcp_socket
        return CodexConfig(codex_bin=self.codex_bin, cwd=ctx.cwd, config_overrides=overrides, env=env)

    def thread_params(self, ctx: ProviderContext) -> dict[str, Any]:
        params: dict[str, Any] = {"cwd": ctx.cwd, "approvalPolicy": "on-request", "approvalsReviewer": "user",
                                  "sandbox": "workspace-write", "config": {"projects": projects_untrusted(ctx.cwd)}}
        if ctx.kind == RESEARCH_KIND:
            params["approvalPolicy"] = "never"
            params["config"] = {**research_settings(ctx.max_subagents), **params["config"]}
        else:
            del params["sandbox"]
            params["config"] = {**read_config(self.read_scope(ctx)), **params["config"]}
            if ctx.baseline:
                params["developerInstructions"] = ctx.baseline
        if ctx.model:
            params["model"] = ctx.model
        return params

    async def _research_mcp_check(self) -> None:
        try:
            servers = await self.mcp_status()
        except Exception as exc:
            log.error("research: mcpServerStatus/list failed (%s)", type(exc).__name__)
            raise ValueError(CODEX_MCP_UNREAD) from None
        offered = sorted(name for name, tools in servers.items() if tools)
        if offered:
            raise ValueError(CODEX_MCP_SEEN.format(names=", ".join(offered)[:200]))
        if servers:
            raise ValueError(CODEX_MCP_LISTED.format(names=", ".join(sorted(servers))[:200]))

    async def start(self, ctx: ProviderContext) -> None:
        if ctx.kind == RESEARCH_KIND:
            refusal = research_refusal()
            if refusal:
                raise ValueError(refusal)
        from openai_codex.client import CodexClient

        self.ctx = ctx
        self.loop = asyncio.get_running_loop()
        self.client = CodexClient(config=self.config(ctx), approval_handler=self._on_server_request)
        await asyncio.to_thread(self.client.start)
        await asyncio.to_thread(self.client.initialize)
        if ctx.resume:
            response = await asyncio.to_thread(self.client.thread_resume, ctx.resume, self.thread_params(ctx))
        else:
            response = await asyncio.to_thread(self.client.thread_start, self.thread_params(ctx))
        self.thread_id = response.thread.id
        if ctx.kind == RESEARCH_KIND:
            await self._research_mcp_check()
        threading.Thread(target=self._read_global, daemon=True).start()
        await self.start_effort()
        self.start_thinking()

    def start_thinking(self) -> None:
        if self.ctx is not None and self.ctx.thinking:
            self.thinking = True
            self.overrides["summary"] = words.CODEX_SUMMARY["on"]

    async def start_effort(self) -> None:
        assert self.ctx is not None
        word = self.ctx.effort
        if not word:
            return
        supported = await self._supported_efforts(self.ctx.model)
        effort = words.CODEX_EFFORT.get(word) or words.codex_max_effort(supported)
        if word in words.CODEX_EFFORT and supported and effort not in supported:
            self.start_refused["effort"] = (f"the model does not support {word}; "
                                            "the session runs on the default effort")
            return
        self.effort = effort
        self.overrides["effort"] = effort

    def _read_global(self) -> None:
        while self.client is not None and self.loop is not None:
            try:
                notification = self.client.next_notification()
            except BaseException:
                return
            raw = to_raw(notification)
            if self.ctx is not None and self.ctx.raw_frame is not None:
                self.loop.call_soon_threadsafe(self.ctx.raw_frame, raw)
            self.loop.call_soon_threadsafe(self._global.put_nowait, raw)

    def seen(self, raw: dict[str, Any]) -> None:
        item = (raw.get("params") or {}).get("item") or {}
        if raw.get("method") == "item/completed" and item.get("type") == "fileChange":
            with self._changes_seen:
                self._changes.pop(item.get("id"), None)
            return
        if raw.get("method") != "item/started":
            return
        if item.get("type") == "fileChange" and item.get("id"):
            with self._changes_seen:
                self._changes[item["id"]] = item.get("changes")
                self._changes_seen.notify_all()

    def repeated(self, raw: dict[str, Any]) -> bool:
        if raw.get("method") != "account/rateLimits/updated":
            return False
        same = raw.get("params") == self._last_limits
        self._last_limits = raw.get("params")
        return same

    def changes_for(self, item_id: Any, wait: float = 2.0) -> Any:
        if not item_id:
            return None
        with self._changes_seen:
            self._changes_seen.wait_for(lambda: item_id in self._changes, timeout=wait)
            return self._changes.pop(item_id, None)

    def _line(self, text: str, tool: str) -> None:
        if self.ctx is not None and self.loop is not None:
            asyncio.run_coroutine_threadsafe(refusals.note(self.ctx, refusals.line(text, tool)), self.loop)

    def _question(self, params: dict[str, Any]) -> dict[str, Any]:
        none: dict[str, Any] = {"answers": {}}
        if self.ctx is None or self.loop is None:
            return none
        questions = normalise.codex_questions(params)
        if self.ctx.kind != CODING_KIND or not questions or any(q.get("isSecret") for q in questions):
            log.info("answered a question with no answers (item %s)", params.get("itemId"))
            self._line(refusals.CODEX_QUESTION_LINE, normalise.CODEX_QUESTION_TOOL)
            return none
        future = asyncio.run_coroutine_threadsafe(self.ctx.request_approval(normalise.codex_question(params)),
                                                  self.loop)
        decision = future.result()
        reply = None if decision.allow else normalise.owner_reply(decision.message)
        if reply is None:
            if decision.message:
                self.loop.call_soon_threadsafe(self._pending_feedback.append, decision.message)
            return none
        return {"answers": normalise.codex_answers(reply, questions)}

    # Called on the SDK's reader thread, which blocks until this returns. An escalation is declined at once
    # and its reason steered into the turn. Any other approval waits for the relay.
    def _on_server_request(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        if self.ctx is not None and self.loop is not None and refusals.codex_refuses(self.ctx, method, params):
            log.info("declined a request to run a command outside the sandbox (item %s)", params.get("itemId"))
            asyncio.run_coroutine_threadsafe(refusals.note(self.ctx, refusals.line(
                refusals.CODEX_LINE, "command", params.get("command"), params.get("reason"))), self.loop)
            self.loop.call_soon_threadsafe(self._pending_feedback.append, refusals.CODEX_ESCALATION)
            return {"decision": "decline"}
        if method in normalise.APPROVAL_METHODS and self.ctx is not None and self.loop is not None:
            changes = self.changes_for(params.get("itemId")) if method == normalise.APPROVAL_METHODS[1] else None
            payload = normalise.codex_approval(method, params, changes)
            future = asyncio.run_coroutine_threadsafe(self.ctx.request_approval(payload), self.loop)
            decision = future.result()
            if decision.allow:
                return {"decision": "accept"}
            if decision.message:
                self.loop.call_soon_threadsafe(self._pending_feedback.append, decision.message)
            return {"decision": "decline"}
        if method == normalise.CODEX_QUESTION_METHOD:
            return self._question(params)
        log.info("declined server request %s", method)
        self._line(refusals.CODEX_DECLINED_LINE.format(method=clip(method, 100)), clip(method, 100))
        if method == "item/permissions/requestApproval":
            return {"permissions": {}, "scope": "turn"}
        if method == MCP_ELICITATION:
            return {"action": "decline"}
        return {}

    async def mcp_status(self) -> dict[str, list[str]]:
        from openai_codex.generated.v2_all import ListMcpServerStatusResponse

        response = await asyncio.to_thread(self.client.request, "mcpServerStatus/list", {"threadId": self.thread_id},
                                           response_model=ListMcpServerStatusResponse)
        return {str(s.name): sorted(str(name) for name in (s.tools or {})) for s in response.data}

    async def turn(self, text: str) -> AsyncIterator[dict[str, Any]]:
        assert self.ctx is not None and self.thread_id is not None
        params = {"outputSchema": self.ctx.output_schema} if self.ctx.output_schema else None
        if self.overrides:
            params = {**(params or {}), **self.overrides}
        started = await asyncio.to_thread(self.client.turn_start, self.thread_id, text, params)
        self.overrides = {}
        self.turn_id = started.turn.id
        try:
            while True:
                notification = await asyncio.to_thread(self.client.next_turn_notification, self.turn_id)
                raw = to_raw(notification)
                if self.ctx.raw_frame is not None:
                    self.ctx.raw_frame(raw)
                self.seen(raw)
                yield raw
                while not self._global.empty():
                    extra = self._global.get_nowait()
                    if not self.repeated(extra):
                        yield extra
                while self._pending_feedback:
                    await self.steer(self._pending_feedback.pop(0))
                if raw["method"] == "turn/completed":
                    return
        finally:
            self.turn_id = None

    async def steer(self, text: str) -> bool:
        if self.turn_id is None or self.thread_id is None:
            return False
        try:
            await asyncio.to_thread(self.client.turn_steer, self.thread_id, self.turn_id, text)
        except Exception:
            return False
        return True

    async def interrupt(self) -> None:
        if self.turn_id is not None and self.thread_id is not None:
            await asyncio.to_thread(self.client.turn_interrupt, self.thread_id, self.turn_id)

    async def set_permission_mode(self, mode: str) -> None:
        raise NotImplementedError("Codex permission mode changes per turn, not mid-turn")

    async def terminate(self) -> None:
        await self.close()

    async def close(self) -> None:
        try:
            if self.client is not None:
                client, self.client = self.client, None
                await asyncio.to_thread(client.close)
        finally:
            if self.temp is not None:
                temp, self.temp = self.temp, None
                temp.close()

