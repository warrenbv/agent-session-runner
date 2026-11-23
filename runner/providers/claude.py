"""Claude Code through the Claude Agent SDK (`claude-agent-sdk` 0.2.180, which bundles the CLI).

Not exercised by the offline tests beyond import and construction: a live session needs a Claude login.

- `ClaudeSDKClient` in streaming mode: `connect()`, then per turn `query(text)` and
  `receive_response()` up to the ResultMessage. Interrupt is `interrupt()`, after which the turn's
  iterator keeps reading to the ResultMessage, the drain the SDK requires before the next query.
- Options: never `bypassPermissions`. Narrow `allowed_tools` and scoped `disallowed_tools`. One MCP
  server with `strict_mcp_config`. `setting_sources=["user"]`, so managed and user settings load but a
  repository's own do not. Read confinement through `settings`, `add_dirs` and temp variables.
- Every start and reconnect first checks the managed settings file (claude_policy.py) and refuses to
  launch the CLI when it does not require the sandbox. A coding session's CLI starts every Bash call in
  the worktree (PROJECT_CWD_ENV), which the test verdict relies on.
  `can_use_tool` for the approval relay, the typed result as `output_format`, and full model ids only.
- The SDK fixes effort, thinking and the output schema when the client connects, so changing them
  reconnects the client and resumes the same conversation before the next turn. Model and permission
  mode change in place.
- Research sessions get a separate, smaller tool set with reads and edits held to their scratch
  directory.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import re
import uuid
from typing import Any, AsyncIterator

from runner import mcp_tools, normalise
from runner.providers import confine
from runner import claude_policy, refusals
from runner import settings as words
from runner.events import Event
from runner.providers.base import (RESEARCH_KIND, Change, ProviderContext, ToolNames, applied, refused,
                                  web_allowed)

MODES = ("default", "acceptEdits", "plan", "dontAsk", "auto")
AUTO_MODE_SETTINGS_MISSING = (
    "auto mode is not available on this machine: its managed settings are missing, so install them at "
    "{path} (examples/managed-settings.json shows the required content)")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FABLE_MODEL = "claude-fable-5-2"
OPUS_MODEL = "claude-opus-5-5"
DEFAULT_MODEL = OPUS_MODEL
FULL_ID = re.compile(r"^claude-[a-z]+(?:-[0-9]+)+(?:-[0-9]{8})?(?:\[1m\])?$")
FABLE_ALIAS_ENV = "ANTHROPIC_DEFAULT_FABLE_MODEL"
# With this set, the bundled CLI (2.1.321) puts the shell back in the session's starting directory after
# each main-thread Bash call, sandboxed calls included. A `cd` inside one command still applies within it.
PROJECT_CWD_ENV = "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"
SUBAGENT_MODEL_ENV = "CLAUDE_CODE_SUBAGENT_MODEL"
FABLE_SUBAGENT_MODEL = OPUS_MODEL
PICK_ANOTHER = ("Pick a model from the start form's Model list: Fable 5.2, another model listed there, or "
                "Default.")


WEB_TOOLS = ("WebSearch", "WebFetch")


def allowed_web_tools(ctx: ProviderContext) -> list[str]:
    return [t for t in WEB_TOOLS if t not in ctx.disallowed_tools] if web_allowed(ctx) else []


def refused_web_tools(ctx: ProviderContext) -> list[str]:
    return [] if ctx.kind == RESEARCH_KIND or web_allowed(ctx) else list(WEB_TOOLS)


BLOCK_READS = "blockReadsOutsideWorkingDirectories"
CLAUDE_TMPDIR_ENV = "CLAUDE_CODE_TMPDIR"
CLAUDE_TMPDIR_MAX = 44


# The CLI keeps these launch settings and the managed ones even with setting_sources=["user"], and merges
# the managed ones last, so managed values win. The sandbox flags here only supply defaults. The start
# check (claude_policy.py) requires the managed file to carry them too.
def read_settings(scope: confine.ReadScope) -> dict[str, Any]:
    return {"permissions": {BLOCK_READS: True},
            "sandbox": {**claude_policy.SANDBOX_FLAGS,
                        "filesystem": {"denyRead": list(confine.SHARED_TEMP), "allowRead": [scope.temp]}},
            GIT_INSTRUCTIONS: False}


GIT_INSTRUCTIONS = "includeGitInstructions"
AUTO_MEMORY_ENV = "CLAUDE_CODE_DISABLE_AUTO_MEMORY"


def system_prompt(ctx: ProviderContext) -> dict[str, Any] | None:
    if ctx.kind == RESEARCH_KIND:
        return None
    preset: dict[str, Any] = {"type": "preset", "preset": "claude_code"}
    if ctx.baseline:
        preset["append"] = ctx.baseline
    return preset


def session_temp_env(temp: str) -> dict[str, str]:
    if len(f"{temp}/claude-{os.getuid()}".encode()) > CLAUDE_TMPDIR_MAX:
        raise ValueError("the session's temp folder's path is too long for Claude Code's sandbox")
    return {CLAUDE_TMPDIR_ENV: temp, "TMPDIR": temp}


RESEARCH_TOOLS = ("Read", "Write", "Edit", "WebSearch", "WebFetch", "Task")
SUBAGENT_TOOLS = ("Task", "Agent")
SCRUB_ENV = "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"
SEARCH_CAP_ENV = "CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION"
BACKGROUND_OFF_ENV = "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"
SAFE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
RESEARCH_REFUSED = "A research session may use only its own tools; this was refused."


def research_rules(cwd: str) -> list[str]:
    scratch = os.path.realpath(cwd)
    if not SAFE_PATH.match(scratch) or scratch == "/":
        raise ValueError("the scratch directory's path cannot ride in a permission rule")
    under = f"/{scratch}/**"
    return [f"Read({under})", f"Edit({under})", *WEB_TOOLS]


def research_env(ctx: ProviderContext) -> dict[str, str]:
    env = {SCRUB_ENV: "1", BACKGROUND_OFF_ENV: "1"}
    if isinstance(ctx.max_searches, int) and not isinstance(ctx.max_searches, bool) and ctx.max_searches >= 1:
        env[SEARCH_CAP_ENV] = str(ctx.max_searches)
    return env


def research_tools(ctx: ProviderContext) -> tuple[list[str], list[str]]:
    if ctx.max_subagents == 0:
        return [t for t in RESEARCH_TOOLS if t not in SUBAGENT_TOOLS], list(SUBAGENT_TOOLS)
    return list(RESEARCH_TOOLS), []


def is_fable(model: str | None) -> bool:
    return isinstance(model, str) and (model.startswith("fable") or model.startswith("claude-fable-"))


# Only full model ids: an alias can resolve to a different model in a different CLI version.
def model_refusal(model: Any) -> str | None:
    if model is None:
        return None
    if not isinstance(model, str) or not FULL_ID.match(model):
        return (f"The model {str(model)[:80]!r} is not a full Claude model id (an alias such as `fable` or "
                f"`opus` can mean an older model), so the session was not started. {PICK_ANOTHER}")
    if is_fable(model) and model.removesuffix("[1m]") != FABLE_MODEL:
        return (f"The model {model!r} is not Fable 5.2, and only Fable 5.2 may run, so the session was not "
                f"started. {PICK_ANOTHER}")
    return None


def session_model(model: str | None) -> str:
    reason = model_refusal(model)
    if reason is not None:
        raise ValueError(reason)
    return model or DEFAULT_MODEL


def auto_mode_refusal(path: str | None = None) -> str | None:
    documented = claude_policy.managed_settings_path()
    if documented is None:
        return "auto mode is not available on this machine: the runner checks managed settings only on Linux and macOS"
    try:
        with open(path or claude_policy.MANAGED_SETTINGS or documented, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return AUTO_MODE_SETTINGS_MISSING.format(path=documented)
    except (OSError, ValueError):
        return "auto mode is not available on this machine: its managed settings could not be read"
    perms = data.get("permissions") if isinstance(data, dict) else None
    if (isinstance(perms, dict) and perms.get("disableAutoMode") == "disable") \
            or (isinstance(data, dict) and data.get("disableAutoMode") == "disable"):
        return "auto mode is not available on this machine: its managed settings turn it off (disableAutoMode)"
    return None


def change_refusal(model: str) -> str | None:
    if not FULL_ID.match(model):
        return "not a full Claude model id (an alias such as `fable` or `opus` can mean an older model)"
    if is_fable(model) and model.removesuffix("[1m]") != FABLE_MODEL:
        return "not Fable 5.2, and only Fable 5.2 may run"
    return None


def to_raw(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = to_raw(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {str(k): to_raw(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_raw(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


# A resume id is passed to the CLI as an argument, so only a UUID is accepted: anything else could be
# read as another flag.
def check_resume(value: str | None) -> str | None:
    if value is None:
        return None
    return str(uuid.UUID(value))


NO_CONVERSATION = "No conversation found with session ID"


def resume_refused(exc: BaseException) -> bool:
    errors = getattr(exc, "errors", None)
    said = [str(e) for e in errors] if isinstance(errors, list) else []
    return any(NO_CONVERSATION in text for text in [*said, str(exc)])


# A streamed user message with `priority: "next"` is delivered at the model's next step instead of
# after the turn. It is documented for the TypeScript SDK and was checked live from Python.
def steer_message(text: str, session_id: str = "default") -> dict[str, Any]:
    return {"type": "user", "message": {"role": "user", "content": text}, "parent_tool_use_id": None,
            "session_id": session_id, "priority": "next"}


def mcp_options(ctx: ProviderContext) -> dict[str, Any]:
    if not ctx.mcp_socket:
        return {}
    return {"mcp_servers": {mcp_tools.SERVER_NAME: {"type": "stdio", "command": mcp_tools.wrapper_path(), "args": [],
                                                    "env": {mcp_tools.SOCKET_ENV: ctx.mcp_socket}}},
            "strict_mcp_config": True}


def mcp_allowed(ctx: ProviderContext) -> list[str]:
    return mcp_tools.claude_names(ctx.propose is True, ctx.lead is True) if ctx.mcp_socket else []


def mcp_refused(ctx: ProviderContext) -> list[str]:
    if not ctx.mcp_socket:
        return []
    return [mcp_tools.claude_name(n) for n in mcp_tools.withheld(ctx.propose is True, ctx.lead is True)]


async def send_steer(client: Any, text: str) -> None:
    async def one() -> AsyncIterator[dict[str, Any]]:
        yield steer_message(text)
    await client.query(one())


class ClaudeProvider:
    name = "claude-code"
    enforces_max_turns = True
    schema_per_turn = True
    cost_carries_over = True

    def __init__(self, cli_path: str | None = None, oauth_token: str = "",
                 temp_parent: str | None = None) -> None:
        self.cli_path = cli_path
        self.temp_parent = temp_parent
        self.temp: confine.SessionTemp | None = None
        self._oauth_token = oauth_token
        self.ctx: ProviderContext | None = None
        self.client: Any = None
        self.bash_starts_in_worktree = False
        self.tool_names = ToolNames()
        self.context = normalise.ContextMeter(self.name)
        self.session_id: str | None = None
        self._schema: dict[str, Any] | None = None
        self.reconnects = 0
        self._stale = False
        self.managed_settings: str | None = None
        self.subagents_allowed = 0

    def normalise(self, raw: dict[str, Any]) -> list[Event]:
        self.context.see(raw)
        return self.context.annotate(self.tool_names.annotate(normalise.claude(raw, ids=bool(self.ctx and self.ctx.drafts))))

    def conversation_id(self) -> str | None:
        return self.session_id

    def resume_refused(self, exc: BaseException) -> bool:
        return resume_refused(exc)

    def settings_now(self) -> dict[str, str | None]:
        assert self.ctx is not None
        return {"model": self.ctx.model or DEFAULT_MODEL, "effort": self.ctx.effort,
                "mode": words.CLAUDE_MODE_WORD.get(self.ctx.permission_mode, self.ctx.permission_mode),
                "thinking": "on" if self.ctx.thinking else "off"}

    async def change_settings(self, changes: dict[str, str]) -> dict[str, Change]:
        assert self.ctx is not None
        out: dict[str, Change] = {}
        if "model" in changes:
            model = changes["model"]
            reason = change_refusal(model)
            if reason is not None:
                out["model"] = refused(reason)
            else:
                before = self.ctx.model or DEFAULT_MODEL
                try:
                    if is_fable(model) != is_fable(before):
                        self._stale = True
                    else:
                        await self.client.set_model(model)
                except Exception as exc:
                    out["model"] = refused(f"Claude Code refused it: {words.one_line(exc)}")
                else:
                    self.ctx.model = model
                    out["model"] = applied(model)
        if "effort" in changes:
            effort = words.CLAUDE_EFFORT[changes["effort"]]
            if effort != self.ctx.effort:
                self.ctx.effort = effort
                self._stale = True
            out["effort"] = applied(effort)
        if "thinking" in changes:
            on = changes["thinking"] == "on"
            if on != self.ctx.thinking:
                self.ctx.thinking = on
                self._stale = True
            out["thinking"] = applied("display summarized" if on else "display omitted (the default)")
        if "mode" in changes:
            mode = words.CLAUDE_MODE[changes["mode"]]
            reason = auto_mode_refusal(self.managed_settings) if mode == "auto" else None
            if reason is not None:
                out["mode"] = refused(reason)
            else:
                try:
                    await self.set_permission_mode(mode)
                except Exception as exc:
                    why = words.one_line(exc)
                    out["mode"] = refused(f"{changes['mode']} mode is not available on this machine: Claude Code "
                                          f"refused it ({why})" if mode == "auto" else f"Claude Code refused it: {why}")
                else:
                    out["mode"] = applied(mode)
        return out

    def read_scope(self, ctx: ProviderContext) -> confine.ReadScope:
        if self.temp is None:
            self.temp = confine.SessionTemp(self.temp_parent)
        return confine.read_scope(ctx.cwd, self.temp.path)

    def options(self, ctx: ProviderContext, resume: str | None = None) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        if ctx.permission_mode not in MODES:
            raise ValueError(f"permission mode must be one of {', '.join(MODES)}")
        if ctx.effort is not None and ctx.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        model = session_model(ctx.model)
        env = {"CLAUDE_CODE_OAUTH_TOKEN": self._oauth_token} if self._oauth_token else {}
        env[FABLE_ALIAS_ENV] = FABLE_MODEL
        if is_fable(model):
            env[SUBAGENT_MODEL_ENV] = FABLE_SUBAGENT_MODEL
        if ctx.kind == RESEARCH_KIND:
            tools, denied = research_tools(ctx)
            return ClaudeAgentOptions(
                cwd=ctx.cwd,
                permission_mode="default",
                tools=tools,
                allowed_tools=research_rules(ctx.cwd),
                disallowed_tools=denied,
                mcp_servers={},
                strict_mcp_config=True,
                max_turns=ctx.max_turns,
                max_budget_usd=ctx.max_budget_usd,
                model=model,
                effort=ctx.effort,
                resume=check_resume(resume or ctx.resume),
                output_format={"type": "json_schema", "schema": ctx.output_schema} if ctx.output_schema else None,
                setting_sources=["user"],
                can_use_tool=self._can_use_tool,
                cli_path=self.cli_path,
                env={**env, **research_env(ctx)},
            )
        scope = self.read_scope(ctx)
        env.update(session_temp_env(scope.temp))
        env[AUTO_MEMORY_ENV] = "1"
        env[PROJECT_CWD_ENV] = "1"
        return ClaudeAgentOptions(
            cwd=ctx.cwd,
            system_prompt=system_prompt(ctx),
            settings=json.dumps(read_settings(scope)),
            add_dirs=[scope.temp],
            permission_mode=ctx.permission_mode,
            allowed_tools=list(ctx.allowed_tools) + mcp_allowed(ctx)
            + [t for t in allowed_web_tools(ctx) if t not in ctx.allowed_tools],
            **mcp_options(ctx),
            disallowed_tools=list(ctx.disallowed_tools)
            + [t for t in refused_web_tools(ctx) if t not in ctx.disallowed_tools]
            + [t for t in mcp_refused(ctx) if t not in ctx.disallowed_tools],
            max_turns=ctx.max_turns,
            max_budget_usd=ctx.max_budget_usd,
            model=model,
            effort=ctx.effort,
            thinking=dict(words.CLAUDE_THINKING_ON) if ctx.thinking else None,
            resume=check_resume(resume or ctx.resume),
            output_format={"type": "json_schema", "schema": ctx.output_schema} if ctx.output_schema else None,
            setting_sources=["user"],
            can_use_tool=self._can_use_tool,
            include_partial_messages=bool(ctx.drafts or ctx.raw_frame),
            cli_path=self.cli_path,
            env=env,
        )

    async def start(self, ctx: ProviderContext) -> None:
        claude_policy.require_managed_settings(self.managed_settings)
        self.ctx = ctx
        self._schema = ctx.output_schema
        await self._connect(self.options(ctx))

    # The test verdict trusts a Claude Bash record only when the launch that produced it set PROJECT_CWD_ENV.
    async def _connect(self, options: Any) -> None:
        from claude_agent_sdk import ClaudeSDKClient

        self.bash_starts_in_worktree = False
        self.client = ClaudeSDKClient(options)
        await self.client.connect()
        self.bash_starts_in_worktree = options.env.get(PROJECT_CWD_ENV) == "1"

    async def _reconnect(self) -> None:
        assert self.ctx is not None
        claude_policy.require_managed_settings(self.managed_settings)
        options = self.options(self.ctx, resume=self.session_id)
        old, self.client = self.client, None
        if old is not None:
            await old.disconnect()
        self._schema = self.ctx.output_schema
        self._stale = False
        await self._connect(options)
        self.reconnects += 1

    # The approval callback. In auto mode a shell request that reaches it is refused at once (refusals.py).
    # anything else becomes an approval card through the relay (docs/security-model.md).
    async def _can_use_tool(self, tool_name: str, tool_input: dict[str, Any], context: Any) -> Any:
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        assert self.ctx is not None
        if self.ctx.kind == RESEARCH_KIND:
            return self._research_decision(tool_name)
        if self.ctx.mcp_socket and tool_name.startswith("mcp__") and tool_name not in mcp_allowed(self.ctx):
            return PermissionResultDeny(message="Only the project tools are available in this session.",
                                        interrupt=False)
        if refusals.claude_refuses(self.ctx, tool_name):
            command = tool_input.get("command") if isinstance(tool_input, dict) else None
            message, text = refusals.claude_answer(tool_input)
            await refusals.note(self.ctx, refusals.line(text, tool_name, command,
                                                        getattr(context, "decision_reason", None)))
            return PermissionResultDeny(message=message, interrupt=False)
        payload = normalise.claude_approval(tool_name, tool_input, getattr(context, "tool_use_id", None)
                                            or f"claude-{uuid.uuid4()}", getattr(context, "agent_id", None),
                                            getattr(context, "title", None))
        decision = await self.ctx.request_approval(payload)
        if decision.allow:
            return PermissionResultAllow()
        reply = normalise.owner_reply(decision.message) if tool_name in normalise.QUESTION_TOOLS else None
        answered = normalise.claude_answers(reply, tool_input) if reply is not None else None
        if answered is not None:
            return PermissionResultAllow(updated_input=answered)
        return PermissionResultDeny(message=decision.message, interrupt=False)

    def _research_decision(self, tool_name: str) -> Any:
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        assert self.ctx is not None
        cap = self.ctx.max_subagents
        if tool_name in SUBAGENT_TOOLS and isinstance(cap, int) and self.subagents_allowed < cap:
            self.subagents_allowed += 1
            return PermissionResultAllow()
        return PermissionResultDeny(message=RESEARCH_REFUSED, interrupt=False)

    async def mcp_status(self) -> dict[str, list[str]]:
        status = await self.client.get_mcp_status()
        out: dict[str, list[str]] = {}
        for server in (status or {}).get("mcpServers") or []:
            out[str(server.get("name"))] = sorted(str(t.get("name")) for t in server.get("tools") or [])
        return out

    async def turn(self, text: str) -> AsyncIterator[dict[str, Any]]:
        assert self.ctx is not None
        if self.ctx.output_schema != self._schema or self._stale:
            await self._reconnect()
        yield {"type": "QuerySent"}
        await self.client.query(text)
        async for raw in self._receive():
            yield raw

    async def follow(self) -> AsyncIterator[dict[str, Any]]:
        assert self.ctx is not None
        async for raw in self._receive():
            yield raw

    async def _receive(self) -> AsyncIterator[dict[str, Any]]:
        async for message in self.client.receive_response():
            raw = to_raw(message)
            if self.ctx is not None and self.ctx.raw_frame is not None:
                self.ctx.raw_frame(raw)
            if isinstance(raw, dict) and isinstance(raw.get("session_id"), str) and raw.get("session_id"):
                self.session_id = raw["session_id"]
            yield raw

    async def steer(self, text: str) -> bool:
        try:
            await send_steer(self.client, text)
        except Exception:
            return False
        return True

    async def interrupt(self) -> None:
        await self.client.interrupt()

    async def set_permission_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"permission mode must be one of {', '.join(MODES)}")
        await self.client.set_permission_mode(mode)
        if self.ctx is not None:
            self.ctx.permission_mode = mode

    async def terminate(self) -> None:
        await self.close()

    async def close(self) -> None:
        try:
            if self.client is not None:
                client, self.client = self.client, None
                await client.disconnect()
        finally:
            if self.temp is not None:
                temp, self.temp = self.temp, None
                temp.close()

