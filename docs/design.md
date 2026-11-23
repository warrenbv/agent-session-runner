# Design

This document describes how the runner works. The security controls have their own document, [security-model.md](security-model.md). File names below are relative to the repository root.

## Components

The runner is one Python process per session. It talks to two things outside itself.

- A **control plane**: the service that owns sessions, queues the owner's messages, shows approval cards and stores the event stream. The runner calls it over HTTP with a token bound to one session. The control plane is not part of this repository. [`runner/stub_control_plane.py`](../runner/stub_control_plane.py) is an in-process stand-in used by the tests and the demo.
- A **coding agent CLI**: Claude Code, started by the Claude Agent SDK, or `codex app-server`, started by the Codex SDK. Each provider module owns one of them.

In the original system a separate launcher on each machine prepared a git worktree, wrote the brief and the test command to files and started `python -m runner` for each session. That launcher is not included. Whatever starts the runner must do those things. For a linked worktree it must also pass `--git-dir` and `--git-common-dir`, because the runner never lets git find the repository from the worktree's own `.git` file. A plain checkout needs neither.

Inside the process:

| Module | Role |
|---|---|
| `runner/__main__.py`, `runner/config.py` | Command line, environment, secrets |
| `runner/session.py` | The session driver |
| `runner/providers/` | The provider interface and the Claude, Codex and fake providers |
| `runner/normalise.py`, `runner/events.py` | Raw vendor messages into one event stream |
| `runner/relay.py`, `runner/refusals.py` | Approvals, and requests refused without asking |
| `runner/testrun.py`, `runner/handoff_check.py` | The test verdict and the hand-off checks |
| `runner/handoff.py`, `runner/gitdiff.py` | Commit, push, pull request, the turn diffs, and the repository checks before any git call |
| `runner/claude_policy.py` | The managed settings check before every Claude start |
| `runner/http_transport.py` | No redirects for the two clients that carry a credential |
| `runner/mcp_shim.py`, `runner/session_socket.py`, `runner/mcp_tools.py` | The MCP bridge |
| `runner/control_plane.py` | The HTTP client and the pull protocol |

## How a session runs

1. **Start.** `__main__` reads the token and other secrets (config.py), removes them from the environment, trims what is left to the short list of variables the CLIs may inherit, builds the provider and the `Session`, and calls `Session.run(prompt)`.
2. **Before the first turn.** The driver records which protected paths already exist (placeholders.py). For a coding session it checks the repository: a linked worktree's paths must verify and the repository's git configuration must hold no key that could run a program ([`gitdiff.py`](../runner/gitdiff.py), `CONFIG_RULES`). A refusal ends the session with that sentence as the `end` event's `detail`. The driver then opens the tools socket if the session has tools, starts the provider, and posts one `message` saying which baseline prompt the session runs under. The Claude provider refuses to start, again with a sentence in `detail`, unless the machine's managed settings file requires the sandbox ([`claude_policy.py`](../runner/claude_policy.py)). The first prompt carries the runner's standing note and the repository's own `CLAUDE.md` or `AGENTS.md`, labelled as repository content.
3. **Each turn.** The driver snapshots the worktree into a git tree (the repository check runs again first, and a refusal there skips the diff), sends the prompt, and posts every normalised event in order while the provider streams. A pull loop runs beside the turn every second. It delivers the owner's messages, approval answers and controls:
   - `queue` holds a message until the turn ends, then sends it as the next turn.
   - `steer` hands it to the provider at the model's next step (Claude: a streamed message with `priority: "next"`, Codex: `turn/steer`). If the provider cannot steer, it is held as `queue`.
   - `now` denies any waiting approval, interrupts, lets the turn drain, and sends the message first.
   - Controls: `interrupt`, `finish`, `withdraw` (take back a message the runner still holds), `settings` (model, effort, mode or thinking, from the next turn) and `keep_open` (extend a waiting session's idle limit).
   At the turn's end the worktree is snapshotted again and the difference is posted as a `diff` event.
4. **Caps.** Turns, budget and wall clock (30 minutes by default) are reported as `limit` and `end` events and never extended. The wall clock interrupts, waits a grace period, then terminates. A bounded coding session that reaches a cap, or is asked to finish, wraps up instead of stopping: the running turn is interrupted and one runner-written closing turn asks for the typed result.
5. **Driven sessions** wait after each turn for the owner's next message, until a `finish` control, a stop, or the idle limit. Their closing turn asks for one summary result.
6. **Hand-off checks.** After work that completed, [`handoff_check.py`](../runner/handoff_check.py) compares the typed result with what the runner can see: a missing or thin result, fewer items than the brief's numbered list, changed files not listed, junk files. A thin result gets at most one repair turn. Findings are reported, never enforced.
7. **The test turn.** If the project has a test command and the work changed files, one more runner-written turn asks the agent to run exactly that command, bare, from the directory the session started in. The verdict is read from the provider's record of the call ([`testrun.py`](../runner/testrun.py)): `passed`, `failed`, `not_run` or `tree_changed`. The recorded command must equal the configured one with only surrounding whitespace trimmed. Where it ran comes from the provider: a Codex record must carry the worktree as its `cwd`, and a Claude record counts only when the provider confirms it launched the CLI with each Bash call starting in the worktree. Without that evidence the verdict is `not_run`. A final check compares the model's own `tests_claim` with this verdict. The test turn's own result never replaces the work's.
8. **Hand-off.** With a repository configured, the repository is checked again, then leftover changes are committed on `agent/<slug>` under a fixed agent identity, the branch is pushed (never forced) over HTTPS to the GitHub repository named by `--repo`, whatever remotes the checkout has, and a pull request is opened there with the typed result and the findings in its body. The test verdict goes with the hand-off record on the `end` event.
9. **End.** The tools socket closes, the provider closes, processes the session left running are ended (Linux only, [`reaper.py`](../runner/reaper.py)), sandbox placeholder files are removed, and the `end` event is posted. Exit codes: 0 completed, 2 a cap, 3 token revoked, 1 an error.

A resumed run continues the same event numbering, picks the conversation back up by id when the provider still has it, and starts a new conversation with a carry-over note when it does not.

## The provider interface

[`runner/providers/base.py`](../runner/providers/base.py) defines a `Provider` protocol:

| Method | Contract |
|---|---|
| `start(ctx)` | Start or resume the conversation with a `ProviderContext` |
| `turn(text)` | Send a prompt and yield raw messages up to and including the turn's end |
| `steer(text)` | Deliver text at the next step, or return False so the driver holds it |
| `interrupt()` | End the running turn. Its iterator still drains to the end |
| `set_permission_mode(mode)` | Change the permission mode |
| `change_settings(changes)` | Between turns only. Answers per field: applied, or refused with a reason |
| `settings_now()` | The model, effort, mode and thinking the next turn runs on |
| `terminate()`, `close()` | Stop the process, release resources |
| `normalise(raw)` | Raw message to events |

`ProviderContext` carries the approval callback and every option: kind (coding or research), caps, permission mode, tool lists, resume id, output schema, MCP socket, web access and the baseline. Every option defaults to the safe value. The typed result schema (`summary`, `files_changed`, `tests_run`, `tests_claim`, `branch`, `open_questions`, `next_steps`, `items`) is defined in the same file.

Two vendor constraints shape the providers. The Claude Agent SDK fixes effort, thinking and the output schema when the client connects, so changing them reconnects and resumes the same conversation. Every Claude start and reconnect checks the managed settings again before the client is built, and records whether the launch set the Bash directory reset that the test verdict relies on. The Codex SDK's high-level client accepts every approval by default, so the Codex provider uses the low-level client with its own approval handler.

## The event schema

Every event is `{"sequence", "kind", "provider", "payload"}`. `provider` is `claude-code` or `codex`. The payload is capped at 16 KB of compact UTF-8 JSON, and a cut payload carries `"truncated": true`.

| Kind | Main payload fields |
|---|---|
| `turn` | `phase` (started, completed, waiting). Completed: `is_error`, `interrupted`, `duration_ms`, and `subtype` and `session_id` (Claude) or `status` and `turn_id` (Codex) |
| `message` | `role` (assistant, user, runner) and `text`. User messages carry `author` and `mode`. Runner messages carry a `purpose` such as `baseline`, `settings`, `refused`, `tests-verdict` or `cleanup` |
| `tool` | `phase`, `id`, `name`, and `input_preview` (started) or `is_error` and `output_preview` (completed) |
| `approval` | `request_id`, `tool`, `input_preview`, `requester`. Questions also carry `questions` |
| `subagent` | `phase` (requested, started, progress, completed) and the ids that link it to its tool call |
| `diff` | `scope` (turn, session, session_file). A turn diff carries `files`, `paths`, `stat`, `patch` and `patch_truncated` |
| `usage` | `total_cost_usd` (Claude) or `total` token counts (Codex) |
| `limit` | `limit` (rate_window, max_turns, budget, wall_clock and others) and `source` (provider or runner) |
| `thinking` | The provider's readable thinking, only when the session asked for it |
| `end` | `reason`, `turns`, `cost_usd`, and for a completed run `checks` and `result` |

[`tests/helpers.py`](../tests/helpers.py) holds the table of every field the control plane relies on, and [`tests/test_contract.py`](../tests/test_contract.py) checks it against every recorded fixture and every runner-written event. The golden files in `tests/fixtures/` pin the exact mapping from recorded raw messages.

## The MCP bridge

An agent reaches tools outside its session through one small stdio MCP server. Each CLI starts [`runner/mcp_shim.py`](../runner/mcp_shim.py) as a child process. The shim holds no credential. It relays every tool call over a unix socket to the runner ([`runner/session_socket.py`](../runner/session_socket.py)), which forwards it to the control plane as `sessions.tool_call` with the session's own token. The answer goes back the same way.

The catalogue ([`runner/mcp_tools.py`](../runner/mcp_tools.py)) has thirteen tools, each with a fixed description and a closed JSON Schema, and a test pins a hash of the whole of it.

- **Eight work on the session's own project in the tracker:** list, read, create, update, complete and comment on tasks, read the project's entry and search its notes. Every coding session with tools is offered these.
- **One proposes separate pieces of work** for the owner to start as their own sessions. It is offered only when the session is started with it enabled.
- **Four are for a lead session:** start worker sessions, read their changes, send them back with notes and ask for them to be merged. They are offered only to a lead, which is not offered the proposing tool.

The catalogue only describes. The control plane enforces each session's scope and every argument rule on each call, whatever was offered. [security-model.md](security-model.md#the-mcp-server-and-its-scope) has the controls around the socket and how each CLI is held to this one server.

## The control-plane protocol

Calls are `POST <base>/api/v1/call/<operation>` with `Authorization: Bearer <token>` and JSON bodies. The runner uses `sessions.event_post`, `sessions.pull`, `sessions.tool_call` and `sessions.draft_post`, and research sessions also use `research.inputs_get`, `research.output_part` and `research.output_done`. Events are numbered from 1 with no gaps. A resend after a network error reuses the number, and the control plane answers a repeat as a duplicate. A 413 is resent as a stub. A 401 or 403 means the session has ended. A redirect is never followed, so the token never goes to a second address: the call fails as rejected.

## Research sessions

A research session runs in an empty scratch directory with no repository, no diff, no test turn and no pull request. Its tools are web search and fetch plus file tools held to the scratch directory. It writes two files for its role, which [`research_out.py`](../runner/research_out.py) checks and posts in parts at every ending.

## Testing without credentials

The fake provider replays raw messages in each vendor's real shape, so the real normaliser, driver, relay, test-turn and hand-off code run in the tests. The stub control plane serves the HTTP API on loopback. The recorded fixtures are real sessions captured from both CLIs, with user names, hosts and account details replaced. The real providers are only imported and constructed in the tests.

The fixtures whose names contain `recorded` were captured from real sessions of the two CLIs. The Codex recordings and the Claude recordings with `2_1_321` in their names come from the pinned versions. The other Claude recordings come from an earlier Claude Code release, 2.1.319, as their init messages show. After capture, user names, paths, hosts and account fields were replaced with neutral values, and the account's rate-limit figures were set to round numbers. `claude_turn.raw.json` and `codex_turn.raw.json` were written by hand in the documented shapes. The golden tests run each raw file through the real normaliser and compare the output with its `.golden.json`, and a missing golden file fails unless `RUNNER_UPDATE_GOLDEN=1` is set. They prove that the mapping from these exact messages is stable and that both vendors' messages become the same event kinds. They do not prove that a newer CLI still sends these shapes, and they say nothing about a live session, which the tests never start.
