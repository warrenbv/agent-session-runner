# Security model

The runner starts a coding agent that writes and runs code on the machine. This document describes the controls the runner itself applies, in generic terms. It does not describe any particular deployment. The examples under [`examples/`](../examples) are neutral starting points, not a hardened configuration.

## Threat model in brief

The model, the repository it works on and the pages it reads are all treated as untrusted. The runner tries to make sure that a session can only read its own work, cannot reach the credentials that start it, cannot widen its own permissions, and cannot report its own success without evidence. It relies on each vendor's sandbox for process isolation and does not replace it.

## Secrets and the environment

Both CLIs run as child processes that inherit the runner's environment, so anything left there would sit inside the model's sandbox. Both SDKs start their CLI from a copy of the runner's environment and cannot drop a variable, so the runner trims its own environment before any provider starts ([`config.py`](../runner/config.py)).

- The session token, the GitHub token, `ANTHROPIC_API_KEY` and any `CLAUDE_CODE_OAUTH_TOKEN` are read and deleted from the environment. The runner's other variables (token file paths, the control-plane URL, anything starting `RUNNER_`) are removed once read.
- Every other variable is removed unless it is on a short list, whatever it is called. The list is `PASSED` in `config.py`:
  - `PATH`, `HOME`, `USER`, `LOGNAME` and `SHELL`, to find programs, the home directory and the user's shell
  - `TMPDIR`, `LANG`, `TERM` and `TZ`, for the temp directory, language, terminal type and time zone
  - the locale categories by name: `LC_ALL`, the six POSIX categories (`LC_COLLATE`, `LC_CTYPE`, `LC_MESSAGES`, `LC_MONETARY`, `LC_NUMERIC`, `LC_TIME`) and the six glibc adds (`LC_ADDRESS`, `LC_IDENTIFICATION`, `LC_MEASUREMENT`, `LC_NAME`, `LC_PAPER`, `LC_TELEPHONE`). Any other name starting `LC_` is removed, because SSH setups commonly forward every `LC_` variable, so the prefix could carry anything.
  - `__CF_USER_TEXT_ENCODING`, the macOS text encoding, which Codex also gives its own MCP servers
  - `SSL_CERT_FILE`, `SSL_CERT_DIR`, `NODE_EXTRA_CA_CERTS` and `CODEX_CA_CERTIFICATE`, the trust stores a private certificate authority needs
  - `CLAUDE_CONFIG_DIR` and `CODEX_HOME`, each CLI's configuration directory when it is not the default under the home directory
- The runner trusts the listed variables as the operator set them and does not check their values, although several change what a CLI does: a relative entry in `PATH`, a configuration directory the model can write to, or a trust store that adds a certificate authority.
- Proxy variables are not on the list, because a proxy URL can carry a password. The runner's own HTTP clients ignore proxies.
- An operator who needs something else passed names it in `RUNNER_PASS_ENV`, a comma-separated list such as `RUNNER_PASS_ENV=CLAUDE_CODE_USE_BEDROCK,AWS_REGION`. The runner reads it and removes it like its other variables. Each entry must be a variable name. An entry that is not one is refused by its position in the list, never quoted, so a value typed by mistake as `NAME=value` does not reach the error output: "RUNNER_PASS_ENV entry 2 is not a variable name". A name the runner removes (the secrets above, the token file paths, anything starting `RUNNER_`) is refused before any provider starts: "RUNNER_PASS_ENV may not name RUNNER_TOKEN: the runner removes it before any CLI starts". Another name starting `RUNNER_` is refused by its position in the same way. A variable named there reaches both CLIs and the model's shell, so name only what a session should see.
- The providers and the SDKs then add variables to each child on top of this:
  - Claude Code, every session: `CLAUDE_CODE_OAUTH_TOKEN` when a token file is set, `ANTHROPIC_DEFAULT_FABLE_MODEL`, and `CLAUDE_CODE_SUBAGENT_MODEL` when the session runs Fable
  - Claude Code, coding sessions: `CLAUDE_CODE_TMPDIR` and `TMPDIR` (the session's temp folder), `CLAUDE_CODE_DISABLE_AUTO_MEMORY` and `CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR`
  - Claude Code, research sessions: `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`, `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS`, and `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` when a search cap is set
  - the Claude Agent SDK, every session: `CLAUDE_CODE_ENTRYPOINT`, `CLAUDE_AGENT_SDK_VERSION`, `CLAUDE_CODE_SDK_READS_SESSION_STATE` and `PWD`, plus `TRACEPARENT` and `TRACESTATE` when an OpenTelemetry span is active, which the runner never starts. It also removes `CLAUDECODE`.
  - Codex, coding sessions: `TMPDIR` (the session's temp folder) and, when the session has tools, `AGENT_MCP_SOCKET`, the MCP socket path. Research sessions add nothing.
  - the Codex SDK, every session: the directory of the bundled Codex tools is put first in `PATH`.
- Claude Code's launch settings and MCP server definition, which carries the socket path, travel as command-line arguments (`--settings`, `--mcp-config`), not in the environment. So do Codex's configuration overrides (`--config`).
- A session started from a shell that relied on some other variable, such as a cloud provider setting for Claude Code, needs that variable named in `RUNNER_PASS_ENV`.
- This covers the environment only. Each CLI still reads its own configuration and stored login under the home directory (`~/.claude`, `~/.codex` or the directories named above). What the model can read there is decided by the vendors' sandboxes and the read rules in "Confinement", not by this list.
- The Claude login token is read from a 0600 file and never enters the runner's own environment. The Claude provider puts it in the Claude CLI's environment through the SDK's per-child environment, so it is in the CLI process's environment. Codex never receives it. A research session's tool set has no shell, and its provider also sets `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`, Claude Code's opt-in setting for removing credential variables from the commands it runs. Coding sessions do not set it.
- The GitHub token reaches git only through the child's environment, never its arguments, and the REST API only through the `Authorization` header.
- Neither credential follows a redirect. The control-plane client and the GitHub client both turn off urllib's redirect handling, which would send the `Authorization` header on to the new address ([`http_transport.py`](../runner/http_transport.py)). A redirect answer stops the call: the control-plane client raises `Rejected` ("control plane refused the request: 302 redirect") and the GitHub client raises `ForgeError` naming the step, such as "open_pr: HTTP 302". A repository renamed on GitHub therefore fails that hand-off step instead of being followed to its new name. The hand-off's git push follows no redirect either (see "The runner's own git").
- Token files must be regular files, mode 0600 or stricter, owned by the runner's user.
- No log or print call may take an argument whose name mentions a token, a secret or the brief. A test checks this on the source.
- `config.py` reads the runner's environment. The one other read is in `mcp_shim.py`, which reads the path of its unix socket and nothing else. `gitdiff.py` is the only module that starts processes. Tests check all three.

## The control-plane address

The runner talks to the control plane over https. Plain http is accepted only to the loopback, by the name `localhost` or by a loopback address such as `127.0.0.1` or `::1` ([`config.py`](../runner/config.py), `check_url`). An address of one of this machine's other interfaces is refused too, because that address can later belong to another machine.

The session token and the whole event stream travel over that connection, so plain http never crosses a network. A control plane on another machine needs https, with `RUNNER_CA_FILE` naming the certificate authority when it is a private one.

Everything else is refused before any request: http to any other host, any scheme other than https and http, a URL with a user name, password, query or fragment, and a URL with no host. The refusal for http to another host reads:

> RUNNER_CONTROL_URL may use http only to the loopback, by the name localhost or a loopback address, and 192.0.2.10 is neither, so use https

## Confinement

A coding session may read its worktree, the git directory that worktree's objects live in, and a 0700 temp folder of its own ([`providers/confine.py`](../runner/providers/confine.py)). The scope denies the user's home and the shared temp directories, then re-allows those three paths, so a project cloned later is out of reach too.

- **Claude Code:** `blockReadsOutsideWorkingDirectories`, sandbox read rules that deny the shared temp directories and allow the session's own, the session's temp folder as `TMPDIR`, and auto memory off. The launch settings also turn the sandbox on, make it fail when it cannot start and refuse unsandboxed commands. Claude Code merges its managed settings after these, so the managed values win where they are present.
- **Codex:** named permission profiles that list what may be read and written, with network off. The legacy workspace-write sandbox has no read limit, so it is not used for coding sessions.

The vendor sandboxes do the enforcing. Network policy for Claude Code comes from managed settings (see [`examples/managed-settings.json`](../examples/managed-settings.json)). The runner checks one managed settings file and the drop-in directory beside it, described next, and nothing else. Claude Code gives two kinds of source a higher rank than that file: settings its vendor's servers deliver to an organisation, and OS-level policy such as a macOS configuration profile. When one of those delivers any policy on the machine, Claude Code uses it and by default ignores the file, and the runner cannot see that. On Windows, and on any other platform the check has no path for, the Claude provider refuses to start.

### What the runner checks before Claude Code starts

Because managed values win, the launch settings alone cannot guarantee the sandbox. Before every Claude start and every reconnect, and before the SDK client exists, the runner opens this platform's managed settings file, the one Claude Code reads when no higher-ranked source delivers a policy (`MANAGED_PATHS` in [`claude_policy.py`](../runner/claude_policy.py)) and checks the file it opened:

- Linux: `/etc/claude-code/managed-settings.json`
- macOS: `/Library/Application Support/ClaudeCode/managed-settings.json`

It must be a JSON object, not a symbolic link, owned by root, not writable by group or others, in a directory that is also owned by root and not writable by group or others, with `sandbox.enabled` true, `sandbox.failIfUnavailable` true, `sandbox.allowUnsandboxedCommands` false, and a network allowlist: `sandbox.network.allowedDomains` a non-empty list of non-empty strings, and `sandbox.network.allowManagedDomainsOnly` true, so that the CLI ignores network entries from the launch settings and the user's settings and the managed list is the whole list. It must not name a `policyHelper` (or `policyHelpers`), a program Claude Code runs at startup whose output then becomes the policy, because the check never sees that output. A `managed-settings.d` beside the file is allowed only as a root-owned directory not writable by group or others, holding no `*.json` file, because Claude Code would merge those files after it and the check does not read them. The directory that holds the file is opened once, and the file and `managed-settings.d` are opened through that descriptor without following a link and judged from their own descriptors. The directories above it are reached by path and trusted as the system's own. Anything else is refused with one sentence and the session ends with it as its `detail`:

> Claude Code could not start because its managed settings {problem}, so install them at {path} as a root-owned file writable only by root, with the sandbox settings and network allowlist that examples/managed-settings.json shows.

The path is the one for this platform. The problem is one of "are missing", "could not be read as a JSON object", "are a symbolic link, which the runner does not follow", "are not owned by root", "are writable by group or other users", "sit in a directory that root does not own or that group or other users can write to", "lack the required sandbox settings or network allowlist", "name a policyHelper program, whose output the runner cannot check", "have a managed-settings.d beside them that is not a root-owned directory writable only by root, or that the runner cannot list", or "are extended by files in managed-settings.d, which the runner does not check". [`examples/managed-settings.json`](../examples/managed-settings.json) passes the check, and a test holds it to that. A switch to auto mode is also refused when the file is missing.

On any platform other than Linux and macOS the Claude provider refuses to start, because the check has no file it can vouch for there:

> Claude Code could not start because the runner can check its managed settings only on Linux and macOS, and this machine reports {platform}.

What stays with Claude Code and the machine: the CLI reads the managed settings itself and its sandbox does the enforcing. The check vouches for the managed settings file, not for whatever policy is in force. It checks that the file asks for the sandbox. It does not check that the sandbox works, that the allowlist is narrow, whether a host outside the list is denied or asked about (`sandbox.network.strictAllowlist`, which the check does not require), or that the file is unchanged after the check. Changing the file or its directory needs root.

## Permission modes and approvals

- Claude Code never runs with `bypassPermissions`. The runner refuses that mode before the CLI starts. The command line starts a session in auto mode only through a settings change after connecting, and that change is refused when the managed settings disable auto mode.
- Tools are allowed narrowly and denied explicitly. Web search and fetch are allowed only for a session whose order says `web: true`, and denied otherwise. Codex web search is `disabled` unless the same flag is set.
- An approval that reaches the runner becomes an `approval` event, and the session waits for the owner's answer from the control plane ([`relay.py`](../runner/relay.py)). Silence denies after ten minutes, and stopping the session denies everything still waiting.
- The Codex provider uses the SDK's low-level client because the high-level one accepts every approval by default. Threads start with `approvalPolicy: "on-request"` and `approvalsReviewer: "user"`.

## Refusals

Some requests are refused at once, with words the model can act on, instead of paging the owner ([`refusals.py`](../runner/refusals.py)):

- In Claude Code's auto mode, a shell request that still reaches the approval callback is refused. Auto mode only falls back to the callback for what its classifier would not allow, so no command pattern is needed.
- In every Codex mode, a command approval that asks to run outside the sandbox is declined, and the reason is steered back into the same turn. Approving it would run the command without the read confinement.

Each refusal leaves one `message` line on the stream. Turn, budget and time caps still apply to a session that keeps trying.

## Test verdicts

The runner never runs the project's test command itself, so code a session wrote runs only inside the provider's sandbox. After the work, one more turn asks the agent to run the exact configured command. The verdict is then read from the provider's own record of that shell call ([`testrun.py`](../runner/testrun.py)):

- Claude Code: the `Bash` tool result for that command, its `is_error` flag and the CLI's own "Exit code N" line, plus `returnCodeInterpretation` for commands such as `grep -q` whose non-zero exit the CLI reports as an ordinary answer.
- Codex: the completed `commandExecution` item for that command and its `exitCode`.

The recorded command must be the configured command as written. Only whitespace around a single line is trimmed. Spacing inside the command, inside quotes included, must match, and a command with a second line or a Unicode line separator never matches. A `cd x && <command>` in one call is a different command and does not match.

A pass also needs evidence that the command ran in the worktree, and that evidence comes from the provider, not from a prefix the model was asked to keep. Asking for `cd <worktree> && <command>` was tried and failed: a model drops a `cd` to the directory it is already in.

- Codex: the completed item must carry its own `cwd`, as an absolute path without NUL that resolves to the worktree. A missing, relative or different `cwd` makes the verdict `not_run`.
- Claude Code: the provider launches a coding session's CLI with `CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1`, which makes the CLI put the shell back in the starting directory after each main-thread Bash call. The provider records that the connected client was launched this way, at every start and reconnect. Without that confirmation a Claude verdict is `not_run`. A `cd` in an earlier call does not carry into the test call.

The model's words and the command's printed output are never read for the verdict. A model can say "all tests pass" in prose, print a fake summary, or run a different command. None of these changes the verdict. A call that was denied, never ran, ran in the background, or left the worktree different from before is not a pass. The model's own `tests_claim` is kept as evidence and compared with the verdict, and a mismatch is reported as a hand-off finding ([`handoff_check.py`](../runner/handoff_check.py)). The demo shows this case.

## The MCP server and its scope

[`mcp_shim.py`](../runner/mcp_shim.py) is a small stdio MCP server that the CLI starts. It holds no credential and reads one environment variable, the path of a unix socket. Each tool call goes over that socket to the runner ([`session_socket.py`](../runner/session_socket.py)), which forwards it to the control plane with the session's own token.

The command is [`bin/runner-mcp`](../bin/runner-mcp). From a clone, run it as `uv run bin/runner-mcp` so that `python3` is the project's environment. Its first line is `#!/usr/bin/env -S python3 -I`, which on Linux needs GNU coreutils 8.30 or later for `env -S`, and it takes `python3` from PATH unless a deployment rewrites that line.

- The socket is 0600 in a 0700 directory under the user's runtime directory. The example managed settings deny that directory to sandboxed commands, so the model's shell cannot call the socket directly.
- Claude Code gets this server as its only MCP server, with `strict_mcp_config`. Only the offered tools are allowed and any other MCP tool is refused without asking. Codex gets the server defined per session, off unless the session has tools, with an explicit list of enabled tools.
- The catalogue ([`mcp_tools.py`](../runner/mcp_tools.py)) has closed JSON Schemas and a pinned hash. It only describes. Scope (the session's one project) and every argument rule must be enforced by the control plane on each `sessions.tool_call`. That enforcement is not in this repository, and the stub control plane does not enforce it.

## Repository content is untrusted

- Claude Code loads `setting_sources=["user"]`, so managed and user settings apply but a repository's own `.claude/` settings do not.
- Codex marks the worktree and its main checkout `untrusted` on every start, which stops Codex from trusting the project on its own and loading its `.codex/` config, hooks and MCP servers.
- The repository's `CLAUDE.md` or `AGENTS.md` is given to the model labelled as repository content, not the owner's words, and capped at 32 KB.
- The runner's own git calls (snapshot, diff, commit, push) are covered in the next section.
- A resume id is checked differently by each provider. The Claude provider accepts only a UUID and passes it on in canonical form, because it reaches the Claude CLI as a command-line argument and anything else could be read as another flag. The Codex provider does no check of its own: the thread id goes to the app-server inside a JSON-RPC `thread/resume` request, never as an argument. An id the app-server does not know comes back as an error, and the runner then starts a fresh conversation and says so on the stream.

## The runner's own git

The runner runs git on the session's repository to snapshot the worktree, diff it, commit and push. A repository's own configuration could make those calls run a program in the runner, outside the session's sandbox, for example a clean filter during the snapshot's `git add --all`. [`gitdiff.py`](../runner/gitdiff.py) closes that in three ways.

- **No discovery.** Git is told where the repository is on every call (`GIT_DIR`, `GIT_COMMON_DIR`, `GIT_WORK_TREE`). For a plain checkout those come from the working directory and its real `.git` directory. For a linked worktree they come from the command line (`--git-dir`, `--git-common-dir`), and before any git call the runner checks that the worktree's `.git` pointer, the registered directory under the checkout's `.git/worktrees` and its `commondir` still name each other. A linked worktree started without those paths is refused, and so is a pointer, registration or `commondir` that does not verify.
- **The configuration is listed and checked first.** Before each group of git calls the runner lists the repository's config file with `git config --file - --no-includes --null --list`, fed from a descriptor opened without following links. A key that could run a program or change where git connects is refused: every `filter.*`, merge drivers, diff commands and textconv, external diff, hooks and `core.hooksPath` other than the null device, `core.fsmonitor` other than off, pagers, editors, askpass and credential helpers, signing programs, aliases, includes, transport commands, remote proxies, every `http.*` key (proxies, TLS verification, certificate authorities, client certificates and keys, cookie files, extra headers, redirects and the rest, whether general or keyed to a URL), push and fetch URL rewrites, remotes named by a URL, partial clone and promisor remotes, archive compressors and a few more. The full table, each row with its reason, is `CONFIG_RULES`. System and user git configuration and system attributes are not read at all.
- **Fixed settings on every call.** fsmonitor, external diff, textconv, hooks, the pager and submodule recursion are off. Only git's own transports are allowed and no lazy fetch runs.

The hand-off's push goes to an address built from the `--repo` name, `https://github.com/<owner>/<name>.git` ([`handoff.py`](../runner/handoff.py), `GitHub.push`). No remote in the checkout, `origin` included, decides where the commits go, so they land only in the repository the pull request is opened on, and a checkout with no remote at all can still be handed off. That one call allows HTTPS alone and follows no redirect. The repository's config may hold no `http.*` key when it is checked, but git reads that file again when it pushes, so a process able to rewrite it in between (see the end of this section) could still change the push. Four http settings are pinned for that call, keyed to the exact address because git lets a URL-specific setting outrank a general one: no redirect, certificate verification on, no proxy, and no address given for the host in place of the one its name resolves to. A certificate authority file cannot be pinned this way. A rewrite that adds `insteadOf`, `pushInsteadOf` or a remote named by the address would send the push itself elsewhere, and no setting undoes that. Whatever the config says, the push also carries no tags and no push options and is never signed, so it creates only the agent branch and runs no signing program. Git looks a push address up as a remote name before using it as a URL, so a remote named by a URL is refused with the other keys above. The token is attached only for `https://github.com/`.

A refused key ends the session at its start, or fails the hand-off, with one sentence:

> Git refused 'filter.lfs.clean'. Ask the machine's administrator to remove that key from the project's Git config before restarting the session.

The check is a second line. The runner reads the configuration and git then reads it again, so a process able to write that file between the two could race the check. The first line is the session sandbox refusing the write. In a linked worktree the config and `commondir` sit in the main checkout's `.git`, which a session may read but not write, and only the `.git` pointer sits inside the worktree. On the original system a live probe had a session of each provider try to replace a linked worktree's `.git` pointer, rewrite `commondir` and append to the repository's config, and both sandboxes refused all three. In a plain checkout the config sits inside the worktree itself. Whether each vendor sandbox refuses a write to `.git/config` there was not probed, so for a plain checkout the check may be the only line.

## Known gaps

- A hand-off finding about the test claim is raised only when the model's `tests_claim` says `passed` for the configured command and the recorded call ran that same command. A claim of `passed` for some other command raises no finding, even when the configured command failed or never ran. The verdict line on the stream still reports what the configured command did.
- A `tree_changed` verdict, where the test run left the worktree different from before, raises no hand-off finding either, whatever the model claimed. The stream says the result does not count, but the hand-off checks only report `failed` and `not_run` ([`handoff_check.py`](../runner/handoff_check.py), `test_claim_findings`).
- With no test command configured there is no test turn and no test finding.
- The Claude evidence of where the tests ran is the launch setting and the pinned CLI's behaviour with it, not a directory recorded per call. A newer CLI that dropped or changed `CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR` would need this checked again. A settings `env` entry that overrides it is not looked for.
- The managed settings check reads the managed settings file, looks for drop-in files beside it, and reads nothing else. Claude Code ranks settings its vendor's servers deliver to an organisation, and OS-level policy such as a macOS configuration profile, above the file. When one of those delivers any policy on the machine, Claude Code uses it and by default ignores the file, and the runner cannot see that.
- The managed settings check requires a non-empty allowlist that only managed settings can set. It does not judge what the list allows, and it leaves hosts outside the list to the CLI's own choice between denying and asking.
- The git configuration check is a second line behind the session sandbox, and it can be raced by anything able to write the repository's config (see "The runner's own git"). For a plain checkout, whose config sits inside the worktree, the sandbox side was not probed. Submodules' own configuration is not listed. The runner's calls do not recurse into submodules and ignore their dirty state.
- A repository that uses git-lfs, git-crypt or any other filter driver is refused. No driver is allowed by name.

## What this does not protect against

- A flaw in a vendor's sandbox or CLI. The runner configures them and depends on them.
- Prompt injection through web pages or repository files. The runner limits what a session can reach, but the model still reads that content.
- Anything the control plane allows. Approval policy, tool scope and merge decisions belong to it.
