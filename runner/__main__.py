"""The command line for one session: `python -m runner --provider claude|codex|fake ...`.

Whatever starts the runner prepares the worktree, writes the prompt (and optionally the test command
and a resumed run's notes) to files, and passes the session's token through the environment or a
0600 file (config.py). The runner never names a session: the token decides which session it is.
A linked worktree also needs `--git-dir` and `--git-common-dir`, because the runner does not let git
find them from the worktree's own `.git` file (gitdiff.py). A plain checkout needs neither.

Exit codes: 0 the run completed, 2 a cap ended it, 3 the control plane revoked the token, 1 an error.
A research session also exits 1 when its files could not be handed over.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import re
import sys

from runner import config, gitdiff, handoff, research_out, testrun
from runner.control_plane import Client, ResearchClient
from runner.providers.base import CODING_KIND, RESEARCH_KIND
from runner.providers.claude import EFFORTS
from runner.session import BASE_FETCH_OUTCOMES, Caps, Options, Session, Timing, base_fetch_line
from runner.settings import MODES

COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="runner", description="Run one agent session and post its events to the control plane.")
    p.add_argument("--provider", choices=("claude", "codex", "fake"), required=True)
    p.add_argument("--cwd", required=True, help="the session's worktree")
    p.add_argument("--git-dir", help="a linked worktree's own git directory (<checkout>/.git/worktrees/<name>)")
    p.add_argument("--git-common-dir", help="a linked worktree's common git directory (<checkout>/.git)")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--fixture", help="the fake provider's script")
    p.add_argument("--model")
    p.add_argument("--effort", choices=EFFORTS, help="Claude Code at start; Codex from its first turn")
    p.add_argument("--permission-mode", default="default", choices=("default", "acceptEdits", "plan", "dontAsk"))
    p.add_argument("--start-mode", choices=MODES,
                   help="the mode the session starts in, as Session settings names it (the start order's"
                        " permission_mode); applied before the first turn, on either provider")
    p.add_argument("--allowed-tool", action="append", default=[])
    p.add_argument("--disallowed-tool", action="append", default=[])
    p.add_argument("--resume", help="a Claude session id (UUID) or a Codex thread id")
    p.add_argument("--max-turns", type=int)
    p.add_argument("--max-budget-usd", type=float)
    p.add_argument("--wall-clock", type=float, default=1800.0)
    p.add_argument("--grace", type=float, default=30.0)
    p.add_argument("--pull-interval", type=float, default=1.0)
    p.add_argument("--slug", help="the branch is agent/<slug>; with --repo, the run is handed off as a PR")
    p.add_argument("--repo", help="owner/name on GitHub")
    p.add_argument("--base", default="main", help="the pull request's base branch")
    p.add_argument("--base-ref", type=commit_id,
                   help="the commit the worktree was cut from; the hand-off counts from it (default: --base)")
    p.add_argument("--title", help="the session's title (the start order's), which titles the pull request;"
                                   " without it, a plain line of the brief (handoff.pr_title)")
    p.add_argument("--test-cmd-file",
                   help="a file holding the project's test command, which the session runs in one last turn")
    p.add_argument("--tools", action="store_true",
                   help="offer the session the project tools (the MCP server; the start order's tools)")
    p.add_argument("--propose", action="store_true",
                   help="with --tools, offer the ninth project tool, sessions_propose, and say so in the standing"
                        " note (the start order's propose)")
    p.add_argument("--lead", action="store_true",
                   help="with --tools, offer the four lead tools (workers_start, worker_changes, worker_send_back,"
                        " worker_merge) instead of sessions_propose, and say so in the standing note"
                        " (the start order's lead)")
    p.add_argument("--web", action="store_true",
                   help="let the session search and fetch the web (the start order's web, set for a"
                        " project that allows it); without it the web tools are refused")
    p.add_argument("--driven", action="store_true",
                   help="wait for the owner's next message after each turn (the start order's mode driven)")
    p.add_argument("--idle-seconds", type=int, default=14400,
                   help="a driven session is finished after this long with no message (default four hours)")
    p.add_argument("--drafts", action="store_true",
                   help="post replies as they are written to sessions.draft_post (the start order's drafts)")
    p.add_argument("--thinking", action="store_true",
                   help="ask the provider for its readable thinking from the first turn (the start order's thinking)")
    p.add_argument("--file-diffs", action="store_true",
                   help="after each turn that changed files, also post the whole session's change against its"
                        " base, one event per file (the start order's file_diffs)")
    p.add_argument("--no-baseline", dest="baseline", action="store_false",
                   help="give the session no platform baseline (runner/baseline.md; the start order's baseline"
                        " as a literal false): Claude Code's preset without the append, Codex with no developer"
                        " instructions")
    resumed = p.add_argument_group("a resumed run (the start order's resume)")
    resumed.add_argument("--first-sequence", type=positive, default=1,
                         help="the number of the run's first event, where the session's stream left off")
    resumed.add_argument("--run", type=positive, default=1, help="the session's run number")
    resumed.add_argument("--resume-text-file",
                         help="a file holding the owner's message that picks the session back up; makes the run a"
                              " resumed one (its brief is then never sent as a prompt)")
    resumed.add_argument("--carry-over-file",
                         help="a file holding the code-written carry-over note, sent before the owner's message only when"
                              " the run starts a new conversation")
    resumed.add_argument("--cost-before", type=cost, help="the cost the session had reached, in USD")
    resumed.add_argument("--base-fetch", choices=BASE_FETCH_OUTCOMES,
                         help="whether the fetch of the default branch (--base) into origin/<base> before"
                              " this resumed run worked; the session is told in one line either way")
    resumed.add_argument("--merge-note-file",
                         help="a file holding the line about the merge of the default branch into the branch"
                              " before a lead's send-back (made by whatever prepared the worktree), sent after the owner's message")
    research = p.add_argument_group("a research session (--kind research)")
    research.add_argument("--kind", choices=(CODING_KIND, RESEARCH_KIND), default=CODING_KIND,
                          help="research: --cwd is the session's empty scratch directory; no repository, diff,"
                               " test turn or pull request")
    research.add_argument("--role", choices=research_out.ROLES)
    research.add_argument("--session-id", type=positive, help="the session's id, which the research operations name")
    research.add_argument("--research-run-id", type=positive)
    research.add_argument("--max-subagents", type=int, help="subagent starts allowed (counted in the stream)")
    research.add_argument("--max-searches", type=positive, help="web searches allowed (Claude Code enforces it)")
    return p.parse_args(argv)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("a whole number of at least 1")
    return number


def cost(value: str) -> float:
    number = float(value)
    if not 0 <= number < 1e6:
        raise argparse.ArgumentTypeError("a cost of at least 0")
    return number


RESUME_TEXT_CAP = 20000
CARRY_CAP = 64 * 1024


def read_resume(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if not args.resume_text_file:
        if args.carry_over_file:
            raise SystemExit("--carry-over-file is for a resumed run (--resume-text-file)")
        return None, None
    found = []
    for path, cap, what in ((args.resume_text_file, RESUME_TEXT_CAP, "message"),
                            (args.carry_over_file, CARRY_CAP, "carry-over note")):
        if not path:
            found.append(None)
            continue
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"the resumed run's {what} file could not be read ({type(exc).__name__})") from None
        if not text.strip() or len(text) > cap:
            raise SystemExit(f"the resumed run's {what} is empty or longer than {cap} characters")
        found.append(text)
    return found[0], found[1]


MERGE_NOTE_CAP = 16 * 1024


def read_merge_note(args: argparse.Namespace) -> str | None:
    if not args.merge_note_file:
        return None
    if not args.resume_text_file:
        raise SystemExit("--merge-note-file is for a resumed run (--resume-text-file)")
    try:
        text = pathlib.Path(args.merge_note_file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logging.getLogger("runner").warning("the send-back merge note could not be read (%s)", type(exc).__name__)
        return None
    return text.strip()[:MERGE_NOTE_CAP] or None


def check_research(args: argparse.Namespace) -> None:
    if args.kind != RESEARCH_KIND:
        return
    if args.provider == "codex":
        from runner.providers.codex import research_refusal
        refusal = research_refusal(args.role)
        if refusal:
            raise SystemExit(refusal)
    missing = [flag for flag, value in (("--role", args.role), ("--session-id", args.session_id),
                                        ("--max-subagents", args.max_subagents), ("--max-searches", args.max_searches),
                                        ("--max-turns", args.max_turns)) if value is None]
    if missing:
        raise SystemExit(f"a research session needs {', '.join(missing)}")
    if args.max_subagents < 0:
        raise SystemExit("--max-subagents is at least 0")
    if args.slug or args.repo or args.base_ref or args.test_cmd_file or args.tools or args.propose or args.lead \
            or args.web or args.driven:
        raise SystemExit("a research session takes no repository, branch, test command, tools, proposals, lead,"
                         " web flag or driving")


def read_test_cmd(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return testrun.check_test_cmd(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"the test command file could not be read ({type(exc).__name__})") from None
    except ValueError as exc:
        raise SystemExit(f"the test command is not usable: {exc}") from None


def commit_id(value: str) -> str:
    if not COMMIT.match(value):
        raise argparse.ArgumentTypeError("a full commit id (40 or 64 hex digits)")
    return value


def provider_for(args: argparse.Namespace, settings: config.Settings | None = None):
    if args.provider == "fake":
        from runner.providers.fake import FakeProvider
        if not args.fixture:
            raise SystemExit("--fixture is required with --provider fake")
        return FakeProvider(args.fixture)
    if args.provider == "claude":
        from runner.providers.claude import ClaudeProvider
        return ClaudeProvider(oauth_token=settings.claude_token if settings else "")
    from runner.providers.codex import CodexProvider
    return CodexProvider()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse(sys.argv[1:] if argv is None else argv)
    check_research(args)
    settings = config.take_secrets()
    prompt = pathlib.Path(args.prompt_file).read_text(encoding="utf-8")
    client = Client(settings.control_url, settings.token, settings.ca_file, host=settings.control_host)
    if args.kind == RESEARCH_KIND:
        return run_research(args, settings, client, prompt)
    test_cmd = read_test_cmd(args.test_cmd_file)
    resume_text, carry_over = read_resume(args)
    lead = bool(args.lead and args.tools)
    propose = bool(args.propose and args.tools and not lead)
    hand, note = None, handoff.standing_note(propose, lead)
    if args.slug and args.repo:
        forge = handoff.GitHub(args.repo, settings.github_token)
        slug, base, since, cwd, title = args.slug, args.base, args.base_ref, args.cwd, args.title
        note = handoff.driven_note(slug, False, propose, lead) if args.driven \
            else handoff.brief_note(slug, propose, lead)
        reuse = resume_text is not None

        def hand(result):
            return handoff.hand_off(cwd, slug, base, result, forge, since=since, existed=session.existed,
                                    title=title, brief=prompt, latest=session.title, reuse=reuse,
                                    wrapped_up=session.wrapped_up, checks=session.checks, run=args.run)

    worktree = gitdiff.Worktree(str(pathlib.Path(args.cwd).absolute()), args.git_dir or "", args.git_common_dir or "") \
        if args.git_dir is not None or args.git_common_dir is not None else None
    session = Session(
        provider_for(args, settings), client,
        Options(cwd=args.cwd, model=args.model, effort=args.effort, permission_mode=args.permission_mode,
                allowed_tools=args.allowed_tool, disallowed_tools=args.disallowed_tool, resume=args.resume, brief_note=note,
                test_cmd=test_cmd, tools=args.tools, propose=propose, lead=lead, driven=args.driven,
                drafts=args.drafts,
                file_diffs=args.file_diffs, base_ref=args.base_ref, start_mode=args.start_mode,
                thinking=args.thinking, web=args.web,
                idle_seconds=float(max(args.idle_seconds, 1)), first_sequence=args.first_sequence,
                resume_text=resume_text, carry_over=carry_over, cost_before=args.cost_before,
                base_fetch=base_fetch_line(args.base_fetch, args.base) if resume_text is not None else None,
                merge_note=read_merge_note(args),
                baseline=args.baseline, worktree=worktree),
        Caps(max_turns=args.max_turns, max_budget_usd=args.max_budget_usd, wall_clock=args.wall_clock,
             grace=args.grace),
        Timing(pull_interval=args.pull_interval), handoff=hand)
    outcome = asyncio.run(session.run(prompt))
    logging.getLogger("runner").info("session ended: %s after %d turns", outcome.reason, outcome.turns)
    return outcome.exit_code


def research_session(args: argparse.Namespace, provider, client: Client) -> Session:
    out = research_out.ResearchOut(ResearchClient(client, args.session_id), args.role, args.cwd)
    return Session(
        provider, client,
        Options(cwd=args.cwd, model=args.model, effort=args.effort, output_schema=None, diff=False,
                kind=RESEARCH_KIND, max_subagents=args.max_subagents, max_searches=args.max_searches),
        Caps(max_turns=args.max_turns, max_budget_usd=args.max_budget_usd, wall_clock=args.wall_clock,
             grace=args.grace),
        Timing(pull_interval=args.pull_interval), research=out)


def run_research(args: argparse.Namespace, settings: config.Settings, client: Client, prompt: str) -> int:
    session = research_session(args, provider_for(args, settings), client)
    logging.getLogger("runner").info("research session %d (run %s, role %s) starting", args.session_id,
                                     args.research_run_id, args.role)
    outcome = asyncio.run(session.run(prompt))
    logging.getLogger("runner").info("research session ended: %s after %d turns", outcome.reason, outcome.turns)
    if not session.research.done:
        return outcome.exit_code if outcome.exit_code not in (0, 2) else 1
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

