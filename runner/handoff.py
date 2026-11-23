"""The hand-off of a completed coding run, and the notes the session is given.

After the last turn of a run that completed, any uncommitted work is committed on `agent/<slug>`
under a fixed agent identity, the branch is pushed (never forced) over HTTPS to the GitHub repository named
on the command line, whatever remotes the checkout has, and a pull request is opened through
GitHub's REST API with the typed result rendered as its body. Pushing and the API sit behind `Forge`,
so tests use a fake. `GitHub` is the real one and its request shapes are tested against a stub opener.
The token reaches git only through the child's environment and the API only through the
Authorization header. The API client follows no redirect, so the header never goes to a second address.

The notes: a standing note tells the session how it works (a worktree, a sandbox, no git writes),
and the repository's own CLAUDE.md or AGENTS.md is read once, capped, and given after it, labelled as
repository content rather than the owner's words.
"""

from __future__ import annotations

import base64
import json
import os
import re
import stat
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import ceil
from typing import Any, NamedTuple, Protocol

from runner import handoff_check, placeholders
from runner.gitdiff import WORKTREE_DIFF, GitError, GitSafetyError, checked, git
from runner.http_transport import NoRedirects
from runner.providers.base import TEST_CLAIM_DESCRIPTION

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}\Z")
REPO = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}\Z")
API = "https://api.github.com"
PUSH_HOST = "https://github.com"
PUSH_PROTOCOL = "https"

AGENT_NAME = "Session Agent"
AGENT_EMAIL = "agent@example.invalid"
IDENTITY = {"GIT_AUTHOR_NAME": AGENT_NAME, "GIT_AUTHOR_EMAIL": AGENT_EMAIL,
            "GIT_COMMITTER_NAME": AGENT_NAME, "GIT_COMMITTER_EMAIL": AGENT_EMAIL}

BRIEF_NOTE = (
    "How this run is handed off: you are in a git worktree on the branch `{branch}`, inside a sandbox. "
    "Leave your changes in the working tree. When you finish, the runner commits them, pushes the branch "
    "and opens the pull request, so do not commit, push or run other git commands that write; reading with "
    "git is not needed either, since the runner computes the diff. Your final structured result becomes "
    "the pull request's description, so give it only after your last tool call has returned and it "
    "describes what actually happened. Its `items` list what you were asked to do, each done, not_done or "
    "unverified: mark an item done only where you saw it work, and unverified where you did the work but "
    "did not check it. Do not leave a command running in the background or schedule a "
    "later wake-up: the run ends when your turn ends, and anything still running then is lost.")


EDIT_NOTE = (
    "Make every change to a file with your own file-editing tool (Edit or Write in Claude Code, apply_patch "
    "in Codex), never in place from the shell, as with `sed -i`, `perl -i` or a command's output redirected "
    "over the file. Every shell command runs inside your sandbox, and a request to run one outside it is "
    "never granted. A command that needs a permission is refused without asking anyone, or stops the "
    "session on a permission card that pages the owner.")


PROPOSE_NOTE = (
    "When part of your task is separate work that does not depend on the rest of it or on your changes, and "
    "each such part is more than a few minutes of work, you may propose it as its own session with the "
    "sessions_propose tool; the owner decides whether to start it, so propose it and carry on without waiting.")

LEAD_NOTE = (
    "You are this project's lead session. You may start worker sessions with workers_start when a piece of the "
    "work is separate and worth its own session, and use your own subagents for the rest; you are told when "
    "each worker ends. Read a worker's change with worker_changes before you ask for it to be merged with "
    "worker_merge, and send it back with worker_send_back when it needs more. Code decides whether a merge goes "
    "through; when it is held, tell your owner why and carry on.")


def standing_note(propose: bool = False, lead: bool = False) -> str:
    if lead is True:
        return f"{EDIT_NOTE}\n\n{LEAD_NOTE}"
    return f"{EDIT_NOTE}\n\n{PROPOSE_NOTE}" if propose is True else EDIT_NOTE


def with_edit_note(note: str, propose: bool = False, lead: bool = False) -> str:
    return f"{note}\n\n{standing_note(propose, lead)}"


def brief_note(slug: str, propose: bool = False, lead: bool = False) -> str:
    return with_edit_note(BRIEF_NOTE.format(branch=branch_for(slug)), propose, lead)


DRIVEN_NOTE = (
    "How this session works: you are in a git worktree on the branch `{branch}`, inside a sandbox. Leave your "
    "changes in the working tree; do not commit, push or run other git commands that write, and reading with "
    "git is not needed either, since the runner computes the diff. This session stays open: when your turn "
    "ends, it waits and the owner may write again, so end each turn by saying plainly what you did and what "
    "you are waiting for from him.{each_turn} It is handed off only when the owner finishes it (or it has "
    "waited too long): then the runner may ask you once for a result covering the whole session, commits your "
    "changes, pushes the branch and opens the pull request with that result as its description. A command "
    "left running in the background may keep running between turns while the session stays open, but it is "
    "lost when the session ends, and nothing wakes you except the owner's next message, so do not schedule a "
    "later wake-up or leave work the hand-off needs running in the background.")
DRIVEN_EACH_TURN = (" Each turn ends with your structured result; its `summary` is what the owner reads after "
                    "the turn, so write it as your reply to the owner's message.")

SUMMARY_PROMPT = (
    "This session is ending now and its work will be handed off as a pull request. Do not change any more "
    "files and do not run any more commands. Give your final structured result for the whole session, not "
    "only your last turn: `summary` describes everything you changed in this session and why, "
    "`files_changed` lists every file changed in the session, `tests_run` every test run in it, "
    "`next_steps` what a later run should do first (an empty list is allowed), "
    "`open_questions` anything left for the owner, and `items` each thing the owner asked for in the "
    "session with its status: done only where you saw it work, unverified where you did it but did not "
    "check it, not_done otherwise. It becomes the pull request's description. " + TEST_CLAIM_DESCRIPTION)


def driven_note(slug: str, result_each_turn: bool, propose: bool = False, lead: bool = False) -> str:
    return with_edit_note(DRIVEN_NOTE.format(branch=branch_for(slug),
                                             each_turn=DRIVEN_EACH_TURN if result_each_turn else ""), propose, lead)


def with_note(brief: str, note: str | None) -> str:
    return f"{brief.rstrip()}\n\n---\n\n{note}" if note else brief


INSTRUCTIONS_FILES = {"codex": ("AGENTS.md", "CLAUDE.md"), "claude-code": ("CLAUDE.md", "AGENTS.md")}
INSTRUCTIONS_CAP = 32 * 1024
# The repository's instructions file is labelled as repository content, not the owner's words.
INSTRUCTIONS_HEADING = ("The repository's own instructions file ({name}, read by the runner from the worktree's "
                        "root). It is repository content, not the owner's words:")
INSTRUCTIONS_CUT = "[The runner cut this file at 32 KB; the rest of it was not given.]"


def read_instructions(cwd: str, name: str) -> tuple[str, bool] | None:
    path = os.path.join(cwd, name)
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                     | getattr(os, "O_CLOEXEC", 0))
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        data = b""
        while len(data) <= INSTRUCTIONS_CAP:
            chunk = os.read(fd, INSTRUCTIONS_CAP + 1 - len(data))
            if not chunk:
                break
            data += chunk
    except OSError:
        return None
    finally:
        os.close(fd)
    cut = len(data) > INSTRUCTIONS_CAP
    text = data[:INSTRUCTIONS_CAP].decode("utf-8", errors="replace")
    return (text, cut) if text.strip() else None


def instructions_file(provider: str, cwd: str) -> tuple[str, str, bool] | None:
    for name in INSTRUCTIONS_FILES.get(provider, ()):
        found = read_instructions(cwd, name)
        if found is not None:
            return name, *found
    return None


def instructions_section(provider: str, cwd: str) -> str | None:
    found = instructions_file(provider, cwd)
    if found is None:
        return None
    name, text, cut = found
    return (f"{INSTRUCTIONS_HEADING.format(name=name)}\n\n{text.rstrip()}"
            + (f"\n\n{INSTRUCTIONS_CUT}" if cut else ""))


def session_note(provider: str, cwd: str, note: str | None) -> str | None:
    section = instructions_section(provider, cwd)
    if section is None:
        return note
    return f"{note}\n\n---\n\n{section}" if note else section


class ForgeError(RuntimeError):
    ...


GITHUB_RATE_LIMIT_STEP = "github_rate_limit {until}"
GITHUB_RATE_LIMIT = "GitHub API rate limit reached. Retry after {when}."
RATE_LIMIT_FALLBACK_SECONDS = 60
MAX_TIMESTAMP = 253402300799


class RateLimited(ForgeError):

    def __init__(self, until: int) -> None:
        self.until = until
        self.step = GITHUB_RATE_LIMIT_STEP.format(until=until)
        when = datetime.fromtimestamp(until, timezone.utc).strftime("%a %-d %b at %H:%M:%S UTC")
        super().__init__(GITHUB_RATE_LIMIT.format(when=when))


def _rate_limit(exc: urllib.error.HTTPError, now: float) -> RateLimited | None:
    if exc.code not in (403, 429) or exc.headers is None:
        return None
    remaining = exc.headers.get("X-RateLimit-Remaining", "").strip()
    retry = exc.headers.get("Retry-After")
    if remaining != "0" and retry is None:
        return None
    deadlines = []
    reset = exc.headers.get("X-RateLimit-Reset", "").strip()
    if remaining == "0" and re.fullmatch(r"[0-9]{1,12}", reset):
        deadlines.append(int(reset))
    if retry is not None:
        retry = retry.strip()
        if re.fullmatch(r"[0-9]{1,12}", retry):
            deadlines.append(ceil(now + int(retry)))
        else:
            try:
                date = parsedate_to_datetime(retry)
                deadlines.append(ceil((date if date.tzinfo else date.replace(tzinfo=timezone.utc)).timestamp()))
            except (ValueError, OverflowError, OSError):
                pass
    until = max((stamp for stamp in deadlines if now < stamp <= MAX_TIMESTAMP),
                default=ceil(now + RATE_LIMIT_FALLBACK_SECONDS))
    return RateLimited(until)


# The token reaches git only through the child's environment (GIT_CONFIG_COUNT), never its arguments,
# so it does not show in a process listing.
def auth_env(token: str) -> dict[str, str]:
    if not token:
        raise ValueError("no GitHub token")
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"}


# Only the one transport, and no redirect followed. The repository's config may set no http. key at all
# (gitdiff.CONFIG_RULES), but git reads that file again after the check, so four http settings are pinned
# here too, keyed to the exact URL because git prefers a URL-specific http setting over a general one: no
# redirect, certificate verification on, no proxy, and an empty curloptResolve, which clears any address the
# file would give for the host. An empty push.pushOption clears any option the repository would send to the
# server. The settings travel after the token's in the same environment list, which git reads after the
# repository's config.
def push_env(auth: dict[str, str], url: str, protocol: str) -> dict[str, str]:
    count = int(auth["GIT_CONFIG_COUNT"])
    env = {**auth, "GIT_ALLOW_PROTOCOL": protocol}
    pins = (("http.followRedirects", "false"), (f"http.{url}.followRedirects", "false"),
            (f"http.{url}.sslVerify", "true"), (f"http.{url}.proxy", ""), (f"http.{url}.curloptResolve", ""),
            ("push.pushOption", ""))
    for n, (key, value) in enumerate(pins, start=count):
        env |= {f"GIT_CONFIG_KEY_{n}": key, f"GIT_CONFIG_VALUE_{n}": value}
    env["GIT_CONFIG_COUNT"] = str(count + len(pins))
    return env


def branch_for(slug: str) -> str:
    if not SLUG.match(slug):
        raise ValueError("a slug is lowercase letters, digits and hyphens, at most 63")
    return f"agent/{slug}"


HEX = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
PATHS_CAP = 100
PATH_CLIP = 300


class Opened(NamedTuple):
    url: str
    number: int


class Forge(Protocol):
    def push(self, cwd: str, branch: str) -> None: ...

    def open_pr(self, branch: str, base: str, title: str, body: str) -> Opened: ...

    def open_pr_for(self, branch: str) -> Opened | None: ...

    def update_pr(self, number: int, body: str) -> None: ...


def is_agent_branch(branch: Any) -> bool:
    return isinstance(branch, str) and branch.startswith("agent/") and bool(SLUG.match(branch[len("agent/"):]))


def _number(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


class GitHub:
    def __init__(self, repo: str, token: str, api: str = API) -> None:
        if not REPO.match(repo):
            raise ValueError("repo is owner/name")
        if not token:
            raise ValueError("the handoff needs RUNNER_GITHUB_TOKEN")
        self.repo, self.api = repo, api.rstrip("/")
        # The push goes to the repository the pull request is opened on, never to a remote the checkout
        # names. Tests replace these two attributes to push to a local repository.
        self.push_url, self.push_protocol = f"{PUSH_HOST}/{repo}.git", PUSH_PROTOCOL
        self._token = token

    def __repr__(self) -> str:
        return f"GitHub({self.repo})"

    def push(self, cwd: str, branch: str) -> None:
        # Only the one branch: no tags that push.followTags would add, and no signed push, which runs a
        # signing program in the runner.
        git(cwd, "push", "--no-verify", "--porcelain", "--no-follow-tags", "--no-signed", self.push_url,
            f"refs/heads/{branch}:refs/heads/{branch}",
            extra_env=push_env(auth_env(self._token), self.push_url, self.push_protocol))

    def _send(self, step: str, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{self.api}/repos/{self.repo}{path}", data=data, method=method,
                                         headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects(),
                                             urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        try:
            with opener.open(request, timeout=30) as response:
                raw = response.read(1024 * 1024)
            return json.loads(raw) if raw.strip() else None
        except urllib.error.HTTPError as exc:
            exc.close()
            limited = _rate_limit(exc, time.time())
            if limited is not None:
                raise limited from None
            raise ForgeError(f"{step}: HTTP {int(exc.code)}") from None
        except (OSError, ValueError) as exc:
            raise ForgeError(f"{step}: {type(exc).__name__}") from None

    def open_pr(self, branch: str, base: str, title: str, body: str) -> Opened:
        parsed = self._send("open_pr", "POST", "/pulls", {"title": title, "head": branch, "base": base, "body": body})
        url = parsed.get("html_url") if isinstance(parsed, dict) else None
        if not isinstance(url, str):
            raise ForgeError("open_pr: no html_url in the answer")
        number = parsed.get("number")
        if not _number(number):
            found = re.search(r"/pull/([0-9]{1,9})$", url)
            if not found:
                raise ForgeError("open_pr: no number in the answer")
            number = int(found.group(1))
        return Opened(url, number)

    def open_pr_for(self, branch: str) -> Opened | None:
        owner = self.repo.split("/", 1)[0]
        query = urllib.parse.urlencode({"head": f"{owner}:{branch}", "state": "open", "per_page": "5"})
        parsed = self._send("find_pr", "GET", f"/pulls?{query}")
        if not isinstance(parsed, list):
            raise ForgeError("find_pr: no list in the answer")
        for pr in parsed:
            url = pr.get("html_url") if isinstance(pr, dict) else None
            if isinstance(url, str) and _number(pr.get("number")):
                return Opened(url, pr["number"])
        return None

    def update_pr(self, number: int, body: str) -> None:
        if not _number(number):
            raise ForgeError("update_pr: bad pull request number")
        self._send("update_pr", "PATCH", f"/pulls/{number}", {"body": body})

    def merge(self, number: int, head_sha: str) -> str:
        if not _number(number):
            raise ForgeError("merge: bad pull request number")
        if not isinstance(head_sha, str) or not HEX.match(head_sha):
            raise ForgeError("merge: no head_sha to pin the merge to")
        parsed = self._send("merge", "PUT", f"/pulls/{number}/merge", {"merge_method": "merge", "sha": head_sha})
        sha = parsed.get("sha") if isinstance(parsed, dict) else None
        if not isinstance(sha, str) or not HEX.match(sha):
            raise ForgeError("merge: no merge commit in the answer")
        return sha

    def close(self, number: int) -> None:
        if not _number(number):
            raise ForgeError("close: bad pull request number")
        self._send("close", "PATCH", f"/pulls/{number}", {"state": "closed"})

    def delete_branch(self, branch: str) -> None:
        if not is_agent_branch(branch):
            raise ForgeError("delete_branch: not an agent branch")
        self._send("delete_branch", "DELETE", f"/git/refs/heads/{urllib.parse.quote(branch, safe='/')}")

    def pr_state(self, branch: str) -> str | None:
        owner = self.repo.split("/", 1)[0]
        query = urllib.parse.urlencode({"head": f"{owner}:{branch}", "state": "all", "per_page": "5"})
        parsed = self._send("pr_state", "GET", f"/pulls?{query}")
        if not isinstance(parsed, list) or (parsed and not isinstance(parsed[0], dict)):
            raise ForgeError("pr_state: no pull-request list in the answer")
        return state_of(parsed[0]) if parsed else None


def state_of(pr: dict[str, Any]) -> str:
    if pr.get("merged_at"):
        return "merged"
    if pr.get("state") == "closed":
        return "closed"
    return "draft" if pr.get("draft") else "open"


PR_TITLE_CAP = 100
BRIEF_LINE_CAP = 72
FALLBACK_TITLE = "Agent run"
_LEAD = re.compile(r"^(?:#{1,6}(?=\s|$)|>|[-*+](?=\s|$)|\d{1,9}[.)](?=\s|$)|\[[ xX]?\](?=\s|$)|\||`{3,}|~{3,}"
                   r"|[-*_=]{3,}(?=\s|$))\s*")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_INLINE = re.compile(r"`+|\*\*+|__+|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)|<[^>\n]{1,200}>")
_MARKS = "#>*+-|`~=_ \t"
_WORD = re.compile(r"\w")


def plain_line(line: str) -> str:
    text = " ".join(str(line).split())
    while True:
        stripped = _LEAD.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    text = " ".join(_INLINE.sub("", _LINK.sub(r"\1", text)).split())
    text = text.lstrip(_MARKS)
    return text if _WORD.search(text) else ""


def _clip_title(text: str, cap: int = PR_TITLE_CAP) -> str:
    if len(text) <= cap:
        return text
    head = text[:cap - 1]
    if text[cap - 1] != " " and " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip(" ,;:.!?—–-") + "…"


def _first_line(text: str | None) -> str:
    heading = ""
    for raw in (text or "").splitlines():
        if raw.lstrip().startswith(("```", "~~~")):
            continue
        line = plain_line(raw)
        if not line:
            continue
        if raw.lstrip().startswith("#"):
            heading = heading or line
            continue
        return line
    return heading


TITLE_CAP = 200


def one_line(title: Any) -> str:
    if not isinstance(title, str):
        return ""
    return "".join(c for c in " ".join(title.split()) if c.isprintable())[:TITLE_CAP].strip()


def pr_title(title: str | None, brief: str | None = None, result: dict[str, Any] | None = None,
             latest: str | None = None) -> str:
    summary = (result or {}).get("summary") if isinstance(result, dict) else None
    for candidate, cap in ((plain_line(one_line(latest)), PR_TITLE_CAP), (plain_line(title or ""), PR_TITLE_CAP),
                           (_first_line(brief), BRIEF_LINE_CAP),
                           (_first_line(summary if isinstance(summary, str) else None), BRIEF_LINE_CAP)):
        if candidate:
            return _clip_title(candidate, cap)
    return FALLBACK_TITLE


ITEM_WORDS = {"done": "Done", "not_done": "Not done", "unverified": "Unverified"}

WRAPPED_UP_LINE = "Wrapped up at its limit; it may be unfinished."
CHECKS_HEADING = "## What the runner checked"
NEXT_STEPS_HEADING = "## For a later run"
NOTHING_FOUND = "Nothing found."
RUN_LINE = "Result from run {run}."
PR_UPDATE_FAILED = "The runner could not update the pull request description; it may still show an earlier result."


def render(result: dict[str, Any] | None, fallback: str = "", wrapped_up: str | None = None,
           checks: dict[str, Any] | None = None) -> str:
    lead = [WRAPPED_UP_LINE, ""] if wrapped_up and wrapped_up != "asked" else []
    if not isinstance(result, dict):
        lines = [*lead, fallback or "The run returned no typed result.", ""]
        result = {}
    else:
        lines = [*lead, str(result.get("summary") or "").strip() or "(no summary)", ""]
    asked = ([i for i in result.get("items") or [] if isinstance(i, dict)]
             if isinstance(result.get("items"), list) else [])
    if asked:
        lines += ["## What was asked (as the session reports it)",
                  *[f"- {ITEM_WORDS.get(str(i.get('status')), str(i.get('status')))}: {i.get('label')}"
                    + (f" ({i['where']})" if i.get("where") else "") for i in asked[:100]], ""]
    for key, title in (("files_changed", "Files changed"), ("tests_run", "Tests run"),
                       ("open_questions", "Open questions")):
        items = result.get(key) or []
        if isinstance(items, list) and items:
            lines += [f"## {title}", *[f"- {item}" for item in items[:100]], ""]
    findings = (checks or {}).get("findings") or []
    lines += [CHECKS_HEADING, *([f"- {finding['text']}" for finding in findings] or [NOTHING_FOUND]), ""]
    next_steps = result.get("next_steps") or []
    if next_steps:
        lines += [NEXT_STEPS_HEADING, *[f"- {step}" for step in next_steps], ""]
    lines.append("Opened by the session runner.")
    return "\n".join(lines)[:60000]


def merge_base(cwd: str, start: str, branch: str) -> str:
    try:
        found = git(cwd, "merge-base", start, f"refs/heads/{branch}").strip()
    except GitSafetyError:
        raise
    except GitError:
        return start
    return found if HEX.match(found) else start


def hand_off(cwd: str, slug: str, base: str, result: dict[str, Any] | None, forge: Forge,
             since: str | None = None, existed: frozenset[str] | None = None, title: str | None = None,
             brief: str | None = None, latest: str | None = None, reuse: bool = False,
             wrapped_up: str | None = None, checks: dict[str, Any] | None = None,
             run: int = 1) -> dict[str, Any]:
    branch = branch_for(slug)
    current = git(cwd, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if current != branch:
        raise GitError(f"the worktree is on {current!r}, not {branch!r}")
    heading = pr_title(title, brief, result, latest=latest)
    start = merge_base(cwd, since or base, branch)
    if git(cwd, "status", *WORKTREE_DIFF, "--porcelain", "--untracked-files=all").strip():
        skip = placeholders.ignorable(cwd, existed)
        skip = (*skip, *handoff_check.junk_paths(worktree_status(cwd, start)))
        git(cwd, "add", "--all", "--", ".")
        if skip:
            git(cwd, "rm", "-r", "--cached", "--quiet", "--ignore-unmatch", "--", *skip,
                extra_env={"GIT_LITERAL_PATHSPECS": "1"})
        if git(cwd, "diff", "--cached", "--no-ext-diff", "--name-only", "-z").strip("\0"):
            git(cwd, "commit", "--no-verify", "--quiet", "-m", heading, extra_env=IDENTITY)
            start = merge_base(cwd, since or base, branch)
    ahead = int(git(cwd, "rev-list", "--count", f"{start}..{branch}").strip() or 0)
    if ahead == 0:
        return {"handoff": "nothing to push", "branch": branch}
    head_sha = git(cwd, "rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}").strip()
    if not HEX.match(head_sha):
        raise GitError("git rev-parse gave no commit id")
    stats = change_stats(cwd, start, head_sha)
    forge.push(cwd, branch)
    opened = forge.open_pr_for(branch) if reuse else None
    body = render(result, wrapped_up=wrapped_up, checks=checks)
    if reuse:
        body = f"{RUN_LINE.format(run=run)}\n\n{body}"
    update_failed = False
    if opened is None:
        opened = forge.open_pr(branch, base, heading, body)
    else:
        try:
            forge.update_pr(opened.number, body)
        except Exception as exc:
            update_failed = True
            if checks is None:
                checks = {"version": handoff_check.VERSION, "findings": []}
            text = f"{PR_UPDATE_FAILED} {exc}" if isinstance(exc, RateLimited) else PR_UPDATE_FAILED
            checks.setdefault("findings", []).append({"key": "pr_update_failed", "text": text})
    return {"handoff": "pull request", "branch": branch, "commits": ahead, "pr_url": opened.url,
            "pr_number": opened.number, "head_sha": head_sha, "base": base, "stats": stats,
            **({"checks": checks} if update_failed else {})}


def worktree_status(cwd: str, base: str | None = None) -> handoff_check.Status:
    with checked(cwd):
        start = git(cwd, "merge-base", base or "HEAD", "HEAD").strip()
        changed = git(cwd, "diff", *WORKTREE_DIFF, "--no-ext-diff", "--no-textconv", "--no-renames",
                      "--name-only", "-z", start)
        untracked = git(cwd, "ls-files", "--others", "--exclude-standard", "-z")
        tracked = git(cwd, "ls-tree", "-r", "--name-only", "-z", start)
    others = tuple(p for p in untracked.split("\0") if p)
    return handoff_check.Status(tuple(sorted(set(p for p in changed.split("\0") if p) | set(others))),
                                others, frozenset(p for p in tracked.split("\0") if p))


def has_changes(cwd: str, base: str) -> bool:
    return bool(git(cwd, "--no-optional-locks", "-c", "core.quotePath=true", "status",
                    *WORKTREE_DIFF, "--porcelain=v1", "--untracked-files=all")
                or git(cwd, "rev-list", "--max-count=1", f"{base}..HEAD"))


def change_stats(cwd: str, start: str, tip: str) -> dict[str, Any]:
    out = git(cwd, "diff", "--no-ext-diff", "--no-textconv", "--no-color", "--no-renames", "--numstat", "-z",
              start, tip, "--")
    files = insertions = deletions = 0
    paths: list[str] = []
    for record in out.split("\0"):
        if not record:
            continue
        added, deleted, path = record.split("\t", 2)
        files += 1
        insertions += int(added) if added.isdigit() else 0
        deletions += int(deleted) if deleted.isdigit() else 0
        if len(paths) < PATHS_CAP:
            paths.append(path[:PATH_CLIP])
    return {"files": files, "insertions": insertions, "deletions": deletions, "paths": paths,
            "paths_truncated": files > PATHS_CAP}

