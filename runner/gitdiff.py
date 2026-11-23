"""Git with fixed arguments and no shell, and the turn-end diff.

Every call is `git` plus FIXED plus literal arguments, run with an argument list, a reduced
environment and a timeout. fsmonitor, external diff, textconv and the pager are off and hooks point
nowhere. This is the only module allowed to start processes (a test enforces it).

Git is never left to discover the repository. Every call names its git directory, common directory and
worktree in the environment. For a plain checkout they come from the working directory. For a linked
worktree they come from whatever created it (`--git-dir` and `--git-common-dir`), and the pointer
files are checked against them first. Before a group of git calls the repository's own configuration
is listed without following includes, and any key that could make git run a program or change where it
connects is refused (CONFIG_RULES). System and user git configuration are not read at all.

The turn diff: at a turn's start the worktree is snapshotted into a tree object through a private index
file (tracked and untracked files, .gitignore respected, the real index untouched). At its end it is
snapshotted again and the two trees are diffed. With `file_diffs`, the whole session's change against
its base is also posted, one event per file.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from runner.config import git_environment
from runner.events import clip, size

FIXED = ("--no-pager", "-c", "core.fsmonitor=false", "-c", "diff.external=", "-c", "diff.noprefix=false",
         "-c", "core.hooksPath=/dev/null", "-c", "core.pager=", "-c", "color.ui=false",
         "-c", "submodule.recurse=false", "-c", "fetch.recurseSubmodules=false",
         "-c", "push.recurseSubmodules=no", "-c", "status.submoduleSummary=false", "-c", "diff.submodule=short")
DIFF = ("--no-ext-diff", "--no-textconv", "--no-color")
WORKTREE_DIFF = ("--ignore-submodules=dirty",)
PATCH_CAP = 12000
STAT_CAP = 3000
TIMEOUT = 120

# Each pattern matches the whole lower-cased key, subsections included. The second field lists the
# only values that are allowed for that key.
CONFIG_RULES = (
    (r"alias\..+", ()),                                       # an alias can run a shell command
    (r"branch\..+\.mergeoptions", ()),                        # can pick an external merge strategy
    (r"core\.alternaterefscommand", ()),                      # lists an alternate's refs through a command
    (r"core\.askpass", ()),                                   # runs a password prompt program
    (r"core\.editor|sequence\.editor", ()),                   # starts an editor
    (r"core\.fsmonitor", (b"false", b"no", b"off", b"0", b"")),  # can run a filesystem monitor
    (r"core\.gitproxy|core\.sshcommand", ()),                 # picks a transport command
    (r"core\.hookspath", (b"/dev/null",)),                     # says where git looks for hooks
    (r"core\.pager|pager\..+", ()),                           # can pick a pager command
    (r"credential\.(?:.+\.)?helper", ()),                     # runs a program for credentials
    (r"diff\.external|diff\..+\.(?:command|textconv)", ()),   # runs a diff or conversion program
    (r"extensions\.worktreeconfig", ()),                      # opens a second config file in a linked gitdir
    (r"extensions\.partialclone|remote\..+\.promisor", ()),   # an object read can start a fetch
    (r"filter\..+", ()),                                      # attributes can pick clean, smudge or process programs
    (r"gpg\.(?:.+\.)?program|gpg\.ssh\.defaultkeycommand", ()),  # runs a signer or its key lookup
    (r"hook\..+", ()),                                        # configured hooks run without a hooks directory
    (r"http\..+", ()),                                        # proxies, TLS checks, certificates, headers, redirects
    (r"remote\..+\.proxy", ()),                               # reroutes a connection
    (r"include\..+|includeif\..+", ()),                       # opens another configuration file
    (r"interactive\.difffilter", ()),                         # pipes diffs through a command
    (r"merge\..+\.driver", ()),                               # attributes can pick a merge program
    (r"receive\..+|uploadpack\.packobjectshook", ()),          # receive hooks or a pack-objects substitute
    (r"remote\..+\.(?:receivepack|uploadpack|vcs)", ()),       # picks a remote helper or endpoint command
    (r"remote\..+\.pushurl|url\..+\.(?:insteadof|pushinsteadof)", ()),  # changes where a push or fetch goes
    (r"remote\..*://.*", ()),                                 # a remote named by a URL captures a push to that URL
    (r"tar\..+\.command", ()),                                # can run a compressor for an archive
)
_CONFIG_RULES = tuple((re.compile(pattern, re.DOTALL), allowed) for pattern, allowed in CONFIG_RULES)

# The refusals name a step and what to repair, never git's output.
BAD_REGISTRATION = ("Git found no single registration for this worktree in the project's checkout. "
                    "Ask the machine's administrator to restore its Git registration before restarting the session.")
BAD_DIRECTORY = ("The registered Git directory is missing or is not a direct directory under the checkout's "
                 ".git/worktrees folder. Ask the machine's administrator to restore it before restarting the session.")
BAD_POINTER = ("Git could not verify the worktree's .git as a regular file at the expected path naming the "
               "registered Git directory. "
               "Ask the machine's administrator to restore it before restarting the session.")
BAD_COMMON_DIR = ("The commondir file is missing, is not a regular file, or names a different common directory. "
                  "Ask the machine's administrator to restore it before restarting the session.")
BAD_CONFIG = ("The checkout's Git config is missing, unreadable, or invalid. "
              "Ask the machine's administrator to repair the checkout's config before restarting the session.")
BLOCKED_CONFIG = ("Git refused {key}. Ask the machine's administrator to remove that key from the project's Git "
                  "config before restarting the session.")


class GitError(Exception):
    ...


class GitSafetyError(GitError):
    ...


def _regular_file(path: str) -> Any:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise OSError("expected a regular file")
    return os.fdopen(fd, "rb")


# Reads a git metadata file (`.git` or `commondir`) only to compare it with the path already known.
def _names(path: str, target: str, prefix: bytes = b"", *, resolve: bool = True) -> bool:
    try:
        with _regular_file(path) as handle:
            raw = handle.read(4097).rstrip(b"\r\n")
        if not raw.startswith(prefix) or len(raw) > 4096 or b"\0" in raw or not raw[len(prefix):]:
            return False
        named = os.path.join(os.path.dirname(path), os.fsdecode(raw[len(prefix):]))
        return (os.path.realpath(named) if resolve else os.path.abspath(named)) == target
    except OSError:
        return False


# A linked worktree's three paths, as given by whatever created it.
@dataclass(frozen=True)
class Worktree:
    path: str
    git_dir: str
    common_dir: str

    # Finds the registration from the checkout's side, by the backlink in its `gitdir` file.
    @classmethod
    def registered(cls, checkout: str, path: str) -> Worktree:
        path = os.path.abspath(path)
        common_dir = os.path.join(os.path.realpath(checkout), ".git")
        registrations = os.path.join(common_dir, "worktrees")
        if os.path.realpath(registrations) != registrations:
            raise GitSafetyError(BAD_DIRECTORY)
        try:
            with os.scandir(registrations) as entries:
                found = [entry.path for entry in entries if entry.is_dir(follow_symlinks=False)
                         and _names(os.path.join(entry.path, "gitdir"), os.path.join(path, ".git"), resolve=False)]
        except OSError:
            found = []
        if len(found) != 1:
            raise GitSafetyError(BAD_REGISTRATION)
        worktree = cls(path, found[0], common_dir)
        worktree.verify()
        return worktree

    def verify(self) -> None:
        root = os.path.join(self.common_dir, "worktrees")
        if (os.path.dirname(self.git_dir) != root or not os.path.isdir(self.git_dir)
                or os.path.realpath(self.git_dir) != self.git_dir):
            raise GitSafetyError(BAD_DIRECTORY)
        if os.path.realpath(self.path) != self.path or not _names(
                os.path.join(self.path, ".git"), self.git_dir, b"gitdir: "):
            raise GitSafetyError(BAD_POINTER)
        if not _names(os.path.join(self.git_dir, "commondir"), self.common_dir):
            raise GitSafetyError(BAD_COMMON_DIR)

    def environment(self) -> dict[str, str]:
        return {"GIT_DIR": self.git_dir, "GIT_COMMON_DIR": self.common_dir, "GIT_WORK_TREE": self.path}


# Context variables, so the binding also reaches git calls made through asyncio.to_thread.
_WORKTREE: ContextVar[Worktree | None] = ContextVar("runner_git_worktree", default=None)
_CHECKED: ContextVar[str | None] = ContextVar("runner_git_checked", default=None)


@contextmanager
def use_worktree(worktree: Worktree | None) -> Iterator[None]:
    token = _WORKTREE.set(worktree)
    try:
        yield
    finally:
        _WORKTREE.reset(token)


# The bound worktree, a plain checkout with a real `.git` directory, or a bare repository. A `.git`
# file (a linked worktree) with no binding is refused.
def _repository(cwd: str) -> tuple[str | None, dict[str, str]]:
    path = os.path.abspath(cwd)
    worktree = _WORKTREE.get()
    if worktree and path == worktree.path:
        return worktree.common_dir, worktree.environment()
    dotgit = os.path.join(path, ".git")
    if os.path.lexists(dotgit):
        if not os.path.isdir(dotgit) or os.path.islink(dotgit):
            raise GitSafetyError(BAD_REGISTRATION)
        return dotgit, {"GIT_DIR": dotgit, "GIT_COMMON_DIR": dotgit, "GIT_WORK_TREE": path}
    if os.path.isdir(os.path.join(path, "objects")) and os.path.isfile(os.path.join(path, "HEAD")):
        return path, {"GIT_DIR": path, "GIT_COMMON_DIR": path}
    return None, {"GIT_CEILING_DIRECTORIES": os.path.dirname(path)}


def verify_repository(cwd: str) -> None:
    worktree = _WORKTREE.get()
    if worktree and os.path.abspath(cwd) == worktree.path:
        worktree.verify()
    else:
        _repository(cwd)


# Lists the common config through stdin, from a descriptor opened without following a link, so git
# neither discovers a repository nor follows an include while reading it.
def _check_config(common_dir: str) -> None:
    try:
        with _regular_file(os.path.join(common_dir, "config")) as config:
            result = subprocess.run(["git", "--no-pager", "config", "--file", "-", "--no-includes", "--null", "--list"],
                                    cwd="/", env=git_environment(), stdin=config, capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitSafetyError(BAD_CONFIG) from exc
    if result.returncode:
        raise GitSafetyError(BAD_CONFIG)
    for entry in result.stdout.split(b"\0"):
        raw_key, separator, value = entry.partition(b"\n")
        key = os.fsdecode(raw_key)
        setting = value if separator else None
        if key == "core.fsmonitor" and setting is not None:
            setting = setting.lower()
        for pattern, allowed in _CONFIG_RULES:
            if pattern.fullmatch(key.lower()) and setting not in allowed:
                # A URL subsection can carry a user name. Control characters are escaped, and the key is cut
                # so that the sentence fits the hand-off's step field.
                display_key = re.sub(r"(?<=://)[^/@\s]+@", "<redacted>@", key)
                raise GitSafetyError(BLOCKED_CONFIG.format(key=clip(ascii(display_key), 62)))


def check_repository(cwd: str) -> None:
    verify_repository(cwd)
    common_dir, _ = _repository(cwd)
    if common_dir is not None:
        _check_config(common_dir)


# One check for a group of git calls that make up one operation.
@contextmanager
def checked(cwd: str) -> Iterator[None]:
    path = os.path.abspath(cwd)
    if _CHECKED.get() == path:
        yield
        return
    check_repository(cwd)
    token = _CHECKED.set(path)
    try:
        yield
    finally:
        _CHECKED.reset(token)


PROTOCOLS = ("file", "git", "ssh", "http", "https", "ftp", "ftps")


def _environment(cwd: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    _, paths = _repository(cwd)
    extra = dict(extra or {})
    # Git's own transports only, local bare remotes included, and no remote helper. No lazy fetch. A caller
    # may narrow the transports for one call, never widen them.
    asked = extra.pop("GIT_ALLOW_PROTOCOL", None)
    allowed = PROTOCOLS if asked is None else [p for p in asked.split(":") if p in PROTOCOLS]
    return git_environment({**extra, **paths, "GIT_ALLOW_PROTOCOL": ":".join(allowed) or "none",
                            "GIT_NO_LAZY_FETCH": "1"})


def git(cwd: str, *args: str, extra_env: dict[str, str] | None = None, check: bool = True) -> str:
    with checked(cwd):
        try:
            done = subprocess.run(["git", *FIXED, *args], cwd=cwd, env=_environment(cwd, extra_env),
                                  capture_output=True, text=True, timeout=TIMEOUT, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitError(f"git {args[0]}: {type(exc).__name__}") from None
    if check and done.returncode != 0:
        raise GitError(f"git {args[0]} exited {done.returncode}")
    return done.stdout


def snapshot(cwd: str, exclude: Sequence[str] = ()) -> str:
    with checked(cwd), tempfile.TemporaryDirectory(prefix="runner-index-") as scratch:
        index = {"GIT_INDEX_FILE": os.path.join(scratch, "index")}
        has_head = git(cwd, "rev-parse", "--verify", "--quiet", "HEAD", check=False).strip()
        if has_head:
            git(cwd, "read-tree", "HEAD", extra_env=index)
        git(cwd, "add", "--all", "--", ".", extra_env=index)
        if exclude:
            git(cwd, "rm", "-r", "--cached", "--quiet", "--ignore-unmatch", "--", *exclude,
                extra_env={**index, "GIT_LITERAL_PATHSPECS": "1"})
        return git(cwd, "write-tree", extra_env=index).strip()


def diff(cwd: str, before: str, after: str) -> dict[str, Any] | None:
    if before == after:
        return None
    with checked(cwd):
        numstat = git(cwd, "diff", *DIFF, "--numstat", before, after)
        stat = git(cwd, "diff", *DIFF, "--stat=200", before, after)
        patch = git(cwd, "diff", *DIFF, "--patch", before, after)
    files = [line.split("\t", 2)[2] for line in numstat.splitlines() if line.count("\t") >= 2]
    return {"scope": "turn", "from": before, "to": after, "files": len(files), "paths": files[:200],
            "stat": clip(stat, STAT_CAP), "patch": clip(patch, PATCH_CAP), "patch_truncated": len(patch) > PATCH_CAP}


SESSION = "session"
SESSION_FILE = "session_file"
FILE_PATCH_BOUND = 15000
CHANGE_CEILING = 400 * 1024
SUMMARY_BOUND = 15000
FILES_CAP = 300
STATUS = {"A": "added", "D": "deleted", "M": "changed", "T": "changed", "R": "renamed", "C": "added"}
_PLAIN = ("-c", "core.quotePath=false")


def _raw_entries(cwd: str, base: str, after: str) -> list[dict[str, Any]]:
    out = git(cwd, *_PLAIN, "diff", *DIFF, "--raw", "-z", "-M", "--no-abbrev", base, after)
    fields = out.split("\0")
    entries: list[dict[str, Any]] = []
    i = 0
    while i < len(fields) - 1:
        head = fields[i]
        if not head.startswith(":"):
            i += 1
            continue
        parts = head[1:].split(" ")
        if len(parts) < 5:
            break
        old_mode, new_mode, old_sha, new_sha, letter = parts[0], parts[1], parts[2], parts[3], parts[4]
        code = letter[:1]
        if code in ("R", "C"):
            old_path, path = fields[i + 1], fields[i + 2]
            i += 3
        else:
            old_path, path = None, fields[i + 1]
            i += 2
        sig = f"{old_sha[:12]}{new_sha[:12]}{old_mode[-3:]}{new_mode[-3:]}{code}"
        entries.append({"path": path, "old_path": old_path if code == "R" else None,
                        "status": STATUS.get(code, "changed"), "sig": sig})
    return entries


def _numstat(cwd: str, base: str, after: str) -> dict[str, tuple[int | None, int | None]]:
    out = git(cwd, *_PLAIN, "diff", *DIFF, "--numstat", "-z", "-M", base, after)
    fields = out.split("\0")
    counts: dict[str, tuple[int | None, int | None]] = {}
    i = 0
    while i < len(fields):
        head = fields[i]
        if head.count("\t") < 2:
            i += 1
            continue
        added, removed, path = head.split("\t", 2)
        if path == "":
            path = fields[i + 2] if i + 2 < len(fields) else ""
            i += 3
        else:
            i += 1
        counts[path] = (int(added) if added.isdigit() else None, int(removed) if removed.isdigit() else None)
    return counts


def _file_patch(cwd: str, base: str, after: str, entry: dict[str, Any]) -> str:
    paths = [entry["old_path"], entry["path"]] if entry["old_path"] else [entry["path"]]
    return git(cwd, *_PLAIN, "diff", *DIFF, "--patch", "-M", base, after, "--", *paths,
               extra_env={"GIT_LITERAL_PATHSPECS": "1"})


def file_event(entry: dict[str, Any], counts: tuple[int | None, int | None], patch: str | None,
               to: str, bound: int = FILE_PATCH_BOUND) -> dict[str, Any]:
    added, removed = counts
    binary = added is None
    payload = {"scope": SESSION_FILE, "to": to, "path": entry["path"], "old_path": entry["old_path"],
               "status": entry["status"], "added": added or 0, "removed": removed or 0, "binary": binary,
               "sig": entry["sig"], "patch": patch if patch is not None else "", "too_large": False}
    if size(payload) > bound:
        payload["patch"], payload["too_large"] = "", True
    return payload


PART_DIGITS = 99999


def _json_bytes(char: str, cache: dict[str, int]) -> int:
    found = cache.get(char)
    if found is None:
        found = cache[char] = len(json.dumps(char, ensure_ascii=False).encode("utf-8")) - 2
    return found


def split_patch(patch: str, room: int) -> list[str]:
    if room < 64:
        return []
    cache: dict[str, int] = {}
    pieces: list[str] = []
    start = 0
    while start < len(patch):
        used, end, newline = 0, start, -1
        while end < len(patch):
            width = _json_bytes(patch[end], cache)
            if used + width > room:
                break
            used += width
            if patch[end] == "\n":
                newline = end
            end += 1
        if end < len(patch) and newline >= start and newline + 1 - start >= (end - start) // 2:
            end = newline + 1
        pieces.append(patch[start:end])
        start = end
    return pieces


def file_events(entry: dict[str, Any], counts: tuple[int | None, int | None], patch: str | None, to: str,
                parts_allowed: bool, bound: int = FILE_PATCH_BOUND) -> list[dict[str, Any]]:
    single = file_event(entry, counts, patch, to, bound)
    if not single["too_large"] or patch is None or not parts_allowed:
        return [single]
    shell = {**single, "patch": "", "too_large": False}
    pieces = split_patch(patch, bound - size({**shell, "part": PART_DIGITS, "parts": PART_DIGITS}))
    if len(pieces) < 2:
        return [single]
    return [{**shell, "patch": piece, "part": n, "parts": len(pieces)} for n, piece in enumerate(pieces, 1)]


def session_diff(cwd: str, base: str, after: str, sent: dict[str, str],
                 sizes: dict[str, tuple[str, int]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with checked(cwd):
        return _session_diff(cwd, base, after, sent, sizes)


def _session_diff(cwd: str, base: str, after: str, sent: dict[str, str],
                  sizes: dict[str, tuple[str, int]] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = _raw_entries(cwd, base, after)
    counts = _numstat(cwd, base, after)
    sizes = {} if sizes is None else sizes
    files: list[dict[str, Any]] = []
    change = 0
    for n, entry in enumerate(entries):
        if n >= FILES_CAP:
            break
        path, sig = entry["path"], entry["sig"]
        found = counts.get(path, (0, 0))
        again = sent.get(path) != sig
        patch = None
        if found[0] is not None and (again or sizes.get(path, ("", 0))[0] != sig):
            patch = _file_patch(cwd, base, after, entry)
        if patch is not None:
            sizes[path] = (sig, len(patch.encode("utf-8")))
        elif found[0] is None:
            sizes[path] = (sig, 0)
        change += sizes[path][1]
        if not again:
            continue
        files.extend(file_events(entry, found, patch, after, parts_allowed=change <= CHANGE_CEILING))
        sent[path] = sig
    listed: list[dict[str, Any]] = []
    total_added = total_removed = 0
    summary: dict[str, Any] = {"scope": SESSION, "base": base, "to": after, "files": len(entries),
                               "added": 0, "removed": 0, "listed": listed, "complete": True}
    for n, entry in enumerate(entries):
        added, removed = counts.get(entry["path"], (0, 0))
        total_added += added or 0
        total_removed += removed or 0
        if n >= FILES_CAP or not summary["complete"]:
            summary["complete"] = False
            continue
        row = {"path": entry["path"], "status": entry["status"], "added": added or 0, "removed": removed or 0,
               "binary": added is None, "sig": entry["sig"]}
        if entry["old_path"]:
            row["old_path"] = entry["old_path"]
        listed.append(row)
        if size(summary) > SUMMARY_BOUND:
            listed.pop()
            summary["complete"] = False
    summary["added"], summary["removed"] = total_added, total_removed
    return files, summary

