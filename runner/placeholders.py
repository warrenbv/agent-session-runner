"""Empty placeholder files that Claude Code's Linux sandbox can leave in the worktree.

The sandbox protects certain configuration paths in the working directory by mounting over them, and
creates an empty file or directory to mount over where none exists. The CLI removes them when its last
sandboxed command ends, but not when it exits while one is still running or is killed. Without this
module a hand-off could commit two dozen empty files.

A path counts as a placeholder only when it is in PROTECTED, untracked, empty (never a symlink) and did
not exist when the session started. Nothing is unlinked while the CLI may be alive, because unlinking
a mount point would lift the write protection for a command still running: between turns the
snapshots leave placeholders out, and after the provider has closed `settle` waits until no process
uses the worktree before `sweep` removes them.
"""

from __future__ import annotations

import logging
import os
import stat
import time

from runner.gitdiff import GitError, git

log = logging.getLogger("runner.placeholders")

PROTECTED = (
    ".gitconfig", ".gitmodules", ".bashrc", ".bash_profile", ".zshrc", ".zprofile", ".profile",
    ".ripgreprc", ".mcp.json", ".vscode", ".idea",
    ".claude/settings.json", ".claude/settings.local.json", ".claude/skills", ".claude/hooks",
    ".claude/commands", ".claude/agents", ".claude/launch.json", ".claude/workflows", ".claude/routines",
    ".claude/output-styles", ".claude/scheduled_tasks.json", ".claude/loop.md",
)
PARENTS = (".claude",)


def present(cwd: str) -> frozenset[str]:
    return frozenset(name for name in (*PROTECTED, *PARENTS) if os.path.lexists(os.path.join(cwd, name)))


def _empty(path: str) -> bool:
    try:
        info = os.lstat(path)
        if stat.S_ISREG(info.st_mode):
            return info.st_size == 0
        if stat.S_ISDIR(info.st_mode):
            return not os.listdir(path)
    except OSError:
        pass
    return False


def ignorable(cwd: str, existed: frozenset[str] | None) -> list[str]:
    if existed is None:
        return []
    candidates = [name for name in PROTECTED if name not in existed and _empty(os.path.join(cwd, name))]
    if not candidates:
        return []
    try:
        tracked = set(git(cwd, "ls-files", "-z", "--", *candidates,
                          extra_env={"GIT_LITERAL_PATHSPECS": "1"}).split("\0"))
    except GitError as exc:
        log.warning("placeholders not recognised: %s", exc)
        return []
    return [name for name in candidates if name not in tracked]


def _ancestors() -> set[str]:
    pids, pid = set(), os.getpid()
    while pid > 1 and str(pid) not in pids:
        pids.add(str(pid))
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
                pid = int(handle.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def busy(cwd: str) -> bool:
    if not os.path.isdir("/proc/self"):
        return False
    root = os.path.realpath(cwd)
    inside = root.rstrip("/") + "/"
    own_ns = os.readlink("/proc/self/ns/mnt")
    skip = _ancestors()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid in skip:
            continue
        try:
            where = os.readlink(f"/proc/{pid}/cwd")
            if where == root or where.startswith(inside):
                return True
            if os.readlink(f"/proc/{pid}/ns/mnt") == own_ns:
                continue
            with open(f"/proc/{pid}/mountinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    fields = line.split(" ")
                    if len(fields) > 4 and fields[4].startswith(inside):
                        return True
        except OSError:
            continue
    return False


def settle(cwd: str, wait: float = 10.0, step: float = 0.25) -> bool:
    deadline = time.monotonic() + wait
    while busy(cwd):
        if time.monotonic() >= deadline:
            return False
        time.sleep(step)
    return True


# Only after the provider has closed and no process still uses the worktree: unlinking a mount point
# while a sandboxed command runs would lift that path's write protection.
def sweep(cwd: str, existed: frozenset[str] | None) -> list[str]:
    removed = []
    for name in ignorable(cwd, existed):
        path = os.path.join(cwd, name)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                os.rmdir(path)
            else:
                os.unlink(path)
            removed.append(name)
        except OSError as exc:
            log.warning("placeholder %s kept: %s", name, type(exc).__name__)
    for parent in PARENTS if existed is not None else ():
        path = os.path.join(cwd, parent)
        if parent not in existed and _empty(path) and os.path.isdir(path):
            try:
                os.rmdir(path)
            except OSError:
                pass
    if removed:
        log.info("removed %d sandbox placeholder(s): %s", len(removed), ", ".join(removed))
    return removed

