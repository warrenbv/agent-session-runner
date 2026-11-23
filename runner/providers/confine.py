"""What a coding session may read: its worktree, the git directory that worktree's objects live in,
and a temp folder of its own.

The scope denies the service user's home and the shared temp directories, then re-allows those three
paths inside them. Denying the parents rather than naming each project keeps a project cloned later
out of reach too. Outside the home and temp directories nothing changes (toolchains, system libraries,
/etc). The same scope feeds both providers' settings. Research sessions are not changed by this: they
already run without a shell, or with reads held to their scratch directory. See docs/security-model.md,
"Confinement".
"""

from __future__ import annotations

import os
import pwd
import shutil
import tempfile
from dataclasses import dataclass

TEMP_PARENT: str | None = None
# Short on purpose: Claude Code appends `/claude-<uid>` to its temp root and falls back to the shared
# /tmp/claude-<uid> when the whole path is longer than CLAUDE_TMPDIR_MAX bytes.
TEMP_PREFIX = "rs-"
SHARED_TEMP = ("/tmp", "/var/tmp")


def home() -> str:
    try:
        return os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        return os.path.realpath(os.path.expanduser("~"))


# A worktree's `.git` is a file naming `<repo>/.git/worktrees/<id>`, whose `commondir` names
# `<repo>/.git`. Git reads objects, refs and config there for every status, diff and commit, so that
# directory must stay readable. The main checkout's working files need not.
def git_common_dir(cwd: str) -> str | None:
    dot = os.path.join(cwd, ".git")
    if os.path.isdir(dot):
        return os.path.realpath(dot)
    if not os.path.isfile(dot):
        return None
    try:
        with open(dot, encoding="utf-8") as handle:
            line = handle.read(4096).strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not line.startswith("gitdir:"):
        return None
    gitdir = os.path.join(cwd, line.removeprefix("gitdir:").strip())
    try:
        with open(os.path.join(gitdir, "commondir"), encoding="utf-8") as handle:
            return os.path.realpath(os.path.join(gitdir, handle.read(4096).strip()))
    except (OSError, UnicodeDecodeError):
        return os.path.realpath(gitdir)


@dataclass(frozen=True)
class ReadScope:

    worktree: str
    git_dir: str | None
    temp: str
    denied: tuple[str, ...]

    @property
    def readable(self) -> tuple[str, ...]:
        out: list[str] = []
        for path in (self.worktree, self.git_dir, self.temp):
            if path and path not in out:
                out.append(path)
        return tuple(out)


def read_scope(cwd: str, temp: str, user_home: str | None = None) -> ReadScope:
    worktree = os.path.realpath(cwd)
    temp = os.path.realpath(temp)
    denied = tuple(dict.fromkeys(os.path.realpath(p) for p in (user_home or home(), *SHARED_TEMP)))
    git_dir = git_common_dir(worktree)
    for path in (worktree, git_dir, temp):
        if path and any(path == d or d.startswith(path.rstrip("/") + "/") for d in denied):
            raise ValueError("a session's own directory may not be the home, a temp root, or a parent of one")
    return ReadScope(worktree=worktree, git_dir=git_dir, temp=temp, denied=denied)


class SessionTemp:

    made: list[str] = []

    def __init__(self, parent: str | None = None) -> None:
        self.path = os.path.realpath(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=parent or TEMP_PARENT))
        os.chmod(self.path, 0o700)
        SessionTemp.made.append(self.path)

    def close(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

