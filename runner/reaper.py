"""Ending what a session left running, at its end only.

When the runner runs inside its own systemd scope (named SCOPE), everything it started is in the same
cgroup, whatever it did with process groups or reparenting. At the session's end, after the provider
has closed, `Reaper.sweep` reads the runner's own cgroup (cgroup v2) and signals every process but the
runner and its ancestors: TERM, a short grace, then KILL, until the group is empty. A runner started
by hand, outside such a scope, sweeps nothing. Off Linux there is no group to read and `sweep`
returns None.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

log = logging.getLogger("runner.reaper")

SCOPE = re.compile(r"^runner-session-\d+\.scope$")
CGROUP_ROOT = "/sys/fs/cgroup"
GRACE = 3.0
STEP = 0.1
KILL_ROUNDS = 5
NAME_CAP = 15


@dataclass
class Swept:
    ended: int = 0
    names: dict[str, int] = field(default_factory=dict)
    killed: int = 0
    left: int = 0

    def add(self, other: "Swept | None") -> None:
        if other is None:
            return
        self.ended += other.ended
        self.killed += other.killed
        self.left = other.left
        for name, count in other.names.items():
            self.names[name] = self.names.get(name, 0) + count

    def sentence(self) -> str:
        listed = ", ".join(f"{name} ({count})" if count > 1 else name for name, count in sorted(self.names.items()))
        noun = "process" if self.ended == 1 else "processes"
        text = f"At the session's end the runner stopped {self.ended} {noun} it had left running: {listed}."
        if self.killed:
            text += f" {self.killed} of them did not stop on SIGTERM within {GRACE:g} s and needed SIGKILL."
        if self.left:
            text += f" {self.left} could not be stopped; whatever started the session's scope should stop it when the runner exits."
        return text


class Group(Protocol):

    def members(self) -> set[int] | None: ...
    def name(self, pid: int) -> str: ...
    def signal(self, pid: int, sig: int) -> None: ...


def _clean(name: str) -> str:
    return "".join(ch for ch in name if ch.isprintable())[:NAME_CAP] or "?"


class Cgroup:

    def __init__(self, proc: str = "/proc", root: str = CGROUP_ROOT) -> None:
        self.proc, self.root = proc, root
        self._path: str | None | bool = False

    def path(self) -> str | None:
        if self._path is False:
            self._path = None
            try:
                with open(os.path.join(self.proc, "self", "cgroup"), encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("0::"):
                            rel = line[3:].strip()
                            if SCOPE.match(os.path.basename(rel)):
                                self._path = os.path.join(self.root, rel.lstrip("/"))
                            break
            except OSError:
                pass
        return self._path or None

    def members(self) -> set[int] | None:
        path = self.path()
        if path is None or not os.path.isdir(path):
            return None
        pids: set[int] = set()
        for where, _dirs, files in os.walk(path):
            if "cgroup.procs" not in files:
                continue
            try:
                with open(os.path.join(where, "cgroup.procs"), encoding="utf-8") as handle:
                    pids.update(int(line) for line in handle if line.strip().isdigit())
            except OSError:
                continue
        return {pid for pid in pids if not self._zombie(pid)}

    def _zombie(self, pid: int) -> bool:
        try:
            with open(os.path.join(self.proc, str(pid), "stat"), encoding="utf-8", errors="replace") as handle:
                return handle.read().rsplit(")", 1)[1].split()[0] in ("Z", "X")
        except (OSError, IndexError):
            return True

    def name(self, pid: int) -> str:
        try:
            with open(os.path.join(self.proc, str(pid), "comm"), encoding="utf-8", errors="replace") as handle:
                return _clean(handle.read().strip())
        except OSError:
            return "?"

    def signal(self, pid: int, sig: int) -> None:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def ancestors(proc: str = "/proc") -> set[int]:
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            with open(os.path.join(proc, str(pid), "stat"), encoding="utf-8", errors="replace") as handle:
                pid = int(handle.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


class Reaper:
    def __init__(self, group: Group | None = None, spare: Callable[[], set[int]] = ancestors,
                 grace: float = GRACE, step: float = STEP, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.group = group if group is not None else Cgroup()
        self.spare, self.grace, self.step, self.sleep, self.clock = spare, grace, step, sleep, clock

    def _alive(self, spared: set[int]) -> set[int] | None:
        members = self.group.members()
        return None if members is None else members - spared

    def sweep(self) -> Swept | None:
        spared = self.spare()
        alive = self._alive(spared)
        if alive is None:
            return None
        names: dict[int, str] = {}
        swept = Swept()
        if not alive:
            return swept
        deadline = self.clock() + self.grace
        while alive:
            for pid in alive - names.keys():
                names[pid] = self.group.name(pid)
                self.group.signal(pid, signal.SIGTERM)
            if self.clock() >= deadline:
                break
            self.sleep(self.step)
            alive = self._alive(spared) or set()
        killed: set[int] = set()
        for _ in range(KILL_ROUNDS):
            if not alive:
                break
            for pid in alive:
                names.setdefault(pid, self.group.name(pid))
                killed.add(pid)
                self.group.signal(pid, signal.SIGKILL)
            self.sleep(self.step)
            alive = self._alive(spared) or set()
        swept.ended, swept.killed, swept.left = len(names), len(killed), len(alive)
        for name in names.values():
            swept.names[name] = swept.names.get(name, 0) + 1
        log.info("ended %d leftover process(es) (%d killed, %d left): %s", swept.ended, swept.killed,
                 swept.left, ", ".join(sorted(swept.names)))
        return swept

