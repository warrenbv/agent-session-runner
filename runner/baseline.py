"""The platform baseline: standing guidance every coding session gets after its brief and the
repository's own instructions file.

The text is baseline.md beside this module. Its first line is `Baseline N`, and the tests pin its
SHA-256 beside N, so an edit is always a deliberate change. Claude Code gets it appended to the
`claude_code` system prompt preset. Codex gets it as developer instructions. A conversation resumed
by id keeps what its first run was given, so the stream then says `recorded` rather than claiming the
text was sent again. Which baseline a run went under is posted once, before the first turn.
"""

from __future__ import annotations

import functools
import hashlib
import pathlib
import re
from dataclasses import dataclass
from typing import Any

PATH = pathlib.Path(__file__).with_name("baseline.md")
PURPOSE = "baseline"
FIRST_LINE = re.compile(r"^Baseline ([1-9][0-9]{0,3})\Z")
SHA_SHOWN = 12

APPEND, DEVELOPER, RECORDED, OFF = "append", "developer", "recorded", "off"
HOW = {"claude-code": APPEND, "codex": DEVELOPER}


@dataclass(frozen=True)
class Baseline:
    version: int
    text: str
    sha: str


@functools.cache
def load() -> Baseline:
    data = PATH.read_bytes()
    text = data.decode("utf-8").rstrip("\n")
    found = FIRST_LINE.match(text.split("\n", 1)[0])
    if found is None:
        raise ValueError("baseline.md must start with a line `Baseline N`")
    return Baseline(int(found.group(1)), text, hashlib.sha256(data).hexdigest())


def applied(provider: str, on: bool, resumed_by_id: bool) -> str:
    if resumed_by_id:
        return RECORDED
    return HOW.get(provider, APPEND) if on else OFF


def event(how: str, project: tuple[str, int, bool] | None) -> dict[str, Any]:
    base = load()
    name, size, cut = project if project is not None else (None, 0, False)
    return {"role": "runner", "purpose": PURPOSE, "version": base.version, "sha": base.sha[:SHA_SHOWN],
            "applied": how, "project_file": name, "project_file_bytes": size, "cut": cut}

