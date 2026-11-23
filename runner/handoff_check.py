"""Deterministic, report-only checks of a coding run's hand-off.

The findings compare what the session said with what the runner can see: a missing or thin typed
result, fewer items than the brief's numbered list, a "tests passed" claim contradicted by the
runner's own test verdict, changed files not listed in the result, and junk files left out of the
commit. No finding blocks a hand-off. A thin or short result gets at most one result-only repair turn.
The functions read only their arguments. The session driver does the git reads.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Iterable

from runner.providers.base import tidy_test_claim

VERSION = 2
PATHS_SHOWN = 20
NUMBERED = re.compile(r"^(\d{1,2})[.)]\s")
JUNK_DIRS = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules")
JUNK_FILES = ("*.pyc", ".DS_Store", "*.orig", "*.rej")
RESULT_MISSING = "The session returned no typed result."
RESULT_THIN = "The result has an empty summary or items."
ITEMS_SHORT = "The brief has {expected} numbered items; the result reports {actual}."
TESTS_CLAIM_FAILED = "The session reports that tests passed, but the runner recorded a failed final test check."
TESTS_CLAIM_NOT_RUN = "The session reports that tests passed, but the runner recorded no verified final test run."
FILES_UNLISTED = "Changed files not named in files_changed: {paths}."
JUNK_LEFT_OUT = "Untracked junk left out of the hand-off commit, kept on disk: {paths}."
MORE_PATHS = "{paths}; and {more} more"
REPAIR_SECONDS = 300.0
REPAIR_PURPOSE = "handoff_repair"
REPAIR_TIMEOUT = "The session's result repair ran out of time."
REPAIR_PROMPT = (
    "Your result lacks {missing}; give it again, covering the whole brief and every numbered request, "
    "with all required result fields; change no files. Report only work you did and checks you saw. "
    "This repair turn is cut off after 300 seconds.")


@dataclass(frozen=True)
class Status:

    changed: tuple[str, ...] = ()
    untracked: tuple[str, ...] = ()
    base_tracked: frozenset[str] = frozenset()


def numbered_items(brief: str) -> int:
    count = 0
    for line in brief.splitlines():
        match = NUMBERED.match(line)
        if not match:
            if count and line.strip() and not line[0].isspace():
                break
            continue
        if int(match[1]) != count + 1:
            break
        count += 1
    return count if count >= 2 else 0


def junk_paths(status: Status) -> tuple[str, ...]:
    return tuple(sorted(path for path in set(status.untracked) if path not in status.base_tracked and
                        (any(part in JUNK_DIRS for part in path.split("/")[:-1]) or
                         any(fnmatch.fnmatchcase(path.split("/")[-1], pattern) for pattern in JUNK_FILES))))


def path_list(paths: Iterable[str]) -> str:
    paths = sorted(set(paths))
    text = ", ".join(paths[:PATHS_SHOWN])
    return MORE_PATHS.format(paths=text, more=len(paths) - PATHS_SHOWN) if len(paths) > PATHS_SHOWN else text


def result_findings(brief: str, result: dict[str, Any] | None) -> list[dict[str, str]]:
    if result is None:
        return [{"key": "result_missing", "text": RESULT_MISSING}]
    findings = []
    items = result.get("items") or []
    if not str(result.get("summary") or "").strip() or not items:
        findings.append({"key": "result_thin", "text": RESULT_THIN})
    expected = numbered_items(brief)
    if len(items) < expected:
        findings.append({"key": "items_short", "text": ITEMS_SHORT.format(expected=expected, actual=len(items))})
    return findings


def repair_prompt(brief: str, result: dict[str, Any] | None) -> str:
    missing = []
    if result is None:
        missing.append("a typed result")
    if not result or not str(result.get("summary") or "").strip():
        missing.append("a non-empty summary")
    if not result or not result.get("items"):
        missing.append("non-empty items")
    expected = numbered_items(brief)
    if expected and len((result or {}).get("items") or []) < expected:
        missing.append(f"items covering all {expected} numbered requests")
    return REPAIR_PROMPT.format(missing=", ".join(missing))


def test_claim_findings(result: dict[str, Any] | None, tests: dict[str, Any] | None,
                        test_cmd: str | None) -> list[dict[str, str]]:
    claim = tidy_test_claim((result or {}).get("tests_claim"))
    recorded = tidy_test_claim({"command": tests.get("command"), "status": "unverified"}) \
        if isinstance(tests, dict) else None
    expected = " ".join(test_cmd.split()) if test_cmd else None
    if not claim or claim["status"] != "passed" or claim["command"] != expected \
            or not recorded or recorded["command"] != expected:
        return []
    code = tests.get("exit_code")
    if "exit_code" not in tests or (code is not None and
                                   (not isinstance(code, int) or isinstance(code, bool) or abs(code) > 65535)):
        return []
    status = tests.get("status")
    if status == "failed":
        return [{"key": "tests_claim_failed", "text": TESTS_CLAIM_FAILED}]
    if status == "not_run":
        return [{"key": "tests_claim_not_run", "text": TESTS_CLAIM_NOT_RUN}]
    return []


def check(brief: str, result: dict[str, Any] | None, status: Status,
          tests: dict[str, Any] | None = None, test_cmd: str | None = None) -> dict[str, Any]:
    findings = result_findings(brief, result)
    junk = junk_paths(status)
    findings.extend(test_claim_findings(result, tests, test_cmd))
    unlisted = set(status.changed) - set((result or {}).get("files_changed") or []) - set(junk)
    if unlisted:
        findings.append({"key": "files_unlisted", "text": FILES_UNLISTED.format(paths=path_list(unlisted))})
    if junk:
        findings.append({"key": "junk_left_out", "text": JUNK_LEFT_OUT.format(paths=", ".join(junk))})
    return {"version": VERSION, "findings": findings}

