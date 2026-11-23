"""A research session's hand-off: inputs written into its scratch directory before the first turn,
and its output files checked and posted at the end.

Research sessions have no repository. Lead and challenger roles first fetch their inputs in parts,
join them, check each file's SHA-256 and write them under `inputs/` with a manifest. At every ending
except a revoked token, the role's two output files are read (regular files only, never through a
link, capped), checked against the contract's shapes, posted in parts and then confirmed once with
every file's part count and hash. A refused count or hash is resent a bounded number of times. The
fixtures under tests/fixtures/research_contract pin the exact messages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from dataclasses import dataclass, field
from typing import Any

from runner.control_plane import PART_BYTES, Rejected, ResearchClient, TooLarge

log = logging.getLogger("runner.research")

ROLES = ("provider", "lead_brief", "lead_report", "challenger")
LEAD_ROLES = ("lead_brief", "lead_report")
INPUT_ROLES = (*LEAD_ROLES, "challenger")
OUTPUTS = {"provider": ("synthesis.md", "claims.json"), "lead_brief": ("brief.json", "brief.md"),
           "lead_report": ("report.md", "report.json"), "challenger": ("challenge.md", "verdicts.json")}
INPUTS_DIR = "inputs"
MANIFEST_NAME = "manifest.md"
FILE_CAP = 512 * 1024
MAX_PARTS = 11
INPUTS_CAP = 8 * 1024 * 1024
INPUT_PARTS_CAP = 400
SUMMARY_CAP = 2000
REASON_CAP = 1000
OUTPUT_ATTEMPTS = 3
INPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,99}){0,2}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

CLAIM_HEADING = re.compile(r"^##[ \t]+(C(\d+))(?:[ \t:.\-–—].*)?$")
H2 = re.compile(r"^##[ \t]+(.*?)[ \t]*$")
CLAIM_ID = re.compile(r"^C\d+$")
CONFIDENCE = ("high", "medium", "low")
CHECK_STATUSES = ("checked", "not_checked", "differs")
BRIEF_KEYS = ("question", "sub_questions", "scope", "exclusions", "time_window", "source_types", "depth", "output",
              "questions_for_owner")
REPORT_KEYS = ("findings", "disagreements", "gaps", "overlap", "source_checks", "providers_reached")
CHALLENGE_HEADINGS = ("Verdicts", "The case against", "Pre-mortem", "Methods and limits")
VERDICT_HEADING = re.compile(r"^###[ \t]+V(\d+)\b")
VERDICTS = ("held", "weakened", "overturned", "could_not_check")
CLAIM_REF = re.compile(r"^[a-z]+:C[1-9]\d{0,3}$")
CLAIMS_CAP = 300
SOURCES_CAP = 50
PRE_MORTEM_CAP = 5


class InputsError(Exception):
    ...


def manifest(texts: dict[str, str]) -> str:
    lines = ["# The inputs of this step", "",
             "Every file under `inputs/` is listed here; no other input file exists.", ""]
    if not texts:
        lines.append("This step has no input files.")
    for name, text in texts.items():
        size = len(text.encode("utf-8"))
        lines.append(f"- `inputs/{name}` ({size} bytes{'' if text.strip() else ', empty'})")
    return "\n".join(lines) + "\n"


def split(text: str, size: int = PART_BYTES) -> list[str]:
    data = text.encode("utf-8")
    parts, start = [], 0
    while start < len(data):
        end = min(start + size, len(data))
        while end < len(data) and (data[end] & 0xC0) == 0x80:
            end -= 1
        parts.append(data[start:end].decode("utf-8"))
        start = end
    return parts or [""]


def synthesis_ids(text: str) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    ids: list[str] = []
    numbers: list[int] = []
    last_claim = methods = -1
    h2 = []
    for line in text.splitlines():
        m = H2.match(line)
        if m:
            h2.append(m.group(1))
    for i, heading in enumerate(h2):
        claim = CLAIM_HEADING.match(f"## {heading}")
        if claim:
            ids.append(claim.group(1))
            numbers.append(int(claim.group(2)))
            last_claim = i
        elif heading.strip().lower() == "methods and limits":
            methods = i
    if not ids:
        problems.append("synthesis.md has no `## C<n>` claim headings")
    elif len(set(ids)) != len(ids) or numbers != sorted(numbers):
        problems.append("synthesis.md's claim headings are repeated or out of order")
    if methods < 0:
        problems.append("synthesis.md has no closing `## Methods and limits` heading")
    elif methods < last_claim or methods != len(h2) - 1:
        problems.append("synthesis.md's `## Methods and limits` is not its closing heading")
    return ids, problems


def _str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _count(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def check_claims(data: Any, ids: list[str] | None) -> list[str]:
    if not isinstance(data, list):
        return ["claims.json is not a list"]
    problems: list[str] = []
    for n, claim in enumerate(data, 1):
        if not isinstance(claim, dict):
            problems.append(f"claims.json entry {n} is not an object")
            continue
        missing = [k for k in ("id", "text", "confidence", "how_found", "sources") if k not in claim]
        if missing:
            problems.append(f"claims.json entry {n} lacks {', '.join(missing)}")
            continue
        if not (isinstance(claim["id"], str) and CLAIM_ID.match(claim["id"])) or not _str(claim["text"]) \
                or claim["confidence"] not in CONFIDENCE or not isinstance(claim["how_found"], str):
            problems.append(f"claims.json entry {n} has a malformed id, text, confidence or how_found")
        sources = claim["sources"]
        if not isinstance(sources, list) or not all(
                isinstance(s, dict) and _str(s.get("url")) and isinstance(s.get("title"), str)
                and isinstance(s.get("accessed"), str) for s in sources):
            problems.append(f"claims.json entry {n} has malformed sources")
    if ids is not None and not problems:
        found = [c["id"] for c in data]
        if found != ids:
            problems.append("claims.json's ids do not match synthesis.md's claim headings")
    return problems[:10]


def check_brief(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["brief.json is not an object"]
    missing = [k for k in BRIEF_KEYS if k not in data]
    if missing:
        return [f"brief.json lacks {', '.join(missing)}"]
    problems = []
    if not _str(data["question"]):
        problems.append("brief.json's question is empty")
    if not isinstance(data["sub_questions"], list):
        problems.append("brief.json's sub_questions is not a list")
    asks = data["questions_for_owner"]
    if not isinstance(asks, list) or len(asks) > 3 or not all(isinstance(q, str) for q in asks):
        problems.append("brief.json's questions_for_owner is not a list of at most three questions")
    return problems


def check_report(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["report.json is not an object"]
    missing = [k for k in REPORT_KEYS if k not in data]
    if missing:
        return [f"report.json lacks {', '.join(missing)}"]
    problems = []
    if not isinstance(data["findings"], list) or not all(
            isinstance(f, dict) and _str(f.get("text")) and isinstance(f.get("cites"), list)
            and all(isinstance(c, dict) and _str(c.get("provider")) and _str(c.get("claim")) for c in f["cites"])
            for f in data["findings"]):
        problems.append("report.json's findings are malformed")
    for key in ("disagreements", "gaps", "providers_reached"):
        if not isinstance(data[key], list):
            problems.append(f"report.json's {key} is not a list")
    if not isinstance(data["overlap"], list) or not all(
            isinstance(o, dict) and _str(o.get("url")) and isinstance(o.get("providers"), list) for o in data["overlap"]):
        problems.append("report.json's overlap is malformed")
    if not isinstance(data["source_checks"], list) or not all(
            isinstance(s, dict) and _str(s.get("url")) and s.get("status") in CHECK_STATUSES
            and isinstance(s.get("note", ""), str) for s in data["source_checks"]):
        problems.append("report.json's source_checks are malformed")
    return problems


def challenge_ids(text: str) -> tuple[list[int], list[str]]:
    h2 = [m.group(1) for line in text.splitlines() if (m := H2.match(line))]
    numbers = [int(m.group(1)) for line in text.splitlines() if (m := VERDICT_HEADING.match(line))]
    problems = []
    if h2 != list(CHALLENGE_HEADINGS):
        problems.append("challenge.md's `## ` headings are not " + ", ".join(CHALLENGE_HEADINGS) + ", in that order"
                        " and no other")
    return numbers, problems


def _sources_ok(value: Any, required: bool) -> bool:
    return isinstance(value, list) and (1 if required else 0) <= len(value) <= SOURCES_CAP and all(
        isinstance(s, dict) and isinstance(s.get("url"), str)
        and s["url"].strip().lower().startswith(("http://", "https://")) for s in value)


def _refs_ok(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and CLAIM_REF.match(v) for v in value)


def check_verdicts(data: Any, numbers: list[int] | None) -> list[str]:
    if not isinstance(data, dict):
        return ["verdicts.json is not an object"]
    verdicts = data.get("verdicts")
    if not isinstance(verdicts, list) or not 1 <= len(verdicts) <= CLAIMS_CAP:
        return [f"verdicts.json's verdicts is not a list of 1 to {CLAIMS_CAP}"]
    problems: list[str] = []
    ids = [v.get("id") if isinstance(v, dict) else None for v in verdicts]
    if ids != [f"V{n}" for n in range(1, len(verdicts) + 1)]:
        problems.append("verdicts.json's verdicts are not numbered V1, V2, ... in order")
    elif numbers is not None and numbers != list(range(1, len(verdicts) + 1)):
        problems.append("verdicts.json's ids do not match challenge.md's `### V<n>` headings")
    claims = [v.get("claim") for v in verdicts if isinstance(v, dict)]
    if not all(isinstance(c, str) and CLAIM_REF.match(c) for c in claims):
        problems.append("verdicts.json names a claim that is not <provider>:C<n>")
    elif len(set(claims)) != len(claims):
        problems.append("verdicts.json judges a claim twice")
    for n, v in enumerate(verdicts, 1):
        if not isinstance(v, dict) or v.get("verdict") not in VERDICTS:
            problems.append(f"verdicts.json's verdict V{n} is not one of {', '.join(VERDICTS)}")
        elif v["verdict"] == "could_not_check":
            if not _str(v.get("reason")):
                problems.append(f"verdicts.json's verdict V{n} (could_not_check) has no reason")
        elif not _sources_ok(v.get("sources"), True):
            problems.append(f"verdicts.json's verdict V{n} has no sources or malformed ones")
    case = data.get("case_against")
    if not isinstance(case, list):
        problems.append("verdicts.json's case_against is not a list")
    elif not case and not _str(data.get("case_against_none")):
        problems.append("verdicts.json's case_against is empty and case_against_none says nothing")
    elif not all(isinstance(k, dict) and _str(k.get("text")) and _sources_ok(k.get("sources"), True)
                 and _refs_ok(k.get("against")) for k in case):
        problems.append("verdicts.json's case_against is malformed")
    pre = data.get("pre_mortem", [])
    if not isinstance(pre, list) or len(pre) > PRE_MORTEM_CAP or not all(
            isinstance(p, dict) and _str(p.get("text")) and _refs_ok(p.get("rests_on", [])) for p in pre):
        problems.append(f"verdicts.json's pre_mortem is not a list of at most {PRE_MORTEM_CAP} points")
    return problems[:10]


def validate(role: str, texts: dict[str, str]) -> tuple[dict[str, list[str]], int | None]:
    out: dict[str, list[str]] = {}
    claims: int | None = None
    parsed: dict[str, Any] = {}
    for name, text in texts.items():
        if name.endswith(".md"):
            out[name] = [] if text.strip() else [f"{name} is empty"]
        else:
            try:
                parsed[name] = json.loads(text)
                out[name] = []
            except json.JSONDecodeError:
                out[name] = [f"{name} is not valid JSON"]
    if role == "provider":
        ids = None
        if "synthesis.md" in texts:
            ids, problems = synthesis_ids(texts["synthesis.md"])
            out["synthesis.md"] += problems
        if "claims.json" in parsed:
            out["claims.json"] += check_claims(parsed["claims.json"], ids)
            if isinstance(parsed["claims.json"], list):
                claims = len(parsed["claims.json"])
    elif role == "lead_brief" and "brief.json" in parsed:
        out["brief.json"] += check_brief(parsed["brief.json"])
    elif role == "lead_report" and "report.json" in parsed:
        out["report.json"] += check_report(parsed["report.json"])
    elif role == "challenger":
        numbers = None
        if "challenge.md" in texts:
            numbers, problems = challenge_ids(texts["challenge.md"])
            out["challenge.md"] += problems
        if "verdicts.json" in parsed:
            out["verdicts.json"] += check_verdicts(parsed["verdicts.json"], numbers)
            if isinstance(parsed["verdicts.json"], dict) and isinstance(parsed["verdicts.json"].get("verdicts"), list):
                claims = len(parsed["verdicts.json"]["verdicts"])
    return out, claims


def read_output(cwd: str, name: str) -> tuple[str | None, str]:
    path = os.path.join(cwd, name)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "not_regular" if os.path.islink(path) else "unreadable"
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None, "not_regular"
        if info.st_size > FILE_CAP:
            return None, "too_large"
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(FILE_CAP + 1)
    except OSError:
        return None, "unreadable"
    finally:
        os.close(fd)
    if len(data) > FILE_CAP:
        return None, "too_large"
    try:
        return data.decode("utf-8"), "read"
    except UnicodeDecodeError:
        return None, "unreadable"


STATUS_WORDS = {"missing": "was not written", "not_regular": "is not a regular file", "too_large":
                f"is larger than {FILE_CAP // 1024} KB", "unreadable": "could not be read as UTF-8 text",
                "refused": "was refused by the control plane"}

ENDINGS = {"wall_clock": "The run's wall clock ran out before it finished.",
           "max_turns": "The run used all its turns before it finished.",
           "budget": "The run reached its budget before it finished.",
           "max_subagents": "The run started more subagents than its cap allows, so it was stopped.",
           "shell_used": "A command ran in this research session, whose shell must be off, so the runner stopped"
                         " it at once and its files are not to be trusted."}
NOT_KEPT = "The control plane did not keep the files: {reason}"
MISMATCH = ("The control plane refused the files' part counts or hashes {n} times ({status} {code}), so they were"
            " not kept.")
NOT_TAKEN = ("The control plane would not take the files ({status} {code}): the step had already ended or its files"
             " were already kept.")


def ending(reason: str, detail: str | None = None) -> str | None:
    if reason == "completed":
        return None
    if reason == "error":
        return f"The session failed ({detail or 'no detail'})."
    return ENDINGS.get(reason, f"The run ended early ({reason}).")


@dataclass
class ResearchOut:

    client: ResearchClient
    role: str
    cwd: str
    posted: list[dict[str, Any]] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)
    claims: int | None = None
    done: bool = False

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"a research role is one of {', '.join(ROLES)}")

    @property
    def lead(self) -> bool:
        return self.role in LEAD_ROLES

    @property
    def has_inputs(self) -> bool:
        return self.role in INPUT_ROLES

    def fetch_inputs(self) -> int:
        if not self.has_inputs:
            return 0
        files: dict[str, dict[str, Any]] = {}
        size = part = 0
        total = 1
        while part < total:
            out = self.client.inputs_get(part)
            count = out.get("total")
            if not _count(count, 0, INPUT_PARTS_CAP):
                raise InputsError("the inputs' part count was missing or out of range")
            if part == 0:
                total = count
            elif count != total:
                raise InputsError("the inputs' part count changed while they were fetched")
            if out.get("part") != part:
                raise InputsError("an inputs answer named another part than the one asked for")
            if total == 0:
                break
            name, index, parts, digest, content = (out.get(k) for k in ("name", "index", "parts", "sha256", "content"))
            if not isinstance(name, str) or not INPUT_NAME.match(name) or ".." in name.split("/") \
                    or name == MANIFEST_NAME or not isinstance(content, str):
                raise InputsError("an input file's name or content was not usable")
            if not _count(parts, 1, INPUT_PARTS_CAP) or not _count(index, 0, parts - 1) \
                    or not isinstance(digest, str) or not SHA256.match(digest):
                raise InputsError(f"input {name}: its piece number, count of pieces or SHA-256 was not usable")
            entry = files.setdefault(name, {"parts": parts, "sha256": digest, "pieces": {}})
            if entry["parts"] != parts or entry["sha256"] != digest or index in entry["pieces"]:
                raise InputsError(f"input {name}: its pieces disagree about the file")
            size += len(content.encode("utf-8"))
            if size > INPUTS_CAP:
                raise InputsError(f"the inputs are larger than {INPUTS_CAP // (1024 * 1024)} MB")
            entry["pieces"][index] = content
            part += 1
        texts: dict[str, str] = {}
        for name, entry in files.items():
            if sorted(entry["pieces"]) != list(range(entry["parts"])):
                raise InputsError(f"input {name} arrived with pieces missing")
            text = "".join(entry["pieces"][i] for i in range(entry["parts"]))
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != entry["sha256"]:
                raise InputsError(f"input {name} does not match its SHA-256")
            texts[name] = text
        root = os.path.join(self.cwd, INPUTS_DIR)
        os.makedirs(root, mode=0o700, exist_ok=True)
        for name, text in {**texts, MANIFEST_NAME: manifest(texts)}.items():
            path = os.path.join(root, name)
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
        log.info("research inputs: %d files in %d parts written under %s/ with %s", len(texts), total, INPUTS_DIR,
                 MANIFEST_NAME)
        return len(texts)

    def missing(self) -> list[str]:
        return [name for name in OUTPUTS[self.role] if read_output(self.cwd, name)[1] != "read"]

    def post_outputs(self, reason: str, detail: str | None = None, note: str | None = None) -> tuple[bool, str | None]:
        texts: dict[str, str] = {}
        for name in OUTPUTS[self.role]:
            text, status = read_output(self.cwd, name)
            self.statuses[name] = status
            if text is not None:
                texts[name] = text
        problems, self.claims = validate(self.role, texts)
        read = dict(self.statuses)
        for attempt in range(1, OUTPUT_ATTEMPTS + 1):
            self.statuses = dict(read)
            files, sentences = self._send_parts(texts, problems, reason, detail, note)
            partial = reason != "completed" or any(s != "posted" for s in self.statuses.values())
            why = " ".join(sentences)[:REASON_CAP] if partial else None
            try:
                answer = self.client.output_done(files, partial, why)
            except Rejected as exc:
                if exc.status == 400 and attempt < OUTPUT_ATTEMPTS:
                    log.warning("research output_done refused (%s %s), attempt %d; sending the parts again",
                                exc.status, exc.code, attempt)
                    continue
                words = (MISMATCH.format(n=attempt, status=exc.status, code=exc.code) if exc.status == 400
                         else NOT_TAKEN.format(status=exc.status, code=exc.code))
                log.error("research outputs not taken (%s %s) after %d attempts", exc.status, exc.code, attempt)
                self._not_kept()
                return True, " ".join([*sentences, words])[:REASON_CAP]
            self.posted, self.done = files, True
            if isinstance(answer, dict) and answer.get("accepted") is False:
                refusal = " ".join(str(answer.get("reason") or "no reason given").split())
                log.warning("research outputs not kept by the control plane (their shape)")
                self._not_kept()
                return True, " ".join([*sentences, NOT_KEPT.format(reason=refusal)])[:REASON_CAP]
            log.info("research outputs: %s; partial %s", ", ".join(f"{k} {v}" for k, v in self.statuses.items()),
                     partial)
            return partial, why
        raise AssertionError("not reached")

    def _send_parts(self, texts: dict[str, str], problems: dict[str, list[str]], reason: str,
                    detail: str | None, note: str | None = None) -> tuple[list[dict[str, Any]], list[str]]:
        sentences = [s for s in [ending(reason, detail), note] if s]
        files = []
        for name in OUTPUTS[self.role]:
            text = texts.get(name)
            entry: dict[str, Any] = {"name": name, "parts": 0, "sha256": None}
            if text is not None:
                parts = split(text)
                try:
                    if len(parts) > MAX_PARTS:
                        raise TooLarge()
                    for index, content in enumerate(parts):
                        self.client.output_part(name, index, len(parts), content)
                except Rejected as exc:
                    log.warning("research output %s refused (%s %s)", name, exc.status, exc.code)
                    self.statuses[name] = "refused"
                except TooLarge:
                    log.warning("research output %s refused as too large", name)
                    self.statuses[name] = "too_large"
                else:
                    entry = {"name": name, "parts": len(parts),
                             "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                    self.statuses[name] = "invalid" if problems.get(name) else "posted"
            if self.statuses[name] in STATUS_WORDS:
                sentences.append(f"{name} {STATUS_WORDS[self.statuses[name]]}.")
            sentences += [f"{p}." for p in problems.get(name, [])]
            files.append(entry)
        return files, sentences

    def _not_kept(self) -> None:
        for name, status in self.statuses.items():
            if status in ("posted", "invalid"):
                self.statuses[name] = "not_kept"

    def result(self, summary: str, limits: list[str], partial: bool, reason: str | None) -> dict[str, Any]:
        text = " ".join((summary or "").split())
        return {"summary": text[:SUMMARY_CAP], "claims": self.claims, "limits": list(dict.fromkeys(limits)),
                "partial": partial, "reason": reason,
                "files": [{"name": n, "status": self.statuses.get(n, "not_posted")} for n in OUTPUTS[self.role]]}

