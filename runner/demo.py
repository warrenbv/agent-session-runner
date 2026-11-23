"""A credential-free tour of the runner: `python -m runner.demo`.

Part 1 replays two recorded sessions, one from Claude Code and one from Codex, through the real
normaliser (normalise.py) and prints the single event stream both become. Part 2 runs one whole
coding session end to end: the real session driver (session.py) drives the fake provider
(providers/fake.py) against the stub control plane (stub_control_plane.py) over loopback HTTP, with
an approval, a file change, the final test turn and the hand-off checks.

Nothing here is a mock-up of the runner's behaviour. Only the model and the control plane are
stand-ins: the recordings stand in for the model in part 1, a short script stands in for it in
part 2, and the stub stands in for the control plane. See docs/design.md for the path an event takes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import tempfile
from collections import Counter
from typing import Any

from runner import gitdiff, normalise
from runner.control_plane import Client, Retry
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session, Timing
from runner.stub_control_plane import TOKEN, StubControlPlane

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
RECORDINGS = (("claude-code", "claude_recorded_2_1_321.raw.json"), ("codex", "codex_recorded.raw.json"))
WIDTH = 96
TEST_COMMAND = "uv run pytest -q"


def _short(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= WIDTH else text[: WIDTH - 3] + "..."


def replay() -> None:
    print("Part 1: two recorded sessions, one normaliser, one event stream\n")
    seen: dict[str, Counter[str]] = {}
    for provider, name in RECORDINGS:
        raw = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        kinds: Counter[str] = Counter()
        print(f"--- {provider}: {len(raw['messages'])} raw messages from {name}")
        for message in raw["messages"]:
            for event in normalise.NORMALISERS[provider](message):
                kinds[event.kind] += 1
                print(f"{provider:<11} {event.kind:<9} {_short(event.payload)}")
        seen[provider] = kinds
        print()
    print("Event kinds per provider (the same vocabulary for both):")
    for provider, kinds in seen.items():
        print(f"  {provider:<11} " + ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items())))
    print()


def _say(text: str) -> dict[str, Any]:
    return {"type": "AssistantMessage", "model": "claude-opus-5-5", "parent_tool_use_id": None,
            "content": [{"type": "TextBlock", "text": text}]}


def _bash(tool_id: str, command: str) -> dict[str, Any]:
    return {"type": "AssistantMessage", "model": "claude-opus-5-5", "parent_tool_use_id": None,
            "content": [{"type": "ToolUseBlock", "id": tool_id, "name": "Bash", "input": {"command": command}}]}


def _bash_result(tool_id: str, ok: bool, text: str) -> dict[str, Any]:
    return {"type": "UserMessage", "parent_tool_use_id": None,
            "tool_use_result": {"stdout": text, "stderr": "", "interrupted": False} if ok else f"Error: {text}",
            "content": [{"type": "ToolResultBlock", "tool_use_id": tool_id, "content": text, "is_error": not ok}]}


def _result(structured: dict[str, Any], cost: float) -> dict[str, Any]:
    return {"type": "ResultMessage", "subtype": "success", "duration_ms": 1200, "duration_api_ms": 1000,
            "is_error": False, "num_turns": 3, "session_id": "0b7c4a52-8f1e-4d3a-9c6b-2e5f7a8d9e10",
            "total_cost_usd": cost, "terminal_reason": "completed", "structured_output": structured}


def _typed(summary: str, **fields: Any) -> dict[str, Any]:
    return {"summary": summary, "files_changed": [], "tests_run": [], "tests_claim": None,
            "branch": "agent/demo", "open_questions": [], "next_steps": [],
            "items": [{"label": "Exclude the cache directory", "status": "done", "where": "scripts/backup.sh"}],
            **fields}


def script() -> dict[str, Any]:
    work = [
        _say("I'll clear the old build output, then fix the backup exclusion."),
        _bash("toolu_demo_1", "rm -rf build/"),
        {"fake": "approval", "tool_name": "Bash", "input": {"command": "rm -rf build/"}, "tool_use_id": "toolu_demo_1",
         "on_allow": [_bash_result("toolu_demo_1", True, "")],
         "on_deny": [_bash_result("toolu_demo_1", False, "$DENY_MESSAGE")]},
        {"fake": "write", "path": "scripts/backup.sh", "text": "rsync -a --exclude cache /srv/data /backup\n"},
        _say("The backup now skips the cache directory."),
        _result(_typed("Exclude the cache directory from the backup", files_changed=["scripts/backup.sh"],
                       tests_run=[f"{TEST_COMMAND}: passed"],
                       tests_claim={"command": TEST_COMMAND, "status": "passed"}), 0.08),
    ]
    # The model reports a pass in its typed result, but the recorded tool result says exit code 1.
    # The runner reads the verdict from the tool result (docs/security-model.md, "Test verdicts").
    tests = [
        _bash("toolu_demo_2", TEST_COMMAND),
        _bash_result("toolu_demo_2", False, "Exit code 1\nFAILED tests/test_backup.py::test_excludes_cache"),
        _say("All tests pass."),
        _result(_typed("Ran the closing check", tests_run=[f"{TEST_COMMAND}: passed"]), 0.11),
    ]
    return {"provider": "claude-code", "turns": [work, tests], "default_turn": [_say("Nothing more."),
                                                                                _result(_typed("Nothing more."), 0.12)]}


def _worktree(root: pathlib.Path) -> str:
    repo = root / "repo"
    repo.mkdir()
    path = str(repo)
    gitdiff.git(path, "init", "--quiet", "--initial-branch=main")
    for key, value in (("user.name", "Demo"), ("user.email", "demo@example.invalid"), ("commit.gpgsign", "false")):
        gitdiff.git(path, "config", key, value)
    (repo / "README.txt").write_text("demo project\n", encoding="utf-8")
    gitdiff.git(path, "add", "README.txt")
    gitdiff.git(path, "commit", "--quiet", "-m", "initial")
    gitdiff.git(path, "switch", "--quiet", "-c", "agent/demo")
    return path


def session() -> int:
    print("Part 2: one session end to end (fake provider, real driver, stub control plane)\n")
    plane = StubControlPlane()
    # The stub plays the person at the console: it allows the one approval the script asks for.
    plane.when(lambda e: e["kind"] == "approval", lambda e: plane.answer(e["payload"]["request_id"], "allow"))
    try:
        with tempfile.TemporaryDirectory(prefix="runner-demo-") as scratch:
            cwd = _worktree(pathlib.Path(scratch))
            provider = FakeProvider(script())
            client = Client(plane.url, TOKEN, timeout=5)
            options = Options(cwd=cwd, test_cmd=TEST_COMMAND)
            timing = Timing(pull_interval=0.05, retry=Retry(attempts=3, first=0.1))
            outcome = asyncio.run(Session(provider, client, options, Caps(wall_clock=60, grace=2), timing)
                                  .run("Make the backup skip the cache directory."))
    finally:
        plane.close()
    for event in plane.events:
        print(f"{event['sequence']:>3} {event['kind']:<9} {_short(event['payload'])}")
    end = plane.of("end")[-1] if plane.of("end") else {}
    verdicts = plane.of("message", purpose="tests-verdict")
    claim = (outcome.result or {}).get("tests_claim")
    ran = len(plane.of("turn", phase="started"))
    print()
    print(f"Session ended: {outcome.reason}, cost {outcome.cost_usd} USD")
    print(f"The end event says turns {end.get('turns')} although {ran} turns ran: it counts only the turns of work,"
          " not the test turn the runner wrote.")
    print()
    print("The point of the session: the model claims a pass, the tool record shows a failure, the runner says so.")
    print(f"  The model's claim in its typed result: {json.dumps(claim)}")
    print(f"  The runner's verdict, read from the recorded tool result: {verdicts[-1]['text'] if verdicts else None}")
    for finding in (end.get("checks") or {}).get("findings", []):
        print(f"  Hand-off finding {finding['key']}: {finding['text']}")
    return 0 if outcome.reason == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m runner.demo",
                                     description="Replay recorded sessions and run one fake session, offline.")
    parser.add_argument("--part", choices=("replay", "session", "all"), default="all")
    args = parser.parse_args(argv)
    if args.part in ("replay", "all"):
        replay()
    if args.part in ("session", "all"):
        return session()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
