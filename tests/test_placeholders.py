"""Sandbox placeholders: empty files left at protected paths never reach a diff, commit, statistics or
test verdict, and are removed only once nothing uses the worktree."""

from __future__ import annotations

import asyncio
import os

import pytest

from conftest import fast_timing, git, typed_result
from runner import gitdiff
from runner import handoff as handoff_mod
from runner import placeholders
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session

CMD = "uv run pytest -q"
FILES = [n for n in placeholders.PROTECTED if n not in (".vscode", ".idea")]
DIRS = [".vscode", ".idea"]


class FakeForge:
    def __init__(self):
        self.pushed, self.prs = [], []

    def push(self, cwd, branch):
        self.pushed.append(branch)

    def open_pr(self, branch, base, title, body):
        self.prs.append({"title": title, "body": body})
        return handoff_mod.Opened("https://github.example/owner/repo/pull/7", 7)


class Sandboxed(FakeProvider):

    def __init__(self, fixture, cli_cleans=False):
        super().__init__(fixture)
        self.cli_cleans, self.made, self.found = cli_cleans, [], []

    async def turn(self, text):
        cwd = self.ctx.cwd
        self.found.append(sorted(p for p in self.made if os.path.lexists(p)))
        for name in FILES + DIRS:
            path = os.path.join(cwd, name)
            if os.path.lexists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if name in DIRS:
                os.mkdir(path)
            else:
                open(path, "w").close()
            self.made.append(path)
        async for raw in super().turn(text):
            yield raw

    async def close(self):
        await super().close()
        if self.cli_cleans:
            for path in self.made:
                if os.path.isdir(path):
                    os.rmdir(path)
                elif os.path.exists(path) and os.path.getsize(path) == 0:
                    os.unlink(path)


def result(**extra):
    return {"type": "ResultMessage", "subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
            "is_error": False, "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01,
            "terminal_reason": "completed", "structured_output": typed_result(), **extra}


def write(path, text):
    return {"fake": "write", "path": path, "text": text}


def bash(command):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "ToolUseBlock", "id": "toolu_t1", "name": "Bash",
                         "input": {"command": command, "description": "Run the tests"}}]}


def bash_ok():
    return {"type": "UserMessage", "parent_tool_use_id": None,
            "tool_use_result": {"stdout": "3 passed", "stderr": "", "interrupted": False, "isImage": False,
                                "noOutputExpected": False},
            "content": [{"type": "ToolResultBlock", "tool_use_id": "toolu_t1", "content": "3 passed",
                         "is_error": False}]}


def claude(*turns):
    return {"provider": "claude-code", "turns": [list(t) for t in turns], "default_turn": [result()]}


def run(fixture, client, worktree, cli_cleans=False, test_cmd=None):
    provider, forge = Sandboxed(fixture, cli_cleans), FakeForge()
    session = Session(provider, client, Options(cwd=str(worktree), test_cmd=test_cmd), Caps(wall_clock=30, grace=1),
                      fast_timing(),
                      handoff=lambda res: handoff_mod.hand_off(str(worktree), "fix-backup", "main", res, forge,
                                                               existed=session.existed))
    outcome = asyncio.run(asyncio.wait_for(session.run("Fix the backup job."), timeout=30))
    return outcome, provider, forge


@pytest.fixture
def still_running(monkeypatch):
    monkeypatch.setattr(placeholders, "busy", lambda cwd: True)
    monkeypatch.setattr(placeholders, "settle", lambda cwd: False)


def committed(worktree):
    return set(git(worktree, "diff", "--name-only", "main", "agent/fix-backup").split())


def ahead(worktree):
    return int(git(worktree, "rev-list", "--count", "main..agent/fix-backup"))


def test_placeholders_that_outlive_the_cli_are_not_a_pull_request(orch, client, worktree):
    outcome, fake, forge = run(claude([result()]), client, worktree)
    assert len(fake.made) == len(placeholders.PROTECTED) == 23
    assert outcome.handoff == {"handoff": "nothing to push", "branch": "agent/fix-backup"}
    assert forge.pushed == [] and forge.prs == [] and ahead(worktree) == 0
    assert orch.of("diff") == []
    assert git(worktree, "status", "--porcelain", "--untracked-files=all") == ""
    assert not os.path.exists(worktree / ".claude")


def test_placeholders_the_cli_removed_at_exit_are_in_no_turn_diff(orch, client, worktree):
    outcome, _, forge = run(claude([result()]), client, worktree, cli_cleans=True)
    assert orch.of("diff") == []
    assert outcome.handoff["handoff"] == "nothing to push" and forge.prs == [] and ahead(worktree) == 0


def test_only_the_real_change_is_committed(orch, client, worktree):
    outcome, _, forge = run(claude([write("a.txt", "work\n"), result()]), client, worktree)
    assert outcome.handoff["handoff"] == "pull request" and len(forge.prs) == 1
    assert committed(worktree) == {"a.txt"}
    assert outcome.handoff["stats"]["files"] == 1 and outcome.handoff["stats"]["paths"] == ["a.txt"]
    assert orch.of("diff")[0]["paths"] == ["a.txt"]


def test_an_empty_file_off_the_list_is_kept(orch, client, worktree):
    outcome, _, _ = run(claude([write("pkg/__init__.py", ""), write(".gitkeep", ""), result()]), client, worktree)
    assert committed(worktree) == {"pkg/__init__.py", ".gitkeep"}
    assert outcome.handoff["stats"]["files"] == 2


def test_a_listed_name_there_before_the_session_is_untouched(orch, client, worktree):
    (worktree / ".bashrc").write_text("")
    (worktree / ".claude").mkdir()
    (worktree / ".claude" / "settings.json").write_text("")
    outcome, _, _ = run(claude([write("a.txt", "work\n"), result()]), client, worktree)
    assert (worktree / ".bashrc").exists() and (worktree / ".claude" / "settings.json").exists()
    assert committed(worktree) == {"a.txt", ".bashrc", ".claude/settings.json"}


def test_a_listed_name_the_session_filled_is_committed(orch, client, worktree):
    outcome, _, _ = run(claude([write(".mcp.json", "{}\n"), write(".claude/settings.json", "{}\n"), result()]),
                        client, worktree)
    assert committed(worktree) == {".mcp.json", ".claude/settings.json"}
    assert outcome.handoff["stats"]["files"] == 2


def test_a_tracked_empty_listed_file_is_never_removed(worktree):
    (worktree / ".gitmodules").write_text("")
    git(worktree, "add", ".gitmodules")
    assert placeholders.sweep(str(worktree), frozenset()) == []
    assert (worktree / ".gitmodules").exists()


def test_the_test_turn_s_snapshots_match_when_only_placeholders_appeared(orch, client, worktree):
    outcome, _, _ = run(claude([write("a.txt", "work\n"), result()], [bash(CMD), bash_ok(), result()]),
                        client, worktree, test_cmd=CMD)
    assert outcome.handoff["tests"] == {"command": CMD, "status": "passed", "exit_code": None}
    assert committed(worktree) == {"a.txt"}


def test_a_placeholder_stays_on_disk_across_the_turn_s_snapshots(orch, client, worktree):
    outcome, fake, _ = run(claude([result()], [bash(CMD), bash_ok(), result()]), client, worktree, test_cmd=CMD)
    assert fake.found[0] == [] and fake.found[1] == sorted(fake.made)
    assert len(fake.made) == 23
    assert orch.of("diff") == [] and outcome.handoff["handoff"] == "nothing to push"
    assert not os.path.lexists(worktree / ".bashrc")


def test_with_a_process_still_running_nothing_is_unlinked_and_nothing_committed(orch, client, worktree,
                                                                              still_running):
    outcome, fake, forge = run(claude([write("a.txt", "work\n"), result()]), client, worktree)
    assert all(os.path.lexists(p) for p in fake.made) and len(fake.made) == 23
    assert committed(worktree) == {"a.txt"} and outcome.handoff["stats"]["files"] == 1
    assert len(forge.prs) == 1


def test_with_a_process_still_running_placeholders_alone_are_nothing_to_push(orch, client, worktree,
                                                                             still_running):
    outcome, _, forge = run(claude([result()]), client, worktree)
    assert outcome.handoff == {"handoff": "nothing to push", "branch": "agent/fix-backup"}
    assert forge.prs == [] and ahead(worktree) == 0 and orch.of("diff") == []


def test_the_snapshot_leaves_excluded_paths_out(worktree):
    clean = gitdiff.snapshot(str(worktree))
    (worktree / ".zshrc").write_text("")
    assert gitdiff.snapshot(str(worktree)) != clean
    assert gitdiff.snapshot(str(worktree), [".zshrc"]) == clean
    assert (worktree / ".zshrc").exists()


def test_busy_settles_and_off_linux_is_never_busy(worktree, monkeypatch):
    if not os.path.isdir("/proc/self"):
        assert placeholders.busy(str(worktree)) is False
    monkeypatch.setattr(placeholders, "busy", lambda cwd: True)
    assert placeholders.settle(str(worktree), wait=0.05, step=0.01) is False


def test_a_placeholder_directory_and_its_new_parent_are_removed(worktree):
    existed = placeholders.present(str(worktree))
    assert existed == frozenset()
    (worktree / ".claude" / "hooks").mkdir(parents=True)
    (worktree / ".vscode").mkdir()
    (worktree / ".idea").mkdir()
    (worktree / ".idea" / "workspace.xml").write_text("<x/>")
    removed = placeholders.sweep(str(worktree), existed)
    assert sorted(removed) == [".claude/hooks", ".vscode"]
    assert not (worktree / ".claude").exists() and (worktree / ".idea" / "workspace.xml").exists()


def test_no_record_of_the_start_removes_nothing(worktree):
    (worktree / ".bashrc").write_text("")
    assert placeholders.sweep(str(worktree), None) == []
    assert (worktree / ".bashrc").exists()


def test_the_note_asks_for_nothing_left_running():
    note = handoff_mod.brief_note("fix-backup")
    assert note.startswith("How this run is handed off:") and "describes what actually happened." in note
    assert note.endswith("Do not leave a command running in the background or schedule a later wake-up: the run "
                         f"ends when your turn ends, and anything still running then is lost.\n\n{handoff_mod.EDIT_NOTE}")

