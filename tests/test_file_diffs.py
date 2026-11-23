"""The whole session's change against its base, one event per file, split into parts past the bound."""

from __future__ import annotations

import asyncio

from conftest import fast_timing, git
from runner import gitdiff
from runner.events import PAYLOAD_CAP, size
from runner.providers.fake import FakeProvider
from runner.session import Caps, Options, Session


def _say(text="Done."):
    return {"type": "AssistantMessage", "model": "m", "parent_tool_use_id": None,
            "content": [{"type": "TextBlock", "text": text}]}


DONE = {"type": "ResultMessage", "subtype": "success", "duration_ms": 5, "duration_api_ms": 1, "is_error": False,
        "num_turns": 1, "session_id": "s", "total_cost_usd": 0.01, "terminal_reason": "completed"}


def _run(orch, client, worktree, turns, file_diffs=True, base_ref=None, prompts=()):
    for text in prompts:
        orch.queue(text)
    fixture = {"provider": "claude-code", "turns": [[*t, _say(), DONE] for t in turns], "default_turn": [_say(), DONE]}
    session = Session(FakeProvider(fixture), client, Options(cwd=str(worktree), file_diffs=file_diffs,
                                                             base_ref=base_ref),
                      Caps(wall_clock=30, grace=1), fast_timing())
    asyncio.run(asyncio.wait_for(session.run("first"), timeout=30))
    return session


def _diffs(orch):
    return [e["payload"] for e in orch.events if e["kind"] == "diff"]


def _write(path, text):
    return {"fake": "write", "path": path, "text": text}


def test_every_status_with_its_counts_and_patch(worktree):
    (worktree / "old name.txt").write_text("one\ntwo\nthree\nfour\nfive\n")
    (worktree / "gone.txt").write_text("bye\n")
    (worktree / "pic.bin").write_bytes(b"\x00\x01\x02")
    git(worktree, "add", "-A")
    git(worktree, "commit", "--quiet", "-m", "more")
    base = git(worktree, "rev-parse", "HEAD").strip()
    git(worktree, "mv", "old name.txt", "new name.txt")
    (worktree / "gone.txt").unlink()
    (worktree / "README.txt").write_text("hello\nworld\n")
    (worktree / "added.txt").write_text("fresh\n")
    (worktree / "pic.bin").write_bytes(b"\x00\x09\x02\x03")
    after = gitdiff.snapshot(str(worktree))
    sent: dict[str, str] = {}
    files, summary = gitdiff.session_diff(str(worktree), base, after, sent)
    by = {f["path"]: f for f in files}
    assert set(by) == {"new name.txt", "gone.txt", "README.txt", "added.txt", "pic.bin"}
    assert by["new name.txt"]["status"] == "renamed" and by["new name.txt"]["old_path"] == "old name.txt"
    assert (by["gone.txt"]["status"], by["gone.txt"]["removed"]) == ("deleted", 1)
    assert (by["added.txt"]["status"], by["added.txt"]["added"]) == ("added", 1)
    assert by["README.txt"]["status"] == "changed" and "+world" in by["README.txt"]["patch"]
    assert by["pic.bin"]["binary"] is True and by["pic.bin"]["patch"] == ""
    assert all(f["scope"] == "session_file" and f["to"] == after and not f["too_large"] for f in files)
    assert summary["scope"] == "session" and summary["base"] == base and summary["files"] == 5
    assert summary["complete"] is True and {r["path"] for r in summary["listed"]} == set(by)
    assert {r["path"]: r["sig"] for r in summary["listed"]} == {p: f["sig"] for p, f in by.items()} == sent
    again, summary2 = gitdiff.session_diff(str(worktree), base, after, sent)
    assert again == [] and summary2["listed"] == summary["listed"]


def _whole_patch(worktree, base, after, path):
    return gitdiff._file_patch(str(worktree), base, after, {"path": path, "old_path": None})


def _joined(files, path):
    parts = sorted((f for f in files if f["path"] == path), key=lambda f: f["part"])
    return parts, "".join(f["patch"] for f in parts)


def test_a_file_whose_patch_fits_one_event_is_sent_whole_with_no_part_fields(worktree):
    base = gitdiff.snapshot(str(worktree))
    (worktree / "small.txt").write_text("é\n" * 10)
    after = gitdiff.snapshot(str(worktree))
    files, _ = gitdiff.session_diff(str(worktree), base, after, {})
    [small] = files
    assert small["too_large"] is False and small["patch"] == _whole_patch(worktree, base, after, "small.txt")
    assert "part" not in small and "parts" not in small


def test_a_file_past_the_bound_is_sent_whole_in_several_parts(worktree):
    base = gitdiff.snapshot(str(worktree))
    (worktree / "big.txt").write_text("".join(f"line {n} \"quoted\"\ttab é\n" for n in range(5000)))
    (worktree / "small.txt").write_text("é\n" * 10)
    after = gitdiff.snapshot(str(worktree))
    files, summary = gitdiff.session_diff(str(worktree), base, after, {})
    parts, joined = _joined(files, "big.txt")
    assert len(parts) > 2 and joined == _whole_patch(worktree, base, after, "big.txt")
    assert [p["part"] for p in parts] == list(range(1, len(parts) + 1))
    assert {p["parts"] for p in parts} == {len(parts)}
    assert all(p["too_large"] is False and p["added"] == 5000 and p["sig"] == parts[0]["sig"] for p in parts)
    assert all(size(p) <= gitdiff.FILE_PATCH_BOUND for p in parts)
    assert all({k: v for k, v in p.items() if k not in ("patch", "part", "parts")}
               == {k: v for k, v in parts[0].items() if k not in ("patch", "part", "parts")} for p in parts)
    assert [f["path"] for f in files].index("small.txt") > max(i for i, f in enumerate(files) if f["path"] == "big.txt")
    small = next(f for f in files if f["path"] == "small.txt")
    assert small["too_large"] is False and "+é" in small["patch"] and "part" not in small
    assert all(size(f) <= PAYLOAD_CAP for f in files + [summary])


def test_past_the_change_s_ceiling_a_large_file_is_counted_as_before(worktree, monkeypatch):
    monkeypatch.setattr(gitdiff, "CHANGE_CEILING", 120_000)
    base = gitdiff.snapshot(str(worktree))
    body = "".join(f"line {n} of a long file\n" for n in range(3000))
    (worktree / "a-first.txt").write_text(body)
    (worktree / "b-second.txt").write_text(body)
    (worktree / "c-small.txt").write_text("small\n")
    after = gitdiff.snapshot(str(worktree))
    files, summary = gitdiff.session_diff(str(worktree), base, after, {})
    first, joined = _joined(files, "a-first.txt")
    assert len(first) > 1 and joined == _whole_patch(worktree, base, after, "a-first.txt")
    [second] = [f for f in files if f["path"] == "b-second.txt"]
    assert second["too_large"] is True and second["patch"] == "" and second["added"] == 3000
    assert "part" not in second and "parts" not in second
    [small] = [f for f in files if f["path"] == "c-small.txt"]
    assert small["too_large"] is False and small["patch"] and "part" not in small
    assert summary["files"] == 3 and summary["complete"] is True


def test_the_real_ceiling_is_400_kb_and_counts_files_not_sent_again(worktree, monkeypatch):
    assert gitdiff.CHANGE_CEILING == 400 * 1024
    base = gitdiff.snapshot(str(worktree))
    body = "".join(f"line {n} of a file of about two hundred kilobytes\n" for n in range(4400))
    (worktree / "a.txt").write_text(body)
    (worktree / "b.txt").write_text(body)
    after = gitdiff.snapshot(str(worktree))
    sent: dict[str, str] = {}
    sizes: dict[str, tuple[str, int]] = {}
    files, _ = gitdiff.session_diff(str(worktree), base, after, sent, sizes)
    assert all("part" in f for f in files if f["path"] == "a.txt")
    assert [f["too_large"] for f in files if f["path"] == "b.txt"] == [True]
    (worktree / "c.txt").write_text("".join(f"another {n}\n" for n in range(3000)))
    later = gitdiff.snapshot(str(worktree))
    diffed = []
    real = gitdiff._file_patch

    def record(cwd, b, a, entry):
        diffed.append(entry["path"])
        return real(cwd, b, a, entry)

    monkeypatch.setattr(gitdiff, "_file_patch", record)
    again, _ = gitdiff.session_diff(str(worktree), base, later, sent, sizes)
    assert diffed == ["c.txt"] and [f["path"] for f in again] == ["c.txt"]
    assert again[0]["too_large"] is True and "part" not in again[0]


def test_split_patch_keeps_every_character_and_its_bound():
    patch = "".join(f"+{'é' * (n % 7)}\"\\\x01 {n}\n" for n in range(2000)) + "x" * 5000
    pieces = gitdiff.split_patch(patch, 1000)
    assert "".join(pieces) == patch and len(pieces) > 10
    assert all(size({"p": p}) - size({"p": ""}) <= 1000 for p in pieces)
    assert all(p.endswith("\n") for p in pieces if "x" not in p)
    assert gitdiff.split_patch(patch, 10) == []


def test_a_summary_too_long_to_list_every_file_says_so(worktree):
    base = gitdiff.snapshot(str(worktree))
    for n in range(400):
        (worktree / f"{'d' * 60}-{n:03}.txt").write_text(f"{n}\n")
    files, summary = gitdiff.session_diff(str(worktree), base, gitdiff.snapshot(str(worktree)), {})
    assert summary["files"] == 400 and summary["added"] == 400 and summary["complete"] is False
    assert 0 < len(summary["listed"]) < 400 and size(summary) <= gitdiff.SUMMARY_BOUND
    assert len(files) == gitdiff.FILES_CAP


def test_the_file_event_never_carries_a_cut_patch():
    entry = {"path": "a.txt", "old_path": None, "status": "changed", "sig": "x"}
    whole = gitdiff.file_event(entry, (1, 0), "+a\n", "t", bound=1000)
    assert whole["patch"] == "+a\n" and whole["too_large"] is False
    over = gitdiff.file_event(entry, (900, 0), "+" + "a" * 2000, "t", bound=1000)
    assert over["patch"] == "" and over["too_large"] is True and over["added"] == 900


def test_with_the_flag_each_turn_that_changed_files_sends_the_session_s_change(orch, client, worktree):
    base = git(worktree, "rev-parse", "HEAD").strip()
    _run(orch, client, worktree, [[_write("a.txt", "a\n")], [_write("b.txt", "b\n")], []],
         base_ref=base, prompts=["second", "third"])
    scopes = [d["scope"] for d in _diffs(orch)]
    assert scopes == ["turn", "session_file", "session", "turn", "session_file", "session"]
    diffs = _diffs(orch)
    assert diffs[1]["path"] == "a.txt" and diffs[4]["path"] == "b.txt"
    assert diffs[2]["base"] == base and [r["path"] for r in diffs[5]["listed"]] == ["a.txt", "b.txt"]
    assert diffs[0] == {k: diffs[0][k] for k in ("scope", "from", "to", "files", "paths", "stat", "patch",
                                                  "patch_truncated")}


def test_a_session_posts_a_large_file_s_parts_before_the_summary(orch, client, worktree):
    base = git(worktree, "rev-parse", "HEAD").strip()
    _run(orch, client, worktree, [[_write("big.txt", "".join(f"line {n}\n" for n in range(5000)))]], base_ref=base)
    diffs = _diffs(orch)
    parts = [d for d in diffs if d["scope"] == "session_file"]
    assert len(parts) > 1 and [p["part"] for p in parts] == list(range(1, len(parts) + 1))
    assert "".join(p["patch"] for p in parts).count("\n+line ") == 5000
    assert diffs[-1]["scope"] == "session" and all(size(d) <= PAYLOAD_CAP for d in diffs)
    assert not any(d.get("truncated") for d in diffs)


def test_without_the_flag_only_the_turn_diff_is_sent(orch, client, worktree):
    _run(orch, client, worktree, [[_write("a.txt", "a\n")]], file_diffs=False)
    assert [d["scope"] for d in _diffs(orch)] == ["turn"]


def test_without_a_base_ref_the_base_is_the_worktree_as_the_first_turn_found_it(orch, client, worktree):
    (worktree / "before.txt").write_text("already here\n")
    _run(orch, client, worktree, [[_write("a.txt", "a\n")]])
    summary = _diffs(orch)[-1]
    assert summary["scope"] == "session" and [r["path"] for r in summary["listed"]] == ["a.txt"]


def test_the_runner_and_the_launcher_carry_the_flag():
    from runner.__main__ import parse
    base = ["--provider", "fake", "--cwd", "/tmp", "--prompt-file", "/tmp/p"]
    assert parse(base + ["--file-diffs"]).file_diffs is True and parse(base).file_diffs is False

