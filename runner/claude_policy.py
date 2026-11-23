"""The managed settings a Claude Code session requires, checked before every start and reconnect.

Claude Code merges its managed settings after the settings the runner passes at launch, so managed
values win where they are present. The runner therefore checks that the managed file itself turns the
sandbox on, makes it fail when it cannot start, refuses unsandboxed commands and sets the network
allowlist that no other settings file can widen. The file and its directory must be owned by root and
writable by no one else, so a session cannot rewrite or replace it. examples/managed-settings.json shows
the required content.

The check reads one file, this platform's managed settings file, and the drop-in directory beside it.
Claude Code ranks policy from its vendor's servers and OS-level policy above that file. When one of them
delivers a policy, Claude Code by default ignores the file, and the check cannot see that. It cannot vouch
for settings it does not read, so it refuses when drop-in files sit beside the file, when the file names a
policy helper program, and on a platform it has no path for.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import sys
from os import fstat
from typing import Any

# Where Claude Code reads its managed settings file on each platform the check supports.
MANAGED_PATHS = {"linux": "/etc/claude-code/managed-settings.json",
                 "darwin": "/Library/Application Support/ClaudeCode/managed-settings.json"}
# Claude Code merges any *.json files in this directory, beside the file, after it.
DROP_INS = "managed-settings.d"
# Keys that hand the policy to a program Claude Code runs at startup, whose output this check never sees.
HELPER_KEYS = ("policyHelper", "policyHelpers")
PLATFORM = sys.platform
# None means the documented path for PLATFORM. Tests point it at a temp file.
MANAGED_SETTINGS: str | None = None
SANDBOX_FLAGS = {"enabled": True, "failIfUnavailable": True, "allowUnsandboxedCommands": False}
START_REFUSED = (
    "Claude Code could not start because its managed settings {problem}, so install them at "
    "{path} as a root-owned file writable only by root, with the sandbox settings and "
    "network allowlist that examples/managed-settings.json shows.")
UNSUPPORTED = ("Claude Code could not start because the runner can check its managed settings only on Linux "
               "and macOS, and this machine reports {platform}.")
MISSING = "are missing"
UNREADABLE = "could not be read as a JSON object"
LINKED = "are a symbolic link, which the runner does not follow"
WRONG_OWNER = "are not owned by root"
WRITABLE = "are writable by group or other users"
DIRECTORY = "sit in a directory that root does not own or that group or other users can write to"
INCOMPLETE = "lack the required sandbox settings or network allowlist"
HELPER = "name a policyHelper program, whose output the runner cannot check"
DROP_IN_DIRECTORY = (f"have a {DROP_INS} beside them that is not a root-owned directory writable only by root, "
                     "or that the runner cannot list")
EXTENDED = f"are extended by files in {DROP_INS}, which the runner does not check"


class ClaudePolicyError(Exception):
    ...


def managed_settings_path() -> str | None:
    return MANAGED_PATHS.get(PLATFORM)


def allowlist(sandbox: dict[str, Any]) -> bool:
    network = sandbox.get("network")
    if not isinstance(network, dict) or network.get("allowManagedDomainsOnly") is not True:
        return False
    domains = network.get("allowedDomains")
    return isinstance(domains, list) and bool(domains) and all(isinstance(d, str) and d.strip() for d in domains)


def root_only(metadata: Any) -> bool:
    return metadata.st_uid == 0 and not metadata.st_mode & 0o022


def _open_directory(path: str, dir_fd: int | None = None, follow: bool = True) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | (0 if follow else os.O_NOFOLLOW)
    return os.open(path, flags, dir_fd=dir_fd)


# Claude Code would merge a drop-in file that this check never reads. The directory is opened beside the
# file through the checked directory's descriptor, without following a link, and judged from its own
# descriptor. Returns None when there is no such directory, DROP_IN_DIRECTORY when it is anything but a
# listable root-only directory, EXTENDED when it holds a *.json file, and "" otherwise.
def drop_in_state(directory: int) -> str | None:
    try:
        handle = _open_directory(DROP_INS, dir_fd=directory, follow=False)
    except FileNotFoundError:
        return None
    except OSError:
        return DROP_IN_DIRECTORY
    try:
        if not root_only(fstat(handle)):
            return DROP_IN_DIRECTORY
        names = os.listdir(handle)
    except OSError:
        return DROP_IN_DIRECTORY
    finally:
        os.close(handle)
    return EXTENDED if any(name.lower().endswith(".json") for name in names) else ""


# The directory is opened once and the file is opened through that descriptor without following a link, so
# ownership and mode of both come from what was read, not from a second look at the path. The directory
# itself is reached by its path, so its parents are trusted as the system's own.
def read_managed_settings(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    info: dict[str, Any] = {"path": path, "readable": False}
    try:
        directory = _open_directory(os.path.dirname(path))
    except FileNotFoundError:
        return {**info, "missing": True}, {}
    except OSError:
        return info, {}
    try:
        directory_safe = root_only(fstat(directory))

        def opener(name: str, flags: int) -> int:
            return os.open(os.path.basename(name), flags | os.O_NOFOLLOW, dir_fd=directory)

        with open(path, encoding="utf-8", opener=opener) as handle:
            metadata = fstat(handle.fileno())
            data = json.load(handle)
        drop_ins = drop_in_state(directory)
    except FileNotFoundError:
        return {**info, "missing": True}, {}
    except OSError as exc:
        return {**info, "linked": exc.errno == errno.ELOOP}, {}
    except ValueError:
        return info, {}
    finally:
        os.close(directory)
    if not isinstance(data, dict):
        return info, {}
    sandbox = data.get("sandbox")
    sandbox = sandbox if isinstance(sandbox, dict) else {}
    info |= {"readable": True, "owner_root": metadata.st_uid == 0, "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
             "writable_by_others": bool(metadata.st_mode & 0o022),
             "directory_root_only": directory_safe,
             "sandbox_enabled": sandbox.get("enabled") is True,
             "fail_if_unavailable": sandbox.get("failIfUnavailable") is True,
             "unsandboxed_commands_disallowed": sandbox.get("allowUnsandboxedCommands") is False,
             "network_allowlist_present": allowlist(sandbox),
             "policy_helper": any(key in data for key in HELPER_KEYS),
             "drop_ins": drop_ins}
    return info, data


def sandbox_problem(info: dict[str, Any]) -> str | None:
    if not info.get("readable"):
        if info.get("missing"):
            return MISSING
        return LINKED if info.get("linked") else UNREADABLE
    if not info.get("owner_root"):
        return WRONG_OWNER
    if info.get("writable_by_others"):
        return WRITABLE
    if not info.get("directory_root_only"):
        return DIRECTORY
    if not all(info.get(flag) for flag in ("sandbox_enabled", "fail_if_unavailable",
                                           "unsandboxed_commands_disallowed", "network_allowlist_present")):
        return INCOMPLETE
    if info.get("policy_helper"):
        return HELPER
    return info.get("drop_ins") or None


def require_managed_settings(path: str | None = None) -> None:
    documented = managed_settings_path()
    if documented is None:
        raise ClaudePolicyError(UNSUPPORTED.format(platform=PLATFORM))
    info, _ = read_managed_settings(path or MANAGED_SETTINGS or documented)
    if problem := sandbox_problem(info):
        raise ClaudePolicyError(START_REFUSED.format(problem=problem, path=documented))
