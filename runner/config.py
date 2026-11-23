"""The environment the runner reads, and the secrets it removes from it before any provider starts.

    RUNNER_CONTROL_URL     the control plane's base URL (https, or http to the loopback)
    RUNNER_CONTROL_HOST    an optional Host header for every call
    RUNNER_CA_FILE         an optional CA bundle for that URL
    RUNNER_TOKEN           the session's token, or RUNNER_TOKEN_FILE, a 0600 file holding it
    RUNNER_GITHUB_TOKEN    an optional GitHub token for the hand-off, or RUNNER_GITHUB_TOKEN_FILE
    CLAUDE_CODE_OAUTH_TOKEN_FILE  an optional 0600 file with the Claude login token
    RUNNER_PASS_ENV        optional, a comma-separated list of further variables the CLIs may inherit

Apart from the MCP shim's one read of its socket path, this is the only module that touches
`os.environ` (a test enforces both). Both CLIs run as child processes that inherit the runner's
environment, so anything left there would sit inside the model's sandbox. The tokens are therefore
read, held in memory and deleted from the environment. Then every variable that is not on a short
list (PASSED below, plus the names in RUNNER_PASS_ENV) is deleted too, whatever it is called. The
secrets and the runner's own variables never pass, even if RUNNER_PASS_ENV names them. The Claude token
is never in the runner's own environment. The Claude provider puts it in the Claude CLI's environment
through the SDK's per-child environment, and Codex never receives it. The CLIs can still
read their own configuration and credential files under the home directory. See
docs/security-model.md, "Secrets and the environment".
"""

from __future__ import annotations

import ipaddress
import os
import re
import stat
from dataclasses import dataclass, field
from typing import MutableMapping
from urllib.parse import urlsplit

DEFAULT_CONTROL_URL = "https://control.example.internal"
STRIPPED = ("RUNNER_TOKEN", "RUNNER_GITHUB_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")


@dataclass(frozen=True)
class Settings:
    control_url: str = DEFAULT_CONTROL_URL
    ca_file: str | None = None
    control_host: str | None = None
    token: str = field(default="", repr=False)
    github_token: str = field(default="", repr=False)
    claude_token: str = field(default="", repr=False)


def is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


# Plain http only to the loopback, so the session token and the event stream never cross a network in clear
# text. An address of one of this machine's own interfaces is not enough, because that address can later
# belong to another machine. Everything else needs https (docs/security-model.md, "The control-plane address").
def check_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname or parts.query or parts.fragment \
            or parts.username or parts.password:
        raise ValueError("RUNNER_CONTROL_URL must be https://host[:port][/path]")
    if parts.scheme == "http" and not is_loopback(parts.hostname):
        raise ValueError("RUNNER_CONTROL_URL may use http only to the loopback, by the name localhost or a "
                         f"loopback address, and {parts.hostname} is neither, so use https")
    return url.rstrip("/")


HOST_HEADER = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?(:\d{1,5})?$")


def check_host(value: str | None) -> str | None:
    if not value:
        return None
    if not HOST_HEADER.match(value):
        raise ValueError("RUNNER_CONTROL_HOST must be a hostname such as control.example.internal")
    return value


def _one_line(value: str, what: str) -> str:
    value = value.strip()
    if not value or any(c.isspace() for c in value) or len(value) > 4096:
        raise ValueError(f"{what} must be one non-empty line")
    return value


def read_token_file(path: str) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("the token file must be a regular file")
        if info.st_mode & 0o077:
            raise ValueError("the token file must be mode 0600 (no group or other access)")
        if info.st_uid != os.getuid():
            raise ValueError("the token file must belong to the runner's user")
        with os.fdopen(fd, encoding="utf-8", closefd=False) as handle:
            return _one_line(handle.read(4097), "the token file")
    finally:
        os.close(fd)


def git_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ}
    # No system or user git configuration and no system attributes: only the checked repository config.
    env |= {"LANG": "C", "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_ATTR_NOSYSTEM": "1"}
    return env | (extra or {})


def take_secrets(env: MutableMapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    url = check_url(env.get("RUNNER_CONTROL_URL") or DEFAULT_CONTROL_URL)
    ca_file = env.get("RUNNER_CA_FILE") or None
    raw_token = env.get("RUNNER_TOKEN") or ""
    token_file = env.get("RUNNER_TOKEN_FILE") or ""
    github = env.get("RUNNER_GITHUB_TOKEN") or ""
    github_file = env.get("RUNNER_GITHUB_TOKEN_FILE") or ""
    for name in STRIPPED:
        env.pop(name, None)
    if raw_token:
        token = _one_line(raw_token, "RUNNER_TOKEN")
    elif token_file:
        token = read_token_file(token_file)
    else:
        raise ValueError("set RUNNER_TOKEN or RUNNER_TOKEN_FILE")
    if github:
        github = _one_line(github, "RUNNER_GITHUB_TOKEN")
    elif github_file:
        github = read_token_file(github_file)
    settings = Settings(control_url=url, ca_file=ca_file, token=token, github_token=github,
                        control_host=check_host(env.get("RUNNER_CONTROL_HOST")),
                        claude_token=claude_oauth_token(env))
    scrub_for_providers(env)
    return settings


# Variables the runner reads, none of which a CLI needs. They never pass, even if RUNNER_PASS_ENV names them.
RUNNER_ONLY = (*STRIPPED, "RUNNER_TOKEN_FILE", "RUNNER_GITHUB_TOKEN_FILE", "CLAUDE_CODE_OAUTH_TOKEN_FILE",
               "RUNNER_CONTROL_URL", "RUNNER_CONTROL_HOST", "RUNNER_CA_FILE")
RUNNER_PREFIXES = ("RUNNER_",)
PASS_ENV = "RUNNER_PASS_ENV"

# What the CLIs inherit. Both SDKs start their CLI from a copy of os.environ and cannot drop a
# variable, so everything else leaves the environment once the runner has read what it needs.
# Proxy variables are left off because a proxy URL can carry a password.
PASSED = (
    # Finding programs, the home directory (and the CLIs' configuration under it) and the user's shell.
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    # Temp directory, language, terminal type and time zone.
    "TMPDIR", "LANG", "TERM", "TZ",
    # The locale categories by name: LC_ALL, the six POSIX ones, then the six glibc adds. Not every name
    # starting LC_, because SSH setups commonly forward any LC_ variable, so the prefix could carry anything.
    "LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MESSAGES", "LC_MONETARY", "LC_NUMERIC", "LC_TIME",
    "LC_ADDRESS", "LC_IDENTIFICATION", "LC_MEASUREMENT", "LC_NAME", "LC_PAPER", "LC_TELEPHONE",
    # macOS text encoding, part of the base environment Codex gives its own MCP servers.
    "__CF_USER_TEXT_ENCODING",
    # Trust stores for a private certificate authority: OpenSSL's, Node's (Claude Code), Codex's own.
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "CODEX_CA_CERTIFICATE",
    # Each CLI's configuration directory when it is not the default under HOME.
    "CLAUDE_CONFIG_DIR", "CODEX_HOME",
)
VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


def runner_only(name: str) -> bool:
    return name in RUNNER_ONLY or name.startswith(RUNNER_PREFIXES)


# A refusal goes to stderr in a traceback, and an operator who mistypes an entry as NAME=value would see
# the value there, so it names the entry's position instead of quoting it. Only the runner's own variable
# names, which are not values, are quoted.
def operator_names(value: str) -> frozenset[str]:
    names = set()
    for position, part in enumerate(value.split(","), start=1):
        name = part.strip()
        if not name:
            continue
        if not VARIABLE_NAME.fullmatch(name):
            raise ValueError(f"{PASS_ENV} entry {position} is not a variable name. Each entry must be a name "
                             "such as AWS_REGION, with no value: letters, digits and underscores, not starting "
                             "with a digit, at most 128 characters")
        if name in RUNNER_ONLY or name == PASS_ENV:
            raise ValueError(f"{PASS_ENV} may not name {name}: the runner removes it before any CLI starts")
        if runner_only(name):
            raise ValueError(f"{PASS_ENV} entry {position} starts with RUNNER_, and the runner removes every "
                             "such variable before any CLI starts")
        names.add(name)
    return frozenset(names)


def passes(name: str, extra: frozenset[str] = frozenset()) -> bool:
    if runner_only(name):
        return False
    return name in PASSED or name in extra


def scrub_for_providers(env: MutableMapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    extra = operator_names(env.get(PASS_ENV) or "")
    names = sorted(name for name in list(env) if not passes(name, extra))
    for name in names:
        env.pop(name, None)
    return names


def claude_oauth_token(env: MutableMapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    path = env.get("CLAUDE_CODE_OAUTH_TOKEN_FILE") or ""
    return read_token_file(path) if path else ""
