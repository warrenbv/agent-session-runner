"""The socket gate: tests that need sockets are skipped only where the environment refuses them."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import conftest


@pytest.mark.parametrize("failure", [None, "create", "bind", "connect"])
def test_loopback_probe_detects_refusal_and_success(monkeypatch, failure):
    factory = MagicMock()
    peer = factory.return_value.__enter__.return_value
    if failure == "create":
        factory.side_effect = PermissionError("sandbox")
    elif failure is not None:
        getattr(peer, failure).side_effect = PermissionError("sandbox")
    monkeypatch.setattr(conftest.socket, "socket", factory)
    assert conftest._sockets_refused() is (failure is not None)
    if failure != "create":
        factory.return_value.__exit__.assert_called()


def _item(file, name, *, fixtures=(), markers=(), cls=None):
    added = []
    return SimpleNamespace(path=Path(file), originalname=name, name=name + "[parameter]",
                           fixturenames=fixtures, cls=cls, iter_markers=lambda marker: iter(markers),
                           add_marker=added.append, added=added)


@pytest.mark.parametrize("refused", [False, True])
def test_collection_skips_only_socket_users_and_only_under_refusal(monkeypatch, refused):
    needs = [
        _item("test_session.py", "test_session", fixtures=("client", "orch")),
        _item("test_mcp_shim.py", "test_socket", markers=(True,)),
    ]
    ordinary = [
        _item("test_mcp_shim.py", "test_the_socket_handler_times_out_bind_free"),
        _item("test_normalise.py", "test_normalise"),
    ]
    monkeypatch.setattr(conftest, "SOCKETS_REFUSED", refused)
    conftest.pytest_collection_modifyitems(needs + ordinary)
    assert all(len(item.added) == int(refused) for item in needs)
    assert all(item.added == [] for item in ordinary)
    assert conftest.SOCKET_SKIP_REASON == "needs sockets, which this sandbox refuses"
    for item in needs:
        for marker in item.added:
            assert marker.name == "skip" and marker.kwargs == {"reason": conftest.SOCKET_SKIP_REASON}

