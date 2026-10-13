"""Shared fixtures. Offline tests are hermetic: outbound network connections are blocked."""

from __future__ import annotations

import os
import socket

import pytest

from orbitwatch.context import DataContext
from orbitwatch.tools import Toolbox


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="also run tests that hit real services")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live") or os.environ.get("ORBITWATCH_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live test: run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _offline_guard(request, monkeypatch, tmp_path):
    """Block non-loopback sockets and keep caches inside the test's temp dir."""
    monkeypatch.setenv("ORBITWATCH_CACHE_DIR", str(tmp_path / "cache"))
    if "live" in request.keywords:
        return
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(self, address)
        raise RuntimeError(f"Network access is blocked in offline tests (tried {address!r}).")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(scope="session")
def fixture_work_dir(tmp_path_factory):
    """Synthetic rasters are deterministic, so generate them once per test session."""
    return tmp_path_factory.mktemp("fixture-mode")


@pytest.fixture
def fixture_ctx(fixture_work_dir) -> DataContext:
    return DataContext.fixtures(fixture_work_dir)


@pytest.fixture
def toolbox(fixture_ctx) -> Toolbox:
    return Toolbox(fixture_ctx)
