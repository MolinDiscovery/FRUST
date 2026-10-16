"""Keep mocked workflow calculations separate from configured user libraries."""

import pytest


@pytest.fixture(autouse=True)
def isolate_reference_store(monkeypatch):
    """Disable implicit shared publication unless a test explicitly enables it."""
    monkeypatch.setenv("FRUST_REFERENCE_STORE", "")
