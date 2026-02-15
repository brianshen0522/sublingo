"""Tests for subtitle batching."""

import pytest

from sublingo.core.batching import create_batches
from sublingo.core.subtitle_parser import SubtitleEntry


def _make_entries(n: int) -> list[SubtitleEntry]:
    return [
        SubtitleEntry(index=i, start=i * 1000, end=(i + 1) * 1000, text=f"Line {i}")
        for i in range(n)
    ]


def test_single_batch():
    entries = _make_entries(5)
    batches = create_batches(entries, batch_size=10)
    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_multiple_batches():
    entries = _make_entries(25)
    batches = create_batches(entries, batch_size=10)
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5


def test_exact_batch_size():
    entries = _make_entries(20)
    batches = create_batches(entries, batch_size=10)
    assert len(batches) == 2
    assert all(len(b) == 10 for b in batches)


def test_empty_entries():
    batches = create_batches([], batch_size=10)
    assert batches == []


def test_batch_size_one():
    entries = _make_entries(3)
    batches = create_batches(entries, batch_size=1)
    assert len(batches) == 3
    assert all(len(b) == 1 for b in batches)


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        create_batches(_make_entries(5), batch_size=0)
