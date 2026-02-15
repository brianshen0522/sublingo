"""Tests for subtitle parsing."""

from pathlib import Path

from sublingo.core.subtitle_parser import SubtitleEntry, parse_file, parse_string

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_srt():
    entries = parse_file(FIXTURES / "sample.srt")
    assert len(entries) == 5
    assert entries[0].text == "Hello, how are you?"
    assert entries[0].start == 1000
    assert entries[0].end == 4000


def test_parse_vtt():
    entries = parse_file(FIXTURES / "sample.vtt")
    assert len(entries) == 3
    assert entries[1].text == "I'm doing well, thank you."


def test_parse_ass():
    entries = parse_file(FIXTURES / "sample.ass")
    assert len(entries) == 3
    assert entries[2].text == "What a beautiful day it is today."


def test_parse_string_srt():
    content = """\
1
00:00:01,000 --> 00:00:04,000
Test line one

2
00:00:05,000 --> 00:00:08,000
Test line two
"""
    entries = parse_string(content, format="srt")
    assert len(entries) == 2
    assert entries[0].text == "Test line one"


def test_entry_has_timing():
    entries = parse_file(FIXTURES / "sample.srt")
    for entry in entries:
        assert entry.start < entry.end
        assert isinstance(entry.start, int)
        assert isinstance(entry.end, int)
