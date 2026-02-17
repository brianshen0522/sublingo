"""Tests for TVDB context filename parsing."""

import pytest

from sublingo.services.tvdb_context import parse_series_info, resolve_tvdb_language


class TestParseSeriesInfo:
    """Tests for parse_series_info with various filename formats."""

    def test_sonarr_style(self):
        result = parse_series_info("South Park - S01E01 - Cartman Gets an Anal Probe.srt")
        assert result == {"series": "South Park", "season": 1, "episode": 1}

    def test_sonarr_style_high_numbers(self):
        result = parse_series_info("Breaking Bad - S05E16 - Felina.srt")
        assert result == {"series": "Breaking Bad", "season": 5, "episode": 16}

    def test_dot_separated(self):
        result = parse_series_info("South.Park.S01E01.720p.BluRay.srt")
        assert result == {"series": "South Park", "season": 1, "episode": 1}

    def test_dot_separated_complex(self):
        result = parse_series_info("The.Office.US.S02E10.1080p.WEB.srt")
        assert result == {"series": "The Office US", "season": 2, "episode": 10}

    def test_numbered_format(self):
        result = parse_series_info("South Park 1x01.srt")
        assert result == {"series": "South Park", "season": 1, "episode": 1}

    def test_numbered_format_high(self):
        result = parse_series_info("Friends 10x18.srt")
        assert result == {"series": "Friends", "season": 10, "episode": 18}

    def test_case_insensitive(self):
        result = parse_series_info("south.park.s01e01.srt")
        assert result == {"series": "south park", "season": 1, "episode": 1}

    def test_no_match_movie(self):
        result = parse_series_info("Inception.2010.1080p.BluRay.srt")
        assert result is None

    def test_no_match_plain(self):
        result = parse_series_info("subtitles.srt")
        assert result is None

    def test_vtt_extension(self):
        result = parse_series_info("Dark - S01E01 - Secrets.vtt")
        assert result == {"series": "Dark", "season": 1, "episode": 1}

    def test_ass_extension(self):
        result = parse_series_info("Steins;Gate - S01E01.ass")
        assert result == {"series": "Steins;Gate", "season": 1, "episode": 1}


class TestResolveTvdbLanguage:
    """Tests for resolve_tvdb_language."""

    def test_known_code(self):
        assert resolve_tvdb_language("ja") == "jpn"

    def test_chinese_traditional(self):
        assert resolve_tvdb_language("zh-TW") == "zhtw"

    def test_english(self):
        assert resolve_tvdb_language("en") == "eng"

    def test_unknown_code(self):
        assert resolve_tvdb_language("xx-YY") is None
