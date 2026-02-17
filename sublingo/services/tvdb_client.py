"""TVDB API v4 client for fetching series and episode metadata."""

from __future__ import annotations

import time
from typing import Any

import httpx

from sublingo.utils.logger import get_logger

logger = get_logger(__name__)

TVDB_BASE_URL = "https://api4.thetvdb.com/v4"
TOKEN_EXPIRY_SECONDS = 25 * 24 * 3600  # 25 days


class TVDBClient:
    """Lightweight TVDB API v4 client with token caching."""

    def __init__(self, api_key: str, timeout: float = 15.0):
        self.api_key = api_key
        self.timeout = timeout
        self._token: str | None = None
        self._token_time: float = 0
        self._series_cache: dict[str, int | None] = {}

    def _ensure_token(self) -> None:
        """Authenticate with TVDB and cache the bearer token."""
        if self._token and (time.time() - self._token_time) < TOKEN_EXPIRY_SECONDS:
            return
        resp = httpx.post(
            f"{TVDB_BASE_URL}/login",
            json={"apikey": self.api_key},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        self._token = resp.json()["data"]["token"]
        self._token_time = time.time()
        logger.debug("TVDB: authenticated successfully")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Authenticated GET request to TVDB API."""
        self._ensure_token()
        resp = httpx.get(
            f"{TVDB_BASE_URL}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def search_series(self, name: str) -> int | None:
        """Search for a series by name and return its TVDB ID, or None."""
        if name in self._series_cache:
            return self._series_cache[name]
        data = self._get("/search", params={"query": name, "type": "series"})
        results = data.get("data", [])
        series_id = int(results[0]["tvdb_id"]) if results else None
        self._series_cache[name] = series_id
        return series_id

    def get_series_translation(
        self, series_id: int, lang: str
    ) -> dict[str, str] | None:
        """Fetch series translation (name + overview) for a language code."""
        try:
            data = self._get(f"/series/{series_id}/translations/{lang}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("TVDB: no series translation for lang %s", lang)
                return None
            raise
        translation = data.get("data")
        if not translation:
            return None
        return {
            "name": translation.get("name", ""),
            "overview": translation.get("overview", ""),
        }

    def get_episode_id(
        self, series_id: int, season: int, episode: int
    ) -> int | None:
        """Find the episode ID for a given season/episode number."""
        data = self._get(
            f"/series/{series_id}/episodes/default",
            params={"season": season, "episodeNumber": episode, "page": 0},
        )
        episodes = data.get("data", {}).get("episodes", [])
        for ep in episodes:
            if ep.get("seasonNumber") == season and ep.get("number") == episode:
                return int(ep["id"])
        return None

    def get_episode_translation(
        self, episode_id: int, lang: str
    ) -> dict[str, str] | None:
        """Fetch episode translation (name + overview) for a language code."""
        try:
            data = self._get(f"/episodes/{episode_id}/translations/{lang}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("TVDB: no episode translation for lang %s", lang)
                return None
            raise
        translation = data.get("data")
        if not translation:
            return None
        return {
            "name": translation.get("name", ""),
            "overview": translation.get("overview", ""),
        }
