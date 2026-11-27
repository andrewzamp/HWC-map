"""Deterministic geocoding of LLM-extracted place names (OSM Nominatim).

Strategy: try "<place>, <state/province>, <country>"; accept the top hit only
if it falls inside the state/province the article names (guards against
homonyms such as Banff, AB vs Banff, Scotland or the dozens of US
'Springfield's). Otherwise fall back to the state/province centroid and
record the coarser `geocode_level`, so map users and analyses can tell
point-precision from region-precision records.
"""
from __future__ import annotations

import json
import logging
import time
import unicodedata

import requests

from common import GEOCACHE

log = logging.getLogger("geocode")
URL = "https://nominatim.openstreetmap.org/search"


def _norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return s.lower().strip()


class Geocoder:
    def __init__(self, cfg: dict):
        self.ua = cfg["user_agent"]
        self.cc = ",".join(cfg["countrycodes"])
        self.min_interval = cfg["min_interval_s"]
        self._last = 0.0
        self.cache: dict = json.loads(GEOCACHE.read_text()) if GEOCACHE.exists() else {}

    def save(self) -> None:
        GEOCACHE.parent.mkdir(parents=True, exist_ok=True)
        GEOCACHE.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False, sort_keys=True))

    def _query(self, q: str) -> list[dict]:
        if q in self.cache:
            return self.cache[q]
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(URL, params={"q": q, "format": "jsonv2", "addressdetails": 1,
                                          "limit": 5, "countrycodes": self.cc},
                             headers={"User-Agent": self.ua}, timeout=30)
            r.raise_for_status()
            res = [{"lat": float(h["lat"]), "lon": float(h["lon"]),
                    "display_name": h["display_name"], "type": h.get("type"),
                    "state": h.get("address", {}).get("state"),
                    "country_code": h.get("address", {}).get("country_code")} for h in r.json()]
        except Exception as e:
            log.warning("geocode error for %r: %s", q, e)
            return []  # not cached: retry next run
        finally:
            self._last = time.time()
        self.cache[q] = res
        return res

    def geocode(self, place: str | None, admin1: str | None, country: str | None) -> dict:
        country = None if country in (None, "unknown") else country
        if place:
            q = ", ".join(p for p in (place, admin1, country) if p)
            for h in self._query(q):
                if admin1 is None or _norm(h["state"]) == _norm(admin1):
                    return {"lat": h["lat"], "lon": h["lon"], "geocode_level": "place",
                            "geocode_query": q, "geocode_match": h["display_name"]}
        if admin1:
            q = ", ".join(p for p in (admin1, country) if p)
            for h in self._query(q):
                if _norm(h["state"]) == _norm(admin1) or h["type"] in ("administrative", "state", "province"):
                    return {"lat": h["lat"], "lon": h["lon"], "geocode_level": "admin1",
                            "geocode_query": q, "geocode_match": h["display_name"]}
        return {"lat": None, "lon": None, "geocode_level": "none",
                "geocode_query": None, "geocode_match": None}
