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
        cached = self.cache.get(q)
        if cached is not None and all("addresstype" in h for h in cached):
            return cached
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
                    "addresstype": h.get("addresstype"),
                    "state": h.get("address", {}).get("state"),
                    "country_code": h.get("address", {}).get("country_code")} for h in r.json()]
        except Exception as e:
            log.warning("geocode error for %r: %s", q, e)
            return []  # not cached: retry next run
        finally:
            self._last = time.time()
        self.cache[q] = res
        return res

    @staticmethod
    def _level(h: dict) -> str:
        """Precision of an OSM hit: a state/province polygon is not a point location."""
        at = h.get("addresstype") or h.get("type")
        if at in ("state", "province", "territory", "region"):
            return "admin1"
        if at in ("country", "continent"):
            return "none"
        return "place"

    def geocode(self, place: str | None, admin1: str | None, country: str | None) -> dict:
        clean = lambda v: None if (v is None or _norm(v) in ("", "unknown", "null", "none", "n/a")) else v
        place, admin1, country = clean(place), clean(admin1), clean(country)
        cc = {"united states": "us", "canada": "ca", "mexico": "mx"}.get(_norm(country))
        out = lambda h, q, lvl: {"lat": h["lat"], "lon": h["lon"], "geocode_level": lvl,
                                 "geocode_query": q, "geocode_match": h["display_name"]}
        if place:
            q = ", ".join(p for p in (place, admin1, country) if p)
            for h in self._query(q):
                in_admin = admin1 is None or _norm(h["state"]) == _norm(admin1)
                in_country = cc is None or h.get("country_code") == cc
                if in_admin and in_country and self._level(h) != "none":
                    return out(h, q, self._level(h))
        if admin1:
            q = ", ".join(p for p in (admin1, country) if p)
            for h in self._query(q):
                if self._level(h) == "admin1" or _norm(h["state"]) == _norm(admin1):
                    return out(h, q, "admin1")
        return {"lat": None, "lon": None, "geocode_level": "none",
                "geocode_query": None, "geocode_match": None}
