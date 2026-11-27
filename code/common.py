"""Shared paths, config loading and JSONL helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"                 # git-ignored: article full text
ARTICLES = DATA / "articles.jsonl"     # one row per fetched article (metadata only)
EXTRACTIONS = DATA / "extractions.jsonl"  # one row per article: LLM output + geocode
GEOCACHE = DATA / "geocode_cache.json"
GEOJSON = ROOT / "docs" / "data" / "events.geojson"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm = cfg["llm"]
    for key in ("backend", "model", "base_url", "api_key"):
        env = os.getenv(f"HWC_LLM_{key.upper()}")
        if env:
            llm[key] = env
    # Run-time search overrides (set from the workflow's "Run workflow" form)
    s = cfg["search"]
    if os.getenv("HWC_TAXON", "").strip():
        s["taxon_filter"] = os.environ["HWC_TAXON"].strip()
    if os.getenv("HWC_PERIOD", "").strip():
        s["period"] = os.environ["HWC_PERIOD"].strip()
    for key in ("start_date", "end_date"):
        v = os.getenv(f"HWC_{key.upper()}", "").strip()
        if v:
            s[key] = v
    if os.getenv("HWC_MAX_QUERIES", "").strip():
        s["max_queries"] = int(os.environ["HWC_MAX_QUERIES"])
    if os.getenv("HWC_MAX_NEW_ARTICLES", "").strip():
        s["max_new_articles"] = int(os.environ["HWC_MAX_NEW_ARTICLES"])
    _validate_window(s)
    return cfg


def _validate_window(s: dict) -> None:
    """Fail fast on malformed dates; a date window overrides `period`."""
    from datetime import date
    start, end = s.get("start_date"), s.get("end_date")
    if not (start or end):
        return
    try:
        d0 = date.fromisoformat(start) if start else None
        d1 = date.fromisoformat(end) if end else date.today()
    except ValueError as e:
        raise SystemExit(f"start_date/end_date must be YYYY-MM-DD: {e}")
    if d0 and d0 > d1:
        raise SystemExit(f"start_date {d0} is after end_date {d1}")
    if d0 is None:
        raise SystemExit("end_date given without start_date")
    s["end_date"] = d1.isoformat()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_title(title: str) -> str:
    """Google News titles end with ' - Publisher'; strip it and normalise
    so the same wire story syndicated by several outlets collapses."""
    title = re.sub(r"\s+-\s+[^-]+$", "", title or "")
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def article_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
