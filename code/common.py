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
    return cfg


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
