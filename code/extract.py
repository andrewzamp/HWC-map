"""Step 2 — full-text retrieval, LLM metadata extraction, geocoding.

* Incremental: each article is processed once per EXTRACTION_VERSION. Rows
  produced by an older schema/prompt version are dropped and redone, so
  improving the prompt automatically re-processes the archive.
* Full text is (re)attempted here for every article without cached text,
  so transient download failures heal on the next run.
* Each row is appended as soon as it's done (safe against Actions timeouts).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from common import ARTICLES, CACHE, EXTRACTIONS, append_jsonl, load_config, norm_title, read_jsonl, write_jsonl
from fetch_news import download_text, resolve_url
from geocode import Geocoder
from llm import LLMClient
from schema import EXTRACTION_VERSION, Extraction, field_guide

log = logging.getLogger("extract")

SYSTEM = f"""You are an information-extraction assistant for a scientific database of \
human-wildlife conflict involving bears (the animal) in North America.

Rules:
- Extract only what the text states or directly implies. Never invent details.
- When information is missing or ambiguous, use null or "unknown" rather than guessing.
- Many news items mention "bear spray" used by people on other people (robberies, assaults). \
These do NOT involve a bear: real_bear_involved=false, is_specific_event=false.
- event_date is when the incident happened, not when the article was published.
- Read the headline carefully: it often states species, outcome and place.
- Return one JSON object with exactly these fields:

{field_guide()}"""


def get_text(a: dict, cfg: dict) -> str | None:
    p = CACHE / f"{a['id']}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    if not cfg["fulltext"]["enabled"]:
        return None
    if not a.get("url"):
        a["url"] = resolve_url(a["gnews_url"], cfg["fulltext"]["request_interval_s"])
    text = download_text(a["url"]) if a.get("url") else None
    time.sleep(cfg["fulltext"]["request_interval_s"])
    if text and len(text) > 300:
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        a["has_fulltext"] = True
        return text
    return None


def build_prompt(a: dict, text: str | None, max_chars: int) -> tuple[str, str]:
    snippet = a.get("snippet") or ""
    # Google News RSS 'snippets' are usually just the headline repeated.
    snippet_is_headline = norm_title(snippet).startswith(norm_title(a.get("title", ""))[:40])
    if text:
        source, body, tag = "full article text", text[:max_chars], "fulltext"
    elif snippet and not snippet_is_headline:
        source, body, tag = "headline and short snippet only", snippet, "snippet"
    else:
        source, body, tag = "HEADLINE ONLY — leave everything not stated in it null/unknown", "", "headline"
    user = (f"Publication date: {a.get('published') or 'unknown'}\n"
            f"Outlet: {a.get('publisher') or 'unknown'}\n"
            f"Headline: {a.get('title')}\n"
            f"Text available: {source}\n"
            + (f"\n---\n{body}\n---" if body else ""))
    return user, tag


def main() -> int:
    cfg = load_config()
    rows = read_jsonl(EXTRACTIONS)
    current = [r for r in rows if r.get("extraction_version") == EXTRACTION_VERSION]
    if len(current) < len(rows):
        log.info("dropping %d rows from older extraction versions (will be redone)", len(rows) - len(current))
        write_jsonl(EXTRACTIONS, current)
    done = {r["id"] for r in current}
    articles = read_jsonl(ARTICLES)
    todo = [a for a in articles if a["id"] not in done]
    log.info("%d articles to process with %s:%s", len(todo), cfg["llm"]["backend"], cfg["llm"]["model"])
    if not todo:
        return 0

    llm = LLMClient(cfg["llm"])
    geo = Geocoder(cfg["geocoding"])
    n_ok = 0
    try:
        for i, a in enumerate(todo, 1):
            text = get_text(a, cfg)
            user, text_source = build_prompt(a, text, cfg["fulltext"]["max_chars"])
            try:
                ex: Extraction = llm.structured(SYSTEM, user, Extraction)
            except Exception as e:
                log.error("[%d/%d] %s — extraction failed: %s", i, len(todo), a["id"], e)
                continue  # not recorded → retried next run
            g = (geo.geocode(ex.location_name, ex.admin1, ex.country) if ex.is_specific_event
                 else {"lat": None, "lon": None, "geocode_level": "none",
                       "geocode_query": None, "geocode_match": None})
            append_jsonl(EXTRACTIONS, [{
                "id": a["id"],
                "extraction_version": EXTRACTION_VERSION,
                "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "llm_backend": cfg["llm"]["backend"],
                "llm_model": cfg["llm"]["model"],
                "text_source": text_source,
                **ex.model_dump(),
                **g,
            }])
            n_ok += 1
            log.info("[%d/%d] %s | %s | event=%s | %s | %s, %s | geo=%s | %s", i, len(todo), a["id"],
                     text_source, ex.is_specific_event, ex.species, ex.location_name, ex.admin1,
                     g["geocode_level"], (a.get("title") or "")[:70])
            geo.save()
    finally:
        write_jsonl(ARTICLES, articles)  # persist resolved URLs / has_fulltext
    n_ft = sum(a.get("has_fulltext", False) for a in todo)
    log.info("extracted %d/%d (full text for %d)", n_ok, len(todo), n_ft)
    return n_ok


if __name__ == "__main__":
    main()
