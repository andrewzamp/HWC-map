"""Step 2 — LLM metadata extraction + geocoding for not-yet-processed articles.

Incremental and idempotent: each article id is processed once and the row is
appended immediately, so an interrupted run (e.g. Actions timeout) loses
nothing. Model name/backend are stored with every row for provenance.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from common import ARTICLES, CACHE, EXTRACTIONS, append_jsonl, load_config, read_jsonl
from geocode import Geocoder
from llm import LLMClient
from schema import Extraction

log = logging.getLogger("extract")

SYSTEM = """You are an information-extraction assistant for a scientific database of \
human-wildlife conflict involving bears in North America.
Rules:
- Extract only what the text states or directly implies. Never invent details.
- When information is missing or ambiguous, use null or "unknown" rather than guessing.
- event_date is when the incident happened, not when the article was published; \
resolve relative dates against the publication date given.
- Species: 'grizzly' means brown bear. Do not infer species from region alone \
unless the article context makes it clear, and then mark species_evidence accordingly.
- Location names must be copied from the article, not generalised.
Return a single JSON object that follows the provided schema."""


def build_prompt(a: dict, text: str | None, max_chars: int) -> tuple[str, str]:
    source = "full article text" if text else "headline and snippet only (full text unavailable)"
    body = (text or a.get("snippet") or "")[:max_chars]
    user = (f"Publication date: {a.get('published') or 'unknown'}\n"
            f"Outlet: {a.get('publisher') or 'unknown'}\n"
            f"Headline: {a.get('title')}\n"
            f"Text source: {source}\n\n---\n{body}\n---")
    return user, ("fulltext" if text else "snippet")


def main() -> int:
    cfg = load_config()
    done = {r["id"] for r in read_jsonl(EXTRACTIONS)}
    todo = [a for a in read_jsonl(ARTICLES) if a["id"] not in done]
    log.info("%d articles to process with %s:%s", len(todo), cfg["llm"]["backend"], cfg["llm"]["model"])
    if not todo:
        return 0

    llm = LLMClient(cfg["llm"])
    geo = Geocoder(cfg["geocoding"])
    n_ok = 0
    for i, a in enumerate(todo, 1):
        p = CACHE / f"{a['id']}.txt"
        text = p.read_text(encoding="utf-8") if p.exists() else None
        user, text_source = build_prompt(a, text, cfg["fulltext"]["max_chars"])
        try:
            ex: Extraction = llm.structured(SYSTEM, user, Extraction)
        except Exception as e:
            log.error("[%d/%d] %s — extraction failed: %s", i, len(todo), a["id"], e)
            continue  # not recorded → retried next run
        g = (geo.geocode(ex.location_name, ex.admin1, ex.country) if ex.is_specific_event
             else {"lat": None, "lon": None, "geocode_level": "none", "geocode_query": None, "geocode_match": None})
        append_jsonl(EXTRACTIONS, [{
            "id": a["id"],
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "llm_backend": cfg["llm"]["backend"],
            "llm_model": cfg["llm"]["model"],
            "text_source": text_source,
            **ex.model_dump(),
            **g,
        }])
        n_ok += 1
        log.info("[%d/%d] %s | event=%s | %s | %s, %s | geo=%s", i, len(todo), a["id"], ex.is_specific_event,
                 ex.species, ex.location_name, ex.admin1, g["geocode_level"])
        geo.save()
    log.info("extracted %d/%d", n_ok, len(todo))
    return n_ok


if __name__ == "__main__":
    main()
