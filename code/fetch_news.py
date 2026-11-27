"""Step 1 — query Google News (via `gnews`) and store new articles.

Only metadata (title, snippet, URL, outlet, publication date) is committed
to the repo. Full text is cached locally in data/cache/ (git-ignored) and
used transiently by the LLM step, to avoid redistributing copyrighted text.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from gnews import GNews

from common import ARTICLES, CACHE, article_id, append_jsonl, load_config, norm_title, read_jsonl

log = logging.getLogger("fetch")


def _iso(rfc822: str) -> str | None:
    try:
        return parsedate_to_datetime(rfc822).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def search(cfg: dict) -> list[dict]:
    s = cfg["search"]
    # gnews inserts the query into the RSS URL without encoding quotes or
    # parentheses, so encode it ourselves (spaces -> %20, which gnews keeps).
    query = quote(" ".join(s["query"].split()), safe="")
    hits: list[dict] = []
    for country in s["countries"]:
        gn = GNews(
            language=s["language"],
            country=country,
            period=s["period"],
            max_results=s["max_results_per_country"],
            exclude_websites=s.get("exclude_websites") or None,
        )
        res = gn.get_news(query) or []
        log.info("Google News %s edition: %d hits", country, len(res))
        for r in res:
            r["_edition"] = country
        hits.extend(res)
    return hits


def resolve_url(gnews_url: str, interval: float) -> str | None:
    """Google News RSS links are opaque redirects; decode to the publisher URL."""
    try:
        from googlenewsdecoder import gnewsdecoder

        out = gnewsdecoder(gnews_url, interval=interval)
        if out.get("status"):
            return out["decoded_url"]
        log.warning("decode failed: %s", out.get("message"))
    except Exception as e:  # network / format changes upstream
        log.warning("decode error: %s", e)
    return None


def download_text(url: str) -> str | None:
    try:
        import trafilatura

        html = trafilatura.fetch_url(url)
        if html:
            return trafilatura.extract(html, include_comments=False, include_tables=False)
    except Exception as e:
        log.warning("download error %s: %s", url, e)
    return None


def main() -> list[dict]:
    cfg = load_config()
    existing = read_jsonl(ARTICLES)
    seen_ids = {a["id"] for a in existing}
    seen_titles = {norm_title(a["title"]) for a in existing}

    new: list[dict] = []
    for h in search(cfg):
        aid = article_id(h["url"])
        nt = norm_title(h.get("title", ""))
        if aid in seen_ids or nt in seen_titles:
            continue
        seen_ids.add(aid)
        seen_titles.add(nt)
        pub = h.get("publisher") or {}
        new.append({
            "id": aid,
            "title": h.get("title"),
            "snippet": h.get("description"),
            "gnews_url": h["url"],
            "url": None,
            "publisher": pub.get("title"),
            "publisher_href": pub.get("href"),
            "published": _iso(h.get("published date")),
            "edition": h["_edition"],
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "has_fulltext": False,
        })
        if len(new) >= cfg["search"]["max_new_articles"]:
            break

    ft = cfg["fulltext"]
    if ft["enabled"]:
        CACHE.mkdir(parents=True, exist_ok=True)
        for a in new:
            a["url"] = resolve_url(a["gnews_url"], ft["request_interval_s"])
            text = download_text(a["url"]) if a["url"] else None
            if text and len(text) > 300:
                (CACHE / f"{a['id']}.txt").write_text(text, encoding="utf-8")
                a["has_fulltext"] = True
            time.sleep(ft["request_interval_s"])

    append_jsonl(ARTICLES, new)
    log.info("%d new articles stored (%d with full text)", len(new), sum(a["has_fulltext"] for a in new))
    return new


if __name__ == "__main__":
    main()
