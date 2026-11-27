"""Step 1 — query Google News (via `gnews`) and store new article metadata.

Only metadata (title, snippet, URL, outlet, publication date) is committed
to the repo. Full text is cached locally in data/cache/ (git-ignored) and
used transiently by the LLM step, to avoid redistributing copyrighted text.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from itertools import zip_longest
from urllib.parse import quote

from gnews import GNews

import taxa
from common import ARTICLES, article_id, append_jsonl, load_config, norm_title, read_jsonl

log = logging.getLogger("fetch")


def _iso(rfc822: str) -> str | None:
    try:
        return parsedate_to_datetime(rfc822).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def search(cfg: dict) -> list[dict]:
    """Run every (taxon group × edition) query; interleave results round-robin
    so a small max_new_articles still samples all groups and editions."""
    s = cfg["search"]
    per_query: list[list[dict]] = []
    for q in taxa.queries(s):
        gn = GNews(language=q["language"], country=q["country"], period=s["period"],
                   max_results=s["max_results_per_query"],
                   exclude_websites=s.get("exclude_websites") or None)
        try:
            # gnews inserts the query into the RSS URL without encoding quotes or
            # parentheses, so encode it ourselves (spaces -> %20, which gnews keeps).
            res = gn.get_news(quote(q["query"], safe="")) or []
        except Exception as e:
            log.warning("query failed (%s %s %s): %r", q["group"], q["country"], q["language"], e)
            res = []
        log.info("Google News %s/%s, group %-10s: %d hits", q["country"], q["language"], q["group"], len(res))
        for r in res:
            r["_edition"], r["_language"], r["_group"] = q["country"], q["language"], q["group"]
        per_query.append(res)
    return [h for grp in zip_longest(*per_query) for h in grp if h is not None]


def resolve_url(gnews_url: str, interval: float) -> str | None:
    """Google News RSS links are opaque redirects; decode to the publisher URL."""
    try:
        from googlenewsdecoder import gnewsdecoder

        out = gnewsdecoder(gnews_url, interval=interval)
        if out.get("status"):
            return out["decoded_url"]
        log.warning("Google News URL decode failed: %s", out.get("message"))
    except Exception as e:  # network / format changes upstream
        log.warning("Google News URL decode error: %r", e)
    return None


def download_text(url: str) -> str | None:
    try:
        import trafilatura

        html = trafilatura.fetch_url(url)
        if html:
            return trafilatura.extract(html, include_comments=False, include_tables=False)
        log.warning("no HTML returned for %s (paywall/bot block?)", url)
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
            "language": h["_language"],
            "query_group": h["_group"],
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "has_fulltext": False,
        })
        if len(new) >= cfg["search"]["max_new_articles"]:
            break

    append_jsonl(ARTICLES, new)
    log.info("%d new articles stored (full text is retrieved in the extract step)", len(new))
    return new


if __name__ == "__main__":
    main()
