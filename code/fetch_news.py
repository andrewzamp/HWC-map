"""Step 1 — discover articles and store their metadata.

Sources (config: search.source):
* gdelt      — GDELT DOC 2.0 API. Returns direct article URLs, filtered by the
               outlet's country (FIPS) and the article language. Note: the date
               GDELT returns is when it first SAW the article (`seendate`), a
               close proxy for, but not identical to, the publication date.
* googlenews — gnews package. Google's redirect links can no longer be decoded
               (Nov 2026), so these articles are extracted from headlines only.

Only metadata (title, snippet, URL, outlet, publication date) is committed
to the repo. Full text is cached locally in data/cache/ (git-ignored) and
used transiently by the LLM step, to avoid redistributing copyrighted text.
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from itertools import zip_longest
from urllib.parse import quote

import requests
from gnews import GNews

import taxa
from common import ARTICLES, article_id, append_jsonl, load_config, norm_title, read_jsonl

log = logging.getLogger("fetch")

# Diagnostics for the full-text step, summarised at the end of extract.py
STATS = {"decode_ok": 0, "decode_fail": 0, "download_ok": 0, "download_fail": 0, "first_errors": []}


def _note_error(kind: str, msg: str) -> None:
    if len(STATS["first_errors"]) < 3:
        STATS["first_errors"].append(f"{kind}: {msg[:300]}")


def _iso(rfc822: str) -> str | None:
    try:
        return parsedate_to_datetime(rfc822).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def search_googlenews(cfg: dict) -> list[list[dict]]:
    """One result list per (taxon group × edition) query, as normalised records."""
    s = cfg["search"]
    per_query: list[list[dict]] = []
    for q in taxa.queries(s):
        window = ({"start_date": tuple(map(int, s["start_date"].split("-"))),
                   "end_date": tuple(map(int, s["end_date"].split("-")))} if s.get("start_date")
                  else {"period": s["period"]})
        gn = GNews(language=q["language"], country=q["country"], **window,
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
        per_query.append([{
            "key": h["url"], "title": h.get("title"), "snippet": h.get("description"),
            "gnews_url": h["url"], "url": None,
            "publisher": (h.get("publisher") or {}).get("title"),
            "publisher_href": (h.get("publisher") or {}).get("href"),
            "published": _iso(h.get("published date")), "date_source": "googlenews_published",
            "edition": q["country"], "language": q["language"], "query_group": q["group"],
            "source": "googlenews"} for h in res])
    return per_query


GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _gdelt_get(params: dict, interval: float, backoff: list[float]) -> tuple[list[dict] | None, str | None]:
    """Rate-limited GDELT call with progressive back-off on HTTP 429.
    Returns (articles, None) or (None, error_text)."""
    err = None
    for attempt, wait in enumerate([interval, *backoff]):
        time.sleep(wait + random.uniform(0, 2))      # jitter: runners share IPs with other GDELT users
        try:
            r = requests.get(GDELT_URL, params=params, timeout=60,
                             headers={"User-Agent": "HWC-map (+https://github.com/andrewzamp/HWC-map)"})
        except requests.RequestException as e:
            err = repr(e)
            continue
        if r.status_code == 429:
            err = "HTTP 429 rate limited"
            continue
        try:
            return (r.json() or {}).get("articles", []), None     # no results → {} or ""
        except ValueError:
            body = r.text.strip()
            if not body:
                return [], None
            return None, f"HTTP {r.status_code}: {body[:300]}"     # GDELT explains query errors in plain text
    return None, err


def _gdelt_date(seen: str | None) -> str | None:
    try:
        return datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _gdelt_window(s: dict) -> dict:
    """Explicit date window (start/end, inclusive) if given, else rolling `period`.
    The GDELT DOC API only searches roughly the last 3 months."""
    if s.get("start_date"):
        end = s.get("end_date") or datetime.now(timezone.utc).date().isoformat()
        return {"startdatetime": s["start_date"].replace("-", "") + "000000",
                "enddatetime": end.replace("-", "") + "235959"}
    return {"timespan": s["period"]}


def search_gdelt(cfg: dict) -> list[list[dict]]:
    s = cfg["search"]
    units = taxa.gdelt_units(s)
    log.info("GDELT: %d queries | taxon=%s | window=%s", len(units), s.get("taxon_filter") or "all", _gdelt_window(s))
    # rotate the starting unit daily so that, under rate limits/time budget,
    # every unit gets queried first on some day
    k = datetime.now(timezone.utc).timetuple().tm_yday % max(len(units), 1)
    queue = units[k:] + units[:k]
    if s.get("max_queries") and len(queue) > s["max_queries"]:
        log.info("max_queries=%d: running %d of %d queries this run", s["max_queries"], s["max_queries"], len(queue))
        queue = queue[:s["max_queries"]]
    budget = time.time() + s.get("gdelt_time_budget_s", 900)
    interval, backoff = s.get("gdelt_interval_s", 8), s.get("gdelt_backoff_s", [20, 45, 90])
    per_query, n_ok, n_fail, n_rl, n_skip = [], 0, 0, 0, 0
    while queue:
        if time.time() > budget:
            n_skip = len(queue)
            log.warning("GDELT time budget reached; %d queries skipped this run (covered by later runs)", n_skip)
            break
        u = queue.pop(0)
        q = taxa.gdelt_query(u)
        params = {"query": q, "mode": "ArtList", "format": "json", "maxrecords": s["max_results_per_query"],
                  "sort": "DateDesc", **_gdelt_window(s)}
        arts, err = _gdelt_get(params, interval, backoff)
        if arts is None and err and "too short or too long" in err and len(u["terms"]) > 1:
            h = len(u["terms"]) // 2                       # unknown length limit: split and retry
            log.warning("GDELT: query too long (%d chars), splitting: %s", len(q), q)
            queue[:0] = [{**u, "terms": u["terms"][:h]}, {**u, "terms": u["terms"][h:]}]
            continue
        if arts is None:
            n_fail += 1
            n_rl += "429" in (err or "")
            log.warning("GDELT query failed (%s %s): %s | %s", u["group"], u["country"], err, q)
            continue
        n_ok += 1
        log.info("GDELT %s/%s %-10s %-45s: %d hits", u["country"], u["language"], u["group"],
                 " OR ".join(u["terms"])[:45], len(arts))
        per_query.append([{
            "key": a["url"], "title": a.get("title"), "snippet": None, "gnews_url": None, "url": a["url"],
            "publisher": a.get("domain"), "publisher_href": f"https://{a['domain']}" if a.get("domain") else None,
            "published": _gdelt_date(a.get("seendate")), "date_source": "gdelt_seendate",
            "edition": u["country"], "language": u["language"], "query_group": u["group"],
            "query_terms": u["terms"], "gdelt_sourcecountry": a.get("sourcecountry"),
            "gdelt_language": a.get("language"), "source": "gdelt"} for a in arts if a.get("url")])
    log.info("GDELT SUMMARY: %d queries succeeded, %d failed after retries (%d of them rate-limited), "
             "%d skipped (time budget)", n_ok, n_fail, n_rl, n_skip)
    return per_query


def search(cfg: dict) -> list[dict]:
    """All queries, interleaved round-robin so a small max_new_articles still
    samples every group and edition."""
    per_query = search_gdelt(cfg) if cfg["search"].get("source", "gdelt") == "gdelt" else search_googlenews(cfg)
    return [h for grp in zip_longest(*per_query) for h in grp if h is not None]


def resolve_url(gnews_url: str, interval: float) -> str | None:
    """Google News RSS links are opaque redirects; decode to the publisher URL."""
    try:
        from googlenewsdecoder import gnewsdecoder

        out = gnewsdecoder(gnews_url, interval=interval)
        if out.get("status"):
            STATS["decode_ok"] += 1
            return out["decoded_url"]
        log.warning("Google News URL decode failed: %s", out.get("message"))
        _note_error("decode failed", str(out.get("message")))
    except Exception as e:  # network / format changes upstream
        log.warning("Google News URL decode error: %r", e)
        _note_error("decode error", repr(e))
    STATS["decode_fail"] += 1
    return None


def download_text(url: str) -> str | None:
    try:
        import trafilatura

        html = trafilatura.fetch_url(url)
        if html:
            text = trafilatura.extract(html, include_comments=False, include_tables=False)
            if text:
                STATS["download_ok"] += 1
                return text
        log.warning("no usable HTML for %s (paywall/bot block?)", url)
        _note_error("no HTML", url)
    except Exception as e:
        log.warning("download error %s: %s", url, e)
        _note_error("download error", f"{url} {e!r}")
    STATS["download_fail"] += 1
    return None


def main() -> list[dict]:
    cfg = load_config()
    existing = read_jsonl(ARTICLES)
    seen_ids = {a["id"] for a in existing}
    seen_titles = {norm_title(a["title"]) for a in existing}

    new: list[dict] = []
    for h in search(cfg):
        aid = article_id(h.pop("key"))
        nt = norm_title(h.get("title") or "")
        if aid in seen_ids or (nt and nt in seen_titles):
            continue
        seen_ids.add(aid)
        seen_titles.add(nt)
        new.append({"id": aid, **h, "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "has_fulltext": False})
        if len(new) >= cfg["search"]["max_new_articles"]:
            break

    append_jsonl(ARTICLES, new)
    log.info("%d new articles stored (full text is retrieved in the extract step)", len(new))
    return new


if __name__ == "__main__":
    main()
