"""Step 2 — triage, full-text retrieval, LLM metadata extraction, taxon
resolution, geocoding.

* Triage: a cheap headline-only LLM call screens out clearly irrelevant hits
  (sports teams, Fox News, 'bear spray' assaults, smugglers...) before the
  expensive steps. Only 'irrelevant' is skipped; 'unsure' goes through. The
  triage verdict and reasoning are stored for every article (auditable).

* Incremental: each article is processed once per EXTRACTION_VERSION. Rows
  produced by an older schema/prompt version are dropped and redone, so
  improving the prompt automatically re-processes the archive.
* Full text is (re)attempted here for every article without cached text,
  so transient download failures heal on the next run.
* Each row is appended as soon as it's done (safe against Actions timeouts).
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone

from common import ARTICLES, CACHE, EXTRACTIONS, TRIAGE_CACHE, append_jsonl, load_config, norm_title, read_jsonl, write_jsonl
from fetch_news import STATS as FT_STATS, download_text, resolve_url
from geocode import Geocoder
from llm import EarlyStop, LLMClient
import taxa
import textsel
from schema import EXTRACTION_VERSION, Extraction, Triage, field_guide

log = logging.getLogger("extract")

SYSTEM = f"""You are an information-extraction assistant for a scientific database of \
human-wildlife conflict involving wild carnivores (bears, wolves, coyotes, foxes, wild cats, \
weasel family, raccoons, coatis, skunks) in North America. Articles may be in English or Spanish.

Rules:
- Extract only what the text states or directly implies. Never invent details.
- When information is missing or ambiguous, use null or "unknown" rather than guessing.
- Frequent false positives: "bear spray" used on people, sports teams (Bears, Panthers, Jaguars, \
Wolverines, Badgers, Coyotes), "bear market", TV stations named FOX, human smugglers called \
"coyotes", cars (Jaguar), people's surnames (Wolf, Fox). These are NOT wild carnivores.
- event_date is when the incident happened, not when the article was published.
- Read the headline carefully: it often states species, outcome and place.
- Write summary in English whatever the article language.
- Return one JSON object with exactly these fields:

{field_guide(Extraction)}"""

TRIAGE_SYSTEM = f"""You screen news headlines for a database of human-wildlife conflict involving \
wild carnivores. Headlines may be in English or Spanish. Your ONLY job is to tell whether the \
animal word refers to the real animal. Do not judge whether a conflict happened.

Sports: team names are NOT animals. If the headline is about a game, score, season, \
player, athlete, coach, roster or tournament, answer 'irrelevant' (e.g. Grizzlies, Bears, \
Lions, Panthers, Jaguars, Wolves, Timberwolves, Coyotes, Bruins, Otters, Pumas, Wolverines, Badgers).
Headlines that name no animal at all (news round-ups, lists, history columns) are 'unsure', \
never 'relevant'.

Spanish names: oso = bear, lobo = wolf, coyote = coyote, zorro/zorra = fox, puma/león de montaña = \
cougar, jaguar, lince = lynx/bobcat, nutria = OTTER (not coypu), tejón = badger or coati, \
mapache = raccoon, coatí = coati, cacomixtle = ringtail, zorrillo/mofeta = skunk, glotón = wolverine.

Return JSON with these fields:

{field_guide(Triage)}"""

_RX_BOOL = {k: re.compile(rf'"{k}"\s*:\s*(true|false)') for k in ("real_wild_carnivore_involved", "is_specific_event")}
_RX_COUNTRY = re.compile(r'"country"\s*:\s*"([^"]*)"')
_RX_REASON = re.compile(r'"relevance_reasoning"\s*:\s*("(?:[^"\\]|\\.)*")')


def early_stop(partial: str) -> dict | None:
    """Stop generation once the model has decided the article is not a
    North-American conflict event: the remaining ~25 fields would be discarded
    by the build step anyway. Saves most of the generation time for non-events."""
    ev = _RX_BOOL["is_specific_event"].search(partial)
    if not ev:
        return None
    is_event = ev.group(1) == "true"
    country = _RX_COUNTRY.search(partial)
    if is_event and not country:
        return None                      # wait for the country field
    if is_event and country.group(1) != "other country":
        return None                      # a real NA (or unknown-country) event: generate everything
    rw = _RX_BOOL["real_wild_carnivore_involved"].search(partial)
    rr = _RX_REASON.search(partial)
    return {"relevance_reasoning": json.loads(rr.group(1)) if rr else None,
            "real_wild_carnivore_involved": rw.group(1) == "true" if rw else None,
            "is_specific_event": is_event,
            "country": country.group(1) if country else None}


NULL_GEO = {"lat": None, "lon": None, "geocode_level": "none", "geocode_query": None, "geocode_match": None}


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


def build_prompt(a: dict, text: str | None, max_chars: int, lead_chars: int = 1000) -> tuple[str, str]:
    snippet = a.get("snippet") or ""
    # Google News RSS 'snippets' are usually just the headline repeated.
    snippet_is_headline = norm_title(snippet).startswith(norm_title(a.get("title", ""))[:40])
    if text:
        body = textsel.condense(text, max_chars, lead_chars)
        source = "full article text" if body == text.strip() else \
            "article lead + later sentences about the animal or the outcome ('[…]' = text skipped)"
        tag = "fulltext"
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
    # newest-fetched first: a targeted run (one taxon / period) is processed
    # straight away, and fresh news reaches the map before the backlog
    todo.sort(key=lambda a: a.get("fetched") or "", reverse=True)
    cap = cfg["llm"].get("max_articles_per_run")
    if cap and len(todo) > cap:
        log.info("%d unprocessed articles; processing %d this run, the rest carry over", len(todo), cap)
        todo = todo[:cap]
    log.info("%d articles to process with %s:%s", len(todo), cfg["llm"]["backend"], cfg["llm"]["model"])
    if not todo:
        return 0

    t_start = time.time()
    budget = cfg["llm"].get("time_budget_s")
    budget = float("inf") if budget is None else float(budget)
    triage_deadline = t_start + budget * cfg["llm"].get("triage_budget_fraction", 0.35)
    deadline = t_start + budget
    # triage verdicts already computed in earlier runs (articles that then ran out of time)
    cached = {r["id"]: Triage(reasoning=r["reasoning"], verdict=r["verdict"]) for r in read_jsonl(TRIAGE_CACHE)
              if r.get("extraction_version") == EXTRACTION_VERSION}

    llm = LLMClient(cfg["llm"])
    runtime = llm.runtime_version()
    log.info("LLM runtime: %s", runtime)
    geo = Geocoder(cfg["geocoding"])
    n_ok = 0
    stamp = lambda: {"extraction_version": EXTRACTION_VERSION,
                     "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "llm_backend": cfg["llm"]["backend"], "llm_model": cfg["llm"]["model"],
                     "llm_runtime": runtime}
    try:
        # Pass 1 — triage all headlines. Kept separate from pass 2 so that each
        # pass reuses one identical system prompt (Ollama's prompt-prefix cache).
        triage: dict[str, Triage] = {}
        if cfg.get("triage", {}).get("enabled", True):
            for i, a in enumerate(todo, 1):
                if a["id"] in cached:
                    triage[a["id"]] = cached[a["id"]]
                    continue
                if time.time() > triage_deadline:
                    log.info("triage time share used up after %d headlines; the rest carry over", i - 1)
                    break
                try:
                    triage[a["id"]] = tr = llm.structured(
                        TRIAGE_SYSTEM, f"Outlet: {a.get('publisher')}\nHeadline: {a.get('title')}", Triage)
                except Exception as e:
                    log.error("[triage %d/%d] %s failed: %s", i, len(todo), a["id"], e)
                    continue
                if tr.verdict != "irrelevant":   # remembered in case extraction runs out of time
                    append_jsonl(TRIAGE_CACHE, [{"id": a["id"], "extraction_version": EXTRACTION_VERSION,
                                                 "verdict": tr.verdict, "reasoning": tr.reasoning}])
                log.info("[triage %d/%d] %-10s | %s", i, len(todo), tr.verdict, (a.get("title") or "")[:80])
                if tr.verdict == "irrelevant":
                    append_jsonl(EXTRACTIONS, [{"id": a["id"], **stamp(), "triage_verdict": tr.verdict,
                                                "triage_reasoning": tr.reasoning, "text_source": None,
                                                "is_specific_event": False,
                                                "real_wild_carnivore_involved": False, **NULL_GEO}])
                    n_ok += 1
            todo = [a for a in todo if a["id"] in triage and triage[a["id"]].verdict != "irrelevant"]
            log.info("triage kept %d articles for full extraction", len(todo))

        # Pass 2 — full text + structured extraction + taxon + geocoding
        for i, a in enumerate(todo, 1):
            if time.time() > deadline:
                log.info("time budget reached: %d of %d triaged articles extracted; the rest carry over",
                         i - 1, len(todo))
                break
            tr = triage.get(a["id"])
            base = {"id": a["id"], **stamp(), "triage_verdict": tr.verdict if tr else None,
                    "triage_reasoning": tr.reasoning if tr else None}
            t0 = time.time()
            try:
                text = get_text(a, cfg)
                if tr is not None and tr.verdict == "unsure" and not textsel.animal_conflict_context(text):
                    append_jsonl(EXTRACTIONS, [{**base, "text_source": "fulltext", "is_specific_event": False,
                                                "real_wild_carnivore_involved": None,
                                                "text_gate": "no animal name near a conflict word", **NULL_GEO}])
                    n_ok += 1
                    log.info("[%d/%d] %s | unsure + text gate: no animal-conflict context | %s",
                             i, len(todo), a["id"], (a.get("title") or "")[:70])
                    continue
                user, text_source = build_prompt(a, text, cfg["fulltext"]["max_chars"],
                                                 cfg["fulltext"].get("lead_chars", 1000))
                t1 = time.time()
                ex = llm.structured(SYSTEM, user, Extraction, early_stop=early_stop)
            except Exception as e:
                log.error("[%d/%d] %s — extraction failed: %s", i, len(todo), a["id"], e)
                continue  # not recorded → retried next run
            if isinstance(ex, EarlyStop):
                append_jsonl(EXTRACTIONS, [{**base, "text_source": text_source, **ex.fields,
                                            "early_stopped": True, **NULL_GEO}])
                n_ok += 1
                log.info("[%d/%d] %s | %s | early stop: event=%s country=%s | text %.0fs + llm %.0fs | %s",
                         i, len(todo), a["id"], text_source, ex.fields["is_specific_event"], ex.fields["country"],
                         t1 - t0, time.time() - t1, (a.get("title") or "")[:70])
                continue
            tx = taxa.resolve(ex.taxon, ex.animal_name, f"{a.get('title') or ''}\n{(text or '')[:cfg['fulltext']['max_chars']]}")
            g = geo.geocode(ex.location_name, ex.admin1, ex.country) if ex.is_specific_event else NULL_GEO
            append_jsonl(EXTRACTIONS, [{**base, "text_source": text_source, **ex.model_dump(), **tx, **g}])
            n_ok += 1
            log.info("[%d/%d] %s | %s | event=%s | %s (%s, %s) | %s, %s | geo=%s | text %.0fs + llm %.0fs | %s",
                     i, len(todo), a["id"], text_source, ex.is_specific_event, tx["taxon"], ex.animal_name,
                     tx["taxon_check"], ex.location_name, ex.admin1, g["geocode_level"], t1 - t0, time.time() - t1,
                     (a.get("title") or "")[:70])
            geo.save()
    finally:
        write_jsonl(ARTICLES, articles)  # persist resolved URLs / has_fulltext
    n_ft = sum(a.get("has_fulltext", False) for a in todo)
    log.info("processed %d articles; full text retrieved for %d/%d extracted", n_ok, n_ft, len(todo))
    log.info("FULL-TEXT DIAGNOSTICS: %s", {k: v for k, v in FT_STATS.items() if k != "first_errors"})
    for err in FT_STATS["first_errors"]:
        log.info("FULL-TEXT FIRST ERRORS: %s", err)
    return n_ok


if __name__ == "__main__":
    main()
