"""Step 3 — join articles + extractions, merge duplicate reports, write GeoJSON.

Duplicate handling: the same incident is typically covered by several outlets.
Reports are merged by single-linkage clustering when they share the taxon
(species, or group if unidentified) and
are within `max_distance_km` and `max_days_apart` (config.yaml). Each map
feature is one *event* carrying all its source articles; attribute values come
from the best-documented report (full text > snippet, higher confidence,
finer geocode, finer date). This is a heuristic — thresholds are explicit in
the config and every feature keeps `n_reports` so it can be audited.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from datetime import date, datetime, timezone

from common import ARTICLES, DATA, EXTRACTIONS, GEOJSON, load_config, read_jsonl
import taxa

log = logging.getLogger("build")

RANK = {
    "text_source": {"fulltext": 2, "snippet": 1, "headline": 0},
    "confidence": {"high": 2, "medium": 1, "low": 0},
    "geocode_level": {"place": 2, "admin1": 1, "none": 0},
    "event_date_precision": {"day": 3, "week": 2, "month": 1, "unknown": 0},
}


def haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _date(r: dict) -> date | None:
    for v in (r.get("event_date"), (r.get("published") or "")[:10]):
        try:
            return date.fromisoformat(v)
        except (TypeError, ValueError):
            continue
    return None


def _taxon_key(r: dict) -> str:
    """Same species if identified to species, otherwise same group."""
    return r.get("taxon") if r.get("taxon_rank") == "species" else f"group:{r.get('taxon_group')}"


def cluster(rows: list[dict], cfg: dict) -> list[list[dict]]:
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if _taxon_key(a) != _taxon_key(b):
                continue
            da, db = _date(a), _date(b)
            if da is None or db is None or abs((da - db).days) > cfg["max_days_apart"]:
                continue
            if haversine_km((a["lat"], a["lon"]), (b["lat"], b["lon"])) > cfg["max_distance_km"]:
                continue
            parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(find(i), []).append(r)
    return list(groups.values())


def best(group: list[dict]) -> dict:
    return max(group, key=lambda r: tuple(RANK[k].get(r.get(k), 0) for k in RANK))


def main() -> dict:
    cfg = load_config()
    arts = {a["id"]: a for a in read_jsonl(ARTICLES)}
    ex = read_jsonl(EXTRACTIONS)
    rows = [{**arts.get(e["id"], {}), **e} for e in ex]
    events = [r for r in rows if r.get("is_specific_event") and r.get("real_wild_carnivore_involved")
              and r.get("country") != "other country"]
    located = [r for r in events if r.get("lat") is not None]

    features = []
    for g in cluster(located, cfg["dedup"]):
        rep = best(g)
        d = _date(rep)
        props = {k: rep.get(k) for k in (
            "animal_name", "taxon", "taxon_scientific", "taxon_rank", "taxon_group", "taxon_family",
            "taxon_check", "taxon_llm", "taxon_evidence_source", "n_animals", "conflict_types", "primary_conflict_type",
            "human_activity", "human_outcome", "n_people_injured", "n_people_killed",
            "domestic_animal_type", "domestic_animal_outcome", "rabies_status", "animal_outcome",
            "location_name", "admin1", "country", "geocode_level",
            "geocode_match", "summary", "confidence", "event_date_precision",
            "multiple_events", "text_source", "llm_model")}
        props.update({
            "event_id": min(r["id"] for r in g),
            "date": d.isoformat() if d else None,
            "date_is_publication_date": not rep.get("event_date"),
            "date_source": None if rep.get("event_date") else rep.get("date_source", "publication"),
            "n_reports": len(g),
            "sources": sorted(({"title": r.get("title"), "url": r.get("url") or r.get("gnews_url"),
                                "publisher": r.get("publisher"), "published": r.get("published"),
                                "source": r.get("source", "googlenews")}
                               for r in g), key=lambda s: s["published"] or ""),
        })
        features.append({"type": "Feature",
                         "geometry": {"type": "Point", "coordinates": [rep["lon"], rep["lat"]]},
                         "properties": props})
    features.sort(key=lambda f: f["properties"]["date"] or "", reverse=True)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_articles": len(arts),
        "n_articles_processed": len(ex),
        "n_articles_not_events": len(rows) - len(events),
        "n_out_of_region": sum(r.get("country") == "other country" for r in rows),
        "n_event_reports_not_geocoded": len(events) - len(located),
        "n_events": len(features),
        "dedup": cfg["dedup"],
        "group_colors": taxa.group_colors(),
    }
    GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    GEOJSON.write_text(json.dumps({"type": "FeatureCollection", "metadata": meta, "features": features},
                                  ensure_ascii=False, indent=1))

    # Flat table for analysis (one row per event, sources collapsed)
    cols = ["event_id", "date", "date_is_publication_date", "event_date_precision",
            "taxon", "taxon_scientific", "taxon_rank", "taxon_family", "taxon_group", "animal_name",
            "taxon_check", "taxon_llm", "taxon_evidence_source", "n_animals", "primary_conflict_type", "conflict_types",
            "human_activity", "human_outcome", "n_people_injured", "n_people_killed",
            "domestic_animal_type", "domestic_animal_outcome", "rabies_status", "animal_outcome",
            "location_name", "admin1", "country", "lat", "lon", "geocode_level",
            "confidence", "text_source", "n_reports", "summary", "source_urls"]
    with open(DATA / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ft in features:
            p = ft["properties"]
            w.writerow({**{c: p.get(c) for c in cols},
                        "conflict_types": ";".join(p["conflict_types"] or []),
                        "lon": ft["geometry"]["coordinates"][0], "lat": ft["geometry"]["coordinates"][1],
                        "source_urls": " | ".join(s["url"] for s in p["sources"])})
    log.info("map data: %s", meta)
    return meta


if __name__ == "__main__":
    main()
