"""Offline tests: no network, no LLM. Run with `pytest -q`.

Fixture rows are SYNTHETIC (invented places/dates) and exist only to exercise
schema validation, taxon resolution, extraction plumbing, de-duplication,
geocoding logic and the GeoJSON build.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import build_map_data  # noqa: E402
import extract  # noqa: E402
import taxa  # noqa: E402
from schema import Extraction, Triage, json_schema  # noqa: E402


def _ex(**kw):
    base = dict(relevance_reasoning="A coyote attacked a dog.", real_wild_carnivore_involved=True,
                is_specific_event=True, multiple_events=False, animal_name="coyote", taxon="coyote",
                taxon_evidence="explicitly stated", n_animals=1, event_date="2026-10-01",
                event_date_precision="day", conflict_types=["livestock or pet depredation"],
                primary_conflict_type="livestock or pet depredation", human_activity="walking a pet",
                human_outcome="no injury", n_people_injured=None, n_people_killed=None,
                domestic_animal_type="pets", domestic_animal_outcome="injured",
                rabies_status="not mentioned", animal_outcome="no action",
                location_name="Testville", admin1="Colorado", country="United States",
                location_precision="locality", summary="Synthetic.", confidence="high", notes=None)
    base.update(kw)
    return base


def _patch_extract(monkeypatch, tmp_path, arts_rows, existing=None):
    arts = tmp_path / "articles.jsonl"
    arts.write_text("\n".join(json.dumps(r) for r in arts_rows) + "\n")
    out = tmp_path / "extractions.jsonl"
    if existing is not None:
        out.write_text("\n".join(json.dumps(r) for r in existing) + "\n")
    monkeypatch.setattr(extract, "ARTICLES", arts)
    monkeypatch.setattr(extract, "EXTRACTIONS", out)
    monkeypatch.setattr(extract, "CACHE", tmp_path)
    monkeypatch.setattr(extract, "resolve_url", lambda u, i: None)   # no network
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)
    return out


class FakeGeo:
    def __init__(self, cfg): pass
    def geocode(self, *a): return {"lat": 39.0, "lon": -105.0, "geocode_level": "place",
                                   "geocode_query": "q", "geocode_match": "Testville"}
    def save(self): pass


# --- schema & registry -------------------------------------------------------

def test_schema_is_flat_and_taxon_vocab_comes_from_registry():
    s = json_schema()
    assert "$ref" not in json.dumps(s), "keep schema flat for grammar-constrained decoding"
    enum = s["properties"]["taxon"]["enum"]
    assert {"cougar", "coyote", "raccoon", "unidentified skunks", "other wild carnivore", "unknown"} <= set(enum)
    Extraction.model_validate(_ex())
    with pytest.raises(Exception):
        Extraction.model_validate(_ex(taxon="panda"))


def test_registry_scientific_names_unique_and_binomial():
    labels = taxa.index()["by_label"]
    sci = [r["scientific"] for r in labels.values() if r["rank"] == "species"]
    assert len(sci) == len(set(sci))
    assert all(len(s.split()) == 2 and s[0].isupper() for s in sci)


@pytest.mark.parametrize("name,expected", [
    ("mountain lion", "cougar"), ("rabid gray fox", "gray fox"), ("mother black bear and cubs", "American black bear"),
    ("oso negro", "American black bear"), ("dos lobos", "gray wolf"), ("coyote pups", "coyote"),
    ("fox", "unidentified wolves, coyotes & foxes"), ("zorrillo", "unidentified skunks"),
    ("sea otter", None),            # out-of-scope taxon must not collapse to "otter"
    ("tejón", None),                # ambiguous (coati in Mexico, badger elsewhere)
    ("Chicago Bears", None),
])
def test_match_name(name, expected):
    assert taxa.match_name(name) == expected


def test_resolve_flags():
    assert taxa.resolve("cougar", "mountain lion")["taxon_check"] == "agree"
    r = taxa.resolve("unidentified wolves, coyotes & foxes", "coyote")
    assert r["taxon"] == "coyote" and r["taxon_check"] == "dictionary more specific"
    assert taxa.resolve("gray fox", "fox")["taxon_check"] == "llm more specific"
    r = taxa.resolve("cougar", "bobcat")
    assert r["taxon"] == "cougar" and r["taxon_check"] == "disagree"   # kept, but flagged
    assert taxa.resolve("cougar", "mountain lion")["taxon_scientific"] == "Puma concolor"


def test_queries_cover_groups_and_editions():
    import common
    qs = taxa.queries(common.load_config()["search"])
    groups = {g["id"] for g in taxa.registry()["groups"]}
    assert {q["group"] for q in qs} == groups
    mx = [q for q in qs if q["country"] == "MX"]
    assert mx and all("ataque" in q["query"] for q in mx)
    canid_us = next(q for q in qs if q["group"] == "canids" and q["country"] == "US")
    assert '-"Fox News"' in canid_us["query"]


# --- extraction plumbing -----------------------------------------------------

def test_triage_irrelevant_skips_extraction(tmp_path, monkeypatch):
    out = _patch_extract(monkeypatch, tmp_path, [
        {"id": "a1", "title": "Panthers beat Jaguars 24-17 - ESPN", "gnews_url": "g", "publisher": "ESPN"}])
    calls = []

    class FakeLLM:
        def __init__(self, cfg): pass
        def runtime_version(self): return "ollama test"
        def structured(self, system, user, cls):
            calls.append(cls.__name__)
            return Triage(reasoning="Sports.", verdict="irrelevant")

    monkeypatch.setattr(extract, "LLMClient", FakeLLM)
    monkeypatch.setattr(extract, "Geocoder", FakeGeo)
    assert extract.main() == 1
    row = json.loads(out.read_text())
    assert calls == ["Triage"] and row["triage_verdict"] == "irrelevant" and row["is_specific_event"] is False


def test_extract_full_path_and_versioning(tmp_path, monkeypatch):
    out = _patch_extract(monkeypatch, tmp_path, [
        {"id": "a1", "title": "Coyote attacks dog in Testville - Test Daily", "gnews_url": "g",
         "snippet": "Coyote attacks dog in Testville  Test Daily", "published": "2026-10-02T10:00:00+00:00",
         "publisher": "Test Daily"}],
        existing=[{"id": "a1", "extraction_version": 1}])   # outdated row → redone

    class FakeLLM:
        def __init__(self, cfg): pass
        def runtime_version(self): return "ollama test"
        def structured(self, system, user, cls):
            if cls is Triage:
                return Triage(reasoning="Coyote and dog.", verdict="relevant")
            assert "HEADLINE ONLY" in user and "Publication date: 2026-10-02" in user
            return cls.model_validate(_ex())

    monkeypatch.setattr(extract, "LLMClient", FakeLLM)
    monkeypatch.setattr(extract, "Geocoder", FakeGeo)
    assert extract.main() == 1
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["extraction_version"] == extract.EXTRACTION_VERSION and r["text_source"] == "headline"
    assert r["taxon_scientific"] == "Canis latrans" and r["taxon_check"] == "agree"
    assert r["llm_runtime"] == "ollama test"
    assert extract.main() == 0  # idempotent


def test_prompts_contain_definitions():
    assert "relevance_reasoning" in extract.SYSTEM and "bear spray" in extract.SYSTEM
    assert "cougar" in extract.SYSTEM            # taxon vocabulary is shown to the model
    assert "verdict" in extract.TRIAGE_SYSTEM


# --- geocoding ---------------------------------------------------------------

def test_geocoder_levels_and_unknowns(monkeypatch, tmp_path):
    import geocode
    monkeypatch.setattr(geocode, "GEOCACHE", tmp_path / "g.json")
    g = geocode.Geocoder({"user_agent": "t", "countrycodes": ["us", "ca", "mx"], "min_interval_s": 0})
    fake = {
        "New Mexico, United States": [{"lat": 34.5, "lon": -106.0, "display_name": "New Mexico, United States",
                                       "type": "administrative", "addresstype": "state",
                                       "state": "New Mexico", "country_code": "us"}],
        "Goldstream Park, Canada": [{"lat": 48.47, "lon": -123.55, "display_name": "Goldstream Provincial Park",
                                     "type": "park", "addresstype": "leisure",
                                     "state": "British Columbia", "country_code": "ca"}],
    }
    monkeypatch.setattr(g, "_query", lambda q: fake.get(q, []))
    assert g.geocode("New Mexico", "unknown", "United States")["geocode_level"] == "admin1"
    r = g.geocode("Goldstream Park", "unknown", "Canada")
    assert r["geocode_level"] == "place" and r["lat"] == 48.47


# --- build ---------------------------------------------------------------------

def test_build_dedups_same_event(tmp_path, monkeypatch):
    arts, exs = [], []
    # r1 & r2: same coyote event, 2 outlets, ~8 km and 1 day apart -> merged
    # r3: same place/date but a cougar -> separate; r4: not an event -> excluded
    spec = [("r1", 40.00, -105.00, "2026-10-01", "coyote", True, "fulltext"),
            ("r2", 40.05, -105.08, "2026-10-02", "coyote", True, "snippet"),
            ("r3", 40.00, -105.00, "2026-10-01", "cougar", True, "fulltext"),
            ("r4", None, None, None, "unknown", False, "headline")]
    for rid, lat, lon, d, tx, ev, src in spec:
        arts.append({"id": rid, "title": f"title {rid}", "url": f"https://example.org/{rid}",
                     "publisher": "X", "published": "2026-10-02T00:00:00+00:00"})
        exs.append({"id": rid, **_ex(event_date=d, taxon=tx, is_specific_event=ev,
                                     real_wild_carnivore_involved=ev),
                    **taxa.resolve(tx, None), "text_source": src, "lat": lat, "lon": lon,
                    "geocode_level": "place" if lat else "none", "geocode_match": None, "llm_model": "fake"})
    a, e = tmp_path / "a.jsonl", tmp_path / "e.jsonl"
    a.write_text("\n".join(map(json.dumps, arts)))
    e.write_text("\n".join(map(json.dumps, exs)))
    gj = tmp_path / "events.geojson"
    for name, val in [("ARTICLES", a), ("EXTRACTIONS", e), ("GEOJSON", gj), ("DATA", tmp_path)]:
        monkeypatch.setattr(build_map_data, name, val)
    meta = build_map_data.main()
    fc = json.loads(gj.read_text())
    assert meta["n_events"] == 2 and meta["n_articles_not_events"] == 1
    merged = [f for f in fc["features"] if f["properties"]["n_reports"] == 2][0]["properties"]
    assert merged["text_source"] == "fulltext" and merged["taxon_scientific"] == "Canis latrans"
    assert fc["metadata"]["group_colors"]["Wild cats"].startswith("#")
    assert (tmp_path / "events.csv").exists()
