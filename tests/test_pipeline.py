"""Offline tests: no network, no LLM. Run with `pytest -q`.

The fixture rows are SYNTHETIC (invented places/dates) and exist only to
exercise schema validation, extraction plumbing, de-duplication and the
GeoJSON build.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import build_map_data  # noqa: E402
import extract  # noqa: E402
from schema import Extraction, json_schema  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _ex(**kw):
    base = dict(relevance_reasoning="A bear raided a garage.", real_bear_involved=True, is_specific_event=True, multiple_events=False, event_date="2026-10-01",
                event_date_precision="day", species="American black bear",
                species_evidence="explicitly stated", n_bears=1, conflict_types=["property damage"],
                primary_conflict_type="property damage", human_activity="at home or on own property",
                human_outcome="no injury", n_people_injured=None, n_people_killed=None,
                bear_outcome="hazed or deterred", location_name="Testville", admin1="Colorado",
                country="United States", location_precision="locality", summary="Synthetic.",
                confidence="high", notes=None)
    base.update(kw)
    return base


def test_schema_is_self_contained_and_validates():
    s = json_schema()
    assert "$ref" not in json.dumps(s), "keep schema flat for grammar-constrained decoding"
    Extraction.model_validate(_ex())
    with pytest.raises(Exception):
        Extraction.model_validate(_ex(species="panda"))


def test_extract_with_fake_llm(tmp_path, monkeypatch):
    arts = tmp_path / "articles.jsonl"
    arts.write_text(json.dumps({"id": "a1", "title": "Bear breaks into garage - Test Daily",
                                "snippet": "Bear breaks into garage  Test Daily", "gnews_url": "g", "published": "2026-10-02T10:00:00+00:00",
                                "publisher": "Test Daily"}) + "\n")
    out = tmp_path / "extractions.jsonl"
    monkeypatch.setattr(extract, "ARTICLES", arts)
    monkeypatch.setattr(extract, "EXTRACTIONS", out)
    monkeypatch.setattr(extract, "CACHE", tmp_path)
    monkeypatch.setattr(extract, "resolve_url", lambda u, i: None)   # no network
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)

    class FakeLLM:
        def __init__(self, cfg): pass
        def structured(self, system, user, cls):
            assert "Publication date: 2026-10-02" in user and "HEADLINE ONLY" in user
            return cls.model_validate(_ex())

    class FakeGeo:
        def __init__(self, cfg): pass
        def geocode(self, *a): return {"lat": 39.0, "lon": -105.0, "geocode_level": "place",
                                       "geocode_query": "q", "geocode_match": "Testville"}
        def save(self): pass

    monkeypatch.setattr(extract, "LLMClient", FakeLLM)
    monkeypatch.setattr(extract, "Geocoder", FakeGeo)
    assert extract.main() == 1
    row = json.loads(out.read_text())
    assert row["text_source"] == "headline" and row["lat"] == 39.0
    assert row["extraction_version"] == extract.EXTRACTION_VERSION
    assert extract.main() == 0  # idempotent


def test_build_dedups_same_event(tmp_path, monkeypatch):
    arts, exs = [], []
    # r1 & r2: same event, 2 outlets, 10 km apart, 1 day apart -> merged
    # r3: same place, different species -> separate; r4: not an event -> excluded
    spec = [("r1", 40.00, -105.00, "2026-10-01", "American black bear", True, "fulltext"),
            ("r2", 40.05, -105.08, "2026-10-02", "American black bear", True, "snippet"),
            ("r3", 40.00, -105.00, "2026-10-01", "brown/grizzly bear", True, "fulltext"),
            ("r4", None, None, None, "unknown bear species", False, "snippet")]
    for rid, lat, lon, d, sp, ev, src in spec:
        arts.append({"id": rid, "title": f"title {rid}", "url": f"https://example.org/{rid}",
                     "publisher": "X", "published": "2026-10-02T00:00:00+00:00"})
        exs.append({"id": rid, **_ex(event_date=d, species=sp, is_specific_event=ev), "text_source": src,
                    "lat": lat, "lon": lon, "geocode_level": "place" if lat else "none",
                    "geocode_match": None, "llm_model": "fake"})
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
    assert merged["text_source"] == "fulltext"  # best-documented report is representative
    assert merged["species_scientific"] == "Ursus americanus"
    assert (tmp_path / "events.csv").exists()


def test_outdated_rows_are_redone(tmp_path, monkeypatch):
    arts = tmp_path / "articles.jsonl"
    arts.write_text(json.dumps({"id": "a1", "title": "t", "gnews_url": "g"}) + "\n")
    out = tmp_path / "extractions.jsonl"
    out.write_text(json.dumps({"id": "a1", "extraction_version": 1}) + "\n")
    monkeypatch.setattr(extract, "ARTICLES", arts)
    monkeypatch.setattr(extract, "EXTRACTIONS", out)
    monkeypatch.setattr(extract, "CACHE", tmp_path)
    monkeypatch.setattr(extract, "resolve_url", lambda u, i: None)
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)
    monkeypatch.setattr(extract, "LLMClient", lambda cfg: type("L", (), {
        "structured": lambda self, s, u, cls: cls.model_validate(_ex(is_specific_event=False))})())
    monkeypatch.setattr(extract, "Geocoder", lambda cfg: type("G", (), {"save": lambda self: None})())
    assert extract.main() == 1
    assert [json.loads(l)["extraction_version"] for l in out.read_text().splitlines()] == [extract.EXTRACTION_VERSION]


def test_geocoder_levels_and_unknowns(monkeypatch, tmp_path):
    import geocode
    monkeypatch.setattr(geocode, "GEOCACHE", tmp_path / "g.json")
    g = geocode.Geocoder({"user_agent": "t", "countrycodes": ["us", "ca"], "min_interval_s": 0})
    fake = {
        "New Mexico, United States": [{"lat": 34.5, "lon": -106.0, "display_name": "New Mexico, United States",
                                       "type": "administrative", "addresstype": "state",
                                       "state": "New Mexico", "country_code": "us"}],
        "Goldstream Park, Canada": [{"lat": 48.47, "lon": -123.55, "display_name": "Goldstream Provincial Park",
                                     "type": "park", "addresstype": "leisure",
                                     "state": "British Columbia", "country_code": "ca"}],
    }
    monkeypatch.setattr(g, "_query", lambda q: fake.get(q, []))
    r = g.geocode("New Mexico", "unknown", "United States")
    assert r["geocode_level"] == "admin1"            # a state is not a point location
    r = g.geocode("Goldstream Park", "unknown", "Canada")
    assert r["geocode_level"] == "place" and r["lat"] == 48.47


def test_prompt_contains_field_definitions():
    assert "bear spray" in extract.SYSTEM and "relevance_reasoning" in extract.SYSTEM
