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
                n_animals=1, event_date="2026-10-01",
                event_date_precision="day", conflict_types=["livestock or pet depredation"],
                primary_conflict_type="livestock or pet depredation", human_activity="walking a pet",
                human_outcome="no injury", n_people_injured=None, n_people_killed=None,
                domestic_animal_type="pets", domestic_animal_outcome="injured",
                rabies_status="not mentioned", animal_outcome="no action",
                location_name="Testville", admin1="Colorado", country="United States",
                summary="Synthetic.", confidence="high")
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


def test_resolve_text_evidence_wins():
    assert taxa.resolve("cougar", "mountain lion")["taxon_check"] == "agree"
    r = taxa.resolve("unidentified wolves, coyotes & foxes", "coyote")
    assert r["taxon"] == "coyote" and r["taxon_check"] == "text more specific"
    # generic name: the LLM's species pick is NOT accepted without textual support
    r = taxa.resolve("gray fox", "fox")
    assert r["taxon"] == "unidentified wolves, coyotes & foxes" and r["taxon_llm"] == "gray fox"
    r = taxa.resolve("cougar", "bobcat")
    assert r["taxon"] == "bobcat" and r["taxon_check"] == "disagree"
    assert taxa.resolve("other wild carnivore", None, "no names here")["taxon_check"] == "llm only (no text evidence)"


# Real mistakes from the 2026-11-27 run (headline-only extractions by qwen2.5:7b)
@pytest.mark.parametrize("llm,name,headline,expected", [
    ("gray wolf", None, "4 grizzly bears responsible for killing mother and daughter in B.C., officials say", "brown bear"),
    ("unidentified skunks", None, "Fighting raccoon rabies", "raccoon"),
    ("striped skunk", "skunk", "Two veterinary workers receive boosters after dog kills rabid skunk", "unidentified skunks"),
    ("North American river otter", "otter", "Lake Tahoe swim instructor recounts rare river otter attack",
     "North American river otter"),
    ("American black bear", "bear", "A bear wandered into town; officers said the black bear was relocated",
     "American black bear"),
])
def test_resolve_regressions(llm, name, headline, expected):
    assert taxa.resolve(llm, name, headline)["taxon"] == expected


def test_scan_ignores_sports_plurals():
    assert taxa.scan_text("Lions lose to Chicago Cubs") == []


def test_clean_place():
    from geocode import clean_place
    assert clean_place("near Goldstream Park") == "Goldstream Park"
    assert clean_place("condado de Guilford") == "Guilford County"
    assert clean_place("North Vancouver") == "North Vancouver"     # real names untouched


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
        def structured(self, system, user, cls, early_stop=None):
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
        def structured(self, system, user, cls, early_stop=None):
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
    assert r["taxon_evidence_source"] == "animal name"
    assert r["llm_runtime"] == "ollama test"
    assert extract.main() == 0  # idempotent


def test_prompts_contain_definitions():
    assert "relevance_reasoning" in extract.SYSTEM and "bear spray" in extract.SYSTEM
    assert "cougar" in extract.SYSTEM            # taxon vocabulary is shown to the model
    assert "verdict" in extract.TRIAGE_SYSTEM and "nutria = OTTER" in extract.TRIAGE_SYSTEM


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
            ("r4", None, None, None, "unknown", False, "headline"),
            ("r5", 43.2, -4.8, "2026-10-01", "brown bear", True, "fulltext")]    # Spain: out of region
    for rid, lat, lon, d, tx, ev, src in spec:
        arts.append({"id": rid, "title": f"title {rid}", "url": f"https://example.org/{rid}",
                     "publisher": "X", "published": "2026-10-02T00:00:00+00:00"})
        exs.append({"id": rid, **_ex(event_date=d, taxon=tx, is_specific_event=ev,
                                     real_wild_carnivore_involved=ev,
                                     country="other country" if rid == "r5" else "United States"),
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
    assert meta["n_events"] == 2 and meta["n_articles_not_events"] == 2 and meta["n_out_of_region"] == 1
    merged = [f for f in fc["features"] if f["properties"]["n_reports"] == 2][0]["properties"]
    assert merged["text_source"] == "fulltext" and merged["taxon_scientific"] == "Canis latrans"
    assert fc["metadata"]["group_colors"]["Wild cats"].startswith("#")
    assert (tmp_path / "events.csv").exists()


# --- speed: early stop ---------------------------------------------------------

def test_early_stop_parsing():
    nonevent = '{"relevance_reasoning": "General \\"wolf\\" policy story.", "real_wild_carnivore_involved": true, "is_specific_event": false'
    r = extract.early_stop(nonevent)
    assert r["is_specific_event"] is False and r["relevance_reasoning"] == 'General "wolf" policy story.'
    head = '{"relevance_reasoning": "x", "real_wild_carnivore_involved": true, "is_specific_event": true, "country": "'
    assert extract.early_stop(head + 'Can') is None                 # wait for the value
    assert extract.early_stop(head + 'Canada", ') is None           # NA event: keep generating
    assert extract.early_stop(head + 'other country", ')["country"] == "other country"


def test_early_stop_row_is_written_and_excluded(tmp_path, monkeypatch):
    from llm import EarlyStop
    out = _patch_extract(monkeypatch, tmp_path, [{"id": "a1", "title": "Wolves return to Europe", "url": None,
                                                  "gnews_url": "g"}])

    class FakeLLM:
        def __init__(self, cfg): pass
        def runtime_version(self): return "t"
        def structured(self, system, user, cls, early_stop=None):
            if cls is Triage:
                return Triage(reasoning="Wolves.", verdict="relevant")
            return EarlyStop({"relevance_reasoning": "Policy.", "real_wild_carnivore_involved": True,
                              "is_specific_event": False, "country": None}, "{...")

    monkeypatch.setattr(extract, "LLMClient", FakeLLM)
    monkeypatch.setattr(extract, "Geocoder", FakeGeo)
    assert extract.main() == 1
    row = json.loads(out.read_text())
    assert row["early_stopped"] is True and row["is_specific_event"] is False


def test_schema_decisions_come_first():
    props = list(json_schema()["properties"])
    assert props[:4] == ["relevance_reasoning", "real_wild_carnivore_involved", "is_specific_event", "country"]


# --- GDELT ---------------------------------------------------------------------

def test_gdelt_units_same_conflict_filter_and_length():
    import common
    s = common.load_config()["search"]
    units = taxa.gdelt_units(s)
    for lang in {u["language"] for u in units}:
        assert len({tuple(u["conflict"]) for u in units if u["language"] == lang}) == 1   # identical across taxa
    assert all(len(taxa.gdelt_query(u)) <= s["gdelt_max_query_chars"] for u in units)
    # every animal term of every group is queried somewhere
    for ed in s["editions"]:
        lang = ed.get("terms", ed["language"][:2])
        for g in taxa.registry()["groups"]:
            got = [t for u in units if u["group"] == g["id"] and u["country"] == ed["country"] for t in u["terms"]]
            assert got == g["query_terms"].get(lang, [])
    # exclusions travel only with related terms
    fox = next(u for u in units if u["country"] == "US" and '"gray fox"' in taxa.gdelt_query(u))
    assert '-"Fox News"' in taxa.gdelt_query(fox) and "Timberwolves" not in taxa.gdelt_query(fox)


def test_gdelt_split_on_too_long_and_parse(monkeypatch):
    import common
    import fetch_news
    cfg = common.load_config()
    cfg["search"]["editions"] = [{"country": "MX", "language": "es-419", "terms": "es"}]
    calls = []

    def fake_get(params, interval, backoff):
        q = params["query"]
        calls.append(q)
        if q.startswith("(lobo OR lobos") and "coyote" in q:
            return None, "HTTP 200: Your query was too short or too long."
        if "oso" in q:
            return [{"url": "https://example.mx/oso", "title": "Oso ataca a perro", "seendate": "20261126T101500Z",
                     "domain": "example.mx", "language": "Spanish", "sourcecountry": "Mexico"}], None
        return [], None

    monkeypatch.setattr(fetch_news, "_gdelt_get", fake_get)
    per_query = fetch_news.search_gdelt(cfg)
    assert any("sourcecountry:MX" in c and "sourcelang:spanish" in c for c in calls)
    canid_calls = [c for c in calls if "lobo" in c or "zorro" in c or "coyote" in c]
    assert len(canid_calls) >= 3                            # rejected once, then split in two
    rec = next(r for grp in per_query for r in grp if r["url"] == "https://example.mx/oso")
    assert rec["date_source"] == "gdelt_seendate" and rec["published"].startswith("2026-11-26T10:15")


# --- run-time options (workflow form) --------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("coyote", ("species", "coyote")), ("Puma concolor", ("species", "cougar")),
    ("mountain lion", ("species", "cougar")), ("felids", ("group", "felids")),
    ("Wild cats", ("group", "felids")), ("Ursidae", ("group", "bears")),
])
def test_taxon_filter(name, expected):
    assert taxa.taxon_filter(name) == expected


def test_taxon_filter_unknown_fails_fast():
    with pytest.raises(SystemExit):
        taxa.taxon_filter("panda")


def test_single_species_units(monkeypatch):
    import common
    monkeypatch.setenv("HWC_TAXON", "American black bear")
    s = common.load_config()["search"]
    units = taxa.gdelt_units(s)
    terms = [t for u in units for t in u["terms"]]
    assert {u["group"] for u in units} == {"bears"}
    assert "black bear" in terms and "oso negro" in terms
    assert "bear" not in terms and "grizzly" not in terms           # no other species / generic terms
    assert "American black bear" not in terms                        # redundant with "black bear"
    us_terms = [t for u in units if u["country"] == "US" for t in u["terms"]]
    assert not any(t.startswith("oso") for t in us_terms)           # Spanish names only in MX queries


def test_date_window_and_overrides(monkeypatch):
    import common
    import fetch_news
    monkeypatch.setenv("HWC_START_DATE", "2026-11-01")
    monkeypatch.setenv("HWC_END_DATE", "2026-11-15")
    monkeypatch.setenv("HWC_MAX_QUERIES", "3")
    s = common.load_config()["search"]
    assert fetch_news._gdelt_window(s) == {"startdatetime": "20261101000000", "enddatetime": "20261115235959"}
    assert s["max_queries"] == 3
    monkeypatch.setenv("HWC_START_DATE", "2026-11-20")
    with pytest.raises(SystemExit):
        common.load_config()


def test_max_queries_caps_requests(monkeypatch):
    import common
    import fetch_news
    monkeypatch.setenv("HWC_MAX_QUERIES", "4")
    calls = []
    monkeypatch.setattr(fetch_news, "_gdelt_get", lambda p, i, b: (calls.append(p) or ([], None)))
    fetch_news.search_gdelt(common.load_config())
    assert len(calls) == 4
