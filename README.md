# HWC-map

A near-real-time map of **human–wildlife conflict (HWC)** events mined from online news.
Scope: **wild terrestrial Carnivora in the USA, Canada and Mexico** (English and Spanish news). Taxa are defined in [`taxa.yaml`](taxa.yaml).

**🗺️ Live map: <https://andrewzamp.github.io/HWC-map/>**

```
GDELT DOC API ──────► article metadata + full text ──► open-weight LLM (Ollama, JSON-schema constrained)
        │                                                     │
        ▼                                                     ▼
 data/articles.jsonl                               data/extractions.jsonl ◄── Nominatim geocoding
                                                              │
                                     duplicate-report merging (space × time × species)
                                                              ▼
                                 docs/data/events.geojson  +  data/events.csv
                                                              ▼
                                       docs/ (Leaflet map on GitHub Pages)
```

## Pipeline

| Step | Script | What it does |
|---|---|---|
| fetch | `code/fetch_news.py` | Runs one query per **taxon group × edition** (outlets in the US/English, Canada/English and Mexico/Spanish) against the **GDELT DOC 2.0 API**, built from `taxa.yaml` and `config.yaml`. GDELT returns direct article URLs. Results are sampled round-robin across queries, and already-seen URLs and syndicated duplicate headlines are dropped. Stores metadata only. Google News (`source: googlenews`) remains available, but its links can no longer be decoded, so those articles are headline-only. |
| extract | `code/extract.py` | **Pass 1, triage**: a cheap headline-only LLM call skips clearly irrelevant hits (sports teams, Fox News, "bear spray" assaults, smugglers called "coyotes"…). Only *irrelevant* is skipped, *unsure* goes through, and every verdict is stored. **Pass 2, extraction**: downloads the article body (retried on later runs if it fails), then sends headline + date + the first ~3,500 characters to an open-weight LLM with a definition of every field. Output is streamed. The relevance decisions come first in the schema, and generation is stopped as soon as the model says the article is not a specific event or is outside North America, since the remaining fields would be discarded anyway. Output is constrained to the JSON schema in `code/schema.py` and validated with pydantic. Taxon resolution and Nominatim geocoding follow. Incremental: each article is processed once per extraction version. |
| build | `code/build_map_data.py` | Keeps specific events, merges reports of the same incident, writes GeoJSON for the map and a flat CSV for analysis. |

Run everything: `python code/pipeline.py` (or a single step: `python code/pipeline.py fetch|extract|build`).

### GDELT query design
GDELT limits query length (≈200–300 characters) and request rate (≈1 per 5 s per IP; GitHub runners share IPs). So:
- **The same conflict-term filter is used for every taxon** in a language (the first `gdelt_conflict_terms_n` terms in `config.yaml`). A filter that varied by taxon would bias between-taxa comparisons.
- Animal terms are split across several queries rather than dropped. A query GDELT rejects as too long is halved automatically.
- Each query carries only the exclusions that share a word with its animal terms.
- Rate limits are handled with progressive back-off and a time budget. The starting query rotates daily, and the 7-day window overlaps between runs, so a query skipped one day is covered the next. The `GDELT SUMMARY` log line reports successes, failures and skips.

### Taxa (`taxa.yaml`)
The registry is the single source of truth for each taxon group's search terms (EN/ES), excluded terms (`-"Chicago Bears"`, `-"Fox News"`…), the closed species vocabulary the model must choose from, name synonyms and map colours. Adding a species or group there updates queries, the schema and the map; then bump `EXTRACTION_VERSION`. Nomenclature follows the Mammal Diversity Database.

**Species identification is resolved twice, independently.** The LLM picks a standardised label from the registry, and the animal name *as written* (`animal_name`) is matched against registry synonyms. The result is stored in `taxon_check`:
- `agree`: both resolutions give the same taxon.
- `dictionary more specific` / `llm more specific`: one resolution refines the other within the same group.
- `disagree`: the two conflict. The record is kept, flagged on the map, and can be filtered out in analyses.
- `no dictionary match`: the name isn't in the registry.

Generic names ("fox", "skunk", "otter") resolve to *unidentified <group>*, never to a species. Context-dependent names (e.g. *tejón*: coati in Mexico, badger elsewhere) are not resolved by the dictionary at all.

### Extracted fields (`code/schema.py`)
`relevance_reasoning`, `real_wild_carnivore_involved`, `is_specific_event`, `multiple_events`, `animal_name`, `taxon` (+ evidence; scientific name, family and group added from the registry),
`n_animals`, `event_date` (+ precision), `conflict_types` / `primary_conflict_type` (including rabies/disease exposure), `human_activity`, `human_outcome`,
`n_people_injured/killed`, `domestic_animal_type/outcome`, `rabies_status`, `animal_outcome`, `location_name`, `admin1`, `country`,
`location_precision`, `summary`, `confidence`, `notes`. Each row also stores the triage verdict, the LLM model and runtime version,
whether the extraction used the full text, a snippet or only the headline, the extraction version (bumping `EXTRACTION_VERSION` in `schema.py` re-processes all articles), and the geocoding level
(`place` / `admin1` / `none`) with the matched OSM feature.

### Design choices
- **Controlled vocabularies + explicit "unknown"**: every field you filter on is an enum. The model is told to leave a field missing rather than guess.
- **The LLM never outputs coordinates.** It returns place names copied from the text. Geocoding is deterministic, and a result is accepted only if it falls inside the state/province the article names. Otherwise the record falls back to the state/province centroid and is flagged.
- **Event date ≠ publication date.** The model resolves relative dates against the publication date. If no event date is found, the publication date is used and flagged (`date_is_publication_date`).
- **Duplicate reports** are merged by single-linkage clustering (same species, ≤ 30 km, ≤ 3 days apart; thresholds in `config.yaml`). Attribute values come from the best-documented report, and all sources are kept.
- **Copyright**: only headline, snippet, URL and our own summary are committed. Full text is cached in `data/cache/` (git-ignored) and used only during the run.

## Running locally
```bash
pip install -r requirements.txt
# install Ollama (https://ollama.com), then:
ollama pull qwen2.5:7b-instruct
python code/pipeline.py
python -m http.server -d docs 8000   # open http://localhost:8000
```
To use a hosted open-weight model instead (any OpenAI-compatible endpoint):
```bash
export HWC_LLM_BACKEND=openai HWC_LLM_BASE_URL=https://api.groq.com/openai/v1 \
       HWC_LLM_MODEL=llama-3.3-70b-versatile HWC_LLM_API_KEY=...
```

Tests (offline, no LLM): `pytest -q`

## Custom runs (taxon, period, number of queries)
**Actions → Update HWC map → Run workflow** opens a form where every field is optional:

| Field | Example | Effect |
|---|---|---|
| taxon | `coyote`, `Puma concolor`, `mountain lion`, `felids` | Search one species or one group only (species: its specific names, in each edition's language) |
| start_date / end_date | `2026-11-01` / `2026-11-15` | Explicit window (inclusive); overrides `period`. GDELT searches only ~the last 3 months |
| period | `24h`, `7d`, `30d` | Rolling look-back window when no dates are given |
| max_queries | `5` | Cap on GDELT queries this run |
| max_new_articles | `10` | Cap on new articles accepted this run |

Locally, the same overrides are environment variables: `HWC_TAXON`, `HWC_START_DATE`, `HWC_END_DATE`, `HWC_PERIOD`, `HWC_MAX_QUERIES`, `HWC_MAX_NEW_ARTICLES`. An unknown taxon or a malformed date stops the run immediately with a message listing the valid options. Extraction processes the most recently fetched articles first, so a targeted run's articles are handled in that same run.

## GitHub setup (one-time)
1. **Settings → Actions → General → Workflow permissions** → *Read and write permissions* (the workflow commits the data).
2. **Settings → Pages** → *Deploy from a branch* → `main` / `/docs`.
3. **Actions → Update HWC map → Run workflow** for the first run. After that it runs daily (05:17 UTC).

Each run processes at most `max_articles_per_run` articles (config); any backlog carries over to the next run. On a **private** repo, Actions minutes count against your plan's monthly quota; public repos run free.

Ollama is downloaded only when its version changes (binary and model are both cached). Set `OLLAMA_VERSION` in the workflow to pin it; the version used is recorded in every extracted row.

The map is then served at `https://andrewzamp.github.io/HWC-map/`.

## Known limitations / next steps
- Query terms differ in how much noise they pull in ("fox", "jaguar", "coyote", "panther" are heavily polysemous), so **recall and precision differ by taxon and language**. Compare taxa only after measuring this.
- **Dates**: GDELT gives the time it first *saw* an article, not its publication time. It's used only when the text gives no event date, and it's flagged (`date_source`).
- GDELT and Google News return ≤ 250 / ≤ 100 items per query, and their coverage of local outlets is uneven, so the sample is **not** a census. Reporting effort varies by region and outlet, and the map shows *reported* conflicts. Any spatial or temporal analysis must account for this (e.g. by modelling media effort).
- Paywalled or JS-rendered pages fall back to snippet-only extraction (flagged in the data).
- A 7B model makes mistakes. Before scaling, hand-annotate a validation sample (~100–200 articles) and measure per-field accuracy. Precision/recall of `is_specific_event` is the priority.
- Nominatim's free API allows ≤ 1 request/s and suits this volume. At larger scale, self-host it or switch to GeoNames.
