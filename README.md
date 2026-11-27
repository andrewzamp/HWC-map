# HWC-map

A near-real-time map of **human–wildlife conflict (HWC)** events mined from online news.
Prototype scope: **bears (*Ursus americanus*, *U. arctos*, *U. maritimus*) in the USA and Canada**.

```
Google News (gnews) ──► article metadata + full text ──► open-weight LLM (Ollama, JSON-schema constrained)
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
| fetch | `code/fetch_news.py` | Queries Google News (US + CA editions) with the query in `config.yaml`, drops already-seen URLs and syndicated duplicate headlines, decodes the Google redirect, downloads the article body with `trafilatura`. |
| extract | `code/extract.py` | Sends headline + date + body to an open-weight LLM. The JSON schema in `code/schema.py` constrains decoding, and the output is validated with pydantic. Place names are then geocoded with OSM Nominatim (`code/geocode.py`). Incremental: each article is processed once. |
| build | `code/build_map_data.py` | Keeps specific events, merges reports of the same incident, writes GeoJSON for the map and a flat CSV for analysis. |

Run everything: `python code/pipeline.py` (or a single step: `python code/pipeline.py fetch|extract|build`).

### Extracted fields (`code/schema.py`)
`is_specific_event`, `multiple_events`, `event_date` (+ precision), `species` (+ evidence),
`n_bears`, `conflict_types` / `primary_conflict_type`, `human_activity`, `human_outcome`,
`n_people_injured/killed`, `bear_outcome`, `location_name`, `admin1`, `country`,
`location_precision`, `summary`, `confidence`, `notes`. Each row also stores the LLM model,
whether the extraction used the full text or only the snippet, and the geocoding level
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

## GitHub setup (one-time)
1. **Settings → Actions → General → Workflow permissions** → *Read and write permissions* (the workflow commits the data).
2. **Settings → Pages** → *Deploy from a branch* → `main` / `/docs`.
3. **Actions → Update HWC map → Run workflow** for the first run. After that it runs daily (05:17 UTC).

The map is then served at `https://andrewzamp.github.io/HWC-map/`.

## Known limitations / next steps
- Google News returns ≤ 100 items per query and its ranking is opaque, so the sample is **not** a census. Reporting effort varies by region and outlet, and the map shows *reported* conflicts. Any spatial or temporal analysis must account for this (e.g. by modelling media effort).
- Paywalled or JS-rendered pages fall back to snippet-only extraction (flagged in the data).
- A 7B model makes mistakes. Before scaling, hand-annotate a validation sample (~100–200 articles) and measure per-field accuracy. Precision/recall of `is_specific_event` is the priority.
- Nominatim's free API allows ≤ 1 request/s and suits this volume. At larger scale, self-host it or switch to GeoNames.
