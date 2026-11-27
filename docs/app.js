/* HWC-map front-end: Leaflet map + client-side filtering of events.geojson */
(() => {
  "use strict";

  // Group colours come from taxa.yaml via events.geojson metadata
  let GROUP_COLORS = {};
  const colorOf = (p) => GROUP_COLORS[p.taxon_group] || "#999";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : "#");
  const fmtDate = (d) => d ? new Date(d + "T00:00:00").toLocaleDateString("en-GB",
    { day: "numeric", month: "short", year: "numeric" }) : "date unknown";
  const cap = (s) => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;

  // ---- map ------------------------------------------------------------
  const map = L.map("map", { zoomControl: true }).setView([50, -105], 3);
  // Keyless basemaps only (CARTO now requires an API key).
  const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/";
  const base = {
    "Topographic": L.tileLayer(ESRI + "World_Topo_Map/MapServer/tile/{z}/{y}/{x}", {
      attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, USGS, &copy; OpenStreetMap contributors", maxZoom: 19 }),
    "Satellite": L.tileLayer(ESRI + "World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      attribution: "Tiles &copy; Esri &mdash; Maxar, Earthstar Geographics", maxZoom: 19 }),
  };
  base.Topographic.addTo(map);
  L.control.layers(base, null, { position: "topright" }).addTo(map);
  L.control.scale({ imperial: false }).addTo(map);

  const legend = L.control({ position: "bottomright" });
  legend.onAdd = () => L.DomUtil.create("div", "legend");
  legend.addTo(map);
  function drawLegend(groupsPresent) {
    legend.getContainer().innerHTML = groupsPresent.map((g) =>
      `<div><span class="dot" style="background:${GROUP_COLORS[g] || "#999"}"></span>${esc(g)}</div>`).join("") +
      `<div><span class="dot harm" style="background:#fff"></span>person injured</div>` +
      `<div><span class="dot fatal" style="background:#fff"></span>person killed</div>` +
      `<div><span class="dot region" style="background:#999"></span>region-level location</div>`;
  }

  let clusterLayer = L.markerClusterGroup({ maxClusterRadius: 40, showCoverageOnHover: false });
  let plainLayer = L.layerGroup();
  let activeLayer = clusterLayer.addTo(map);

  // ---- state ------------------------------------------------------------
  let FEATURES = [];
  const markers = new Map();   // event_id -> marker
  const $ = (id) => document.getElementById(id);
  const FACETS = {
    group: { el: "f-group", get: (p) => [p.taxon_group], color: (v) => GROUP_COLORS[v] || "#999" },
    taxon: { el: "f-taxon", get: (p) => [p.taxon] },
    conflict: { el: "f-conflict", get: (p) => p.conflict_types || [] },
    human: { el: "f-human", get: (p) => [p.human_outcome] },
    animal: { el: "f-animal", get: (p) => [p.animal_outcome] },
  };
  const selected = {}; // facet -> Set of allowed values

  function buildFacet(key) {
    const f = FACETS[key];
    const counts = {};
    FEATURES.forEach((ft) => f.get(ft.properties).forEach((v) => (counts[v] = (counts[v] || 0) + 1)));
    const values = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    selected[key] = new Set(values);
    $(f.el).innerHTML = values.map((v) => `
      <label class="chk"><input type="checkbox" data-facet="${key}" value="${esc(v)}" checked>
        ${f.color ? `<span class="swatch" style="background:${f.color(v)}"></span>` : ""}
        ${esc(cap(v))}<span class="n" data-count="${key}|${esc(v)}">${counts[v]}</span></label>`).join("");
  }

  function passes(p, skipFacet) {
    for (const key in FACETS) {
      if (key === skipFacet) continue;
      const vals = FACETS[key].get(p);
      if (!vals.some((v) => selected[key].has(v))) return false;
    }
    const from = $("d-from").value, to = $("d-to").value;
    if (from && (!p.date || p.date < from)) return false;
    if (to && (!p.date || p.date > to)) return false;
    if (!$("q-region").checked && p.geocode_level !== "place") return false;
    if (!$("q-low").checked && p.confidence === "low") return false;
    const q = $("search").value.trim().toLowerCase();
    if (q) {
      const hay = [p.summary, p.location_name, p.admin1, ...(p.sources || []).map((s) => `${s.title} ${s.publisher}`)]
        .join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  }

  function markerFor(ft) {
    const p = ft.properties;
    const cls = ["dot"];
    if (p.geocode_level !== "place") cls.push("region");
    if (p.human_outcome === "killed") cls.push("fatal");
    else if (p.human_outcome === "injured") cls.push("harm");
    const size = 12 + Math.min(p.n_reports - 1, 4) * 2;
    const [lon, lat] = ft.geometry.coordinates;
    const m = L.marker([lat, lon], {
      icon: L.divIcon({ className: "", iconSize: [size, size],
        html: `<div class="${cls.join(" ")}" style="width:${size}px;height:${size}px;background:${colorOf(p)}"></div>` }),
      title: `${p.location_name || p.admin1 || ""} — ${p.date || ""}`,
    });
    m.on("click", () => showDetail(ft));
    return m;
  }

  function render() {
    const vis = FEATURES.filter((ft) => passes(ft.properties));
    clusterLayer.clearLayers(); plainLayer.clearLayers();
    vis.forEach((ft) => activeLayer.addLayer(markers.get(ft.properties.event_id)));
    // facet counts reflect all *other* active filters
    for (const key in FACETS) {
      const c = {};
      FEATURES.filter((ft) => passes(ft.properties, key))
        .forEach((ft) => FACETS[key].get(ft.properties).forEach((v) => (c[v] = (c[v] || 0) + 1)));
      document.querySelectorAll(`[data-count^="${key}|"]`).forEach((el) => {
        el.textContent = c[el.dataset.count.split("|").slice(1).join("|")] || 0;
      });
    }
    $("count").textContent = vis.length;
    $("list").innerHTML = vis.map((ft) => {
      const p = ft.properties;
      return `<li data-id="${esc(p.event_id)}"><span class="swatch" style="background:${colorOf(p)}"></span>
        <span class="t">${esc(p.location_name || p.admin1 || "Unknown location")}${p.admin1 && p.location_name ? ", " + esc(p.admin1) : ""}</span>
        <span class="s">${esc(fmtDate(p.date))} · ${esc(p.taxon)} · ${esc(p.primary_conflict_type)}${p.human_outcome === "injured" || p.human_outcome === "killed" ? " · person " + esc(p.human_outcome) : ""}</span></li>`;
    }).join("");
    return vis;
  }

  function showDetail(ft) {
    const p = ft.properties;
    const tags = (p.conflict_types || []).map((t) => `<span class="tag">${esc(t)}</span>`).join("");
    const flags = [
      p.geocode_level !== "place" ? `<span class="tag warn">location approximate (${esc(p.geocode_level)})</span>` : "",
      p.date_is_publication_date ? `<span class="tag warn">date = ${p.date_source === "gdelt_seendate" ? "date first seen online" : "publication date"}</span>` : "",
      p.confidence === "low" ? `<span class="tag warn">low-confidence extraction</span>` : "",
      p.text_source && p.text_source !== "fulltext" ? `<span class="tag warn">extracted from ${p.text_source === "headline" ? "headline" : "headline/snippet"} only</span>` : "",
      p.multiple_events ? `<span class="tag warn">article reports multiple incidents</span>` : "",
      p.taxon_check === "disagree" ? `<span class="tag warn">species identification uncertain</span>` : "",
      p.rabies_status === "confirmed positive" ? `<span class="tag danger">rabies confirmed</span>` : "",
      p.human_outcome === "killed" ? `<span class="tag danger">fatal</span>` : "",
    ].join("");
    const row = (k, v) => (v === null || v === undefined || v === "") ? "" : `<dt>${k}</dt><dd>${v}</dd>`;
    $("detail-body").innerHTML = `
      <h3>${esc(p.location_name || p.admin1 || "Unknown location")}</h3>
      <div class="when">${esc(fmtDate(p.date))}${p.event_date_precision && p.event_date_precision !== "day" ? ` (precision: ${esc(p.event_date_precision)})` : ""} · ${esc([p.admin1, p.country].filter(Boolean).join(", "))}</div>
      <div>${tags}${flags}</div>
      <p class="summary">${esc(p.summary)}</p>
      <dl>
        ${row("Animal", `${esc(cap(p.taxon))}${p.taxon_scientific ? ` <i>${esc(p.taxon_scientific)}</i>` : ""}<br><small>${esc(p.taxon_family || p.taxon_group)} · named in text as “${esc(p.animal_name || "—")}” · species from ${esc(p.taxon_evidence_source === "none" ? "model only" : p.taxon_evidence_source)}${p.taxon_llm && p.taxon_llm !== p.taxon ? ` · model proposed “${esc(p.taxon_llm)}”` : ""}</small>`)}
        ${row("Animals involved", esc(p.n_animals))}
        ${row("Primary type", esc(p.primary_conflict_type))}
        ${row("Human activity", esc(p.human_activity))}
        ${row("Human outcome", esc(p.human_outcome) + (p.n_people_injured ? ` (${esc(p.n_people_injured)} injured)` : "") + (p.n_people_killed ? ` (${esc(p.n_people_killed)} killed)` : ""))}
        ${row("Domestic animals", p.domestic_animal_type && p.domestic_animal_type !== "not applicable" ? esc(`${p.domestic_animal_type} — ${p.domestic_animal_outcome}`) : "")}
        ${row("Rabies", p.rabies_status && p.rabies_status !== "not mentioned" ? esc(p.rabies_status) : "")}
        ${row("Animal outcome", esc(p.animal_outcome))}
        ${row("Geocoded to", esc(p.geocode_match))}
        ${row("Confidence", esc(p.confidence))}
      </dl>
      <h4>Sources (${p.n_reports})</h4>
      <ul class="src">${(p.sources || []).map((s) => `<li><a href="${esc(safeUrl(s.url))}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a><br><small>${esc(s.publisher || "")} · ${esc((s.published || "").slice(0, 10))}</small></li>`).join("")}</ul>
      <div class="prov">Metadata extracted automatically by ${esc(p.llm_model)} and geocoded with OpenStreetMap Nominatim. May contain errors — check the sources. Event ID ${esc(p.event_id)}</div>`;
    $("detail").classList.remove("hidden");
    map.invalidateSize();
    document.querySelectorAll("#list li").forEach((li) => li.classList.toggle("active", li.dataset.id === p.event_id));
    history.replaceState(null, "", `#event=${encodeURIComponent(p.event_id)}`);
  }

  function focus(id) {
    const ft = FEATURES.find((f) => f.properties.event_id === id);
    if (!ft) return;
    const m = markers.get(id);
    if (activeLayer === clusterLayer && clusterLayer.hasLayer(m)) {
      showDetail(ft);
      clusterLayer.zoomToShowLayer(m);
    } else {
      showDetail(ft);
      map.setView(m.getLatLng(), Math.max(map.getZoom(), 8));
    }
  }

  // ---- wiring ----------------------------------------------------------
  document.addEventListener("change", (e) => {
    const t = e.target;
    if (t.dataset.facet) {
      t.checked ? selected[t.dataset.facet].add(t.value) : selected[t.dataset.facet].delete(t.value);
      render();
    } else if (t.id === "q-cluster") {
      map.removeLayer(activeLayer);
      activeLayer = t.checked ? clusterLayer : plainLayer;
      activeLayer.addTo(map); render();
    } else if (["d-from", "d-to", "q-region", "q-low"].includes(t.id)) render();
  });
  $("search").addEventListener("input", render);
  $("list").addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) focus(li.dataset.id); });
  $("close").addEventListener("click", () => { $("detail").classList.add("hidden"); map.invalidateSize(); history.replaceState(null, "", location.pathname); });
  $("reset").addEventListener("click", () => {
    Object.keys(FACETS).forEach(buildFacet);
    ["d-from", "d-to", "search"].forEach((id) => ($(id).value = ""));
    $("q-region").checked = $("q-low").checked = true;
    render();
  });
  $("download").addEventListener("click", () => {
    const fc = { type: "FeatureCollection", features: FEATURES.filter((ft) => passes(ft.properties)) };
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(new Blob([JSON.stringify(fc, null, 1)], { type: "application/geo+json" })),
      download: "hwc_events_filtered.geojson" });
    a.click(); URL.revokeObjectURL(a.href);
  });

  fetch("data/events.geojson", { cache: "no-cache" })
    .then((r) => r.json())
    .then((fc) => {
      FEATURES = fc.features;
      GROUP_COLORS = (fc.metadata && fc.metadata.group_colors) || {};
      drawLegend([...new Set(FEATURES.map((f) => f.properties.taxon_group))].sort());
      FEATURES.forEach((ft) => markers.set(ft.properties.event_id, markerFor(ft)));
      Object.keys(FACETS).forEach(buildFacet);
      const m = fc.metadata || {};
      $("meta").innerHTML = m.generated_at
        ? `${m.n_events ?? FEATURES.length} events from ${m.n_articles_processed ?? "?"} articles` +
          `<br>updated ${esc(m.generated_at.replace("T", " ").slice(0, 16))} UTC`
        : "No data yet — waiting for the first pipeline run";
      const vis = render();
      if (vis.length) map.fitBounds(L.latLngBounds(vis.map((f) => [f.geometry.coordinates[1], f.geometry.coordinates[0]])).pad(0.2), { maxZoom: 7 });
      const h = decodeURIComponent((location.hash.match(/event=([^&]+)/) || [])[1] || "");
      if (h) focus(h);
    })
    .catch((err) => { $("meta").textContent = "Could not load data: " + err; });
})();
