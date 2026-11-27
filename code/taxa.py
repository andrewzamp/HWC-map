"""Taxon registry (taxa.yaml): vocabulary, queries and name resolution.

Species identity is resolved from two independent sources:
  1. the LLM picks a standardised label from the closed registry vocabulary;
  2. TEXT EVIDENCE: the animal name as written by the article (returned by the
     LLM) and, failing that, a scan of headline + text for registry names —
     both matched deterministically against registry synonyms.

Rule: text evidence wins whenever it exists; the LLM label is used only when
the text gives no usable name (e.g. "the animal", out-of-registry species).
Rationale: a 7B model can mislabel even an explicit headline ("4 grizzly bears"
→ gray wolf), while a dictionary match is reproducible and auditable. Generic
names ("skunk") stay at group level even if the LLM proposes a species.
The LLM's own label is always kept (`taxon_llm`) and the relation between the
two sources is recorded in `taxon_check` for QC.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import yaml

from common import ROOT

UNKNOWN = "unknown"

# words that describe an individual, not a taxon (EN + ES)
MODIFIERS = set("""a an the and of two three four several pack group family rabid sick injured dead
young baby babies juvenile adult old large big small aggressive habituated nuisance wild female male
mother mom sow boar cub cubs pup pups kit kits kitten kittens yearling yearlings
un una unos unas el la los las y de dos tres joven jovenes cria crias cachorro cachorros
hembra macho madre adulto grande pequeno enfermo herido muerto rabioso salvaje""".split())


def _norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]+", " ", s).strip()


@lru_cache
def registry() -> dict:
    with open(ROOT / "taxa.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unidentified_label(group: dict) -> str:
    return f"unidentified {group['label'].lower()}"


@lru_cache
def index() -> dict:
    """label → record; synonym → label; plus ordered vocabularies."""
    reg = registry()
    by_label, synonyms, group_labels = {}, {}, []
    ambiguous = {_norm(n) for n in reg.get("ambiguous_names", [])}
    for g in reg["groups"]:
        group_labels.append(g["label"])
        meta = {"group": g["label"], "group_id": g["id"], "family": g["family"], "color": g["color"]}
        for sp in g["species"]:
            by_label[sp["name"]] = {**meta, "scientific": sp["scientific"], "rank": "species"}
            for syn in [sp["name"], *sp.get("synonyms", [])]:
                key = _norm(syn)
                if key and key not in ambiguous:
                    synonyms.setdefault(key, set()).add(sp["name"])
        by_label[unidentified_label(g)] = {**meta, "scientific": None, "rank": "family"}
        for gen in g.get("generic_names", []):
            key = _norm(gen)
            if key and key not in ambiguous:
                synonyms.setdefault(key, set()).add(unidentified_label(g))
    other = reg.get("other_label", "other wild carnivore")
    by_label[other] = {"group": "Other carnivores", "group_id": "other", "family": None,
                       "color": "#999999", "scientific": None, "rank": "order"}
    by_label[UNKNOWN] = {"group": "Unknown", "group_id": "unknown", "family": None,
                         "color": "#BBBBBB", "scientific": None, "rank": None}
    # a synonym shared by >1 species is ambiguous by construction
    synonyms = {k: next(iter(v)) for k, v in synonyms.items() if len(v) == 1}
    return {"by_label": by_label, "synonyms": synonyms, "group_labels": group_labels, "other": other}


def taxon_labels() -> list[str]:
    return list(index()["by_label"].keys())


def match_name(name: str | None) -> str | None:
    """Deterministic match of a name as written → registry label (or None)."""
    key = _norm(name)
    if not key:
        return None
    syn = index()["synonyms"]
    # tolerate NON-taxonomic modifiers only ("rabid gray fox", "mother bear and cubs");
    # "sea otter" or "black-footed ferret" must NOT collapse to "otter"/"ferret".
    words = key.split()
    for n in range(len(words), 0, -1):
        for i in range(len(words) - n + 1):
            cand = " ".join(words[i:i + n])
            rest = words[:i] + words[i + n:]
            hit = _lookup(cand, syn)
            if hit and all(w in MODIFIERS for w in rest):
                return hit
    return None


def _lookup(cand: str, syn: dict) -> str | None:
    """Exact synonym, else simple EN/ES plural stripping on the last word."""
    for c in (cand, cand[:-1] if cand.endswith("s") else None, cand[:-2] if cand.endswith("es") else None):
        if c and c in syn:
            return syn[c]
    return None


@lru_cache
def _scan_regex() -> tuple[re.Pattern, dict]:
    syn = index()["synonyms"]
    forms = {}
    for k, label in syn.items():
        for f in (k, k + "s", k + "es"):
            forms.setdefault(f, label)
    alts = sorted(forms, key=len, reverse=True)     # longest match first ("black bear" before "bear")
    return re.compile(r"\b(" + "|".join(map(re.escape, alts)) + r")\b"), forms


def scan_text(text: str | None) -> list[str]:
    """Registry labels named anywhere in a text, in order of appearance."""
    if not text:
        return []
    rx, forms = _scan_regex()
    return [forms[m.group(1)] for m in rx.finditer(_norm(text))]


def _evidence(name_as_written: str | None, text: str | None) -> tuple[str | None, str]:
    """Best text-grounded taxon label and where it came from."""
    by_label = index()["by_label"]
    found = scan_text(text)
    species_found = {l for l in found if by_label[l]["rank"] == "species"}
    name_label = match_name(name_as_written)
    if name_label:
        rec = by_label[name_label]
        if rec["rank"] != "species":
            # generic name ("bear"); the text may still name the species once ("black bear")
            same_group = {l for l in species_found if by_label[l]["group"] == rec["group"]}
            if len(same_group) == 1:
                return same_group.pop(), "text scan"
        return name_label, "animal name"
    if len(species_found) == 1:
        return species_found.pop(), "text scan"
    groups = {by_label[l]["group"] for l in found}
    if not species_found and len(groups) == 1:
        return next(l for l in found), "text scan"      # only a generic name, one group
    return None, "none"


def resolve(llm_label: str, name_as_written: str | None, text: str | None = None) -> dict:
    """Final taxon fields: text evidence first, LLM label as fallback."""
    by_label = index()["by_label"]
    ev, source = _evidence(name_as_written, text)
    llm = by_label.get(llm_label, by_label[UNKNOWN])
    if ev is None:
        check, final = "llm only (no text evidence)", llm_label
    elif ev == llm_label:
        check, final = "agree", ev
    else:
        e = by_label[ev]
        same_group = e["group"] == llm["group"]
        if same_group and e["rank"] == "species" and llm["rank"] != "species":
            check = "text more specific"
        elif same_group and e["rank"] != "species" and llm["rank"] == "species":
            check = "llm more specific (not supported by text)"
        else:
            check = "disagree"
        final = ev
    rec = by_label.get(final, by_label[UNKNOWN])
    return {"taxon": final, "taxon_scientific": rec["scientific"], "taxon_rank": rec["rank"],
            "taxon_group": rec["group"], "taxon_family": rec["family"],
            "taxon_llm": llm_label, "taxon_evidence_label": ev, "taxon_evidence_source": source,
            "taxon_check": check}


def group_colors() -> dict:
    by_label = index()["by_label"]
    return {r["group"]: r["color"] for r in by_label.values()}


def queries(cfg_search: dict) -> list[dict]:
    """All (group × edition) Google News queries, as plain strings."""
    out = []
    flt = taxon_filter(cfg_search.get("taxon_filter"))
    for ed in cfg_search["editions"]:
        lang = ed.get("terms", ed["language"][:2])
        conflict = cfg_search["conflict_terms"][lang]
        for g in _groups(flt):
            terms = _terms_for(g, lang, flt)
            if not terms:
                continue
            q = f"({' OR '.join(_q(t) for t in terms)}) ({' OR '.join(_q(t) for t in conflict)})"
            excl = g.get("exclude_terms", {}).get(lang) or []
            if excl:
                q += " " + " ".join(f"-{_q(t)}" for t in excl)
            out.append({"group": g["id"], "country": ed["country"], "language": ed["language"], "query": q})
    return out


GDELT_LANG = {"en": "english", "es": "spanish", "fr": "french"}


def taxon_filter(name: str | None) -> tuple[str, str] | None:
    """Resolve a user-supplied taxon (group id/label, species name, synonym or
    scientific name) → ("group", group_id) or ("species", label)."""
    if not name:
        return None
    key = _norm(name)
    for g in registry()["groups"]:
        if key in (_norm(g["id"]), _norm(g["label"]), _norm(g["family"])):
            return ("group", g["id"])
        for sp in g["species"]:
            if key in (_norm(sp["name"]), _norm(sp["scientific"])):
                return ("species", sp["name"])
    label = match_name(name)
    if label and index()["by_label"][label]["rank"] == "species":
        return ("species", label)
    valid = [g["id"] for g in registry()["groups"]] + [sp["name"] for g in registry()["groups"] for sp in g["species"]]
    raise SystemExit(f"Unknown taxon {name!r}. Use a group id or a species, e.g. one of: {', '.join(valid)}")


def _terms_for(g: dict, lang: str, flt: tuple[str, str] | None) -> list[str]:
    """Query terms of a group, restricted to one species if requested."""
    terms = list(g["query_terms"].get(lang) or [])
    if flt is None or flt[0] == "group":
        return terms
    label = flt[1]
    own = [t for t in terms if match_name(t) == label]            # e.g. "mountain lion" → cougar
    sp = next(s for s in g["species"] if s["name"] == label)
    # add the species' multi-word names in this language ("brown bear", "Canada lynx"),
    # unless already covered by a broader term ("grizzly" covers "grizzly bear")
    covered = {w for t in own for w in _norm(t).split() if len(_norm(t).split()) == 1}
    out, seen = list(own), {_norm(t) for t in own}
    cands = sorted({_norm(n): n for n in [sp["name"], *sp.get("synonyms", [])]}.items(), key=lambda kv: len(kv[0]))
    for n, name in cands:                       # shortest first, so "black bear" blocks "american black bear"
        contains_seen = any(re.search(rf"\b{re.escape(s)}\b", n) for s in seen)
        if len(n.split()) < 2 or n in seen or _lang_of(n) != lang or covered & set(n.split()) or contains_seen:
            continue
        seen.add(n)
        out.append(name)
    return out


_ES_HEADS = {"oso", "osa", "lobo", "zorro", "zorra", "zorrita", "lince", "nutria", "marta", "zorrillo",
             "vison", "gato", "leon", "perro", "gloton", "mapache", "coati", "cacomixtle", "tejon"}


def _lang_of(name_norm: str) -> str:
    """Registry synonyms are not language-tagged: Spanish names start with a Spanish noun."""
    return "es" if name_norm.split()[0] in _ES_HEADS else "en"


def _groups(flt: tuple[str, str] | None) -> list[dict]:
    groups = registry()["groups"]
    if flt is None:
        return groups
    if flt[0] == "group":
        return [g for g in groups if g["id"] == flt[1]]
    grp = index()["by_label"][flt[1]]["group_id"]
    return [g for g in groups if g["id"] == grp]


def _or(terms: list[str]) -> str:
    """GDELT: OR-ed terms must be in parentheses; a single term must not be."""
    return _q(terms[0]) if len(terms) == 1 else "(" + " OR ".join(_q(t) for t in terms) + ")"


def gdelt_units(cfg_search: dict) -> list[dict]:
    """GDELT query units, one or more per (group × edition).

    Every unit of a language uses the SAME conflict-term set (first
    `gdelt_conflict_terms_n` terms), so the query filter is identical across
    taxa. Animal terms are split across units to respect the length limit;
    each unit carries the group exclusions that share a word with its animal
    terms (space is reserved for them).
    """
    max_len = cfg_search.get("gdelt_max_query_chars", 230)
    n_c = cfg_search.get("gdelt_conflict_terms_n", 8)
    units = []
    flt = taxon_filter(cfg_search.get("taxon_filter"))
    for ed in cfg_search["editions"]:
        lang = ed.get("terms", ed["language"][:2])
        conflict = cfg_search["conflict_terms"][lang][:n_c]
        filters = f" sourcecountry:{ed['country']} sourcelang:{GDELT_LANG.get(lang, lang)}"
        for g in _groups(flt):
            terms = _terms_for(g, lang, flt)
            if not terms:
                continue
            excl = list(g.get("exclude_terms", {}).get(lang) or [])
            unit = {"group": g["id"], "country": ed["country"], "language": ed["language"],
                    "conflict": conflict, "filters": filters}
            fits = lambda ch: len(gdelt_query({**unit, "terms": ch, "exclude": _relevant(excl, ch)})) <= max_len
            chunk = []
            for term in terms:
                if chunk and not fits(chunk + [term]):
                    units.append({**unit, "terms": chunk, "exclude": _relevant(excl, chunk)})
                    chunk = []
                chunk.append(term)
            if chunk:
                units.append({**unit, "terms": chunk, "exclude": _relevant(excl, chunk)})
    return units


def _relevant(exclusions: list[str], terms: list[str]) -> list[str]:
    """Exclusions sharing a word stem with the query's animal terms
    ("Fox News" ↔ "red fox"; "Timberwolves" ↔ "wolves"; "Chicago Bears" ↔ "bear")."""
    stems = {w[:4] for t in terms for w in _norm(t).split()}
    def hit(x):
        words = _norm(x).split()
        return any(w.startswith(s) or s in w for w in words for s in stems)
    return [x for x in exclusions if hit(x)]


def gdelt_query(unit: dict) -> str:
    q = _or(unit["terms"]) + " " + _or(unit["conflict"])
    q += "".join(f" -{_q(x)}" for x in unit["exclude"])
    return q + unit["filters"]


def _q(term: str) -> str:
    return f'"{term}"' if " " in term else term
