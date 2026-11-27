"""Taxon registry (taxa.yaml): vocabulary, queries and name resolution.

Species identity is resolved twice, independently:
  1. the LLM picks a standardised label from the closed registry vocabulary;
  2. the animal name *as written* in the article is matched against the
     registry synonyms (deterministic, accent/case-insensitive).
Agreement → high trust. Disagreement is kept and flagged (`taxon_check`),
never silently overwritten, so it can be audited or used as a QC filter.
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


def resolve(llm_label: str, name_as_written: str | None) -> dict:
    """Combine LLM label and dictionary match into final taxon fields."""
    by_label = index()["by_label"]
    dict_label = match_name(name_as_written)
    llm = by_label.get(llm_label, by_label[UNKNOWN])
    if dict_label is None:
        check, final = "no dictionary match", llm_label
    elif dict_label == llm_label:
        check, final = "agree", llm_label
    else:
        dic = by_label[dict_label]
        same_group = dic["group"] == llm["group"]
        if same_group and dic["rank"] == "species" and llm["rank"] != "species":
            # LLM: "unidentified wolves, coyotes & foxes"; name: "coyote" → name is more specific
            check, final = "dictionary more specific", dict_label
        elif same_group and llm["rank"] == "species" and dic["rank"] != "species":
            # name is generic ("fox"); LLM picked a species from context ("gray fox")
            check, final = "llm more specific", llm_label
        else:
            # genuine conflict: keep the model's reading of the full context, but flag it
            check, final = "disagree", llm_label
    rec = by_label.get(final, by_label[UNKNOWN])
    return {"taxon": final, "taxon_scientific": rec["scientific"], "taxon_rank": rec["rank"],
            "taxon_group": rec["group"], "taxon_family": rec["family"],
            "taxon_dictionary_match": dict_label, "taxon_check": check}


def group_colors() -> dict:
    by_label = index()["by_label"]
    return {r["group"]: r["color"] for r in by_label.values()}


def queries(cfg_search: dict) -> list[dict]:
    """All (group × edition) Google News queries, as plain strings."""
    out = []
    for ed in cfg_search["editions"]:
        lang = ed.get("terms", ed["language"][:2])
        conflict = cfg_search["conflict_terms"][lang]
        for g in registry()["groups"]:
            terms = g["query_terms"].get(lang) or []
            if not terms:
                continue
            q = f"({' OR '.join(_q(t) for t in terms)}) ({' OR '.join(_q(t) for t in conflict)})"
            excl = g.get("exclude_terms", {}).get(lang) or []
            if excl:
                q += " " + " ".join(f"-{_q(t)}" for t in excl)
            out.append({"group": g["id"], "country": ed["country"], "language": ed["language"], "query": q})
    return out


def _q(term: str) -> str:
    return f'"{term}"' if " " in term else term
