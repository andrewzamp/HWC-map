"""Deterministic text selection for the LLM: what to read, and whether to read.

On the CPU runner, reading input (prompt evaluation, ~15 tokens/s) costs more
than writing output, so the text sent to the model is condensed:
  * the LEAD (first `lead_chars`): news is written as an inverted pyramid, so
    who/what/where/when are usually there;
  * then, in original order, later sentences that name a registry animal or
    contain an outcome/conflict word (killed, euthanized, relocated, hospital…),
    which is where outcomes reported late in a story are found;
  until `max_chars`. The model sees '[…]' where text was skipped.

`animal_conflict_context` is a cheap gate for articles triage marked 'unsure':
does any registry animal name occur within `window` words of a conflict word?
If not (e.g. a sports story, a city news round-up that matched on a stray
word), the article is not sent to full extraction and the reason is recorded.
"""
from __future__ import annotations

import bisect
import re

import taxa

# Whole-word patterns (EN + ES, accent-insensitive). Deliberately NOT bare
# stems: "mat" would match "match", "car" "career", "cat" "category".
CONFLICT_WORDS = [
    r"attack\w*", r"maul\w*", r"bit", r"bites?", r"bitten", r"biting", r"injur\w*", r"wound\w*",
    r"kill\w*", r"dead", r"deaths?", r"died", r"fatal\w*", r"euthani[sz]\w*", r"rabid", r"rabies",
    r"livestock", r"cattle", r"calf", r"calves", r"sheep", r"lambs?", r"goats?", r"chickens?", r"poultry",
    r"hens?", r"pets?", r"dogs?", r"cats?", r"garbage", r"trash", r"encounter\w*", r"charg(?:ed|ing)",
    r"stalk\w*", r"sightings?", r"spotted", r"(?:back)?yards?", r"porch", r"cabins?", r"vehicles?",
    r"struck", r"collisions?", r"relocat\w*", r"trapp?(?:ed|ing)", r"captur\w*", r"shot", r"hospitali?\w*",
    r"stitches", r"conflicts?", r"nuisance",
    r"ataq\w*", r"atac\w*", r"mord\w*", r"herid\w*", r"mato", r"mataron", r"matan", r"muert\w*",
    r"rabia", r"rabioso", r"ganado", r"becerr\w*", r"borreg\w*", r"cabras?", r"gallinas?", r"mascotas?",
    r"perros?", r"gatos?", r"basura", r"avist\w*", r"atropell\w*", r"reubic\w*", r"sacrific\w*",
]
_CONFLICT_RX = re.compile(r"\b(?:" + "|".join(CONFLICT_WORDS) + r")\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _conflict_positions(words: list[str]) -> list[int]:
    return [i for i, w in enumerate(words) if _CONFLICT_RX.fullmatch(w)]


def animal_conflict_context(text: str | None, window: int = 25) -> bool:
    """True if a registry animal name occurs within `window` words of a conflict word."""
    if not text:
        return True                                  # nothing to check (headline-only): don't gate
    norm = taxa._norm(text)
    words = norm.split()
    conflict = _conflict_positions(words)
    if not conflict:
        return False
    rx, _ = taxa._scan_regex()
    starts, pos = [], 0                              # char offset of each word, to map matches → word index
    for w in words:
        pos = norm.find(w, pos)
        starts.append(pos)
        pos += len(w)
    for m in rx.finditer(norm):
        wi = bisect.bisect_right(starts, m.start()) - 1
        if any(abs(wi - c) <= window for c in conflict):
            return True
    return False


def condense(text: str, max_chars: int = 2400, lead_chars: int = 1000) -> str:
    """Lead + later animal/conflict sentences, in original order, up to max_chars."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    out, used, i = [], 0, 0
    while i < len(sents) and used + len(sents[i]) <= lead_chars:  # the lead
        out.append(sents[i])
        used += len(sents[i]) + 1
        i += 1
    if not out and sents:                                            # very long first sentence
        out.append(sents[0][:lead_chars])
        used, i = len(out[0]), 1
    rx, _ = taxa._scan_regex()
    gap = False
    for s in sents[i:]:
        n = taxa._norm(s)
        relevant = rx.search(n) or _CONFLICT_RX.search(n)
        if relevant and used + len(s) + 1 <= max_chars:
            if gap:
                out.append("[…]")
            out.append(s)
            used += len(s) + 1
            gap = False
        else:
            gap = True
    if gap:
        out.append("[…]")
    return " ".join(out)
