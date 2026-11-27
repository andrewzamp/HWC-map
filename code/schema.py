"""Extraction schemas — the single source of truth for what the LLM returns.

Design choices
--------------
* Closed vocabularies (Literal enums) for every field used for filtering or
  analysis; free text only for names, summary and notes. The JSON schema is a
  decoding constraint, so output always parses and stays in-vocabulary.
* The taxon vocabulary is generated from taxa.yaml at import time, so adding a
  species there automatically updates the model's allowed answers.
* "unknown" is always allowed: explicit missingness over forced guesses.
  Evidence/precision fields record HOW each key value was obtained.
* The model never produces coordinates or scientific names: place names are
  geocoded deterministically; scientific names come from the registry.
* Fields are generated in schema order under constrained decoding, so a short
  reasoning field comes first and conditions the relevance decisions; the
  decisions (event? country?) come next so extraction can stop early for
  non-events and out-of-region stories (see extract.early_stop).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

import taxa

EXTRACTION_VERSION = 5   # bump when schema/prompt change → old rows are re-extracted

Taxon = Literal[tuple(taxa.taxon_labels())]

ConflictType = Literal[
    "physical attack",
    "non-contact encounter",
    "livestock or pet depredation",
    "property damage",
    "food attractant",
    "crop or beehive damage",
    "disease exposure (e.g. rabies)",
    "vehicle collision",
    "animal in human settlement",
    "other",
]

CONFLICT_DEFINITIONS = (
    "'physical attack' = the animal made physical contact with a person (bites, scratches, maulings, fatal or not); "
    "'non-contact encounter' = charge, stalking, bluff charge or close encounter without contact; "
    "'livestock or pet depredation' = the animal killed or injured livestock, poultry or pets; "
    "'property damage' = homes, cabins, vehicles, coops or structures broken into or damaged; "
    "'food attractant' = garbage, bird feeders, pet food, coolers, campsite food; "
    "'crop or beehive damage' = crops, orchards, beehives; "
    "'disease exposure (e.g. rabies)' = possible or confirmed disease transmission to people or pets; "
    "'vehicle collision' = the animal was struck by a car, truck or train; "
    "'animal in human settlement' = presence in a town, yard or school with none of the above; "
    "'other' = none of these."
)


class Triage(BaseModel):
    """Cheap headline-only screen, run before full-text download/extraction.

    Deliberately narrow: it only resolves WORD SENSE (is the animal meant at
    all?). Whether an actual conflict event happened is decided by the full
    extraction, which sees the article text. A triage false negative is
    invisible downstream, so the bar for 'irrelevant' is high.
    """
    reasoning: str = Field(description="One sentence: does the animal word in the headline refer to the real "
                           "animal, or to something else (sports team, brand, car, TV station, person, smugglers, "
                           "figurative use)? If no animal is named but one could be involved (e.g. a rabies alert, "
                           "an unexplained attack), say so.")
    verdict: Literal["relevant", "unsure", "irrelevant"] = Field(
        description="'relevant' = a real wild carnivore is (or is very likely) part of the story, whatever happened "
        "to it (attacks, sightings, vehicle collisions, rabies, livestock losses, management, killings). "
        "'unsure' = an animal may be involved. 'irrelevant' ONLY when the word clearly does not mean the animal "
        "(sports teams like Bears/Lions/Pumas/Otters/Jaguars, 'bear market', 'bear spray' used on people, FOX TV, "
        "Jaguar cars, human smugglers called 'coyotes', people or places named Wolf/Fox/Lobos) or the story has "
        "nothing to do with animals.")


class Extraction(BaseModel):
    relevance_reasoning: str = Field(
        description="One or two sentences: is a real, living wild carnivore (the animal) part of the story, "
        "and does it interact with people, their animals, vehicles or property? Sports teams, 'bear spray' "
        "used on people, 'bear market', TV stations named FOX, human smugglers called 'coyotes', cars, brands, "
        "mascots and people's names do NOT count. Domestic dogs and cats are not wild carnivores."
    )
    real_wild_carnivore_involved: bool = Field(
        description="True only if a real, living wild carnivore (the animal) is part of the story."
    )
    is_specific_event: bool = Field(
        description="True only if real_wild_carnivore_involved is true AND the article reports at least one "
        "concrete, real-world interaction between people (or their animals/property/vehicles) and the "
        "animal at an identifiable time and place. False for general advice, policy or management debates, "
        "research, opinion, hunting/trapping-season news, or listicles without a specific incident."
    )
    country: Literal["United States", "Canada", "Mexico", "other country", "unknown"] = Field(
        description="Country where the incident happened. 'other country' if outside the USA, Canada and Mexico "
        "(e.g. Spain, Argentina, the Netherlands) — such stories are out of scope."
    )
    multiple_events: bool = Field(
        description="True if the article reports several distinct incidents (e.g. a seasonal round-up). "
        "Fields below then describe the main/most recent one."
    )
    animal_name: Optional[str] = Field(
        description="The animal exactly as the article names it, in the article's language "
        "(e.g. 'mountain lion', 'rabid gray fox', 'oso negro'). null if no animal is named."
    )
    taxon: Taxon = Field(
        description="Standardised taxon for the main animal. Choose a species only if the article names it "
        "or it is unambiguous (e.g. 'mountain lion'/'puma'/'Florida panther' = cougar; 'grizzly' = brown bear). "
        "If only the group is known (e.g. just 'fox', 'bear', 'skunk'), choose the matching 'unidentified ...' "
        "label. Use 'other wild carnivore' for wild carnivores not listed, 'unknown' otherwise."
    )
    n_animals: Optional[int] = Field(description="Number of wild animals involved (e.g. sow with 2 cubs = 3). null if unknown.")
    event_date: Optional[str] = Field(
        description="Date the incident occurred (YYYY-MM-DD), NOT the publication date. Resolve relative "
        "expressions ('Tuesday', 'last week') using the publication date provided. null if the text does "
        "not say when it happened — never copy the publication date."
    )
    event_date_precision: Literal["day", "week", "month", "unknown"] = Field(
        description="How precisely the article pins down event_date."
    )
    conflict_types: list[ConflictType] = Field(
        description="All categories that apply to the main incident. Definitions: " + CONFLICT_DEFINITIONS
    )
    primary_conflict_type: ConflictType = Field(
        description="The single most severe/central category, using the same definitions."
    )
    human_activity: Literal[
        "hiking or recreation", "walking a pet", "hunting or trapping", "camping",
        "at home or on own property", "working", "driving", "other", "unknown", "not applicable"
    ] = Field(description="What the affected person was doing at the time.")
    human_outcome: Literal["no injury", "injured", "killed", "unknown", "not applicable"] = Field(
        description="Worst outcome for people. 'killed' if anyone died (e.g. 'fatally mauled'); "
        "'not applicable' if no person was directly involved (e.g. animal hit by car, garbage raid)."
    )
    n_people_injured: Optional[int] = Field(description="null if not reported")
    n_people_killed: Optional[int] = Field(description="null if not reported")
    domestic_animal_type: Literal[
        "livestock", "poultry", "pets", "other", "unknown", "not applicable"
    ] = Field(description="Domestic animals harmed, if any ('pets' = dogs, cats; 'livestock' = cattle, sheep, goats...).")
    domestic_animal_outcome: Literal["injured", "killed", "unknown", "not applicable"] = Field(
        description="Worst outcome for domestic animals; 'not applicable' if none were involved."
    )
    rabies_status: Literal["confirmed positive", "suspected", "tested negative", "not mentioned"] = Field(
        description="Rabies status of the wild animal as reported."
    )
    animal_outcome: Literal[
        "no action", "hazed or deterred", "trapped and relocated", "killed by authorities",
        "killed by a person in defence", "killed illegally or in retaliation", "killed by vehicle",
        "captured, fate unknown", "tested for disease", "being searched for", "unknown"
    ] = Field(description="What happened to the wild animal(s) as reported.")
    location_name: Optional[str] = Field(
        description="Most specific GEOGRAPHIC place name of the incident, as written in the article "
        "(town, neighbourhood, park, trail, lake). Not a business, building or road description "
        "(for 'a store in Vancouver' write 'Vancouver'). Do NOT include the state/province. null if none."
    )
    admin1: Optional[str] = Field(
        description="US state, Canadian province/territory or Mexican state, full name (e.g. 'British Columbia', "
        "not 'B.C.'; 'Nuevo León'). Fill it if the place clearly implies it. null if unknown."
    )
    summary: str = Field(description="Neutral 1-2 sentence summary of the incident, in English.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Your confidence that the fields above are correct given the text available."
    )


def json_schema(model: type[BaseModel] = Extraction) -> dict:
    return model.model_json_schema()


def field_guide(model: type[BaseModel] = Extraction) -> str:
    """Definition of every field, injected into the prompt.

    Grammar-constrained decoding (Ollama `format`) restricts which tokens the
    model may emit but never shows it the field descriptions.
    """
    lines = []
    for name, prop in json_schema(model)["properties"].items():
        allowed = prop.get("enum") or prop.get("items", {}).get("enum")
        if not allowed and "anyOf" in prop:
            allowed = next((p.get("enum") for p in prop["anyOf"] if "enum" in p), None)
        lines.append(f"- {name}: {prop.get('description', '')}" + (f" Allowed: {allowed}." if allowed else ""))
    return "\n".join(lines)
