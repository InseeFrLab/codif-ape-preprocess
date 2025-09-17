"""
Assign NAF 2025 codes for pet services.

Matching configuration and mask logic are delegated to utils/rules.py
for reusability. See:
    - build_matcher_kwargs
    - build_match_mask
"""

import numpy as np
import pandas as pd

from src.constants.inputs import TEXTUAL_INPUTS_CLEANED
from src.constants.targets import NACE_REV2_1_COLUMN

from src.label_cleaning.core.decorators import rule, track_changes
from src.label_cleaning.utils.rules import build_match_mask, build_matcher_kwargs


@rule(
    name="pet_services_assignment_2025",
    tags=["naf_2025"],
    description="Règle service animaux de compagnie version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def pet_services_rule_9699G_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "coach pour animaux de compagnie",
        "comportementaliste animalier cynologiste",
        "cynologiste",
        "petsitting",
        "pet sitter",
        "pet sitting",
        "cat sitting",
        "dog sitting",
        "garde d animaux",
        "balade canine",
        "promeneur canin",
        "physiotherapeute pour animaux masseur kinesitherapeute pour animaux",
        "pension des chevaux",
        "pension pour chevaux",
        "d animaux de compagnie",
        "pension pour animaux domestiques",
        "promenade d animaux de compagnie",
        "toilettage",
        "dressage d animaux de compagnie",
        "education canins",
        "education canine",
        "educateur canin",
        "comportementaliste pour animaux de compagnie",
        "comportementaliste pour chien",
        "comportementaliste canin",
        "comportementaliste canine",
        "education et comportementaliste canin",
        "comportementaliste felin",
        "dressage de chiens",
        "dresseur de chiens",
        "dresseur chien",
        "dresseur de chats",
        "palfrenier soigneur",
        "palfreniere",
        "exploitation de refuges pour animaux abandonnes",
        "chuchoteur",
        "masseur animalier",
        "naturopathe animalier",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "9699G", df[NACE_REV2_1_COLUMN])
    return df, match_mask
