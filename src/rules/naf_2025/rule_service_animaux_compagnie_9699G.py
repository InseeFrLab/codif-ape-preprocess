"""
    Assign NAF 2025 codes for pet services.

    Matching configuration and mask logic are delegated to utils/rules.py
    for reusability. See:
        - build_matcher_kwargs
        - build_match_mask
    """

import numpy as np
import pandas as pd

from core.decorators import rule, track_changes
from utils.rules import build_matcher_kwargs, build_match_mask
from constants.inputs import TEXTUAL_INPUTS_CLEANED
from constants.targets import NACE_REV2_1_COLUMN


@rule(
    name="pet_services_assignment_2025",
    tags=["naf_2025"],
    description="Règle service animaux de compagnie version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def pet_services_rule_9699G_2025(df: pd.DataFrame,
                                 methods=None,
                                 methods_params=None) -> pd.DataFrame:

    terms = [
        "coach pour animaux de compagnie",
        "comportementaliste animalier cynologiste",
        "coiffeuse mixte à domicile"
        "education canine",
        "physiotherapeute pour animaux masseur kinesitherapeute pour animaux",
        "pratique de massage et de digi pression sur le corps des chevaux",
        "promenade d animaux de compagnie",
        "toilettage"
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "9699G", df[NACE_REV2_1_COLUMN])
    return df, match_mask
