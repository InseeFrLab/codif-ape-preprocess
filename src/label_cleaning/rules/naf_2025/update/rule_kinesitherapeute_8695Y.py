"""
Assign NAF 2025 codes for physiotherapists.

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
    name="physiotherapist_assignment_2025",
    tags=["naf_2025"],
    description="Règle masseur-kinésithérapeute version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def physiotherapist_rule_8695Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "activite de masseur kinesitherapeute",
        "activite de masseur kinesotherapeute liberal",
        "masseur kine",
        "masseuse kine",
        "masseur kine en collaboration",
        "masseur kinesi",
        "masseuse kinesi",
        "masseuse kinesitherapeute",
        "masseur kinesitherapeute assistant",
        "masseur kinesitherapeute assistante",
        "masseur kinesitherapeute liberal",
        "masseur kinesitherapeute liberale",
        "masseur kinesitherapeute osteopathe",
        "masseur kinesitherapeute remplacant",
        "masseur kinesitherapeute remplacante",
        "masseur kinesitherapeute",
        "masseur kinesitherapeute liberal",
        "masso kinesitherapie",
        "masseur physiotherapeute",
        "physiotherapie",
        "physiotherapeute",
        "massotherapeutes medicaux",
        "ergotherapeutes",
        "praxitherapeutes",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "8695Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
