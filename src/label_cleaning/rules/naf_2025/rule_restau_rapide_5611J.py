"""
Assign NAF 2025 codes for fast food.

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
    name="fast_food_assignment_2025",
    tags=["naf_2025"],
    description="Règle restauration rapide version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def fast_food_rule_5611J_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "restauration rapide sur place et a emporter",
        "preparation de plats cuisines a emporter",
        "fast food",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "5611J", df[NACE_REV2_1_COLUMN])
    return df, match_mask
