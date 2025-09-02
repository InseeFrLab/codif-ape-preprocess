"""
Assign NAF 2025 codes for meal delivery.

Matching configuration and mask logic are delegated to utils/rules.py
for reusability. See:
    - build_matcher_kwargs
    - build_match_mask
"""

import numpy as np
import pandas as pd
from core.decorators import rule, track_changes

from constants.inputs import TEXTUAL_INPUTS_CLEANED
from constants.targets import NACE_REV2_1_COLUMN
from utils.rules import build_match_mask, build_matcher_kwargs


@rule(
    name="meal_delivery_assignment_2025",
    tags=["naf_2025"],
    description="Règle LMNP version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def meal_delivery_rule_5320H_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "livraison de repas",
        "livraison de repas a domicile",
        "livraison de repas a domicile a velo",
        "livreur de repas",
        "livreur uber eat",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "5320H", df[NACE_REV2_1_COLUMN])
    return df, match_mask
