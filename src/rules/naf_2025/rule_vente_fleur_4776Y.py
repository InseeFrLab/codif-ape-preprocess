"""
    Assign NAF 2025 codes for retailing of flowers.

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
    name="retailing_flowers_assignment_2025",
    tags=["naf_2025"],
    description="Règle commerce de détails fleurs version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def retailing_flowers_rule_4776Y_2025(df: pd.DataFrame,
                                      methods=None,
                                      methods_params=None) -> pd.DataFrame:

    terms = [
        "achat revente de fleurs sur les marches"
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "4776Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
