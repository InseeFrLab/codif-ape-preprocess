"""
Assign NAF 2025 codes for IT programming.

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
    name="it_programming_assignment_2025",
    tags=["naf_2025"],
    description="Règle programmation informatique version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def it_programming_rule_6210Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "conception de site web developpement vente de solutions informatiques",
        "la creation de sites internet",
        "programmeur",
        "web master",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "6210Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
