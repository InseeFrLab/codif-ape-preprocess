"""
    Assign NAF 2025 codes for chiropodist.

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
    name="chiropodist_assignment_2025",
    tags=["naf_2025"],
    description="Règle podologue version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def chiropodist_rule_8622Y_2025(df: pd.DataFrame,
                                methods=None,
                                methods_params=None) -> pd.DataFrame:

    terms = [
        "peridure podologue",
        "pedicure podologue remplacant",
        "pedicure podologue remplacante",
        "podologue"
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "8622Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
