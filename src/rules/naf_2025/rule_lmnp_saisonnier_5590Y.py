"""
    Assign NAF 2025 codes for LMNP **saisonnier** (Loueur Meublé Non Professionnel).

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
    name="seasonal_lmnp_assignment_2025",
    tags=["naf_2025"],
    description="Règle LMNP saisonnier version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def lmnp_rule_5590Y_2025(df: pd.DataFrame, methods=None, methods_params=None) -> pd.DataFrame:

    terms = [
        "loueur de meuble saisonnier",
        "lmnp saisonnier"
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask & (df["activ_perm_et"] == "S") |
                                      df["activ_perm_et"].isnull(),
                                      "5590Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
