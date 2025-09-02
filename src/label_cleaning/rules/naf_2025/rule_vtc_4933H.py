"""
Assign NAF 2025 codes for tourists driver.

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
    name="tourists_driver_assignment_2025",
    tags=["naf_2025"],
    description="Règle VTC version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def tourists_driver_rule_4933H_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "exploitant de voiture de transport avec chauffeur vtc",
        "exploitation de vehicule de tourisme avec chauffeur",
        "conducteur de voiture de tourisme avec chauffeur vtc",
        "conducteur de voiture de transport avec chauffeur",
        "location de voiture avec chauffeur",
        "voiture de tourisme avec chauffeur",
        "voiture de transport avec chauffeur",
        "vehicule de tourisme avec chauffeur",
        "vtc",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "4933H", df[NACE_REV2_1_COLUMN])
    return df, match_mask
