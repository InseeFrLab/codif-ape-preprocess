"""
Assign NAF 2025 codes for renewable energy production.

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
    name="renewable_energy_production_assignment_2025",
    tags=["naf_2025"],
    description="Règle production d'énergie renouvelable version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def renewable_energy_production_rule_3512Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "photovoltaique",
        "solaire",
        "eolien",
        "renouvelable",
        "hydroelectricite",
        "solaire thermique",
        "geothermique",
        "biocarburant gazeux",
        "energie des marees",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "3512Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
