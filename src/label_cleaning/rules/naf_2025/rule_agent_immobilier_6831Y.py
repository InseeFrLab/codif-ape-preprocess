"""
Assign NAF 2025 codes for real state agents.

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
    name="real_state_agent_assignment_2025",
    tags=["naf_2025"],
    description="Règle agents immobiliers version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def real_estate_agent_rule_6831Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "agent commercial en immobilier inscrit au rsac",
        "agent immobilier",
        "mandataire en immobilier inscrit au rsac",
        "mandataire en immobilier",
        "mandataire immobilier",
        "negociateur immobilier independant",
        ("conseil et evaluation en rapport avec l achat la vente "
         "la location de biens immobiliers pour le compte de tiers"),
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "6831Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
