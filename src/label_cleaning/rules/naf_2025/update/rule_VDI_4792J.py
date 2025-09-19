"""
Assign NAF 2025 codes for distance selling.

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
from src.label_cleaning.utils.rules import (
    build_match_mask,
    build_matcher_kwargs,
    filter_methods,
)


@rule(
    name="distance_selling_2025",
    tags=["naf_2025"],
    description="Règle vente à distance version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def distance_selling_4792J_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    methods = filter_methods(methods, exclude=["similarity"])
    terms = [
        "vendeur a domicile vdi",
        "conseiller vdi mandataire au sein de vorwerk france",
        "conseillere mandataire au sein de vorwerk france",
        "vente a distance sur catalogue specialise",
        "vdi",
        "vdi vente a domicile",
        "vente a domicile vdi",
        "vdi vente a domicile societe",
        "vdi vente a domicile independant",
        "vdi sans stock",
        "vdi vente a domicile independant sous le statut acheteur revendeur",
        "vdi mandataire pour monat france",
        "conseillere vdi mandataire au sein de happymix",
        "vente a domicile sous statut vdi",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "4792J", df[NACE_REV2_1_COLUMN])
    return df, match_mask
