"""
Assign NAF 2025 codes for retailing of cars.

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
    name="retailing_cars_2025",
    tags=["naf_2025"],
    description="Règle commerce détail de véhicules version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def retailing_cars_4781Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "achat vente de vehicules d occasion",
        "achat vente de vehicules neufs",
        "commerce de voitures et de vehicules automobiles legers",
        "achat vente de vehicules",
        "achat vente de vehicule",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(
        match_mask & (df["activ_surf_et"] == 1), "4781Y", df[NACE_REV2_1_COLUMN]
    )
    return df, match_mask
