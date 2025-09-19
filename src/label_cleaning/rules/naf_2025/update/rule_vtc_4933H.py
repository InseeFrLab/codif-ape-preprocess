"""
Assign NAF 2025 codes for tourists driver.

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
    name="tourists_driver_assignment_2025",
    tags=["naf_2025"],
    description="Règle VTC version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def tourists_driver_rule_4933H_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    methods = filter_methods(methods, exclude=["fuzzy", "similarity"])
    terms = [
        "exploitant de voiture de transport avec chauffeur vtc",
        "exploitation de vehicule de tourisme avec chauffeur",
        "conducteur de voiture de tourisme avec chauffeur vtc",
        "conducteur de voiture de transport avec chauffeur",
        "location de voitures avec chauffeur",
        "location de voiture avec chauffeur",
        "vehicule avec chauffeur",
        "vehicules avec chauffeur",
        "voiture avec chauffeur",
        "voitures avec chauffeur",
        "tourisme avec chauffeur",
        "location privee de voiture avec chauffeur",
        "voiture de tourisme avec chauffeur",
        "voiture de transport avec chauffeur",
        "vehicule de tourisme avec chauffeur",
        "autre transport de personnes sur demande par véhicule avec chauffeur",
        "vtc",
        (
            "voiture avec chauffeur le trajet et "
            "les conditions tarifaires etant fixees avant la course"
        ),
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "4933H", df[NACE_REV2_1_COLUMN])
    return df, match_mask
