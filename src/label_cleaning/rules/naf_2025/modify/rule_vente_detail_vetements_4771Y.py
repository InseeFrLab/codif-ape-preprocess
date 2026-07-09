"""
Assign NAF 2025 codes for retailing of clothes.

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
from src.label_cleaning.utils.rules import build_match_mask, build_matcher_kwargs, filter_methods


@rule(
    name="retailing_clothes_assignment_2025",
    tags=["naf_2025"],
    description="Règle vente en ligne sans prédominence alimentaire version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def retailing_clothers_4771Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    methods = filter_methods(methods, exclude=["similarity"])
    terms = [
        "achat revente sur internet habillement accessoires et chaussures",
        "vente d habillement",
        "vente de pret a porter",
        "vente de pret a porter a domicile",
        "vente de pret a porter et accessoires",
        "vente sur internet de pret a porter",
        "vente de pret a porter sur internet",
        "vente de vetements",
        "vente a distance de vetements",
        "vente a domicile de vetements",
        "vente de sous vetements",
        "vente de vetements en ligne",
        "vente de vetements sur internet",
        "vente sur internet de vetements",
        "vente de vetements et d objets personnalise",
        "vente pret a porter et produits de beaute",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "4771Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
