"""
Assign NAF 2025 codes for graphic designer.

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
    name="graphic_designing_assignment_2025",
    tags=["naf_2025"],
    description="Règle graphistes version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def graphic_designing_rule_7412Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    methods = filter_methods(methods, exclude=["similarity"])
    terms = [
        "conseil en creation graphique",
        "design ergonomie",
        "ux ui design",
        "ux design",
        "ui design",
        "design graphique",
        "designer graphique",
        "design ergonomie",
        "designer graphiste",
        "graphiste conception de supports",
        "infographiste",
        "graphisme",
        "graphiste",
        "conseil graphique",
        "creation graphique"
        "conception graphique",
        "web designer",
        "logo",
        "charte graphique",
        "identite graphique",
        "pictogramme",
        "conception graphique",
        "concepteurs en communication",
        "corporate design",
        "design d images visuelles",
        "conception de la marque",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "7412Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
