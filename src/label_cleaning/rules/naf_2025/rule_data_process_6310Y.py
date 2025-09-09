"""
Assign NAF 2025 codes for data processing activities.

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
    name="data_analysis_assignment_2025",
    tags=["naf_2025"],
    description="Règle traitement de données version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def data_analysis_rule_6310Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "activite de service en analyse de données et gestion de programme",
        "infrastructure informatique traitement de donnees hebergement et activites connexes",
        "architecture de donnees",
        "data architect",
        "architecture logicielle",
        "hebergement d applications",
        "cloud computing",
        ("prestation de services en analyse de donnees data science "
         "informatique statistique realisation d etudes developpement "
         "de code activites de recherche"),
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "6310Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
