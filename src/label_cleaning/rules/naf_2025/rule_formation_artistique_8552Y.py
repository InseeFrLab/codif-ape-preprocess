"""
Assign NAF 2025 codes for art teaching.

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
    name="art_teaching_assignment_2025",
    tags=["naf_2025"],
    description="Règle ensignement culturel version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def art_teaching_rule_8552Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "accompagner developper et valoriser la creation artistique multi techniques\
            autour des arts visuels par la mise en place d ateliers",
        "animation d atelier artistique et pedagogique pour tout publiques",
        "enseignement des arts du spectacle vivant",
        "formation au chant choral pratique en atelier et en spectacle",
        "professeur de musique",
        "professeur independant de danse",
        "professeur independant de musique",
        "professeur independant de piano",
        "repetition de theatre hebdomadaire",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "8552Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
