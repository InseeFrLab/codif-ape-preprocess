"""
    Assign NAF 2025 codes for independant facilitators.

    Matching configuration and mask logic are delegated to utils/rules.py
    for reusability. See:
        - build_matcher_kwargs
        - build_match_mask
    """

import numpy as np
import pandas as pd

from core.decorators import rule, track_changes
from utils.rules import build_matcher_kwargs, build_match_mask
from constants.inputs import TEXTUAL_INPUTS_CLEANED
from constants.targets import NACE_REV2_1_COLUMN


@rule(
    name="independant_facilitators_assignment_2025",
    tags=["naf_2025"],
    description="Règle animateurs indépendants version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def independant_facilitator_rule_9392Y_2025(df: pd.DataFrame,
                                            methods=None,
                                            methods_params=None) -> pd.DataFrame:

    terms = [
        "animateur culturel independant",
        "animateur independant intervenant dans des structures maisons de retraite etc",
        "animateur sportif independant",
        "animateur independant"

    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "9392Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
