"""
Assign NAF 2025 codes for alternative therapy.

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
    name="alternative_therapy_assignment_2025",
    tags=["naf_2025"],
    description="Règle médecine alternative version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def alternative_therapy_rule_8696Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    methods = filter_methods(methods, exclude=["fuzzy", "similarity"])
    terms = [
        "hypnotherapeute",
        "kinesiologie",
        "reflexologie",
        "naprapathie",
        "psycho energeticien",
        "magnetiseur energeticien",
        "pratiques holistiques",
        "energeticien",
        "energeticienne",
        "therapie Alexander",
        "aromatherapie",
        "therapie de Bach",
        "therapie corporelle",
        "ayurveda",
        "phytotherapie",
        "naturopathie",
        "therapie nutritionnelle",
        "homeopathie",
        "chiropractie",
        "osteopathie",
        "osteopathe",
        "cristallotheraphie",
        "cristalotherapie",
        "lithotherapie",
        "iridologie",
        "radionique",
        "guerisseur",
        "guerisseurs",
        "reiki",
        "guerison par les cristaux",
        "choromotherapie",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "8696Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
