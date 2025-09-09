"""
Assign NAF 2025 codes for IT programming.

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
    name="it_programming_assignment_2025",
    tags=["naf_2025"],
    description="Règle programmation informatique version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def it_programming_rule_6210Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    # methods = filter_methods(methods, exclude=["similarity"])
    terms = [
        "conception de site web developpement vente de solutions informatiques",
        "la creation de sites internet",
        "programmeur",
        "webmaster",
        "administrateur de site internet",
        "architecte informatique",
        "architecte logiciel",
        "machine learning engineer",
        "pentester",
        "developpeur cybersecurite",
        "ingenieur cybersecurite",
        "chef de projet informatique",
        "developpeur logiciels de systemes",
        "developpeur logiciels et applications pour jeux video",
        "developpeur applications de jeux",
        "developpeur d intergiciels pour jeux video",
        "developpeur d applications logicielles pour les entreprises et la finance",
        "developpeur d applications d apprentissage automatique",
        "developpeur d applications d intelligence vision artificielle",
        "developpeur d applications de cybersécurite",
        "developpeur d applications de registres distribues",
        "administrateur de bases de données et pages web",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "6210Y", df[NACE_REV2_1_COLUMN])
    return df, match_mask
