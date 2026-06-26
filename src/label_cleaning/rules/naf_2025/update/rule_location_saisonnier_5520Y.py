"""
Assign NAF 2025 codes for LMNP **saisonnier** (Loueur Meublé Non Professionnel).

Matching configuration and mask logic are delegated to utils/rules.py
for reusability. See:
    - build_matcher_kwargs
    - build_match_mask
"""

import numpy as np
import pandas as pd

from src.constants.inputs import TEXTUAL_INPUTS_CLEANED, CATEGORICAL_INPUTS
from src.constants.targets import NACE_REV2_1_COLUMN

from src.label_cleaning.core.decorators import rule, track_changes
from src.label_cleaning.utils.rules import build_match_mask, build_matcher_kwargs, filter_methods


@rule(
    name="seasonal_location_assignment_2025",
    tags=["naf_2025"],
    description="Règle location saisonnier version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def seasonal_location_rule_5520Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    methods = filter_methods(methods, exclude=["similarity"])
    terms = ["loueur de meuble saisonnier",
             "loeur meuble non professionnel saisonnier",
             "loueur de meuble saisonniere",
             "loueur en meuble saisonniere",
             "meublee saisonniere",
             "meuble saisonnier",
             "meubles saisonniers",
             "meublees saisonnieres",
             "profesionnel saisonniere",
             "professionnel saisonnier",
             "professionnelle saisonnier",
             "professionnelle saisoniere",
             "location de meuble saisonnier",
             "lmnp saisonnier",
             "lmnp saisonniere",
             "logement saisonnier",
             "location meublee non professionnelle saisonniere",
             "lmnp location saisonniere",
             "location saisonniere",
             "location meublee saisonniere",
             "location meuble saisonniere",
             "locations saisonnieres",
             "lmnp de courte duree",
             "location de courte duree"
             "5520Y",
             ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    # On prend les inputs textuels ET catégorielles, mais on retire les variables de contrôle
    cols_to_remove = CATEGORICAL_INPUTS
    cols_to_exclude = ["cj", "liasse_type", "activ_perm_et"]  # Variables de décision

    cols_to_remove = [c for c in cols_to_remove if c not in cols_to_exclude]

    condition = match_mask & ((~(df["cj"] == 1000)) &
                              (df["activ_perm_et"] == "S") | (df["activ_perm_et"].isnull()))

    df[NACE_REV2_1_COLUMN] = np.where(
        condition,
        "5520Y",
        df[NACE_REV2_1_COLUMN],
    )

    return df, match_mask
