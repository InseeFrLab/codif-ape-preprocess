"""
Assign NAF rev 2 codes for LMNP (Loueur Meublé Non Professionnel).

Matching configuration and mask logic are delegated to utils/rules.py
for reusability. See:
    - build_matcher_kwargs
    - build_match_mask
"""

import numpy as np
import pandas as pd
from core.decorators import rule, track_changes

from constants.inputs import TEXTUAL_INPUTS_CLEANED
from constants.targets import NACE_REV2_COLUMN
from utils.rules import build_match_mask, build_matcher_kwargs


@rule(
    name="lmnp_assignment_rev2",
    tags=["naf_rev2"],
    description="Attribue APE 6820B si LMNP détecté dans le libellé selon NAF Rev. 2",
)
@track_changes(column=NACE_REV2_COLUMN)
def lmnp_rule_rev2(df: pd.DataFrame, methods=None, methods_params=None) -> pd.DataFrame:
    terms = [
        "location de logementacquisition et mise en location d'un bien immobilier",
        "lmnp",
        "lmnp au regime reel simplifie d imposition",
        "loueur en meuble non professionnel",
        "loueur bailleur non professionnel",
        "location meublee non professionnelle",
        "loueur meuble non professionnel",
        "loueurs en meubles non professionnels",
        "loueur en meubl non professionnel",
        "loueur en meubles non professionnel",
        "location d un logement meuble",
        "location de logements meubles",
        "location de logements meubles non professionelle de longue duree",
        "location d un meuble",
        "location en meuble",
        "location immobiliere en meuble",
        "location meublee",
        "location meublee 6820A"
        "location meublee en residence de services avec bail commercial",
        "location meublee non professionel",
        "location meublee non professionnelle",
        "location meubles",
        "locations meublees",
        "loueur de meuble",
        "loueur de meuble dans le cadre de l'economie collaborative",
        "loueur en meuble",
        "loueur en meuble non profesionnel",
        "loueur en meuble non professionnel",
        "loueur en meuble non professionnel - code APE 6820A",
        "loueur en meublee non professionnel",
        "loueur en meublee non professionnel en residence de services",
        "loueur meuble non professionnel",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_COLUMN] = np.where(
        match_mask & df["liasse_type"].isin(["E", "L", "S", "X", "I"])
        | df["liasse_type"].isnull(),
        "6820A",
        df[NACE_REV2_COLUMN],
    )

    return df, match_mask
