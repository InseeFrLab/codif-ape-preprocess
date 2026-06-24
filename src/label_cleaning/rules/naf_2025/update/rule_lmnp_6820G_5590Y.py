"""
Assign NAF 2025 codes for LMNP (Loueur Meublé Non Professionnel).

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
from src.label_cleaning.utils.rules import build_match_mask, build_matcher_kwargs


@rule(
    name="lmnp_assignment_2025",
    tags=["naf_2025"],
    description="Règle LMNP version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def lmnp_rule_6820G_5590Y_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    terms = [
        "location de logement",
        "acquisition et mise en location d'un bien immobilier",
        "lmnp",
        "lmnp au regime reel simplifie d imposition",
        "loueur en meuble non professionnel",
        "loeur meuble non professionnel",
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
        "location meublee 6820A",
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

    # On définit les listes pour garder le code propre
    types_urssaf_impots = ["E", "L", "S", "X", "I"]
    types_commerce_greffe = ["C", "R", "G", "D", "Y"]

    # 1. Définition des colonnes à vide
    # On prend les inputs textuels ET catégorielles, mais on retire les variables de contrôle
    cols_to_remove = CATEGORICAL_INPUTS
    cols_to_exclude = ["cj", "liasse_type", "activ_perm_et"]  # Variables de décision

    cols_to_remove = [c for c in cols_to_remove if c not in cols_to_exclude]

    # 2. Définition des conditions
    conditions = [
        # CAS 1 : CJ=1000 ET Activité non-professionnelle ET (Urssaf/Impôt OU Manquant)
        (match_mask) &
        (df["cj"] == "1000") &
        (df["liasse_type"].isin(types_urssaf_impots) | df["liasse_type"].isna()),

        # CAS 2 : CJ=1000 ET Activité non-professionnelle ET Commerce/Greffe
        (match_mask) &
        (df["cj"] == "1000") &
        (df["liasse_type"].isin(types_commerce_greffe)),

        # CAS 3 : Autre (CJ != 1000 ou Activité pro) ET Activité Permanente
        (match_mask) &
        (~((df["cj"] == "1000") & (df["activ_perm_et"] == "P"))),

        # CAS 4 : Autre (CJ != 1000 ou Activité pro) ET Activité Saisonnière
        (match_mask) &
        (~((df["cj"] == "1000") & (df["activ_perm_et"] == "S")))
    ]

    # 3. Définition des codes cibles correspondantes
    choices = [
        "6820G",
        "5520Y",
        "6820G",
        "5520Y"
    ]

    # 4. Application
    df[NACE_REV2_1_COLUMN] = np.select(
        conditions,
        choices,
        default=df[NACE_REV2_1_COLUMN]
    )

    return df, match_mask
