"""
Assign NAF 2025 codes for services to individuals.

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
    name="services2individuals_assignment_2025",
    tags=["naf_2025"],
    description="Règle services à la personne version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def services2individuals_rule_9699H_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    methods = filter_methods(methods, exclude=[])
    terms = [
        "wedding planner",
        "activite de voyance tarologie conseils spirituels aupres de particuliers",
        "astrologues",
        "spirites",
        "services d escorte",
        "services de rencontres",
        "services des agences matrimoniales",
        "la prestation ou l organisation de services sexuels",
        "organisation d evenements de prostitution",
        "prostitution",
        "exploitation d etablissements de prostitution",
        "activites genealogiques",
        "studios de tatouage",
        "percage corporel",
        "cireurs de chaussures",
        "porteurs",
        "preposes au parcage des vehicules",
        "l exploitation de machines de services personnels fonctionnant avec des pieces de monnaie",
        "photomaton",
        "pese personnes",
        "consignes",
        "l exploitation d'automates photo",
        "activites de rencontres rendez vous galant",
        "activites de réseautage",
        ("tatoueurs utilisant des substances biologiques par exemple henne "
         "pour ornement temporaire"),
        "gardiennage à domicile",
        "organisateurs de ceremonies de mariages",
        "coach et consultante en développement personnel",
        "coach personnel",
        "VOYANCE",
        "COACHING EN DEVELOPPEMENT PERSONNEL ET SPIRITUEL",
        "astrologie tarot",
        "astrologie",
        "tarot",
        "cartomancie",
        "tatouage par effraction cutanee tatouage corporel tatouage artistique",
        "voyance telephonique",
        "tarologie, coaching",
        "esoterique",
        "esoterisme",
        "medium",
        "tarologue et practicienne holistique",
        "tatouage corporel permanent et vente d œuvres artistiques",
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "9699H", df[NACE_REV2_1_COLUMN])
    return df, match_mask
