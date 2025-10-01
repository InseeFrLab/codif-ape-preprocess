"""
    Generate n synthetic rows with cleaning-related labels for oversampling 9699H.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_services2individuals",
      tags=["naf_2025"],
      description="Oversample synthetic rows for sport education => 9699H")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_services2individuals_9699H(df: pd.DataFrame, methods=None, methods_params=None, n=1000):
    base_labels = [
        "Wedding planner",
        "Activité de voyance, tarologie, conseils spirituels auprès de particuliers",
        "astrologues",
        "spirites",
        "services d escorte",
        "services de rencontres",
        "services des agences matrimoniales",
        "la prestation ou l organisation de services sexuels",
        "organisation d'événements de prostitution",
        "exploitation d'établissements de prostitution",
        "activités généalogiques",
        "studios de tatouage",
        "perçage corporel",
        "cireurs de chaussures",
        "porteurs",
        "préposés au parcage des véhicules",
        "l exploitation de machines de services personnels fonctionnant avec des pièces de monnaie",
        "photomaton",
        "pèse-personnes",
        "consignes",
        "l exploitation d'automates photo",
        "activités de rencontres/rendez-vous galant",
        "activités de réseautage",
        ("tatoueurs, utilisant des substances biologiques (par exemple henné), "
         "pour ornement temporaire"),
        "gardiennage à domicile",
        "organisateurs de cérémonies de mariages",
        "Coach et consultante en développement personnel",
        "Coach personnel",
        "VOYANCE",
        "COACHING EN DEVELOPPEMENT PERSONNEL ET SPIRITUEL",
        "astrologie, tarot",
        "astrologie",
        "tarot",
        "cartomancie",
        "Tatouage par effraction cutanée / tatouage corporel / tatouage artistique",
        "VOYANCE TELEPHONIQUE",
        "Tarologie, coaching",
        "esoterique",
        "esoterisme",
        "medium",
        "Tarologue et practicienne holistique",
        "Tatouage corporel permanent et vente d'œuvres artistiques",

    ]
    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug9699H{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "9699H",
        })

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
