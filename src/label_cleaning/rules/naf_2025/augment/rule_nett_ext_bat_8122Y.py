"""
    Generate n synthetic rows with cleaning-related labels for oversampling 8551Y.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_building_exterior_cleaning",
      tags=["naf_2025"],
      description="Oversample synthetic rows for exterior cleaning => 8122Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_building_exterior_cleaning_8122Y(df: pd.DataFrame,
                                             methods=None,
                                             methods_params=None,
                                             n=10000):
    base_labels = [
        "nettoyage des exterieurs d une maison a la vapeur basse pression",
        "nettoyage des toitures et des facades",
        "nettoyage des toitures",
        "nettoyage des facades",
        "nettoyage exterieur des batiments",
        "nettoyage exterieur de batiments",
        "nettoyage exterieur de batiment",
        "nettoyage exterieur batiments",
        "nettoyage de fenetres",
        "entretien de fenetres",
        "nettoyage de vitres",
        "entretien de vitres",
        "ramoneur",
        "activites de nettoyage specialise pour les entreprises",
        "nettoyage des facades et des fenetres",
        "activites de nettoyage specialise pour les batiments",
        "nettoyage des conduits de cheminees et atres",
        "nettoyage des poeles",
        "nettoyage des fours",
        "nettoyage des incinerateurs",
        "nettoyage des chaudieres",
        "nettoyage des conduits de ventilation",
        "nettoyage des unites d extraction",
        "nettoyage industriel specialise",
        "nettoyage de machines industrielles",
        "nettoyage des canalisations d approvisionnement en eau",
        "nettoyage des conduits aerauliques",
        "nettoyage des nouveaux batiments immediatement apres leur construction",
        "sablage pour l exterieur des batiments",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug8122Y{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "8122Y",
        })

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
