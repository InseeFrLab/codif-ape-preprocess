"""
    Generate n synthetic rows with cleaning-related labels for oversampling 4933H.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_vtc",
      tags=["naf_2025"],
      description="Oversample synthetic rows for car rentals with driver => 4933H")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_vtc_4933H(df: pd.DataFrame, methods=None, methods_params=None, n=100000):
    base_labels = [
       "location de voiture avec chauffeur",
       "location de voitures avec chauffeur",
       "voiture avec chauffeur",
       "vehicule avec chauffeur",
       "avec chauffeur",
       ("Mettre à disposition d'une clientèle une voiture avec chauffeur, "
        "le trajet et les conditions tarifaires étant fixées avant la course.")
    ]

    # synthetic generation
    new_rows = [
        {
            "liasse_numero": f"Jaug4933H_{i}",  # ID unique par label
            "libelle": label,
            NACE_REV2_1_COLUMN: "4933H",
            "WEIGHT": n,
        }
        for i, label in enumerate(base_labels)
    ]

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
