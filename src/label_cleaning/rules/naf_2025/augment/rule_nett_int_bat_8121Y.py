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


@rule(name="augment_building_interior_cleaning",
      tags=["naf_2025"],
      description="Oversample synthetic rows for interior cleaning => 8121Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_building_interior_cleaning_8121Y(df: pd.DataFrame,
                                             methods=None,
                                             methods_params=None,
                                             n=100):
    base_labels = [
        "nettoyage intérieur des batiment",
        "nettoyage intérieur de batiments",
        "nettoyage interieur des batiment",
        "nettoyage interieur de batiments",
        "nettoyage courant de batiment",
        "nettoyage courant de batiments",
        "nettoyage courant des batiments",
        "nettoyage courant des batiment",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug8121Y{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "8121Y",
        })

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
