"""
    Generate n synthetic rows with cleaning-related labels for oversampling 4778H.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_sport_events",
      tags=["naf_2025"],
      description="Oversample synthetic rows for distant selling => 4778H")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_distant_selling_4778H(df: pd.DataFrame, methods=None, methods_params=None, n=10000):
    base_labels = [
        "vente à distance",
        "vente a distance via internet",
        "activite de vente a distance",
        "vente a distance des produits diversifies",
        "vente a distance sur catalogue",
        "vente a distance sur catalogue general",
        "vente a distance catalogue general",
        "vente a distance sur catalogue specialise",
        "vente a distance specialise",
    ]

    # synthetic generation
    new_rows = [
        {
            "liasse_numero": f"Jaug4778H_{i}",  # ID unique par label
            "libelle": label,
            NACE_REV2_1_COLUMN: "4778H",
            "WEIGHT": n,
        }
        for i, label in enumerate(base_labels)
    ]

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
