"""
    Generate n synthetic rows with cleaning-related labels for oversampling 9621H.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_home_hairdressing",
      tags=["naf_2025"],
      description="Oversample synthetic rows for car rentals with driver => 9621H")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_home_hairdressing_9621H(df: pd.DataFrame, methods=None, methods_params=None, n=10000):
    base_labels = [
       "coiffure a domicile",
       "coiffure domicile",
       "coiffure mixte a domicile",
       "coiffure hors salons",
    ]

    # synthetic generation
    new_rows = [
        {
            "liasse_numero": f"Jaug9621H_{i}",  # ID unique par label
            "libelle": label,
            NACE_REV2_1_COLUMN: "9621H",
            "WEIGHT": n,
        }
        for i, label in enumerate(base_labels)
    ]

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
