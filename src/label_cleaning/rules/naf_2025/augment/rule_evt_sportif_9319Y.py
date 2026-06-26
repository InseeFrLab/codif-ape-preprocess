"""
    Generate n synthetic rows with cleaning-related labels for oversampling 9319Y.

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
      description="Oversample synthetic rows for sport education => 9319Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_sport_events_9319Y(df: pd.DataFrame, methods=None, methods_params=None, n=10000):
    base_labels = [
        "evenement sportif",
    ]

    # synthetic generation
    new_rows = [
        {
            "liasse_numero": f"Jaug9319Y_{i}",  # ID unique par label
            "libelle": label,
            NACE_REV2_1_COLUMN: "9319Y",
            "WEIGHT": n,
        }
        for i, label in enumerate(base_labels)
    ]

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
