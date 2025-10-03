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


@rule(name="augment_other_medicine",
      tags=["naf_2025"],
      description="Oversample synthetic rows for other medecine => 8699Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_other_medecine_8699Y(df: pd.DataFrame, methods=None, methods_params=None, n=10000):
    base_labels = [
        "psychomotricien",
        "psychomotricienne",
        ("psychomotricienne diplomee d etat realisant des bilans "
         "psychomoteurs puis proposant une prise en charge en lien"),
        "psychomotricite",
        "orthophonie",
        "orthophoniste",
        "orthophoniste liberale",
        "orthophoniste liberal",
        "orthophoniste remplacant",
        "orthophoniste remplacante",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug8699Y{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "8699Y",
        })

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
