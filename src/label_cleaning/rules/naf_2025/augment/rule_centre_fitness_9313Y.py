"""
    Generate n synthetic rows with cleaning-related labels for oversampling 9313Y.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_fitness_camp",
      tags=["naf_2025"],
      description="Oversample synthetic rows for sport education => 9313Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_fitness_camp_9313Y(df: pd.DataFrame, methods=None, methods_params=None, n=1000):
    base_labels = [
        "centre de remises en forme",
        "gestion de centres de remise en forme",
        "gestion de centres de remises en forme",
        "professeur de fitness dans des salles de fitness",
        "cours collectif de fitness",
        "les activités de centre de fitness",
        "les activités de centre de yoga",
        "les activités de centre de pilates",
        "les activités de centre de tai chi",
        "studios de yoga",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug9313Y{i}S",
            "libelle": label,
            "activ_perm_et": "S",
            NACE_REV2_1_COLUMN: "9313Y",
        })

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
