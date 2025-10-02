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


@rule(name="augment_vdi",
      tags=["naf_2025"],
      description="Oversample synthetic rows for to home selling => 4792J")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_vdi_4792J(df: pd.DataFrame, methods=None, methods_params=None, n=1000):
    base_labels = [
       "vendeur a domicile vdi",
       "conseiller vdi mandataire au sein de vorwerk france",
       "conseillere mandataire au sein de vorwerk france",
       "vente a distance sur catalogue specialise",
       "vdi",
       "vdi vente a domicile",
       "vente a domicile vdi",
       "vdi vente a domicile societe",
       "vdi vente a domicile independant",
       "vdi sans stock",
       "vdi vente a domicile independant sous le statut acheteur revendeur",
       "vdi mandataire pour monat france",
       "conseillere vdi mandataire au sein de happymix",
       "vente a domicile sous statut vdi",
       "vente a domicile",
       "vendeur a domicile",
       "vendeur a domicile independant",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug4792J{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "4792J",
        })

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
