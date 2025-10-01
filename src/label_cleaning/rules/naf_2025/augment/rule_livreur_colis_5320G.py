"""
    Generate n synthetic rows with cleaning-related labels for oversampling package delivery.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_package_delivery",
      tags=["naf_2025"],
      description="Oversample synthetic rows for package delivery => 5320G")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_package_delivery_6820G(df: pd.DataFrame, methods=None, methods_params=None, n=1000):
    base_labels = [
        "livreur de colis",
        "livraisons à domicile de colis",
        "Chauffeur livreur, livraison de colis",
        "Livraison à vélo de colis",
        "Livraison des colis a domicile",
        "Activité de livraison de marchandises légères (colis) à vélo, scooter ou voiture.",
        ("Activité de livraison indépendante de colis et marchandises légères, "
         "effectuée pour le compte de particulier"),
        "Livraison de colis en véhicule motorisé",
        "Livraisons de colis",
        ("Service de livraison de coliset marchandises diverses à domicile ou en entreprise, "
         "sur demande ou planifié"),
        "Chauffeur livreur, livraison de colis à vélo. ",
        "Livraison à vélo de petits colis à domicile",
        "Activité de livraison de colis et de marchandises",
        "Coursier à vélo de colis",
        "Livraison à vélo de petits colis à domicile",
        ("Prestation de livraison à domicile de colis, documents "
         "ou marchandises pour des particuliers et des entreprises"),
        ("toutes prestation de services de messagerie, d'expédition, "
         "de transport et de livraison rapide de plis, colis et documents confidentiels")

    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug5320G{i}S",
            "libelle": label,
            NACE_REV2_1_COLUMN: "5320G",
        })

    new_df = pd.DataFrame(new_rows)
    print(new_df)
    df_out = pd.concat([df, new_df], ignore_index=True)
    print(df_out)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
