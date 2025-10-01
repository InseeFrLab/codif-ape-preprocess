"""
    Generate n synthetic rows with cleaning-related labels for oversampling meal delivery.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_meal delivery",
      tags=["naf_2025"],
      description="Oversample synthetic rows for meal delivery => 5320H")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_meal_delivery_5320H(df: pd.DataFrame, methods=None, methods_params=None, n=1000):
    base_labels = [
        "livreur uber eat",
        "livreur uber east",
        "livraison a domicile uber",
        "livraison a domicile et courses rapides de repas",
        "livraispn de repas a domicile",
        "livraison de repas a domicile",
        "livraison a velo de repas",
        "livraison de commandes de repas",
        ("Livraison Repas & Produits Alimentaires offre un service de livraison fiable "
         "et rapide de repas et de produits alimentaires à domicile"),
        "livraison repas",
        "Livraison de repas à domicile en vélo",
        ("Activité principale de livraison de repas, réalisée en véhicule léger "
         "(voiture de tourisme) pour le compte de plateformes numériques (Uber Eat)"),
        "livrer des repas préparés avec UBER",
        "Service de livraison local de repas",
        ("COURSIER À VÉLO. La livraison des courses, des repas "
         "sans préparation en collaboration avec des plateformes en ligne"),
        ("Livraison à domicile de repas, courses et petits colis à vélo "
         "ou scooter en tant que coursier indépendant"),
        "livraison à domicile de courses et de repas à vélo",
        ("Activité de coursier indépendant : livraison à domicile de repas, "
         "courses et petits colis pour le compte de plateformes ou de clients particuliers"),
        "livraison de courses et repas à vélo non motorisée.",
        "Livraison à domicile de repas sans préparation",
        "Livrer des repas a domicile a scooter",
        ("L'entreprise propose un service de livraison de repas, "
         "de produits alimentaires et de courses diverses à domicile"),
        "Livraison à vélo de produits divers et variés dont repas à domicile",
        "Livraison à vélo de repas",
        "Activité de coursier indépendant, livraison de repas",
        "Coursier indépendant en livraison de repas pour plateformes numérique",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug5320H{i}S",
            "libelle": label,
            NACE_REV2_1_COLUMN: "5320H",
        })

    new_df = pd.DataFrame(new_rows)
    print(new_df)
    df_out = pd.concat([df, new_df], ignore_index=True)
    print(df_out)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
