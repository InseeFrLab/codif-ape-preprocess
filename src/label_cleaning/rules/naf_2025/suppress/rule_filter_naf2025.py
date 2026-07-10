import pandas as pd

from src.constants import URL_DF_NAF2025, NACE_REV2_1_COLUMN
from src.label_cleaning.core.decorators import rule, track_deletions
from src.utils.io import download_data


@rule(
    name="filter_naf2025_reference",
    tags=["naf_2025"],
    description="Supprime les lignes dont le code NAF n'est pas présent dans la table de référence NAF 2025."
)
@track_deletions(extra_cols=["liasse_numero", "libelle", "cj", "liasse_type", NACE_REV2_1_COLUMN])
def rule_filter_naf2025(df: pd.DataFrame, methods=None, methods_params=None):
    """
    Supprime les lignes du DataFrame si leur code NAF n'est pas présent dans le fichier
    de référence chargé depuis URL_DF_NAF2025.
    """
    # 1. Charger la table de référence
    print("📥 Loading NAF 2025 reference table...")
    ref_df = download_data(URL_DF_NAF2025)

    # Utilisation de la constante pour la colonne cible
    target_col = NACE_REV2_1_COLUMN
    ref_col = "APE_NIV5"

    # On définit la colonne de code dans le DF de référence
    valid_codes = set(ref_df[ref_col].unique())

    # 2. Identifier les lignes à garder (celles qui sont dans la référence)
    mask_to_keep = df[target_col].isin(valid_codes)

    # 3. Retourner le DataFrame filtré.
    # Le décorateur @track_deletions comparera les index pour générer le journal.
    return df[mask_to_keep].copy()
