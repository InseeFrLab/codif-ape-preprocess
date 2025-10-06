"""
    Generate n synthetic rows with cleaning-related labels
    for oversampling LMNP with PERM S and PERM P.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_LMNP_perm_P_S",
      tags=["naf_2025"],
      description="Oversample synthetic rows for LMNP - perm=S => 5590Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_LMNP_S_5590Y_P_6820G(df: pd.DataFrame, methods=None, methods_params=None, n=100000):
    base_labels = [
        "location de logement",
        "acquisition et mise en location d'un bien immobilier",
        "lmnp",
        "lmnp au regime reel simplifie d imposition",
        "loueur en meuble non professionnel",
        "loeur meuble non professionnel",
        "Loeur meublé non professionnel",
        "loueur bailleur non professionnel",
        "location meublee non professionnelle",
        "loueur meuble non professionnel",
        "loueurs en meubles non professionnels",
        "loueur en meubl non professionnel",
        "loueur en meubles non professionnel",
        "location d un logement meuble",
        "location de logements meubles",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionelle de longue duree",
        "location de logements meubles non professionel",
        "location de logements meubles non professionelle",
        "location d un meuble",
        "location d'un meuble",
        "location en meuble",
        "location immobiliere en meuble",
        "location meublee",
        "location meublee 6820A",
        "location meublee en residence de services avec bail commercial",
        "location meublee non professionel",
        "location meublee non professionnelle",
        "location meubles",
        "locations meublees",
        "loueur de meuble",
        "loueur de meuble dans le cadre de l'economie collaborative",
        "loueur en meuble",
        "loueur en meuble non profesionnel",
        "loueur en meuble non professionnel",
        "loueur en meuble non professionnel - code APE 6820A",
        "loueur en meublee non professionnel",
        "loueur en meublee non professionnel en residence de services",
        "loueur meuble non professionnel",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"JaugLMNP{i}S",
            "libelle": label,
            "activ_perm_et": "S",
            NACE_REV2_1_COLUMN: "5590Y",
        })
        new_rows.append({
            "liasse_numero": f"JaugLMNP{i}P",
            "libelle": label,
            "activ_perm_et": "P",
            NACE_REV2_1_COLUMN: "6820G",
        })
        new_rows.append({
            "liasse_numero": f"JaugLMNP{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "6820G",
        })

    new_df = pd.DataFrame(new_rows)
    print(new_df)
    df_out = pd.concat([df, new_df], ignore_index=True)
    print(df_out)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
