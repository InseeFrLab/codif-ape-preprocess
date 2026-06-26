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
        "loueur meuble non professionnelle",
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
    template_C05_C = {"liasse_type": "C", "cj": "1000", NACE_REV2_1_COLUMN: "5520Y", "WEIGHT": n}
    template_C05_I = {"liasse_type": "I", "cj": "1000", NACE_REV2_1_COLUMN: "6820G", "WEIGHT": n}

    new_rows = []
    for i, label in enumerate(base_labels):
        new_rows.append({"liasse_numero": f"JaugLMNP_C05C_{i}", "libelle": label, **template_C05_C})
        new_rows.append({"liasse_numero": f"JaugLMNP_C05I{i}", "libelle": label, **template_C05_I})

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
