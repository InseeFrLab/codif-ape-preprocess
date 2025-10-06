"""
    Generate n synthetic rows with cleaning-related labels
    for oversampling seasonal LMNP .

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_LMNP_saisonniere",
      tags=["naf_2025"],
      description="Oversample synthetic rows for LMNP - saisonniere => 5590Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_seasonal_LMNP_5590Y(df: pd.DataFrame, methods=None, methods_params=None, n=300000):
    base_labels = [
        "location de logement saisonniere",
        "acquisition et mise en location d'un bien immobilier saisonnier",
        "lmnp saisonniere",
        "lmnp saisonniere au regime reel simplifie d imposition",
        "loueur en meuble non professionnel saisonniere",
        "loeur meuble non professionnel saisonniere",
        "loueur bailleur non professionnel saisonniere",
        "location meublee non professionnelle saisonniere",
        "loueur meuble non professionnel saisonniere",
        "loueurs saisonniers en meubles non professionnels",
        "loueur saisonnier en meubl non professionnel saisonniere",
        "loueur en meubles non professionnel saisonniere",
        "location saisonniere d un logement meuble",
        "location saisonniere de logements meubles",
        "location saisonniere de logements meubles non professionelle de longue duree",
        "location saisonniere de logements meubles non professionel",
        "location saisonniere de logements meubles non professionelle",
        "location saisonniere d un meuble",
        "location d un logement meuble saisonniere",
        "location de logements meubles saisonniere",
        "location de logements meubles non professionelle de longue duree saisonniere",
        "location de logements meubles non professionel saisonniere",
        "location de logements meubles non professionelle saisonniere",
        "location d un meuble saisonniere",
        "location en meuble saisonniere",
        "location immobiliere en meuble saisonniere",
        "location meublee saisonniere",
        "location meublee 5590Z saisonniere",
        "location saisonniere meublee en residence de services avec bail commercial",
        "location meublee non professionel saisonniere",
        "location meublee non professionnelle saisonniere",
        "location meubles saisonniere",
        "locations meublees saisonniere",
        "loueur de meuble saisonniere",
        "loueur de meuble dans le cadre de l'economie collaborative saisonniere",
        "loueur en meuble saisonnier",
        "loueur en meuble non profesionnel saisonnier",
        "loueur en meuble non professionnel saisonniere",
        "loueur en meuble non professionnel - code APE 5590Z",
        "loueur en meublee non professionnel saisonniere",
        "loueur en meublee non professionnel en residence de services saisonniere",
        "loueur meuble non professionnel saisonniere",
        "loueur en meublee non professionnelle saisonniere",
        "loueur meuble non professionnel saisonniere",
        "loueur meublee non professionnel saisonniere",
        "5590Z autres hebergements",
        "foyers de travailleurs",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"JaugLogSais{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "5590Y",
        })

    new_df = pd.DataFrame(new_rows)
    print(new_df)
    df_out = pd.concat([df, new_df], ignore_index=True)
    print(df_out)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
