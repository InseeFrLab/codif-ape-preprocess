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
      description="Oversample synthetic rows for LMNP - saisonniere => 5520Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_seasonal_LMNP_5520Y(df: pd.DataFrame, methods=None, methods_params=None, n=300000):
    base_labels = [
        "location de logement saisonniere",
        "acquisition et mise en location d'un bien immobilier saisonnier",
        "lmnp saisonniere",
        "lmnp saisonniere au regime reel simplifie d imposition",
        "loueur en meuble non professionnel saisonniere",
        "loueur en meuble non profesionnel saisonniere",
        "loueur en meuble non professionnelle saisonniere",
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
        "location logement meuble saisonniere",
        "location de logement meubles saisonniere",
        "location de logement meubles non professionelle de longue duree saisonniere",
        "location de logement meubles non professionel saisonniere",
        "location de logement meubles non professionelle saisonniere",
        "location d un meuble saisonniere",
        "location d'un meuble saisonnier",
        "location saisonniere d'un meublé",
        "location saisonniere d'un meuble",
        "location en meuble saisonniere",
        "location immobiliere en meuble saisonniere",
        "location meublee saisonniere",
        "location meublee 5590Z saisonniere",
        "location saisonniere meublee en residence de services avec bail commercial",
        "location meublee en residence de services avec bail commercial saisonniere ",
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
    # 1. Définition des couples (Code CJ, Ratio (pondération) %)
    configurations = [
        ("6540", 46.56), ("5710", 21.76), ("2110", 12.22),
        ("5499", 9.54), ("5202", 4.23),  ("6599", 2.99),
        ("6534", 0.65), ("3220", 0.44), ("6541", 0.27),
        ("2210", 0.23), ("3120", 0.14), ("9220", 0.14),
        ("5599", 0.11), ("2120", 0.07), ("6538", 0.06),
        ("2310", 0.06), ("6597", 0.04), ("2320", 0.06),
        ("5306", 0.03), ("6521", 0.03), ("9110", 0.03),
        ("5699", 0.03), ("6539", 0.03), ("6598", 0.03),
        ("6316", 0.02), ("6589", 0.02), ("5410", 0.02),
        ("2220", 0.02), ("6596", 0.02), ("5426", 0.01),
        ("5646", 0.01), ("5485", 0.01), ("6542", 0.01),
        ("5800", 0.01), ("5515", 0.01), ("5560", 0.01),
        ("5546", 0.01), ("2800", 0.01), ("5770", 0.01),
        ("4140", 0.01),
    ]

    # 2. Génération ultra-rapide avec double boucle (Labels x Configurations)
    new_rows = [
        {
            "liasse_numero": f"JaugLogSais_{i}_{cj}",  # ID unique combinant index et CJ
            "libelle": label,
            "cj": cj,
            NACE_REV2_1_COLUMN: "5520Y",
            "WEIGHT": (ratio / 100.0) * n  # Conversion du % en poids réel
        }
        for i, label in enumerate(base_labels)
        for cj, ratio in configurations
    ]

    new_df = pd.DataFrame(new_rows)
    print(new_df)
    df_out = pd.concat([df, new_df], ignore_index=True)
    print(df_out)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
