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


@rule(name="augment_sport_education",
      tags=["naf_2025"],
      description="Oversample synthetic rows for sport education => 8551Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_sport_education_8551Y(df: pd.DataFrame, methods=None, methods_params=None, n=1000):
    base_labels = [
        "moniteur de ski",
        "moniteur de tennis",
        "moniteur de plongée",
        "educateur sportif",
        "professeur de yoga",
        "professeure de yoga",
        "instructeur pilates",
        "cours de pilates",
        "enseignant de tennis",
        "educateur sportif",
        "cours de judo et de fitness",
        "entrainement sportif",
        "proposition d'activités physiques adaptées",
        "professeur de karaté d'activités physiques et sportives de bien-être",
        "éducateur sportif pluridisciplinaire",
        "préparation physique, coaching",
        "Enseignement de disciplines sportives et d'activités de loisirs",
        "professeur de fitness",
        "coaching sportif",
        "professeur de golf",
        "Entraineur sportif dans le domaine de la natation",
        "entraineur d'escrime",
        "enseignement de l'équitation",
        "monitrice de ski",
        "Je suis professeur de tennis et je donne des cours particuliers de tennis et de Padel",
        "Moniteur de tennis et de padel",
        "apprentissage du tennis",
        "cours de sport personnalisé",
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug8551Y{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "8551Y",
        })

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
