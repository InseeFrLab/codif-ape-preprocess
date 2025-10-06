"""
    Generate n synthetic rows with cleaning-related labels for oversampling 8696Y.

    Args:
        df (pd.DataFrame): Input dataset.
        n (int): Number of synthetic rows to create.

    Returns:
        pd.DataFrame: Dataset with additional synthetic rows.
"""
import pandas as pd

from src.label_cleaning.core.decorators import rule, track_new
from src.constants.targets import NACE_REV2_1_COLUMN


@rule(name="augment_alternative_medicine",
      tags=["naf_2025"],
      description="Oversample synthetic rows for alternative medecine => 8696Y")
@track_new(column=NACE_REV2_1_COLUMN)
def augment_alt_medecine_8696Y(df: pd.DataFrame, methods=None, methods_params=None, n=60000):
    base_labels = [
        "sophrologie",
        "sophrologue",
        "hypnotherapeute",
        "hypnothérapeute",
        "kinesiologie",
        "kinesiologue",
        "reflexologie",
        "naprapathie",
        "psycho energeticien",
        "psycho energeticienne",
        "magnetiseur energeticien",
        "pratiques holistiques",
        "energeticien",
        "energeticienne",
        "therapie Alexander",
        "aromatherapie",
        "therapie de Bach",
        "therapie corporelle",
        "ayurveda",
        "phytotherapie",
        "naturopathie",
        "therapie nutritionnelle",
        "homeopathie",
        "chiropractie",
        "osteopathie",
        "osteopathe",
        "cristallotheraphie",
        "cristalotherapie",
        "lithotherapie",
        "iridologie",
        "radionique",
        "guerisseur",
        "guerisseurs",
        "reiki",
        "guerison par les cristaux",
        "choromotherapie",
        "medecine traditionnelle",
        "medecine traditionnelle chinoise"
    ]

    # synthetic generation
    new_rows = []
    for i in range(n):
        label = base_labels[i % len(base_labels)]
        new_rows.append({
            "liasse_numero": f"Jaug8696Y{i}",
            "libelle": label,
            NACE_REV2_1_COLUMN: "8696Y",
        })

    new_df = pd.DataFrame(new_rows)
    df_out = pd.concat([df, new_df], ignore_index=True)
    mask = pd.Series([False] * len(df) + [True] * len(new_df), index=df_out.index)
    return df_out, mask
