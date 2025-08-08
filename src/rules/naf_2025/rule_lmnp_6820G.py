import numpy as np
import pandas as pd

from core.decorators import rule, track_changes


@rule(
    name="lmnp_assignment_2025",
    tags=["naf_2025"],
    description="Règle LMNP version NAF 2025",
)
@track_changes(column="nace2025")
def lmnp_rule_2025(df: pd.DataFrame) -> pd.DataFrame:
    pattern = r"lmnp|loueur en meuble non professionnel| \
                loueur bailleur non professionnel| \
                location meublee non professionnelle| \
                loueur meuble non professionnel| \
                loueurs en meubles non professionnels| \
                loueur en meubl non professionnel| \
                loueur en meubles non professionnel"
    mask = df["libelle_clean"].str.contains(pattern, case=False, na=False)

    mask_5590Y = df["activ_perm_et"] == "S"
    mask_6820G = df["activ_perm_et"].isin(["P"]) | df["activ_perm_et"].isnull()

    df["nace2025"] = np.where(mask & mask_5590Y, "5590Y", df["nace2025"])
    df["nace2025"] = np.where(mask & mask_6820G, "6820G", df["nace2025"])
    return df, mask
