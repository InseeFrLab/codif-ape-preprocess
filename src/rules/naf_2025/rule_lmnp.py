import numpy as np
import pandas as pd

from core.decorators import rule


@rule(
    name="lmnp_assignment_2025",
    tags=["naf_2025"],
    description="Règle LMNP version NAF 2025",
)
def lmnp_rule_rev2(df: pd.DataFrame) -> pd.DataFrame:
    pattern = r"lmnp|loueur en meuble.*non professionnel"
    is_lmnp = df["libelle"].str.contains(pattern, case=False, na=False)

    is_S = df["activ_perm_et"] == "S"
    is_P_or_null = df["activ_perm_et"].isin(["P"]) | df["activ_perm_et"].isnull()

    df["nace2025"] = np.where(is_lmnp & is_S, "5590Y", df["nace2025"])
    df["nace2025"] = np.where(is_lmnp & is_P_or_null, "6820G", df["nace2025"])
    return df
