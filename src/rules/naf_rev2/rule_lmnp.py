import numpy as np

from core.decorators import rule


@rule(
    name="lmnp_assignment_rev2",
    tags=["naf_rev2"],
    description="Attribue APE 6820B si LMNP détecté dans le libellé selon NAF Rev. 2",
)
def lmnp_rule_rev2(df):
    pattern = r"lmnp|loueur en meuble.*non professionnel"
    mask = df["libelle"].str.contains(pattern, case=False, na=False) & (
        df["liasse_type"].isin(["E", "L", "S", "X", "I"]) | df["liasse_type"].isnull()
    )
    df["nace2025"] = np.where(mask, "6820B", df["nace2025"])
    return df
