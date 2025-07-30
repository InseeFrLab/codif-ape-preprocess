import numpy as np

from core.decorators import rule


@rule(
    name="lmnp_assignment_rev2",
    tags=["naf_rev2"],
    description="Attribue APE 6820B si LMNP détecté dans le libellé selon NAF Rev. 2",
)
def lmnp_rule_rev2(df):
    pattern = r"lmnp|loueur en meuble.*non professionnel"
    mask = df["LIB_CLEAN"].str.contains(pattern, case=False, na=False) & (
        df["AUTO"].isin(["E", "L", "S", "X", "I"]) | df["AUTO"].isnull()
    )
    df["APE_SICORE"] = np.where(mask, "6820B", df["APE_SICORE"])
    return df
