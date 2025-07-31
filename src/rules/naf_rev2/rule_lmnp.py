import numpy as np

from core.decorators import rule
from core.audit import track_changes


@rule(
    name="lmnp_assignment_rev2",
    tags=["naf_rev2"],
    description="Attribue APE 6820B si LMNP détecté dans le libellé selon NAF Rev. 2",
)
@track_changes(column="APE_SICORE")
def lmnp_rule_rev2(df):
    pattern = r"lmnp|loueur en meuble non professionnel| \
                loueur bailleur non professionnel| \
                location meublee non professionnelle| \
                loueur meuble non professionnel| \
                loueurs en meubles non professionnels| \
                loueur en meubl non professionnel| \
                loueur en meubles non professionnel"
    mask = df["LIB_CLEAN"].str.contains(pattern, case=False, na=False) & (
        df["AUTO"].isin(["E", "L", "S", "X", "I"]) | df["AUTO"].isnull()
    )
    df["APE_SICORE"] = np.where(mask, "6820B", df["APE_SICORE"])
    df = df[
        ["DATE", "APE_SICORE", "LIB_SICORE", "AUTO", "NAT_SICORE", "EVT_SICORE", "SURF"]
    ]
    return df
