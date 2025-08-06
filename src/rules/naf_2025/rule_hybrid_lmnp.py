import pandas as pd

from core.decorators import rule
from core.audit import track_changes
from core.utils.text_utils import fuzzy_mask


@rule(name="hybrid_lmnp", tags=["naf_test", "regex-fuzzy"], description="LMNP hybrid regex + fuzzy")
@track_changes(column="nace2025")
def hybrid_lmnp(df):
    # 1) Regex strict
    regex = r"\blmnp\b"
    mask_regex = df["libelle"].str.contains(regex, case=False, na=False)

    # 2) Fuzzy sur le reste
    mask_fuzzy = fuzzy_mask(
        df.loc[~mask_regex, "libelle"],
        pattern="loueur en meuble non professionnel",
        threshold=82.0
    )
    # mask_fuzzy n’est défini que sur les indices restants
    full_fuzzy = pd.Series(False, index=df.index)
    full_fuzzy.loc[mask_fuzzy.index] = mask_fuzzy

    # 3) Condition cat
    mask_cat = df["activ_perm_et"].isin(["P", None])  # ou d’autres modalités

    # 4) Combine
    combined_mask = (mask_regex | full_fuzzy) & mask_cat

    df.loc[combined_mask, "nace2025"] = "6820G"
    return df
