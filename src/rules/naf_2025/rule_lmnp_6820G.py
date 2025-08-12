import numpy as np
import pandas as pd

from core.decorators import rule, track_changes
from matching.multi_matcher import MultiMatcher
from utils.rules import merge_method_params, build_regex_pattern


@rule(
    name="lmnp_assignment_2025",
    tags=["naf_2025"],
    description="Règle LMNP version NAF 2025",
)
@track_changes(column="nace2025")
def lmnp_rule_2025(df: pd.DataFrame, methods=None, methods_params=None) -> pd.DataFrame:
    methods = ["regex"] if methods is None else methods
    methods_params = {} if methods_params is None else methods_params

    terms = [
        "lmnp",
        "loueur en meuble non professionnel",
        "loueur bailleur non professionnel",
        "location meublee non professionnelle",
        "loueur meuble non professionnel",
        "loueurs en meubles non professionnels",
        "loueur en meubl non professionnel",
        "loueur en meubles non professionnel"
    ]

    matcher_kwargs = {}

    for m in methods:
        params = merge_method_params(m, methods_params, terms)
        if m == "regex":
            # Merge/OR terms into a single regex pattern
            params = {"pattern": build_regex_pattern(params["terms"])}
        else:
            params["terms"] = terms

        matcher_kwargs[m] = params

    textual_inputs = ["libelle_clean"]
    mask = pd.Series(False, index=df.index)
    for col in textual_inputs:
        mm = MultiMatcher(methods, **matcher_kwargs)
        mask |= mm.match(df[col])

    mask_5590Y = df["activ_perm_et"] == "S"
    mask_6820G = df["activ_perm_et"].isin(["P"]) | df["activ_perm_et"].isnull()

    df["nace2025"] = np.where(mask & mask_5590Y, "5590Y", df["nace2025"])
    df["nace2025"] = np.where(mask & mask_6820G, "6820G", df["nace2025"])
    return df, mask
