import pandas as pd
from core.decorators import rule, track_new

@rule(
    name="cj_7342_record_2025",
    tags=["naf_2025"],
    description="Ensure presence of CJ modality 7342 (NAF 2025)",
)
@track_new(column='nace2025')
def add_cj_7342_modality_2025(df: pd.DataFrame,methods=None, methods_params=None) -> pd.DataFrame:
    new_row = {
        "liasse_numero": "J00addCJ7342",
        "cj": "7342",
        "libelle": "LMNP",
        "nace2025": "6820G",
    }
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
