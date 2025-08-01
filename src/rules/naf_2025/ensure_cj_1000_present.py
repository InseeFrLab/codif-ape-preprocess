import numpy as np
import pandas as pd

from core.decorators import rule
from core.audit import track_new


@rule(
    name="cj_1000_record_2025",
    tags=["naf_2025"],
    description="Règle présence modalité 1000 de la cj version NAF 2025",
)
@track_new(column="cj")
def add_cj_1000_modality_2025(df: pd.DataFrame) -> pd.DataFrame:
    # Create a new row with default / placeholder values
    new_row = {
        "cj": "1000",
        "libelle": "LMNP",
        "nace2025": "6820G",
    }

    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return new_df