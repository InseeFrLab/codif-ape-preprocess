"""
Generate one rule file per CJ code (e.g. ensure_cj_1000_present.py) using
the list of codes stored in a JSON file (local or S3).

The JSON is loaded into a Pandas DataFrame (one row per CJ).
"""

import os

import pandas as pd

from src.utils.io import download_json

CJ_JSON_PATH = "s3://projet-ape/data/cj.json"
OUTPUT_DIR = "./src/label_cleaning/rules/naf_2025"

TEMPLATE_2025 = """\
import pandas as pd
from src.label_cleaning.core.decorators import rule, track_new

@rule(
    name="cj_{code}_record_2025",
    tags=["naf_2025"],
    description="Ensure presence of CJ modality {code} (NAF 2025)",
)
@track_new(column='nace2025')
def add_cj_{code}_modality_2025(df: pd.DataFrame,methods=None, methods_params=None) -> pd.DataFrame:
    new_row = {{
        "liasse_numero": "J00addCJ{code}",
        "cj": "{code}",
        "libelle": "LMNP",
        "nace2025": "6820G",
    }}
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
"""

TEMPLATE_REV2 = """\
import pandas as pd
from src.label_cleaning.core.decorators import rule, track_new

@rule(
    name="cj_{code}_record_rev2",
    tags=["naf_rev2"],
    description="Ensure presence of CJ modality {code} (NAF rev 2)",
)
@track_new(column='apet_finale')
def add_cj_{code}_modality_rev2(df: pd.DataFrame,methods=None, methods_params=None) -> pd.DataFrame:
    new_row = {{
        "liasse_numero": "J00addCJ{code}",
        "cj": "{code}",
        "libelle": "LMNP",
        "nace2025": "6820A",
    }}
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
"""


def main():
    # Load JSON into a DataFrame
    raw_json = download_json(CJ_JSON_PATH)
    df_codes = pd.json_normalize(raw_json)  # → DataFrame with column "code"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for code in df_codes["code"].dropna():
        file_name = f"ensure_cj_{code}_present_2025.py"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        content_2025 = TEMPLATE_2025.format(code=code)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content_2025)

        print(f"✅ Generated {file_path}")

        file_name = f"ensure_cj_{code}_present_rev2.py"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        content_rev2 = TEMPLATE_REV2.format(code=code)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content_rev2)

        print(f"✅ Generated {file_path}")


if __name__ == "__main__":
    main()
