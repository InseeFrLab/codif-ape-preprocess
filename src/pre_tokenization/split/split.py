import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.constants import (
    CATEGORICAL_FEATURES,
    COL_RENAMING,
    NACE_REV2_1_COLUMN,
    NACE_REV2_COLUMN,
    SURFACE_COLS,
    TEXT_FEATURE,
    TEXTUAL_FEATURES,
    URL_DF_NAF2008,
    URL_DF_NAF2025,
    URL_OUTPUT_NAF2025,
    URL_OUTPUT_NAFREV2,
)
from src.pre_tokenization.ngram_tokenizer.utils import (
    clean_df_naf,
)
from src.utils.io import download_parquet, get_filesystem
from src.utils.logger import get_logger

logger = get_logger(name=__name__)


def split(revision, test_size=0.2):
    if revision == "NAF2008":
        path_raw_cleansed = URL_OUTPUT_NAFREV2
    elif revision == "NAF2025":
        path_raw_cleansed = URL_OUTPUT_NAF2025
    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    fs = get_filesystem()

    logger.info(f"🔎 Reading raw cleansed data from {path_raw_cleansed}")
    df = pq.read_table(path_raw_cleansed, filesystem=fs).to_pandas()

    df = df.rename(columns=COL_RENAMING)

    df_naf = download_parquet(
        URL_DF_NAF2008 if revision == "NAF2008" else URL_DF_NAF2025
    )

    Y = NACE_REV2_COLUMN if revision == "NAF2008" else NACE_REV2_1_COLUMN

    df = df[[Y, TEXT_FEATURE] + TEXTUAL_FEATURES + CATEGORICAL_FEATURES]
    df_naf = clean_df_naf(
        df_naf,
        text_feature=TEXT_FEATURE,
        Y=Y,
        categorical_features=CATEGORICAL_FEATURES,
    )

    variables = [Y] + [TEXT_FEATURE]
    if TEXTUAL_FEATURES is not None:
        variables += TEXTUAL_FEATURES
        for feature in TEXTUAL_FEATURES:
            df[feature] = df[feature].fillna(value="")
    if CATEGORICAL_FEATURES is not None:
        variables += CATEGORICAL_FEATURES
        for feature in CATEGORICAL_FEATURES:
            if feature not in SURFACE_COLS:
                df[feature] = df[feature].fillna(value="NaN")

    df = df.dropna(subset=[Y] + [TEXT_FEATURE], axis=0)

    for col in SURFACE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)
            df[col] = df[col].fillna(value=0.0)

    df[TEXT_FEATURE] = df[TEXT_FEATURE] + df[TEXTUAL_FEATURES].apply(
        lambda x: " ".join(x), axis=1
    )  # drop the textual additional var as they are already concatenated to the libelle

    df_train, df_test = train_test_split(
        df.drop(columns=TEXTUAL_FEATURES),
        test_size=test_size,
        random_state=0,
        shuffle=False,
    )

    df_train, df_val = train_test_split(
        df_train,
        test_size=test_size,
        random_state=0,
        shuffle=False,
    )

    df_train = pd.concat([df_train, df_naf], axis=0)

    # # Adding the true labels to the training set
    logger.info("✅ Data split into train, val and test sets.")
    logger.info(f"df_train looks like: {df_train.head()}")

    return df_train, df_val, df_test
