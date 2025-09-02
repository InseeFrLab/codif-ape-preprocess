import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.constants import (
    CATEGORICAL_FEATURES,
    COL_RENAMING,
    NACE_REV2_1_COLUMN,
    NACE_REV2_COLUMN,
    TEXT_FEATURE,
    TEXTUAL_FEATURES,
    URL_DF_NAF2008,
    URL_DF_NAF2025,
    URL_OUTPUT_NAF2025,
    URL_OUTPUT_NAFREV2,
)
from src.pre_tokenization.ngram_tokenizer.utils import (
    clean_categorical_features,
    clean_df_naf,
    clean_text_feature,
)
from src.utils.io import download_parquet, get_filesystem


def ngram_pretokenize(
    revision, test_size=0.2, remove_stop_words=True, stem=True, **kwargs
):
    if revision == "NAF2008":
        path_raw_cleansed = URL_OUTPUT_NAFREV2
    elif revision == "NAF2025":
        path_raw_cleansed = URL_OUTPUT_NAF2025
    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    fs = get_filesystem()

    df = pq.read_table(path_raw_cleansed, filesystem=fs).to_pandas()

    df = df.rename(columns=COL_RENAMING)

    Y = NACE_REV2_COLUMN if revision == "NAF2008" else NACE_REV2_1_COLUMN

    df = df[[Y, TEXT_FEATURE] + TEXTUAL_FEATURES + CATEGORICAL_FEATURES]

    df_naf = download_parquet(
        URL_DF_NAF2008 if revision == "NAF2008" else URL_DF_NAF2025
    )
    df_naf = clean_df_naf(
        df_naf,
        remove_stop_words=remove_stop_words,
        stem=stem,
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
            df[feature] = df[feature].fillna(value="NaN")

    df = df.dropna(subset=[Y] + [TEXT_FEATURE], axis=0)

    df[TEXT_FEATURE] = clean_text_feature(
        df[TEXT_FEATURE], remove_stop_words=remove_stop_words, stem=stem
    )
    for textual_feature in TEXTUAL_FEATURES:
        df[textual_feature] = clean_text_feature(
            df[textual_feature], remove_stop_words=remove_stop_words, stem=stem
        )

    df[TEXT_FEATURE] = df[TEXT_FEATURE] + df[TEXTUAL_FEATURES].apply(
        lambda x: " ".join(x), axis=1
    )

    # # Clean categorical features
    df = clean_categorical_features(df, categorical_features=CATEGORICAL_FEATURES, y=Y)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(
            columns=[Y, *TEXTUAL_FEATURES]
        ),  # drop the textual additional var as they are already concatenated to the libelle
        df[Y],
        test_size=test_size,
        random_state=0,
        shuffle=True,
    )

    df_test = pd.concat([X_test, y_test], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=0,
        shuffle=True,
    )

    # Adding the true labels to the training set

    df_train = pd.concat([X_train, y_train], axis=1)
    df_train = pd.concat([df_train, df_naf], axis=0)
    df_val = pd.concat([X_val, y_val], axis=1)

    return df_train, df_val, df_test
