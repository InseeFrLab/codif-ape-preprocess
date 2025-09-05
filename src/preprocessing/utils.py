import pandas as pd

from src.constants import SURFACE_COLS


def clean_df_naf(df_naf, text_feature, Y, categorical_features):
    """
    Cleans the NAF DataFrame for concatenation with df_train.

    Args:
        df_naf (pd.DataFrame): NAF DataFrame.
        remove_stop_words (bool): If True, remove stopwords.
        stem (bool): If True, apply stemming.
        text_feature (str): Name of the text feature.
        Y (str): Name of the variable to predict.
        categorical_features (List[str]): Names of the categorical features.
    Returns:
        df_naf (pd.DataFrame): Cleaned NAF DataFrame.
    """

    df_naf = df_naf[["APE_NIV5", "LIB_NIV5"]]
    df_naf = df_naf.rename(columns={"LIB_NIV5": text_feature, "APE_NIV5": Y})
    for cat_feat in categorical_features:
        if cat_feat not in SURFACE_COLS:
            df_naf[cat_feat] = pd.Series(["NaN"] * len(df_naf))
        else:
            df_naf[cat_feat] = pd.Series([0.0] * len(df_naf))

    return df_naf
