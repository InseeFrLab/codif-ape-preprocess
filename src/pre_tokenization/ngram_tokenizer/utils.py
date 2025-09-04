"""
PytorchPreprocessor class.
"""

import re
import string
from typing import List, Optional

import nltk
import numpy as np
import pandas as pd
import unidecode
from joblib import Parallel, delayed
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from src.constants import SURFACE_COLS, URL_MAPPINGS
from src.utils.io import download_json

mappings = download_json(URL_MAPPINGS)
# Make sure stopwords are available
nltk.download("stopwords", quiet=True)

# Preload into a global constant (thread-/process-safe)
FRENCH_STOPWORDS = set(ntlk_stopwords.words("french")) | set(string.ascii_lowercase)


def clean_categorical_features(
    df: pd.DataFrame, categorical_features: List[str], y: Optional[str] = None
) -> pd.DataFrame:
    """
    Cleans the categorical features for pd.DataFrame `df`.

    Args:
        df (pd.DataFrame): DataFrame.
        categorical_features (List[str]): Names of the categorical features.
        y (str): Name of the variable to predict.

    Returns:
        df (pd.DataFrame): DataFrame.
    """
    for surface_col in SURFACE_COLS:
        if surface_col in categorical_features:
            df[surface_col] = df[surface_col].astype(float)
            df = categorize_surface(df, surface_col)
    for variable in categorical_features:
        if variable not in SURFACE_COLS:  # Mapping already done for this variable
            if len(set(df[variable].unique()) - set(mappings[variable].keys())) > 0:
                raise ValueError(
                    f"Missing values in mapping for {variable} ",
                    set(df[variable].unique()) - set(mappings[variable].keys()),
                )
            df[variable] = df[variable].apply(mappings[variable].get)
    if y is not None:
        if len(set(df[y].unique()) - set(mappings[y].keys())) > 0:
            raise ValueError(
                f"Missing values in mapping for {y}, ",
                set(df[y].unique()) - set(mappings[y].keys()),
            )
        df[y] = df[y].apply(mappings[y].get)

    return df


def _build_text_preprocessor(text, remove_stop_words=False, stem=False, n_jobs=None):
    stemmer = SnowballStemmer("french") if stem else None
    if remove_stop_words:
        stopwords = FRENCH_STOPWORDS
    else:
        stopwords = set()

    def preprocess(doc: str) -> str:
        # 1. Remove accents (é -> e, ç -> c, etc.)
        doc = unidecode.unidecode(doc)
        # 2. Lowercase
        doc = doc.lower()
        # 3. Remove punctuation, keep letters (a-z), numbers (0-9), and French chars (àâçéèêëîïôûùüÿñæœ), plus spaces
        doc = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\s]", " ", doc)
        # 4. Collapse multiple spaces into one and strip leading/trailing spaces
        doc = re.sub(r"\s+", " ", doc).strip()
        # 5. Tokenize
        words = doc.split()
        # 6. Remove one-letter tokens (e.g., stray letters)
        words = [w for w in words if len(w) > 1]
        # 7. Stopword removal
        if remove_stop_words:
            words = [w for w in words if w not in stopwords]
        # 8. Stemming
        if stem:
            words = [stemmer.stem(w) for w in words]
        return " ".join(words)

    return preprocess


def clean_text_feature(
    text: list[str],
    remove_stop_words: bool = False,
    stem: bool = False,
    n_jobs: int = -1,
    threshold: int = 50_000,
) -> list[str]:
    """
    Hybrid text cleaning for FastText.
    Uses list comprehension for small corpora,
    joblib.Parallel for large corpora.

    Args:
        text (list[str]): List of documents.
        remove_stop_words (bool): If True, remove stopwords.
        stem (bool): If True, apply stemming.
        n_jobs (int): Number of CPU cores (-1 = all).
        threshold (int): Switch to parallel if len(text) >= threshold.

    Returns:
        list[str]: Cleaned text.
    """
    preprocess = _build_text_preprocessor(remove_stop_words, stem)

    if len(text) < threshold:
        # Small corpus → fastest with list comprehension
        return [preprocess(doc) for doc in tqdm(text, desc="Preprocessing")]
    else:
        # Large corpus → parallelize across CPUs
        return Parallel(n_jobs=n_jobs)(
            delayed(preprocess)(doc) for doc in tqdm(text, desc="Preprocessing")
        )


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

    # df_naf["LIB_NIV5_cleaned"] = clean_text_feature(
    #     df_naf["LIB_NIV5"], remove_stop_words=remove_stop_words, stem=stem
    # )
    df_naf = df_naf[["APE_NIV5", "LIB_NIV5"]]
    df_naf = df_naf.rename(columns={"LIB_NIV5": text_feature, "APE_NIV5": Y})
    for cat_feat in categorical_features:
        if cat_feat not in SURFACE_COLS:
            df_naf[cat_feat] = pd.Series(["NaN"] * len(df_naf))
        else:
            df_naf[cat_feat] = pd.Series([0.0] * len(df_naf))

    return df_naf


def categorize_surface(
    df: pd.DataFrame, surface_feature_name: str, like_sirene_3: bool = True
) -> pd.DataFrame:
    """
    Categorize the surface of the activity.

    Args:
        df (pd.DataFrame): DataFrame to categorize.
        surface_feature_name (str): Name of the surface feature.
        like_sirene_3 (bool): If True, categorize like Sirene 3.

    Returns:
        pd.DataFrame: DataFrame with a new column "surf_cat".
    """
    df_copy = df.copy()
    # Check surface feature exists
    if surface_feature_name not in df.columns:
        raise ValueError(
            f"Surface feature {surface_feature_name} not found in DataFrame."
        )
    # Check surface feature is a float variable
    if not (pd.api.types.is_float_dtype(df[surface_feature_name])):
        raise ValueError(
            f"Surface feature {surface_feature_name} must be a float variable."
        )

    if like_sirene_3:
        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy[surface_feature_name],
            bins=[0, 120, 400, 2500, np.inf],
            labels=["1", "2", "3", "4"],
        ).astype(str)
    else:
        # Log transform the surface
        df_copy["surf_log"] = np.log(df[surface_feature_name])

        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy.surf_log,
            bins=[0, 3, 4, 5, 12],
            labels=["1", "2", "3", "4"],
        ).astype(str)

    df_copy[surface_feature_name] = df_copy["surf_cat"].replace("nan", "0")
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(int)
    df_copy = df_copy.drop(columns=["surf_log", "surf_cat"], errors="ignore")
    return df_copy
