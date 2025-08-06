# core/utils/text_utils.py
import pandas as pd
from rapidfuzz import fuzz


def fuzzy_mask(series: pd.Series, pattern: str, threshold: float = 85.0) -> pd.Series:
    """
    Retourne un masque booléen où chaque valeur de `series` est suffisamment
    similaire à `pattern` selon `fuzz.token_sort_ratio`.
    """
    # On remplace les NaN par vide pour éviter les erreurs
    filled = series.fillna("")
    return filled.apply(lambda s: fuzz.token_sort_ratio(s, pattern) >= threshold)
