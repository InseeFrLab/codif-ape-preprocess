from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def apply_replacements(
    series: pd.Series,
    patterns: Sequence[Tuple[str, Optional[str]]],
) -> pd.Series:
    """
    Apply a sequence of (pattern, replacement) to a pandas Series of strings.
    - If replacement is None, replace matches with NaN.
    - Patterns are applied in order, vectorized via pandas.
    """
    s = series.str.lower().fillna("")
    for pat, repl in patterns:
        if repl is None:
            s = s.replace(pat, np.nan, regex=True)
        else:
            s = s.str.replace(pat, repl, regex=True)
    return s


def remove_single_letters(series: pd.Series) -> pd.Series:
    """
    Remove any isolated single-character tokens (e.g. ' a ', ' b ').
    """
    return series.str.replace(r"\b\w\b", "", regex=True)
