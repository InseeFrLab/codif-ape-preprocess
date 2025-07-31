"""
audit.py

This module defines the `@track_changes` decorator which tracks changes applied
by a rule to a specific column. It logs the original and updated values, along with
other metadata, and returns both the modified DataFrame and a journal DataFrame.

The journal is used for auditing which rules changed what.
"""

from functools import wraps

import pandas as pd


def track_changes(column: str, extra_cols=["libelle"]):
    def decorator(func):
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            before = df[column].copy()
            result = func(df, *args, **kwargs)
            after = df[column]

            changed = before != after
            journal = df.loc[changed, extra_cols].copy()
            journal["APE_BEFORE"] = before[changed]
            journal["APE_AFTER"] = after[changed]
            journal["_log_rules_applied"] = func.__name__

            return result, journal

        return wrapper

    return decorator
