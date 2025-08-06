"""
audit.py

This module defines the `@track_changes` decorator which tracks changes applied
by a rule to a specific column. It logs the original and updated values, along with
other metadata, and returns both the modified DataFrame and a journal DataFrame.

The journal is used for auditing which rules changed what.
"""

from functools import wraps

import pandas as pd


def track_changes(column: str, extra_cols=["liasse_numero", "libelle"]):
    def decorator(func):
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            before = df[column].copy()
            result = func(df, *args, **kwargs)
            after = df[column]

            if isinstance(result, tuple):
                df_out, mask = result
            else:
                raise ValueError("Rule must return (df, mask)")

            changed = mask
            journal = df.loc[changed, extra_cols].copy()
            journal["APE_BEFORE"] = before[changed]
            journal["APE_AFTER"] = after[changed]
            journal["_change_type"] = "modification"
            journal["_log_rules_applied"] = func.__name__

            return result, journal

        return wrapper

    return decorator


def track_new(extra_cols=["liasse_numero", "libelle", "cj"], column: str = "nace2025"):
    """
    Decorator to track newly added rows.
    Assumes the wrapped function returns (df_out, new_mask) where new_mask
    is a boolean Series marking the newly created rows.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            old_len = len(df)

            # Exécution de la règle (qui ne retourne que df_modifié)
            df_out = func(df, *args, **kwargs)

            # Les nouvelles lignes sont celles dont l'index >= old_len
            new_idx = list(range(old_len, len(df_out)))

            # 4) On construit le journal
            if not new_idx:
                journal = pd.DataFrame(columns=extra_cols +
                                       ["APE_BEFORE",
                                        "APE_AFTER",
                                        "_log_rules_applied",
                                        "_change_type"])
            else:
                journal = df_out.loc[list(new_idx), extra_cols].copy()
                journal["APE_BEFORE"] = None
                journal["APE_AFTER"] = df_out.loc[list(new_idx), column]
                journal["_log_rules_applied"] = func.__name__
                journal["_change_type"] = "creation"

            return df_out, journal
        return wrapper
    return decorator
