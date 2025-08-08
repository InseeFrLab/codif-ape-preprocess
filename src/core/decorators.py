"""
core/decorators.py

This module defines decorators used in the rule-processing framework.

Available decorators:
- @rule: Registers a transformation rule in the central registry.
- @track_changes: Tracks modifications made to a specific column.
- @track_new: Tracks creation of new rows.

These decorators are mainly intended for rule fixing, not for general
data cleaning or preprocessing for model training.
"""

from functools import wraps
import pandas as pd

from core.registry import register_rule


def rule(name=None, tags=None, description=""):
    """
    Decorator to register a transformation rule in the central registry.

    Args:
        name (str, optional): Custom name for the rule. Defaults to the function name.
        tags (list, optional): List of tags to categorize the rule.
        description (str, optional): Rule description for documentation purposes.

    Usage:
        @rule(name="my_rule", tags=["naf_2025"], description="Fixes category codes.")
        def my_rule(df):
            ...
    """
    def decorator(func):
        register_rule(
            func=func,
            name=name or func.__name__,
            tags=tags or [],
            description=description,
        )
        return func
    return decorator


def track_changes(column: str, extra_cols=None):
    """
    Decorator to track changes made to a specific column by a rule.

    The wrapped function must return (df_out, mask), where:
        - df_out: modified DataFrame
        - mask: boolean Series indicating which rows were changed

    A journal DataFrame is created containing:
        - extra_cols: user-defined columns for context
        - 'APE_BEFORE': value before change
        - 'APE_AFTER': value after change
        - '_log_rules_applied': name of the applied rule
        - '_change_type': always "modification"

    Args:
        column (str): Column to track for changes.
        extra_cols (list, optional): Additional columns to include in the audit log.
                                     Defaults to ["liasse_numero", "libelle"].
    """
    if extra_cols is None:
        extra_cols = ["liasse_numero", "libelle"]

    def decorator(func):
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            before = df[column].copy()
            result = func(df, *args, **kwargs)

            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError("Rule must return (df_out, mask)")

            df_out, mask = result
            after = df_out[column]

            journal = df_out.loc[mask, extra_cols].copy()
            journal["APE_BEFORE"] = before[mask]
            journal["APE_AFTER"] = after[mask]
            journal["_log_rules_applied"] = func.__name__
            journal["_change_type"] = "modification"

            return (df_out, mask), journal
        return wrapper
    return decorator


def track_new(extra_cols=None, column: str = "nace2025"):
    """
    Decorator to track newly created rows.

    The wrapped function must return the modified DataFrame.
    New rows are detected by comparing DataFrame length before and after execution.

    A journal DataFrame is created containing:
        - extra_cols: user-defined columns for context
        - 'APE_BEFORE': None (no previous value)
        - 'APE_AFTER': value in `column`
        - '_log_rules_applied': name of the applied rule
        - '_change_type': always "creation"

    Args:
        extra_cols (list, optional): Additional columns to include in the audit log.
                                     Defaults to ["liasse_numero", "libelle", "cj"].
        column (str): Column from which to log the "after" value for new rows.
    """
    if extra_cols is None:
        extra_cols = ["liasse_numero", "libelle", "cj"]

    def decorator(func):
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            old_len = len(df)
            df_out = func(df, *args, **kwargs)

            new_idx = list(range(old_len, len(df_out)))
            if not new_idx:
                journal = pd.DataFrame(columns=extra_cols + [
                    "APE_BEFORE", "APE_AFTER", "_log_rules_applied", "_change_type"
                ])
            else:
                journal = df_out.loc[new_idx, extra_cols].copy()
                journal["APE_BEFORE"] = None
                journal["APE_AFTER"] = df_out.loc[new_idx, column]
                journal["_log_rules_applied"] = func.__name__
                journal["_change_type"] = "creation"

            return df_out, journal
        return wrapper
    return decorator
