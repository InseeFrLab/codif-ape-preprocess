import pandas as pd

from utils.cleaning import apply_replacements


def pattern_cleaning_pipeline(series: pd.Series, **pattern_steps) -> pd.Series:
    """
    Configurable regex-based cleaning pipeline for rule-fixing.
    Applies each provided (pattern, replacement) set in order
    """
    steps = list(pattern_steps.items())

    for idx, (_, patterns) in enumerate(steps):
        for pattern, repl in patterns:
            series = apply_replacements(series, patterns)

    return series
