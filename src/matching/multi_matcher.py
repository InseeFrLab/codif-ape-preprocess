"""
multi_matchers.py

Provides the MultiMatcher class that can apply one or more matching methods
to a Pandas Series and combine their results with a logical OR.
"""

from . import MATCHERS
import pandas as pd


class MultiMatcher:
    """
    Apply one or more matching methods (OR-combined) over a column.

    Args:
        methods (list[str]): List of method names (keys in MATCHERS) to apply.
        **kwargs: Optional per-method keyword arguments.

    Example:
        matcher = MultiMatcher(
            methods=["regex", "fuzzy"],
            regex={"pattern": r"lmnp|location meubl"},
            fuzzy={"terms": ["lmnp", "loueur meubl"]}
        )
        mask = matcher.match(df["libelle"])
    """
    def __init__(self, methods, **kwargs):
        self.methods = methods
        self.kwargs = kwargs

    def match(self, series: pd.Series) -> pd.Series:
        """
        Apply the selected methods to the series and OR-combine their results.

        Args:
            series (pd.Series): The data to match against.

        Returns:
            pd.Series: Boolean mask indicating matching rows.
        """
        mask = pd.Series(False, index=series.index)
        for m in self.methods:
            fn = MATCHERS[m]
            mask |= fn(series, **self.kwargs.get(m, {}))
        return mask
