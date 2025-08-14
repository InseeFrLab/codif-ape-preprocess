import pandas as pd

from matching.multi_matcher import MultiMatcher


def merge_method_params(methods, methods_params, terms):
    """
    Merge external method parameters with default terms, allowing 'terms' override.
    """
    params = methods_params.get(methods, {})
    return params if "terms" in params else {**params, "terms": terms}


def build_regex_pattern(terms):
    """Merge/OR terms into a single regex pattern."""
    return "|".join(terms)


def build_matcher_kwargs(methods, methods_params, terms, pattern_builder=None):
    """
    Build matcher kwargs for MultiMatcher, merging default terms with provided method params.

    Args:
        methods (list[str]): list of method names (e.g. ["regex", "fuzzy"]).
        methods_params (dict): parameters for each method.
        terms (list[str]): default terms for matching.
        pattern_builder (callable, optional): custom function to build regex pattern from terms.

    Returns:
        dict: matcher kwargs ready to be passed to MultiMatcher.
    """
    kwargs = {}
    for m in methods:
        params = merge_method_params(m, methods_params, terms)

        if m == "regex":
            builder = pattern_builder or build_regex_pattern
            params = {"pattern": builder(params["terms"])}
        else:
            params["terms"] = terms

        kwargs[m] = params
    print(methods)
    print(kwargs)
    return kwargs


def build_match_mask(df, columns, methods, matcher_kwargs):
    """
    Apply MultiMatcher across multiple columns and combine the results into a single mask.

    Args:
        df (pd.DataFrame): input data.
        columns (list[str]): columns to match on.
        methods (list[str]): matching methods.
        matcher_kwargs (dict): parameters for each method.

    Returns:
        pd.Series: boolean mask where any column matched.
    """
    mask = pd.Series(False, index=df.index)
    mm = MultiMatcher(methods, **matcher_kwargs)

    for col in columns:
        mask |= mm.match(df[col].fillna(""))

    return mask
