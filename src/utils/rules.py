def merge_method_params(methods, methods_params, terms):
    """
    Merge external method parameters with default terms, allowing 'terms' override.
    """
    params = methods_params.get(methods, {})
    return params if "terms" in params else {**params, "terms": terms}


def build_regex_pattern(terms):
    """Merge/OR terms into a single regex pattern."""
    return "|".join(terms)
