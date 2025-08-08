# STEP1_PATTERNS for rule-cleaning context
STEP1_RULE_PATTERNS = [
    (r"[^\w\s]", " "),    # purge punctuation
    (r"\d+", " "),        # purge digits
    (r"\s{2,}", " "),     # multiple spaces → single space
    (r"^\s+|\s+$", ""),   # strip leading/trailing
    (r"^$", None),        # empty → NaN
]

# STEP2_PATTERNS (further normalization)
STEP2_RULE_PATTERNS = [
    (r"\b\w\b", ""),      # remove single letters
    (r"\s{2,}", " "),     # re-normalize spaces
    (r"^\s+|\s+$", ""),   # strip again
]
