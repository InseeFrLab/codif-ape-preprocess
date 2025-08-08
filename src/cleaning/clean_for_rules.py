from utils.cleaning import apply_replacements, remove_single_letters

from constants.regex_patterns import (
    STEP1_RULE_PATTERNS,
    STEP2_RULE_PATTERNS
)
import pandas as pd


def cleaning_pipeline(series: pd.Series) -> pd.Series:
    """
    Lightweight pipeline for rule-fixing:
      1. apply STEP1_RULE_PATTERNS via apply_replacements
      2. remove single letters
      3. apply STEP2_RULE_PATTERNS via apply_replacements
    """
    s = apply_replacements(series, STEP1_RULE_PATTERNS)
    s = remove_single_letters(s)
    s = apply_replacements(s, STEP2_RULE_PATTERNS)
    return s
