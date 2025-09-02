from .clean_for_rules import pattern_cleaning_pipeline


def clean_dataset(
    df, textual_inputs, textual_inputs_cleaned, step1_patterns, step2_patterns
):
    """
    Clean textual columns using provided regex patterns and columns names.

    Args:
        df (pd.DataFrame): Input dataframe.
        textual_inputs (list[str]): Names of raw text columns.
        textual_inputs_cleaned (list[str]): Names of cleaned text columns.
        step1_patterns (dict): First-step regex patterns.
        step2_patterns (dict): Second-step regex patterns.

    Returns:
        pd.DataFrame: Dataframe with cleaned columns added.
    """
    for raw_col, clean_col in zip(textual_inputs, textual_inputs_cleaned):
        df[clean_col] = pattern_cleaning_pipeline(
            df[raw_col], step1=step1_patterns, step2=step2_patterns
        )
    return df
