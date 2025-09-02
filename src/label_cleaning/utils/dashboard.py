def compute_embeddings(df, text_column, embedding_model):
    """
    Compute embeddings for a text column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column to embed.
        embedding_model (callable): Function that returns vector embedding for text.

    Returns:
        pd.DataFrame: DataFrame with 'embedding' column added.
    """
    df["embedding"] = df[text_column].apply(embedding_model)
    return df


def compute_fuzzy_ratios(df, col1, col2):
    """
    Compute fuzzy string similarity ratio between two columns row-wise.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col1, col2 (str): Column names to compare.

    Returns:
        pd.DataFrame: DataFrame with 'ratio' column added.
    """
    from fuzzywuzzy import fuzz

    df["ratio"] = df.apply(lambda row: fuzz.ratio(row[col1], row[col2]), axis=1)
    return df
