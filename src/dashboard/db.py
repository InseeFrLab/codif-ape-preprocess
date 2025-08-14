import duckdb


def create_duckdb_connection(db_path=":memory:"):
    """
    Create a DuckDB connection.

    Args:
        db_path (str): Path to the DuckDB file. Use ":memory:" for in-memory DB.

    Returns:
        duckdb.DuckDBPyConnection: Active DuckDB connection.
    """
    return duckdb.connect(database=db_path)


def register_parquet_as_table(con, parquet_path, table_name):
    """
    Register a parquet file as a DuckDB table.

    Args:
        con (duckdb.DuckDBPyConnection): DuckDB connection.
        parquet_path (str): Path to parquet file (local or S3).
        table_name (str): Name of the table to register.
    """
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{parquet_path}'")


def query_threshold(con, table_name, target_column, method, threshold):
    """
    Query rows in a table where the target_column exceeds a threshold.

    Args:
        con (duckdb.DuckDBPyConnection): DuckDB connection.
        table_name (str): Name of the table.
        target_column (str): Column to apply threshold.
        method (str): Method filter (regex, fuzzy, similarity, etc.)
        threshold (float): Threshold value to filter.

    Returns:
        pd.DataFrame: Filtered rows.
    """
    sql = f"""
        SELECT *
        FROM {table_name}
        WHERE method = '{method}'
        AND {target_column} >= {threshold}
    """
    return con.execute(sql).df()
