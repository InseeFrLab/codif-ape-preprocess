import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import s3fs


def get_filesystem():
    """
    Configure and return a S3-compatible filesystem (MinIO / AWS S3).
    """
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def is_s3_path(path: str) -> bool:
    """Check if the path points to S3."""
    return path.startswith("s3://")


def download_parquet(path: str, use_arrow: bool = False,
                     in_local_dir: bool = False) -> pd.DataFrame:
    """
    Load Parquet from local or S3 (based on path), using pandas or Arrow.
    """
    fs = get_filesystem()
    if use_arrow:
        path = path.removeprefix("s3://")
        data = ds.dataset(path, format="parquet", filesystem=fs).to_table().to_pandas()
        if in_local_dir:
            pq.write_table(pa.Table.from_pandas(data), path)
        return ds.dataset(path, format="parquet", filesystem=fs).to_table().to_pandas()
    else:
        print(path)
        return pd.read_parquet(path, filesystem=fs)


def upload_parquet(df: pd.DataFrame, path: str):
    """
    Save DataFrame as Parquet to local or S3.
    """
    fs = get_filesystem() if is_s3_path(path) else None
    return df.to_parquet(path, index=False, filesystem=fs)


def download_csv(path: str) -> pd.DataFrame:
    """
    Load CSV from local or S3 (based on path).
    The `use_arrow` flag is included for consistency but not used.
    """
    fs = get_filesystem() if is_s3_path(path) else None

    if fs:
        return pd.read_csv(fs.open(path, mode="rb"))
    else:
        return pd.read_csv(path)


def upload_csv(df: pd.DataFrame, path: str):
    """
    Save DataFrame as CSV to local or S3.
    """
    fs = get_filesystem() if is_s3_path(path) else None

    if fs:
        with fs.open(path, mode="w") as f:
            df.to_csv(f, index=False)
    else:
        df.to_csv(path, index=False)


def download_data(path: str, in_local_dir: bool = True) -> pd.DataFrame:
    """
    Load a Parquet or CSV file from local disk or S3.

    Parameters:
        path (str): Path to the file (local or S3)

    Returns:
        pd.DataFrame: Loaded DataFrame
    """

    if path.endswith(".parquet"):
        return download_parquet(path, use_arrow=True, in_local_dir=False)
    elif path.endswith(".csv"):
        return download_csv(path)
    else:
        raise ValueError("Only .parquet or .csv files are supported")


def upload_data(df: pd.DataFrame, path: str):
    """
    Save a DataFrame to Parquet or CSV locally or to S3.

    Parameters:
        df (pd.DataFrame): The DataFrame to save
        path (str): Destination path (local or S3)
    """

    if path.endswith(".parquet"):
        upload_parquet(df, path)
    elif path.endswith(".csv"):
        upload_csv(df, path)
    else:
        raise ValueError("Only .parquet or .csv output is supported")
