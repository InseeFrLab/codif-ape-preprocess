import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import s3fs
import yaml

from src.constants import (
    CATEGORICAL_FEATURES,
    COL_RENAMING,
    NACE_REV2_1_COLUMN,
    NACE_REV2_COLUMN,
    SURFACE_COLS,
    TEXT_FEATURE,
    TEXTUAL_FEATURES,
    URL_MAPPINGS,
    URL_SHARED_CONSTANTS,
)
from src.utils.logger import get_logger

logger = get_logger(name=__name__)


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


def download_parquet(
    path: str, use_arrow: bool = False, in_local_dir: bool = False
) -> pd.DataFrame:
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


def update_shared_constants(key: str, new_path: str):
    """Update JSON in S3 with a new path for the given key (NAF2008 or NAF2025).
    Only if new_path is different from the last one.
    """

    fs = get_filesystem()

    # Load existing dict if it exists
    if fs.exists(URL_SHARED_CONSTANTS):
        with fs.open(URL_SHARED_CONSTANTS, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Ensure key exists with a list
    if key not in data:
        data[key] = []

    # Append only if new_path is different from the last one (or if list is empty)
    if not data[key] or data[key][-1] != new_path:
        data[key].append(new_path)
        logger.info(f"Added {new_path} for {key} in {URL_SHARED_CONSTANTS}")
    else:
        logger.info(f"{new_path} is already the latest path for {key}, not adding.")

    data["URL_MAPPINGS"] = URL_MAPPINGS
    data["TEXT_FEATURE"] = TEXT_FEATURE
    data["TEXTUAL_FEATURES"] = TEXTUAL_FEATURES
    data["CATEGORICAL_FEATURES"] = CATEGORICAL_FEATURES
    data["SURFACE_COLS"] = SURFACE_COLS
    data["COL_RENAMING"] = COL_RENAMING
    data["NAF2008_TARGET"] = NACE_REV2_COLUMN
    data["NAF2025_TARGET"] = NACE_REV2_1_COLUMN

    # Save back to S3
    with fs.open(URL_SHARED_CONSTANTS, "w") as f:
        json.dump(data, f, indent=2)


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


def download_json(path: str):
    """
    Load a JSON file from local or S3 (based on path).
    Returns Python objects (dict or list).

    """
    fs = get_filesystem() if is_s3_path(path) else None

    if fs:
        with fs.open(path, mode="rb") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return pd.json_normalize(json.load(f))


def upload_json(dict_like, path: str):
    """
    Save a dictionary or list as a JSON file to local or S3.
    """
    fs = get_filesystem() if is_s3_path(path) else None
    with fs.open(path, mode="w") as f:
        json.dump(dict_like, f, indent=3)


def upload_yaml(dict_like, path: str):
    """
    Save a dictionary or list as a YAML file to local or S3.
    """
    fs = get_filesystem() if is_s3_path(path) else None

    with fs.open(path, mode="w") as f:
        yaml.dump(dict_like, f, indent=3)


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
        raise ValueError(
            f"Only .parquet or .csv files are supported. Incorrect path: {path}"
        )


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
