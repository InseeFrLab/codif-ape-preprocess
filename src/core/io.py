import os

import pandas as pd
import pyarrow.dataset as ds
import s3fs


def get_filesystem():
    """
    Configure and return a S3-compatible filesystem (MinIO / AWS S3).
    """
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def download_parquet(path: str, use_arrow: bool = False) -> pd.DataFrame:
    """
    Load Parquet from local or S3 (based on path), using pandas or Arrow.
    """
    fs = get_filesystem() if path.startswith("s3://") else None

    if use_arrow:
        return ds.dataset(path, format="parquet", filesystem=fs).to_table().to_pandas()
    else:
        return pd.read_parquet(path, filesystem=fs)


def upload_parquet(df: pd.DataFrame, path: str):
    """
    Save DataFrame as Parquet to local or S3.
    """
    fs = get_filesystem() if path.startswith("s3://") else None
    df.to_parquet(path, index=False, filesystem=fs)
