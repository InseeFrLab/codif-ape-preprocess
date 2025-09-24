import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import s3fs
import yaml
from qdrant_client import QdrantClient, models

from src.constants import (
    CATEGORICAL_FEATURES,
    COL_RENAMING,
    COLLECTION_NAME,
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


def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns the Qdrant client.
    """
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        port=443,
        https=True,
    )


def ensure_collection_exists(
    client: QdrantClient, collection_name: str = COLLECTION_NAME, vector_size: int = 384
):
    """
    Ensures a Qdrant collection exists with the specified configuration.

    The collection is created if it does not already exist.

    Args:
        client (QdrantClient): An initialized Qdrant client.
        collection_name (str): The name of the collection.
        vector_size (int): The dimensionality of the vectors in the collection.
    """
    try:
        client.get_collection(collection_name=collection_name)
        logger.info(f"✅ Collection '{collection_name}' already exists.")
    except Exception:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
        logger.info(f"✨ Collection '{collection_name}' created successfully.")


def import_points(
    client: QdrantClient,
    embeddings: list,
    ids: list,
    texts: list,
    collection_name: str = COLLECTION_NAME,
):
    """
    Imports embeddings, their IDs, and associated text (payload) into the collection.

    Args:
        client (QdrantClient): An initialized Qdrant client.
        embeddings (list): A list of vectors (embeddings) to import.
        ids (list): A list of unique IDs for the points.
        texts (list): A list of original texts corresponding to the embeddings.
        collection_name (str): The name of the collection to import to.
    """
    points = [
        models.PointStruct(id=i, vector=vec, payload={"text": text})
        for i, vec, text in zip(ids, embeddings, texts)
    ]
    client.upsert(collection_name=collection_name, points=points)
    logger.info(
        f"✅ {len(points)} points successfully imported"
        " into collection '{collection_name}'."
    )


def get_all_vectors_with_payload(
    client: QdrantClient, collection_name: str = COLLECTION_NAME, limit: int = 2000000
) -> tuple[list, list, list]:
    """
    Retrieves all vectors, their IDs, and text payload from a collection.

    Note: The payload must contain a 'text' key for this function to work correctly.

    Args:
        client (QdrantClient): An initialized Qdrant client.
        collection_name (str): The name of the collection to retrieve from.
        limit (int): The maximum number of points to retrieve.

    Returns:
        A tuple containing three lists: vectors, IDs, and texts.
    """
    all_points, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_vectors=True,
        with_payload=True,
    )
    vectors = [point.vector for point in all_points]
    ids = [point.id for point in all_points]
    texts = [point.payload.get("text") for point in all_points]

    logger.info(
        f"✅ {len(vectors)} vectors, IDs, and payloads"
        " retrieved from collection '{collection_name}'."
    )

    return vectors, ids, texts


# A global cache to store the text-to-ID mapping
_TEXT_TO_ID_MAPPING: dict = None


def get_text_to_id_mapping(client: QdrantClient, collection_name: str) -> dict:
    """
    Retrieves all texts and their IDs from the collection and caches the mapping.
    The cache is loaded only once per application lifecycle.

    Args:
        client (QdrantClient): An initialized Qdrant client.
        collection_name (str): The name of the Qdrant collection to retrieve from.

    Returns:
        dict: A dictionary mapping text strings to their Qdrant IDs.
    """
    global _TEXT_TO_ID_MAPPING
    if _TEXT_TO_ID_MAPPING is None:
        print("⏳ Text-to-ID mapping not found in cache. Loading from Qdrant...")
        _, ids, texts = get_all_vectors_with_payload(client, collection_name)
        _TEXT_TO_ID_MAPPING = {text: idx for idx, text in zip(ids, texts)}
        print("✅ Text-to-ID mapping loaded successfully.")
    else:
        print("💾 Using cached text-to-ID mapping.")
    return _TEXT_TO_ID_MAPPING


def get_embeddings_by_text(
    series: pd.Series, client: QdrantClient, collection_name: str
) -> pd.Series:
    """
    Given a pd.Series of strings, returns a pd.Series containing the associated embeddings
    from the Qdrant collection.

    This function uses a local text-to-ID mapping for efficient lookup.

    Args:
        series (pd.Series): The text strings for which to find embeddings.
        client (QdrantClient): An initialized Qdrant client.
        collection_name (str): The name of the Qdrant collection to query.

    Returns:
        pd.Series: A Series of embeddings with their corresponding Qdrant IDs as the index.
    """
    text_to_id_mapping = get_text_to_id_mapping(client, collection_name)

    target_ids = [
        text_to_id_mapping[text] for text in series if text in text_to_id_mapping
    ]

    if not target_ids:
        print("❌ None of the provided texts were found in the collection.")
        return pd.Series(dtype="object")

    points = client.retrieve(
        collection_name=collection_name, ids=target_ids, with_vectors=True
    )

    embeddings = [point.vector for point in points]
    result_series = pd.Series(embeddings, index=target_ids)

    return result_series
