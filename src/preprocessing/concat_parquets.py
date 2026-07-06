"""
Script to concatenate two Parquet files from S3 into a single new Parquet file.
This script is intended to be run before the main preprocessing pipeline.

Run using:
uv run -m src.preprocessing.concat_parquets
--path1 s3://bucket/file1.parquet --path2 s3://bucket/file2.parquet
--output s3://bucket/combined.parquet
"""

import argparse
import pandas as pd
from src.utils.io import download_data, upload_data
from src.utils.logger import get_logger
from src.constants.paths import PREFIX, FOLDER, URL_RELABEL_LLM_NAF2025

logger = get_logger(name=__name__)


def concatenate_parquets(path1: str, path2: str, output_path: str):
    """
    Downloads two parquet files from S3 (or local), concatenates them,
    and uploads the result to the specified output path.
    """
    logger.info("🚀 Starting concatenation process...")
    logger.info(f"📄 File 1: {path1}")
    logger.info(f"📄 File 2: {path2}")
    logger.info(f"📤 Output: {output_path}")

    try:
        # 1. Download/Load the data
        # download_data uses download_parquet which handles S3 via s3fs internally
        logger.info("📥 Downloading/Loading dataframes...")
        df1 = download_data(path1)
        df2 = download_data(path2)

        logger.info(f"📊 File 1 shape: {df1.shape}")
        logger.info(f"📊 File 2 shape: {df2.shape}")

        # 2. Concatenate
        logger.info("🔗 Concatenating dataframes...")
        combined_df = pd.concat([df1, df2], ignore_index=True)
        logger.info(f"✅ Combined shape: {combined_df.shape}")

        # 3. Upload/Save the result
        logger.info(f"📤 Uploading combined dataframe to {output_path}...")
        upload_data(combined_df, output_path)
        logger.info("✨ Concatenation and upload completed successfully!")

    except Exception as e:
        logger.error(f"❌ An error occurred during concatenation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Default paths based on constants
    DEFAULT_PATH_REPRISE = PREFIX + FOLDER + "naf2025/" + "raw_reprise.parquet"
    DEFAULT_OUTPUT = PREFIX + FOLDER + "naf2025/" + "concat_synthetic_reprise.parquet"

    parser = argparse.ArgumentParser(description="Concatenate two Parquet files on S3.")
    parser.add_argument(
        "--path1",
        type=str,
        default=URL_RELABEL_LLM_NAF2025,
        help=f"Path to the first Parquet file (default: {URL_RELABEL_LLM_NAF2025})",
    )
    parser.add_argument(
        "--path2",
        type=str,
        default=DEFAULT_PATH_REPRISE,
        help=f"Path to the second Parquet file (default: {DEFAULT_PATH_REPRISE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Destination path for the concatenated Parquet file (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()

    concatenate_parquets(
        path1=args.path1,
        path2=args.path2,
        output_path=args.output
    )
