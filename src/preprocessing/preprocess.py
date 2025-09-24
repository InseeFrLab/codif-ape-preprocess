"""Run using uv run -m src.preprocessing.preprocess --revision NAF2025 from root"""

import argparse

from src.constants.paths import FOLDER, PREFIX
from src.preprocessing.split import split
from src.utils.io import update_shared_constants, upload_parquet
from src.utils.logger import get_logger

logger = get_logger(name=__name__)


def main(revision: str, test_size: float):
    root_save_path = PREFIX + FOLDER

    df_train, df_val, df_test = split(revision, test_size=test_size)

    save_path_split = root_save_path + f"{revision.lower()}/" + "split/"

    logger.info(f"💾 Saving split datasets to {save_path_split}")
    upload_parquet(df_train, save_path_split + "df_train.parquet")
    upload_parquet(df_val, save_path_split + "df_val.parquet")
    upload_parquet(df_test, save_path_split + "df_test.parquet")

    # Update latest path for preocessed data
    update_shared_constants(
        revision.upper(),
        PREFIX + FOLDER + f"{revision.lower()}/",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--revision",
        type=str,
        choices=["NAF2008", "NAF2025"],
        required=True,
        help="Revision of the NACE classification to use.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split.",
    )
    args = parser.parse_args()

    main(**vars(args))
