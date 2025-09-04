PREFIX = "s3://projet-ape/data/"
FOLDER = "08112022_27102024/"
ARTIFACTS_FOLDER = "artifacts/data_cleaning/"
PREPROCESSED_FOLDER = "preprocessed/"

URL_SIRENE4_NAFREV2 = PREFIX + FOLDER + "naf2008/" + "raw.parquet"
URL_SIRENE4_NAF2025 = PREFIX + FOLDER + "naf2025/" + "raw.parquet"

URL_OUTPUT_NAF2025 = PREFIX + FOLDER + "naf2025/" + "raw_cleansed.parquet"
URL_REPORT_NAF2025 = (
    PREFIX + FOLDER + "naf2025/" + ARTIFACTS_FOLDER + "raw_cleansed_report.parquet"
)
URL_OUTPUT_NAFREV2 = PREFIX + FOLDER + "naf2008/" + "raw_cleansed.parquet"
URL_REPORT_NAFREV2 = (
    PREFIX + FOLDER + "naf2008/" + ARTIFACTS_FOLDER + "raw_cleansed_report.parquet"
)

URL_DF_NAF2008 = PREFIX + "naf2008_extended.parquet"
URL_DF_NAF2025 = PREFIX + "naf2025_extended.parquet"
URL_DF_CJ = PREFIX + "cj.json"

URL_MAPPINGS = PREFIX + "mappings.json"

URL_LATEST_PROCESSED_PATH = PREFIX + "latest_processed_data.json"
