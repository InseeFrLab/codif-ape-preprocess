"""
Precompute embeddings and fuzzy ratios for textual columns
and cache the results on S3 for use in the interactive dashboard.
"""

from cleaning.cleaner import clean_dataset
from preprocessing.rules import STEP1_RULE_PATTERNS, STEP2_RULE_PATTERNS

from constants.inputs import TEXTUAL_INPUTS, TEXTUAL_INPUTS_CLEANED
from constants.paths import PREFIX, URL_INPUT_NAF2025
from utils.dashboard_utils import compute_embeddings, compute_fuzzy_ratios
from utils.io import download_data, upload_data

# ---- Load dataset ----
df = download_data(URL_INPUT_NAF2025)

# ---- Clean textual columns ----
df = clean_dataset(
    df,
    textual_inputs=TEXTUAL_INPUTS,
    textual_inputs_cleaned=TEXTUAL_INPUTS_CLEANED,
    step1_patterns=STEP1_RULE_PATTERNS,
    step2_patterns=STEP2_RULE_PATTERNS,
)

# ---- Compute embeddings for methods that require them ----
df_embeddings = compute_embeddings(df, TEXTUAL_INPUTS_CLEANED)
cache_path_embed = f"{PREFIX}/dashboard/cache/naf2025_embedding.parquet"
upload_data(df_embeddings, cache_path_embed)

# ---- Compute fuzzy similarity ratios for pairwise matching ----
df_ratios = compute_fuzzy_ratios(df, TEXTUAL_INPUTS_CLEANED)
cache_path_ratio = f"{PREFIX}/dashboard/cache/naf2025_ratio.parquet"
upload_data(df_ratios, cache_path_ratio)

print("✅ Precompute finished and cached to S3")
