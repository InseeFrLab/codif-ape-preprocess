"""
main.py

This script loads a dataset, dynamically imports all registered rules,
filters them by tag (e.g. "naf_2025"), applies them to the DataFrame,
and saves both the updated dataset and a change log journal as Parquet files.

Outputs:
- Modified DataFrame: outputs/data_with_naf.parquet
- Audit journal: outputs/log_rules_applied.parquet
"""

import os
import sys
import argparse

from utils import io, file
from core.rule_engine import apply_rules
from cleaning.clean_for_rules import pattern_cleaning_pipeline
from constants.inputs import TEXTUAL_INPUTS, TEXTUAL_INPUTS_CLEANED
from constants.paths import (
    URL_SIRENE4_NAF2025,
    URL_OUTPUT_NAF2025,
    URL_REPORT_NAF2025,
)
from constants.regex_patterns import (
    STEP1_RULE_PATTERNS,
    STEP2_RULE_PATTERNS
)
from constants.matchers import DEFAULT_METHOD_PARAMS
from visualisation import run_plot


def load_dataset(path):
    print(f"📥 Loading dataset from: {path or 'default path'}")
    return io.download_data(path)


def save_outputs(training_data, log_rules_applied_training_data, methods):
    """
    Save training data and logs on S3 with dynamic file names based on methods.

    Args:
        training_data (DataFrame): The transformed dataset.
        log_rules_applied_training_data (DataFrame): Audit logs.
        methods (list of str): List of matching methods to save outputs for.
    """
    print("💾 Saving outputs...")
    base_data_name, ext_data = os.path.splitext(URL_OUTPUT_NAF2025)
    base_log_name, ext_log = os.path.splitext(URL_REPORT_NAF2025)

    suffix = file.get_suffix(methods)

    data_path = f"{base_data_name}{suffix}{ext_data}"
    log_path = f"{base_log_name}{suffix}{ext_log}"

    print(f"💾 Saving outputs with suffix '{suffix}':")
    print(f"  - data -> {data_path}")
    print(f"  - log  -> {log_path}")

    io.upload_data(training_data, data_path)
    io.upload_data(log_rules_applied_training_data, log_path)

    print("✅ All done!")


def main(input_data, methods, naf_tag="naf_2025", dry_run=False, show=False):
    df = load_dataset(input_data)
    for raw_col, clean_col in zip(TEXTUAL_INPUTS, TEXTUAL_INPUTS_CLEANED):
        df[clean_col] = pattern_cleaning_pipeline(
            df[raw_col],
            step1=STEP1_RULE_PATTERNS,
            step2=STEP2_RULE_PATTERNS
        )
    (df_out, mask), log_df = apply_rules(df, naf_tag, methods, DEFAULT_METHOD_PARAMS)

    if dry_run:
        print("🚫 Dry run enabled — no output files will be saved.")
        print(df_out)
        print(log_df)
    else:
        save_outputs(df_out, log_df, methods)

    if show:
        run_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply NAF enrichment rules to dataset."
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default=URL_SIRENE4_NAF2025,
        nargs="?",
        help="Path to the input Parquet file",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["regex"],
        help="List of matching methods to apply and save outputs for, e.g., regex fuzzy similarity",
    )
    parser.add_argument(
        "--naf_version",
        type=str,
        default="naf_2025",
        help="Which NAF ruleset to apply (default: naf_2025)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run the process without saving outputs"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Plot dataviz to compare methods",
    )

    args = parser.parse_args()

    if not args.input_data:
        print("❌ Error: You must provide the path to the input Parquet file.")
        parser.print_help()
        sys.exit(1)

    main(args.input_data, args.methods, args.naf_version, args.dry_run, args.show)
