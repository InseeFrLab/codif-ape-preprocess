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

import pandas as pd
from tqdm import tqdm

from core import io
from core.loader import load_rules
from core.registry import RULES_REGISTRY


def load_dataset(path):
    print(f"📥 Loading dataset from: {path or 'default path'}")
    return io.download_data(path)


def apply_rules(training_data, tag):
    print("🔍 Loading and filtering rules...")
    load_rules()
    rules_to_apply = [r for r in RULES_REGISTRY if tag in r.tags]
    print(f"🧩 {len(rules_to_apply)} rule(s) matched with tag '{tag}'")

    all_journals = []

    print("⚙️  Applying rules...")
    for rule in tqdm(rules_to_apply, desc="Processing rules", unit="rule"):
        df, journal = rule.apply(training_data)
        all_journals.append(journal)

    return df, pd.concat(all_journals, ignore_index=True)


def save_outputs(training_data, log_rules_applied_training_data):
    print("💾 Saving outputs...")
    io.upload_data(training_data, "./outputs/fixed_training_data.parquet")
    io.upload_data(log_rules_applied_training_data, "outputs/log_rules_applied.parquet")
    print("✅ All done!")


def main(df_path=None, tag="naf_2025", dry_run=False):
    # Create directories
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./outputs', exist_ok=True)
    df = load_dataset(df_path)
    (df, mask), log_df = apply_rules(df, tag)
    if dry_run:
        print("🚫 Dry run enabled — no output files will be saved.")
    else:
        save_outputs(df, log_df)


if __name__ == "__main__":
    default_df = (
        "s3://projet-ape/NAF-revision/relabeled-data/20241027_sirene4_nace2025.parquet"
    )
    # df_path = sys.argv[1] if len(sys.argv) > 1 else default_df
    # tag = sys.argv[2] if len(sys.argv) > 2 else "naf_rev2"
    parser = argparse.ArgumentParser(
        description="Apply NAF enrichment rules to dataset."
    )
    parser.add_argument(
        "df_path",
        type=str,
        default=default_df,
        nargs="?",
        help="Path to the input Parquet file",
    )
    parser.add_argument(
        "--naf-version",
        type=str,
        default="naf_2025",
        help="Which NAF ruleset to apply (default: naf_2025)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run the process without saving outputs"
    )

    args = parser.parse_args()

    if not args.df_path:
        print("❌ Error: You must provide the path to the input Parquet file.")
        parser.print_help()
        sys.exit(1)
    # main(df_path, tag)
    main(args.df_path, args.naf_version, args.dry_run)
