# Codif-APE Preprocessing Toolkit 📊

Tired of manual **data editing** for NACE classification ? This repository provides a simple, reproducible, and open-source framework for **domain-specific rule-based data cleansing** and **preprocessing**. It's designed to significantly improve the quality and consistency of your training data for NACE classification models.

## Why this project? 🤝

### What challenges ? 🚧

Official statistics rely on data quality. In production, this often means addressing specific business constraints or correcting data that is critical for training accurate models. 
> **Data cleansing**, the process of ensuring that the extracted data is accurate, complete, and consistent—often involves labor-intensive processes that require domain expertise, as detecting and correcting (or removing) corrupt or in- accurate records from a dataset, is a crucial step in the data preparation pipeline. High-quality data is essential for producing reliable and accurate insights in data- driven research and applications. Traditionally, data cleansing has been handled using a combination of rule-based methods, statistical techniques, and manual interventions.
> <cite>(**Domain-specific data gathering and exploitation**: Nabil Moncef B., Davide Buscaldi, Leo Liberti) </cite> 

### What we offer ? ⭐

This toolkit addresses the challenge of **data cleansing** and **preprocessing** by providing a generic and reusable framework for **NLP**.  
Our approach provides a flexible, maintainable alternative to hard-coded expert systems. The pipeline generates a corrected dataset and a transparent log table, documenting every change and the rule applied, ensuring full auditability. It also supports storage solutions compatible with S3 for seamless data handling.

This preprocessing stack is particularly compatible with any [Torch classifiers](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/README.md), making it a standard component of any machine learning pipeline for business classification. It supports any NACE classification system, including country-specific level 5 nomenclatures. 

It also supports storage solutions compatible with S3 for seamless data handling.

## Business Rules Manager 📜

At the core is a transparent Business Rules Manager. The rules are not a black box—they are a configurable layer that can be inspected and adapted. This framework allows you to implement and combine different matching methods (e.g., keyword searches, regular expressions) to compensate for the weaknesses of a single approach.

The preprocessing pipeline not only generates a corrected dataset but also produces an associated log table. This table documents every change made and specifies the exact rule applied to each line, providing full transparency and auditability for every data point.

Here's a high-level view of how the rules are managed:

```bash
rules/*.py
  ↓
@rule(...)  →  src/core/registry.py (rule callable by register_rule) 
  ↓
src/core/loader.py (imports rules)
  ↓
src/core/rule_engine.py (retrieves rules, registers tag selection and applies them to data)  
  ↓
src/main.py (orchestrates the process)
```

## Quick Start 🧪

Here’s a quick test on a sample dataset with a simple rule

Some pre-requisites:
- training data path in ```constants/path.py```. Replace ```s3://``` with ```./``` or don't use PREFIX if you want to load locally. 
```Python
PREFIX = "s3://your-prefix/"
URL_RAW_DATA_NAF2025 = PREFIX + "s3://your-prefix/data/extracted_data.parquet")
URL_CLEANED_OUTPUT_NAF2025 = PREFIX + "full_dataset_cleaned_data.parquet"
URL_REPORT_OUTPUT_NAF2025 = PREFIX + "delta_report_rules_cleansing.parquet"
```
- textual inputs names in ```constants/inputs.py ``` 
```Python
TEXTUAL_INPUTS = ["hauptwirtschaftstätigkeit"]
```
- targets labels names in ```constants/targets.py ```
```Python
NACE_REV2_COLUMN = "WZ_2008"
NACE_REV2_1_COLUMN = "WZ_2025"
```

1. Create a rule file: ```my_rules.py ```
If 'nace' is the target label in your training dataset

```Python
import ...
from constants.inputs import TEXTUAL_INPUTS_CLEANED
from constants.targets import NACE_REV2_1_COLUMN

@rule(
    name="rental_car_services_match",
    tags=["wz_2025"],
    description="Assigns a NACE code for passenger car rental.",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def assign_nace_code_for_rental_cars(df: pd.DataFrame, methods=None, methods_params=None) -> pd.DataFrame:

    """Applies a regex-based rule to identify rental activities."""

    terms = [
        "Autovermietung",
        "Mietwagen",
        "touristische Mietwagen",
        # Add other terms here
    ]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    text_match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(mask, "77.11.1", df[NACE_REV2_1_COLUMN])

    return df
```

2. Run the script:

```Bash
uv run python src/main.py --naf_version "wz_2025" --methods ["regex"]
```

**Get Started quickly** 🚀

Our dependency management is powered by uv—an extremely fast Python package manager written in Rust. It simplifies environment setup and ensures every collaborator works with an identical, dependable environment.

1. Install uv
```Bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone & Sync
```Bash
git clone git@github.com:InseeFrLab/codif-ape-preprocess.git
cd codif-ape-preprocess
uv sync
```
3. Run (--dry-run to run without saving data)
```Bash
uv run python src/main.py --dry-run
```
**Execution Environment** 💻

This repository runs perfectly in a local environment. For users of the datalab SSP Cloud, using a service like [Onyxia](https://www.onyxia.sh/) offers additional benefits. The VS Code service in the SSP Cloud's catalog comes with uv pre-installed, and the platform provides seamless integration with S3-compatible storage for efficient data handling.

**Contribute** 🙏

This is an open-source project aimed at standardizing statistical practices across Europe and beyond. We believe in the power of collaboration and welcome contributions, feedback, and partnerships from other statistical institutes worldwide. Feel free to open an issue or submit a pull request!

**License** 📝

This project is licensed under the Apache-2.0 license.4