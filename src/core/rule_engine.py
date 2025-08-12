import pandas as pd
from tqdm import tqdm

from .loader import load_rules
from .registry import RULES_REGISTRY


def apply_rules(training_data, tag, methods=None, methods_params=None):
    """
    Apply all rules tagged with `tag` to the dataset.

    Loads the registered rules, filters them by tag, applies each one in turn,
    and collects all audit journals into a single DataFrame.

    Args:
        training_data (pd.DataFrame): Data to process.
        tag (str): Tag used to select which rules to apply.
        methods (list[str] or None): Matching methods to inject into rules.
        methods_params (dict): global parameters for each method
            ex: {
                "regex": {"terms": [...], ...},
                "fuzzy": {"terms": [...], "threshold": 85},
                ...
            }

    Returns:
        tuple:
            pd.DataFrame: Updated dataset.
            pd.DataFrame: Combined audit journal of all applied rules.
    """

    print("🔍 Loading and filtering rules...")
    load_rules()
    rules_to_apply = [r for r in RULES_REGISTRY if tag in r.tags]
    print(f"🧩 {len(rules_to_apply)} rule(s) matched with tag '{tag}'")

    all_journals = []

    print("⚙️  Applying rules...")
    for rule in tqdm(rules_to_apply, desc="Processing rules", unit="rule"):
        df, journal = rule.apply(training_data, methods=methods, methods_params=methods_params)
        all_journals.append(journal)

    return df, pd.concat(all_journals, ignore_index=True)
