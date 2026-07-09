import pandas as pd
from tqdm import tqdm

from .loader import load_rules
from .registry import RULES_REGISTRY


def apply_rules(training_data, tag, methods=None, methods_params=None):
    """
    Apply all rules tagged with `tag` to the dataset.

    - Les rules de modification sont appliquées en premier.
    - Les rules de création sont appliquées ensuite.
    - Les rules de suppression sont appliquées en dernier.
    - Le journal final concatène d'abord les modifications, puis les créations,
    puis les suppressions.

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
            pd.DataFrame: Combined journal of modifications.
            pd.DataFrame: Combined journal of creations.
            pd.DataFrame: Combined journal of deletions.
    """

    print("🔍 Loading and filtering rules...")
    load_rules()
    rules_to_apply = [r for r in RULES_REGISTRY if tag in r.tags]
    print(f"🧩 {len(rules_to_apply)} rule(s) matched with tag '{tag}'")

    mods_journals = []
    create_journals = []
    delete_journals = []

    # ⚙️ Appliquer les règles
    print("⚙️  Applying rules...")
    for rule in tqdm(rules_to_apply, desc="Rule applying", unit="rule"):
        result = rule.apply(training_data, methods=methods, methods_params=methods_params)

        if isinstance(result, tuple) and len(result) == 2:
            training_data, journal = result
            if not journal.empty:
                change_type = journal["_change_type"].iloc[0]
                if change_type == "modification":
                    print(f"🔄 Rows updated by rule {rule.name} i.e {rule.description}")
                    mods_journals.append(journal)
                elif change_type == "creation":
                    print(f"🆕 Rows added by rule {rule.name} i.e {rule.description}")
                    create_journals.append(journal)
                elif change_type == "deletion":
                    print(f"🗑️  Rows removed by rule {rule.name} i.e {rule.description}")
                    delete_journals.append(journal)

    # ⏬ Concat journals: modifications first, creations second, deletions last
    all_journals = mods_journals + create_journals + delete_journals
    final_journal = pd.concat(all_journals, ignore_index=True) if all_journals else pd.DataFrame()
    mods_journals = pd.concat(mods_journals, ignore_index=True) if mods_journals else pd.DataFrame()
    create_journals = pd.concat(create_journals, ignore_index=True) if create_journals else pd.DataFrame()
    delete_journals = pd.concat(delete_journals, ignore_index=True) if delete_journals else pd.DataFrame()
    return training_data, final_journal, mods_journals, create_journals, delete_journals
