"""
plots.py — Visualizations and analysis for NACE/WZ coding results

This module contains visualization and analysis functions to compare
classification outputs before and after rule application, as well as
across different matching methods (e.g., regex, fuzzy, hybrid).

Main features:
    - plot_distribution_before_after:
        Displays a histogram comparing the distribution of NACE/NAF codes
        before and after preprocessing.
    - plot_rule_report_comparison:
        Compares the number of changes made per code and per rule
        for different matching methods.
    - plot_methods_overlap:
        Analyzes the overlap of modified codes between matching methods.
    - plot_heatmap_code_method:
        Generates a heatmap showing the distribution of codes by method.
    - show_changed_labels_by_code:
        Lists, for each code, the labels that were changed, along with
        the method and rule applied.

Requirements:
    - Data paths are defined in constants/paths.py.
    - Input and output column names are defined in constants/inputs.py
      and constants/targets.py.

Example usage:
    >>> from plots import plot_distribution_before_after
    >>> plot_distribution_before_after()
"""

from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib_venn import venn2
from plotly.subplots import make_subplots

from src.utils.io import download_data
from src.constants.paths import URL_OUTPUT_NAF2025, URL_REPORT_NAF2025, URL_SIRENE4_NAF2025
from src.constants.targets import NACE_REV2_1_COLUMN

URL_SIRENE4_NAF_2025 ="s3://projet-ape/data/08112022_27102024/naf2025/artifacts/data_cleaning/raw_cleansed_report_regex.parquet"
URL_OUTPUT_NAF_2025 = "s3://projet-ape/extractions/domain_specific_cleaned/delta_report_20241027_sirene4_nace2025"


def plot_distribution_before_after(
    input_path=URL_SIRENE4_NAF2025, output_path=URL_OUTPUT_NAF2025
):
    """
    Plot the distribution of target column before and after cleansing.

    Parameters
    ----------
    input_path : str
        Path to the input dataset before cleansing.
    output_path : str
        Path to the dataset after cleansing.
    target_column : str
        Name of the target column to compare.
    """
    df_before = download_data(input_path).assign(dataset="Avant traitement")
    df_after = download_data(output_path).assign(dataset="Après traitement")

    df_concat = pd.concat([df_before, df_after], ignore_index=True)

    fig = px.histogram(
        df_concat,
        x=NACE_REV2_1_COLUMN,
        color="dataset",
        barmode="group",
        title="Distribution des codes avant/après traitement",
    )
    fig.show()


def plot_rule_report_comparison(
    base_report_path=URL_REPORT_NAF2025, methods=("regex", "fuzzy")
):
    """
    Compare the number of changes per code and per rule across different matching methods.

    This function loads report parquet files for each specified method,
    filters only modifications (_change_type == 'modification'),
    and displays:
      - A grouped bar chart of number of modified rows per code (liasse_numero)
      - A grouped bar chart of number of modified rows per rule

    Parameters
    ----------
    base_report_path : str
        Base path to the report parquet file (without method suffix).
    methods : tuple[str]
        List/tuple of matching method names to compare (suffixes in filenames).
    """
    dfs = []
    for method in methods:
        path = base_report_path.replace(".parquet", f"_{method}.parquet")
        df = download_data(path)
        df = df[df["_change_type"] == "modification"]  # 🔹 Only keep modifications
        df = df.assign(method=method)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # 🔹 Number of changes per code
    fig_code = px.bar(
        df_all.groupby(["APE_AFTER", "method"]).size().reset_index(name="count"),
        x="APE_AFTER",
        y="count",
        color="method",
        barmode="group",
        title="Number of modified rows per code and method",
    )
    fig_code.show()

    # 🔹 Number of changes per rule
    fig_rule = px.bar(
        df_all.groupby(["_log_rules_applied", "method"])
        .size()
        .reset_index(name="count"),
        x="_log_rules_applied",
        y="count",
        color="method",
        barmode="group",
        title="Number of modified rows per rule and method",
    )
    fig_rule.show()


def plot_methods_overlap_0(
    base_report_path=URL_REPORT_NAF2025,
    methods=("regex", "fuzzy"),
    target_column="libelle",
):
    """
    Compare two matching methods and visualize the overlap of changed labels.

    This function generates:
    1. A Venn diagram of unique modified labels for each method.
    2. Interactive Plotly tables showing labels unique to each method and those shared.

    Parameters
    ----------
    base_report_path : str
        Base path to the report parquet file (method suffix will be added).
    methods : tuple of str
        Two methods to compare (must match report file suffixes).
    target_column : str
        Column used for comparison (e.g., cleaned label name).
    """
    if len(methods) != 2:
        raise ValueError("Currently supports exactly two methods.")

    # Load and collect label sets
    sets = {}
    dfs = {}
    for method in methods:
        path = base_report_path.replace(".parquet", f"_{method}.parquet")
        df = download_data(path)
        dfs[method] = df
        sets[method] = set(df[target_column].dropna().unique())

    set_a, set_b = sets[methods[0]], sets[methods[1]]

    # ---------- Venn diagram ----------
    plt.figure(figsize=(6, 6))
    venn2(subsets=(set_a, set_b), set_labels=methods)
    plt.title(f"Overlap of changed '{target_column}' between methods")
    plt.show()

    # ---------- Data for interactive tables ----------
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    both = sorted(set_a & set_b)

    def make_table(labels, title):
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=[title], fill_color="lightgray", align="left"),
                    cells=dict(values=[labels], align="left"),
                )
            ]
        )
        fig.update_layout(width=500, height=400)
        return fig

    # Show interactive tables
    make_table(only_a, f"Unique to {methods[0]}").show()
    make_table(only_b, f"Unique to {methods[1]}").show()
    make_table(both, f"Shared between {methods[0]} and {methods[1]}").show()


def plot_methods_overlap(
    base_report_path=URL_REPORT_NAF2025,
    methods=("regex", "fuzzy"),
    target_column="libelle",
):
    """
    Visualize the overlap of target column values across multiple methods in a single interactive
    Plotly figure.

    For each method, loads the corresponding report (base_report_path suffixed by method),
    filters only rows with _change_type='modification', and compares the target_column values.

    Parameters
    ----------
    base_report_path : str
        Path to the base report parquet file.
    methods : tuple of str
        Names of methods to compare.
        The function will look for files base_report_path suffixed by _{method}.parquet
    target_column : str
        Name of the column containing the labels to compare (e.g., "libelle").
    """
    # 🔹 Load data and build labels per method
    labels_per_method = {}
    for method in methods:
        path = base_report_path.replace(".parquet", f"_{method}.parquet")
        df = download_data(path)
        df = df[df["_change_type"] == "modification"]
        labels_per_method[method] = set(df[target_column].dropna().unique())

    # 🔹 Build intersections for 2..k methods
    intersections = {}
    for r in range(2, len(methods) + 1):
        for combo in combinations(methods, r):
            combo_set = set.intersection(*(labels_per_method[m] for m in combo))
            if combo_set:
                intersections[combo] = combo_set

    # 🔹 Determine unique labels per method
    all_intersection_labels = (
        set().union(*intersections.values()) if intersections else set()
    )
    uniques = {m: list(labels_per_method[m] - all_intersection_labels) for m in methods}

    # 🔹 Layout: tables in subplots
    num_tables = len(methods) + len(intersections)
    fig = make_subplots(
        rows=1,
        cols=num_tables,
        specs=[[{"type": "table"}] * num_tables],
        horizontal_spacing=0.05,
    )

    # 🔹 Add unique tables
    for i, m in enumerate(methods):
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"Unique to {m}"], fill_color="lightgray", align="left"
                ),
                cells=dict(values=[uniques[m]], align="left"),
            ),
            row=1,
            col=i + 1,
        )

    # 🔹 Add intersections tables
    for j, (combo, labels) in enumerate(intersections.items(), start=len(methods)):
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"Shared: {', '.join(combo)}"],
                    fill_color="lightblue",
                    align="left",
                ),
                cells=dict(values=[list(labels)], align="left"),
            ),
            row=1,
            col=j + 1,
        )

    fig.update_layout(
        height=600,
        width=200 * num_tables,
        title_text="Overlap of Labels Across Methods",
    )
    fig.show()


def plot_methods_overlap_by_rule(
    base_report_path=URL_REPORT_NAF2025,
    methods=("regex", "fuzzy"),
    target_column="libelle",
):
    """
    Visualize the overlap of target column values across multiple methods **for each rule**.

    For each rule (_log_rules_applied value), loads the corresponding reports for each method,
    filters only rows with _change_type='modification', and compares the target_column values.

    Displays one Plotly figure per rule.
    """
    from itertools import combinations

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Charger toutes les méthodes
    dfs = {}
    for method in methods:
        path = base_report_path.replace(".parquet", f"_{method}.parquet")
        df = download_data(path)
        df = df[df["_change_type"] == "modification"]
        dfs[method] = df

    # Lister toutes les règles
    all_rules = sorted(
        set().union(
            *(df["_log_rules_applied"].dropna().unique() for df in dfs.values())
        )
    )

    for rule in all_rules:
        # Labels par méthode pour cette règle
        labels_per_method = {}
        for method in methods:
            subset = dfs[method][dfs[method]["_log_rules_applied"] == rule]
            labels_per_method[method] = set(subset[target_column].dropna().unique())

        # Intersections
        intersections = {}
        for r in range(2, len(methods) + 1):
            for combo in combinations(methods, r):
                combo_set = set.intersection(*(labels_per_method[m] for m in combo))
                if combo_set:
                    intersections[combo] = combo_set

        # Labels uniques
        all_intersection_labels = (
            set().union(*intersections.values()) if intersections else set()
        )
        uniques = {
            m: list(labels_per_method[m] - all_intersection_labels) for m in methods
        }

        # Création du subplot
        num_tables = len(methods) + len(intersections)
        fig = make_subplots(
            rows=1,
            cols=num_tables,
            specs=[[{"type": "table"}] * num_tables],
            horizontal_spacing=0.05,
        )

        # Uniques
        for i, m in enumerate(methods):
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=[f"Unique to {m}"], fill_color="lightgray", align="left"
                    ),
                    cells=dict(values=[uniques[m]], align="left"),
                ),
                row=1,
                col=i + 1,
            )

        # Intersections
        for j, (combo, labels) in enumerate(intersections.items(), start=len(methods)):
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=[f"Shared: {', '.join(combo)}"],
                        fill_color="lightblue",
                        align="left",
                    ),
                    cells=dict(values=[list(labels)], align="left"),
                ),
                row=1,
                col=j + 1,
            )

        fig.update_layout(
            height=600,
            width=200 * num_tables,
            title_text=f"Overlap of '{target_column}' for rule: {rule}",
        )
        fig.show()


def plot_heatmap_code_method(
    base_report_path=URL_REPORT_NAF2025, methods=("regex", "fuzzy")
):
    """Affiche une heatmap : lignes = codes, colonnes = méthodes,
    valeurs = nombre de modifications."""
    dfs = []
    for method in methods:
        path = base_report_path.replace(".parquet", f"_{method}.parquet")
        df = download_data(path).assign(method=method)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    pivot_table = (
        df_all.groupby(["APE_AFTER", "method"])
        .size()
        .reset_index(name="count")
        .pivot(index="APE_AFTER", columns="method", values="count")
        .fillna(0)
    )

    fig = px.imshow(
        pivot_table,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Heatmap du nombre de changements par code et par méthode",
    )
    fig.show()


def show_changed_labels_by_code(
    base_report_path=URL_REPORT_NAF2025,
    methods=("regex", "fuzzy"),
    label_column=None,
    max_examples=5,
):
    """
    Affiche pour chaque code : les libellés modifiés, la méthode et la règle.
    label_column : nom de la colonne texte d'origine (par ex. "libelle" ou TEXTUAL_INPUTS[0])
    max_examples : nombre max de libellés affichés par code/méthode/règle
    """
    from constants.inputs import TEXTUAL_INPUTS

    if label_column is None:
        label_column = TEXTUAL_INPUTS[0]

    dfs = []
    for method in methods:
        path = base_report_path.replace(".parquet", f"_{method}.parquet")
        df = download_data(path)
        df["method"] = method
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # On garde les colonnes pertinentes
    cols_to_keep = ["APE_AFTER", "_log_rules_applied", label_column, "method"]
    cols_present = [c for c in cols_to_keep if c in df_all.columns]
    df_all = df_all[cols_present]

    # Groupement par code/méthode/règle
    grouped = (
        df_all.groupby(["APE_AFTER", "method", "_log_rules_applied"])[label_column]
        .apply(lambda x: list(x.nunique())[:max_examples])
        .reset_index()
    )

    for _, row in grouped.iterrows():
        print(
            f"--- Code : {row['"APE_AFTER"']} |\
         Méthode : {row['method']} | Règle : {row['_log_rules_applied']}"
        )
        for label in row[label_column]:
            print(f"   • {label}")
        print()
