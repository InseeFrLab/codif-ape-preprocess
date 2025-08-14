"""
run_plots.py — Centralized execution script for visualizations

This script imports the functions defined in plots.py and executes them
in a logical order to produce a set of visualizations that:
    - Compare the distribution of codes before and after preprocessing.
    - Measure the impact of rules per code and per matching method.
    - Analyze overlap between matching methods.
    - Explore, for each code, which labels were modified, by which method,
      and under which rule.

Workflow:
    1. Load datasets defined in constants/paths.py.
    2. Sequentially call visualization functions from plots.py.
    3. Generate interactive Plotly charts or summary tables in the console.

Usage:
    From a terminal, run:
        uv run python run_plots.py
    (add --naf_version or other parameters if needed)
"""

from visualisation.plots import (
    plot_distribution_before_after,
    plot_rule_report_comparison,
    plot_methods_overlap,
    plot_methods_overlap_by_rule,
    plot_heatmap_code_method,
    show_changed_labels_by_code
)


if __name__ == "__main__":
    print("📊 Distribution of all before/after")
    plot_distribution_before_after()

    print("📊 Comparaison par code et règle")
    plot_rule_report_comparison(methods=("regex", "fuzzy", "similarity"))

    print("📊 Recouvrement des méthodes")
    plot_methods_overlap(methods=("regex", "fuzzy", "similarity"))

    print("📊 Recouvrement des méthodes par rules")
    plot_methods_overlap_by_rule(methods=("regex", "fuzzy", "similarity"))

    print("📊 Heatmap codes vs méthodes")
    plot_heatmap_code_method(methods=("regex", "fuzzy", "similarity"))

    print("📜 Libellés modifiés par code / méthode / règle")
    show_changed_labels_by_code(methods=("regex", "fuzzy", "similarity"), max_examples=5)
