
---

### **`plots.py`**

```python
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

Author: [Your Name or Team]
License: Apache 2.0
"""
```

---

### **`run_plots.py`**

```python
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

Author: [Your Name or Team]
License: Apache 2.0
"""
```

---

### **Internal doc — How to add a new visualization**

````markdown
# Adding a new visualization to the Codif-APE visualization suite

## Step 1 — Create the function in `plots.py`
Follow the existing structure:
- The function name should start with `plot_` if it outputs a chart,
  or `show_` if it outputs a table/list.
- Include a docstring describing the purpose, parameters, and return type.
- Use constants from `constants/` instead of hardcoding paths or column names.
- Use Plotly for charts; return the figure object (`fig`) so it can be reused.

Example:
```python
def plot_example_new_visual(df: pd.DataFrame) -> go.Figure:
    """
    Example: Plot distribution of example_column by method.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        plotly.graph_objects.Figure: The generated figure.
    """
    fig = px.bar(df, x="example_column", y="count", color="method")
    fig.show()
    return fig
````

## Step 2 — Import it in `run_plots.py`

At the top of `run_plots.py`:

```python
from plots import plot_example_new_visual
```

## Step 3 — Call it in the main execution block

In the `if __name__ == "__main__":` block:

```python
if __name__ == "__main__":
    # Existing plots...
    plot_distribution_before_after()
    plot_rule_report_comparison()

    # Your new plot
    plot_example_new_visual(df_output)
```

## Step 4 — Keep functions reusable

Avoid reading files directly inside your function unless it's meant
to be fully standalone. Prefer passing pre-loaded DataFrames as arguments.

## Step 5 — Test locally

Run:

```bash
cd src
uv run python -m visualisation.run_plot
```

Ensure your plot is generated without errors.

```
