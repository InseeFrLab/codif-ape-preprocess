"""
file.py

Module providing utility functions for file name manipulation and handling,
such as extracting suffixes or modifying file paths.
"""


def get_suffix(methods):
    """Return a suffix string representing one or multiple matching methods."""
    if len(methods) == 1:
        return f"_{methods[0]}"
    else:
        return "_" + "_".join(sorted(methods))
