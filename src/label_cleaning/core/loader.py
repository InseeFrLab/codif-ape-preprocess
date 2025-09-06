"""
loader.py

This module dynamically imports all Python modules found under the `rules/` package
using `pkgutil`. It ensures that all rules are loaded and registered in `RULES_REGISTRY`
at runtime.

Usage:
    load_rules()  # loads all rules from the rules/ directory
"""

import importlib
import pkgutil

import src.label_cleaning.rules as rules


def load_rules():
    for finder, name, ispkg in pkgutil.walk_packages(
        rules.__path__, rules.__name__ + "."
    ):
        importlib.import_module(name)
