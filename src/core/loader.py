import importlib
import pkgutil

import rules


def load_rules_from_package():
    for _, module_name, is_pkg in pkgutil.iter_modules(rules.__path__):
        importlib.import_module(f"rules.{module_name}")
