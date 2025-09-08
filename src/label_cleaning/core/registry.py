"""
registry.py

Holds the central registry `RULES_REGISTRY`, a global list used to track all rule functions
registered via the `@rule` decorator. This allows dynamic filtering and application of rules
by name, tag, or description.
"""

from typing import Callable, List


class Rule:
    def __init__(
        self,
        func: Callable,
        name: str,
        tags: List[str],
        methods: List[str],
        description: str,
    ):
        self.func = func
        self.name = name
        self.tags = tags
        self.methods = methods
        self.description = description

    def apply(self, df, **kwargs):
        return self.func(df, **kwargs)


# Le registre global
RULES_REGISTRY: List[Rule] = []


def register_rule(func: Callable, name: str, tags: List[str], description: str = ""):
    rule = Rule(func, name, tags, description)
    RULES_REGISTRY.append(rule)
