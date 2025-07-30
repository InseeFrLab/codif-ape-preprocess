from typing import Callable, List


class Rule:
    def __init__(
        self, func: Callable, name: str, tags: List[str], description: str = ""
    ):
        self.func = func
        self.name = name
        self.tags = tags
        self.description = description

    def apply(self, df):
        return self.func(df)


# Le registre global
RULE_REGISTRY: List[Rule] = []


def register_rule(func: Callable, name: str, tags: List[str], description: str = ""):
    rule = Rule(func, name, tags, description)
    RULE_REGISTRY.append(rule)
