"""
decorators.py

This module defines the `@rule` decorator used to register transformation rules
in a central registry (`registry.py`). Each rule can be tagged and described.

Usage:
    @rule(name="my_rule", tags=["naf_2025"])
    def my_rule(df):
        ...
"""

from core.registry import register_rule


def rule(name=None, tags=None, description=""):
    def decorator(func):
        register_rule(
            func=func,
            name=name or func.__name__,
            tags=tags or [],
            description=description,
        )
        return func

    return decorator
