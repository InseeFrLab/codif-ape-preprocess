"""
Central import hub for matching functions.
This allows other modules to simply do:
    from utils.matching import regex_mask, fuzzy_mask, similarity_mask
instead of importing from multiple files.

These functions are the low-level building blocks for matchers.
"""

from .matchers import MATCHERS

__all__ = ["MATCHERS"]
