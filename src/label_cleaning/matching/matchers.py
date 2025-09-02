"""
matchers.py

Maps human-readable matcher names to their implementation functions.

Selector module for all available matching functions in utils.
By importing from here, you ensure you only use the
officially exposed matching methods.
This keeps implementation details decoupled from usage.

Example:
    from matching.matchers import MATCHERS
    mask = MATCHERS["fuzzy"](series, threshold=80)
"""

# Import selected matchers amongst defined in utils.matching.__init__.py
from utils.matching import fuzzy_mask, regex_mask, similarity_mask

MATCHERS = {
    "regex": regex_mask,
    "fuzzy": fuzzy_mask,
    "similarity": similarity_mask,
}
