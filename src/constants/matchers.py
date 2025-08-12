from .thresholds import FUZZY_THRESHOLD, SIM_THRESHOLD
from .models import SENTENCE_MODEL_NAME

# IMPORTANT: keys here must correspond to those in matching.matchers.MATCHERS
DEFAULT_METHOD_PARAMS = {
    "fuzzy": {
        "threshold": FUZZY_THRESHOLD,
        "terms": [],
    },
    "similarity": {
        "threshold": SIM_THRESHOLD,
        "terms": [],
        "sentence_model": SENTENCE_MODEL_NAME
    },
    "regex": {
        "pattern": "",
    },
}
