"""
Assign NAF 2025 codes for video maker.

Matching configuration and mask logic are delegated to utils/rules.py
for reusability. See:
    - build_matcher_kwargs
    - build_match_mask
"""

import numpy as np
import pandas as pd

from src.constants.inputs import TEXTUAL_INPUTS_CLEANED
from src.constants.targets import NACE_REV2_1_COLUMN

from src.label_cleaning.core.decorators import rule, track_changes
from src.label_cleaning.utils.rules import build_match_mask, build_matcher_kwargs, filter_methods


@rule(
    name="video_maker_assignment_2025",
    tags=["naf_2025"],
    description="Règle videaste version NAF 2025",
)
@track_changes(column=NACE_REV2_1_COLUMN)
def video_maker_5911G_2025(
    df: pd.DataFrame, methods=None, methods_params=None
) -> pd.DataFrame:
    methods = filter_methods(methods, exclude=["fuzzy", "similarity"])
    terms = ["youtuber",
             "youtube",
             "tiktoker",
             "tiktok",
             "streamer",
             "stream",
             "streaming",
             "dailymotion",
             "podcast",
             "vlog",
             "videaste",
             "contenus audiovisuels sur internet",
             "contenus audiovisuels montage video",
             "de contenus video",
             "montage vidéo",
             "production de video et de contenu en ligne",]

    matcher_kwargs = build_matcher_kwargs(methods, methods_params, terms)
    match_mask = build_match_mask(df, TEXTUAL_INPUTS_CLEANED, methods, matcher_kwargs)

    df[NACE_REV2_1_COLUMN] = np.where(match_mask, "5911G", df[NACE_REV2_1_COLUMN])
    return df, match_mask
