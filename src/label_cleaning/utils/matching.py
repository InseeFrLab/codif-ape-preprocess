# enable cudf pandas acceleration on gpu
# import cudf
import pandas as pd
import torch
import torch.nn as nn
from pandarallel import pandarallel
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from src.constants.models import BATCH_SIZE, SENTENCE_MODEL_NAME
from src.constants.thresholds import FUZZY_THRESHOLD, SIM_THRESHOLD

# Set to None initially and loaded only once by _get_model()
_MODEL: SentenceTransformer | None = None

# Initialize parallelization of pandas df
pandarallel.initialize(nb_workers=30, verbose=2, use_memory_fs=False)


def _get_model(model_name: str) -> nn.Module:
    """Load model with automatic multi-GPU support."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(model_name)
        if torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        else:
            print(f"🚀 Using device: {_DEVICE}")
        _MODEL = model.to(_DEVICE)
    return _MODEL


def regex_mask(series: pd.Series, pattern: str) -> pd.Series:
    """Vectorised regex match."""
    # series = cudf.from_pandas(series)
    return series.str.contains(pattern, case=False, regex=True, na=False)


def fuzzy_mask(
    series: pd.Series, terms: list[str], threshold=FUZZY_THRESHOLD
) -> pd.Series:
    """Mask where any term fuzzily matches above threshold."""
    # series = cudf.from_pandas(series)
    return series.fillna("").parallel_map(
        lambda s: any(fuzz.QRatio(s, t) >= threshold for t in terms)
    )


def similarity_mask(
    series: pd.Series,
    terms: list[str],
    threshold: float = SIM_THRESHOLD,
    model_name: str = SENTENCE_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> pd.Series:
    """Mask where max cosine similarity to any term ≥ threshold."""
    model = _get_model(model_name)
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    texts = series.fillna("").tolist()

    term_vecs = base_model.encode(
        terms, convert_to_tensor=True, batch_size=batch_size, normalize_embedding=True
    ).to(_DEVICE)

    text_vecs = base_model.encode(
        texts, convert_to_tensor=True, batch_size=batch_size, normalize_embedding=True
    ).to(_DEVICE)

    sims = (text_vecs @ term_vecs.T).max(dim=1).values
    torch.cuda.empty_cache()
    return pd.Series(sims.cpu().numpy() >= threshold, index=series.index)
