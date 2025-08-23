import pandas as pd
import torch
from rapidfuzz import fuzz
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from constants.thresholds import FUZZY_THRESHOLD, SIM_THRESHOLD
from constants.models import SENTENCE_MODEL_NAME

# Set to None initially and loaded only once by _get_model()
_MODEL: SentenceTransformer | None = None


def _get_model(name):
    global _MODEL
    if _MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {device}")

        dtype = "float16" if device.type == "cuda" else "float32"
        attn = "sdpa"  # fallback sûr, fonctionne sur CPU comme sur GPU

        _MODEL = SentenceTransformer(
            name,
            device=str(device),
            model_kwargs={"attn_implementation": attn, "torch_dtype": dtype}
        )
    return _MODEL


def regex_mask(series: pd.Series, pattern: str) -> pd.Series:
    """Vectorised regex match."""
    return series.str.contains(pattern, case=False, regex=True, na=False)


def fuzzy_mask(series: pd.Series, terms: list[str], threshold=FUZZY_THRESHOLD) -> pd.Series:
    """Mask where any term fuzzily matches above threshold."""
    return series.fillna("").map(
        lambda s: any(fuzz.token_sort_ratio(s, t) >= threshold for t in terms)
    )


def similarity_mask(
    series: pd.Series,
    terms: list[str],
    threshold=SIM_THRESHOLD,
    model_name=SENTENCE_MODEL_NAME
) -> pd.Series:
    """Mask where max cosine similarity to any term ≥ threshold."""
    model = _get_model(model_name)
    texts = series.fillna("").tolist()
    term_vecs = normalize(model.encode(terms,   convert_to_numpy=True), axis=1)
    text_vecs = normalize(model.encode(texts,    convert_to_numpy=True), axis=1)
    sims = (text_vecs @ term_vecs.T).max(axis=1)
    return pd.Series(sims >= threshold, index=series.index)
