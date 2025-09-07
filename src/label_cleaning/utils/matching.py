import pandas as pd
import torch
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from pandarallel import pandarallel

from src.constants.models import SENTENCE_MODEL_NAME, BATCH_SIZE
from src.constants.thresholds import FUZZY_THRESHOLD, SIM_THRESHOLD

# Set to None initially and loaded only once by _get_model()
_MODEL: SentenceTransformer | None = None

# Initialize parallelization of pandas df
pandarallel.initialize(nb_workers=30, verbose=2, use_memory_fs=False)


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
            model_kwargs={"attn_implementation": attn, "torch_dtype": dtype},
        )
    return _MODEL


def regex_mask(series: pd.Series, pattern: str) -> pd.Series:
    """Vectorised regex match."""
    return series.str.contains(pattern, case=False, regex=True, na=False)


def fuzzy_mask(
    series: pd.Series, terms: list[str], threshold=FUZZY_THRESHOLD
) -> pd.Series:
    """Mask where any term fuzzily matches above threshold."""
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
    texts = series.fillna("").tolist()
    term_vecs = model.encode(terms,
                             convert_to_tensor=True,
                             batch_size=batch_size,
                             normalize_embedding=True).to_device()
    text_vecs = model.encode(texts,
                             convert_to_tensor=True,
                             batch_size=batch_size,
                             normalize_embedding=True).to_device()
    sims = (text_vecs @ term_vecs.T).max(axis=1)
    torch.cuda.empty_cache()
    return pd.Series(sims.cpu().numpy() >= threshold, index=series.index)
