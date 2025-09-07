import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from qdrant_client import QdrantClient

# Assuming these modules exist in your project structure
from src.utils import io
from src.label_cleaning.utils import cleaning
from src.constants import (
    SENTENCE_MODEL_NAME, TEXTUAL_INPUTS_CLEANED,
    TEXTUAL_INPUTS,
    STEP1_RULE_PATTERNS,
    STEP2_RULE_PATTERNS,
    COLLECTION_NAME)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the dataset from the specified path.
    """
    print(f"📥 Loading dataset from: {path or 'default path'}")
    return io.download_data(path)


def get_sentence_model() -> SentenceTransformer:
    """
    Initializes and returns the Sentence-Transformer model, optimized for GPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    model = SentenceTransformer(
        SENTENCE_MODEL_NAME,
        device=str(device),
        model_kwargs={"torch_dtype": torch.float16 if device.type == "cuda" else torch.float32},
    )
    return model


def process_and_import_data(df: pd.DataFrame, client: QdrantClient):
    """
    Processes the DataFrame, computes embeddings, and imports them into Qdrant.
    """
    # 1. Identifier les textes uniques de toutes les colonnes
    unique_texts = set()
    for col in TEXTUAL_INPUTS_CLEANED:
        unique_texts.update(df[col].dropna().tolist())

    unique_texts_list = list(unique_texts)

    if not unique_texts_list:
        print("❌ No unique texts found for embedding. Aborting.")
        return

    # 2. Calculer les embeddings pour chaque texte unique
    model = get_sentence_model()
    BATCH_SIZE = 20_000
    print(f"⏳ Encoding {len(unique_texts_list)} unique documents with a batch size of {BATCH_SIZE}...")

    embeddings = model.encode(unique_texts_list, convert_to_numpy=True, batch_size=BATCH_SIZE)
    embeddings = normalize(embeddings, axis=1)

    # 3. Créer un mapping pour associer les textes à leurs embeddings
    text_to_embedding = {text: emb for text, emb in zip(unique_texts_list, embeddings)}

    # 4. Préparer les données pour l'importation dans Qdrant
    points_to_import = []
    ids = []
    texts = []

    point_id_counter = 0

    # Parcourir le DataFrame pour créer un point pour chaque ligne
    for index, row in df.iterrows():
        point_payload = {}
        point_vector = []

        for col in TEXTUAL_INPUTS_CLEANED:
            text = row[col]
            if pd.notna(text) and text in text_to_embedding:
                # Ajouter l'embedding au vecteur du point
                point_vector.extend(text_to_embedding[text].tolist())
                # Ajouter le texte au payload
                point_payload[col] = text

        if point_vector:
            # Créer un point Qdrant avec un ID unique, le vecteur combiné et le payload
            points_to_import.append(models.PointStruct(
                id=point_id_counter, 
                vector=point_vector, 
                payload=point_payload
            ))
            ids.append(point_id_counter)
            texts.append(point_payload) # Storing the payload itself for consistency
            point_id_counter += 1

    # 5. Importer dans Qdrant
    if points_to_import:
        print(f"✅ Encoding complete. Importing {len(points_to_import)} points into Qdrant...")
        client.upsert(collection_name=COLLECTION_NAME, points=points_to_import)
        print(f"✨ Successfully imported {len(points_to_import)} points into collection '{COLLECTION_NAME}'.")
    else:
        print("⚠️ No data to import. Check your TEXTUAL_INPUTS_CLEANED columns.")


if __name__ == "__main__":
    # Example usage:
    # Set this to a path or pass it via an argument
    input_data_path = os.getenv("INPUT_DATA_PATH", "default_dataset.csv")
    
    # Initialize Qdrant client
    qdrant_client = io.get_qdrant_client()
    
    # 1. Load the dataset
    df = load_dataset(input_data_path)
    
    # 2. Clean the dataset (assuming `clean_dataset` is available)
    df = cleaning.clean_dataset(
        df,
        TEXTUAL_INPUTS,
        TEXTUAL_INPUTS_CLEANED,
        STEP1_RULE_PATTERNS,
        STEP2_RULE_PATTERNS,
    )

    # 3. Ensure the collection exists
    vector_size = 384 * len(TEXTUAL_INPUTS_CLEANED)
    # Adjust vector size based on number of columns
    io.ensure_collection_exists(qdrant_client, collection_name=COLLECTION_NAME, vector_size=vector_size)

    # 4. Process data and import to Qdrant
    process_and_import_data(df, qdrant_client)