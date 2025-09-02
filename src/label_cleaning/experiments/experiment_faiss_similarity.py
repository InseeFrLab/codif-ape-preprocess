import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Libellés exemples: idée --> trouver le libellé moyen comme query
libelles = [
    "lmnp",
    "loueur en meuble non professionnel",
    "loueur bailleur non professionnel",
    "location meublee non professionnelle",
    "loueur meuble non professionnel",
    "loueurs en meubles non professionnels",
    "loueur en meubl non professionnel",
    "loueur en meubles non professionnel",
    "masseur kinésithérapeute",
    "masseur kinésithérapeute",
    "lmnp permanent",
    "lmnp saisonnier",
    "vendeur de yaourt",
    "VDI",
    "influenceur",
    "loueur d'appartement de vacances",
    "loueur de airbnb",
    "loueur de meublé saisonnier",
]

# Phrase requête à comparer
query = "lmnp"

# Choix du périphérique
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ⚙️ Charger un modèle pré-entraîné
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=str(device),  # <-- important pour inference GPU
)

# 🔄 Embedding + Normalisation
print("🔎 Encodage des libellés...")
start = time.time()
corpus_embeddings = model.encode(
    libelles, convert_to_numpy=True, show_progress_bar=False
)
corpus_embeddings = normalize(corpus_embeddings, axis=1)  # Normalisation L2
end = time.time()
print(f"✅ Embedding terminé en {end - start:.2f} sec")

# FAISS index avec Inner Product (cosine si normalisé)
index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# 🎯 Embedding et normalisation de la requête
query_vec = model.encode([query], convert_to_numpy=True)
query_vec = normalize(query_vec, axis=1)

# 🔍 Recherche
k = len(libelles) - 1  # top-k
D, indices = index.search(query_vec.astype(np.float32), k)

# 📊 Affichage des résultats
print(f'\n📝 Résultats pour : "{query}"')
for idx, score in tqdm(zip(indices[0], D[0])):
    print(f"  - {libelles[idx]:35s} → Similarité = {score:.2f}")

# ✅ Seuil suggéré pour test de variantes
THRESHOLD = 0.75
start = time.time()
matches = [(libelles[i], s) for i, s in zip(indices[0], D[0]) if s >= THRESHOLD]
end = time.time()
print(f"\n🎯 Candidats avec similarité ≥ {THRESHOLD}:")
print(f"✅ Matching des paires similaires terminés en {end - start:.4f} sec")
for label, score in matches:
    print(f"  - {label:35s} → {score:.4f}")
no_matches = [(libelles[i], s) for i, s in zip(indices[0], D[0]) if s < THRESHOLD]
print(f"\n🎯 Candidats avec similarité < {THRESHOLD}:")
for label, score in no_matches:
    print(f"  - {label:35s} → {score:.4f}")
