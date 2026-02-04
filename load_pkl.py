import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Load PKL
with open("final_hdfc.pkl", "rb") as f:
    data = pickle.load(f)

print("PKL loaded ✅")

# Step 2: Extract text
def extract_text(obj):
    texts = []

    if isinstance(obj, dict):
        for _, v in obj.items():
            texts.extend(extract_text(v))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text(item))
    else:
        texts.append(str(obj))

    return texts

texts = list(set(extract_text(data)))
print(f"Text chunks: {len(texts)}")

# Step 3: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# 🔥 ADD THIS LINE (VERY IMPORTANT)
faiss.normalize_L2(embeddings)

print("Embeddings generated ✅")
print("Embedding shape:", embeddings.shape)


# Step 4: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS index created ✅")
print("Vectors in index:", index.ntotal)

# Step 5: Save FAISS index + metadata
faiss.write_index(index, "hdfc.faiss")

with open("hdfc_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("FAISS index & metadata saved ✅")

# Step 6: Query FAISS (proof it works)

query ="balance transfer benefits"
query_vector = model.encode([query])

# 🔥 ADD THIS LINE
faiss.normalize_L2(query_vector)

D, I = index.search(query_vector, k=5)

print("\nTop matching results:\n")
for idx in I[0]:
    print("-", texts[idx])

