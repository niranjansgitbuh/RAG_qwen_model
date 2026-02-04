import faiss
import pickle
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
from difflib import get_close_matches


DOMAIN_WORDS = {
    "loan", "loans", "personal", "interest", "emi", "tenure",
    "eligibility", "hdfc", "bank", "balance", "transfer",
    "principal", "rate", "charges", "repayment"
}


# ---------------------------
# Load FAISS index + metadata
# ---------------------------
index = faiss.read_index("hdfc.faiss")

with open("hdfc_texts.pkl", "rb") as f:
    texts = pickle.load(f)

print("FAISS index and texts loaded ✅")

# ---------------------------
# Load embedding model
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Spell correction (NEW)
# ---------------------------
spell = SpellChecker()

def correct_spelling(text):
    words = text.lower().split()
    corrected_words = []

    for word in words:
        # 1️⃣ Try fuzzy match with domain words FIRST
        domain_match = get_close_matches(word, DOMAIN_WORDS, n=1, cutoff=0.8)
        if domain_match:
            corrected_words.append(domain_match[0])
            continue

        # 2️⃣ Otherwise use spellchecker
        corrected = spell.correction(word)

        if corrected:
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)

    corrected_query = " ".join(corrected_words)

    if corrected_query != text:
        print(f"📝 Corrected query: {corrected_query}")

    return corrected_query

# ---------------------------
# Prompt Template: Query Rewriter
# ---------------------------
QUERY_REWRITE_PROMPT = """
You are a query understanding assistant.

Fix spelling mistakes.
Preserve domain terms like HDFC, loan, EMI, interest, tenure.
Rewrite the query clearly and concisely.

User query:
"{query}"

Rewritten query:
"""


# ---------------------------
# Retrieve relevant context
# ---------------------------
def retrieve_context(query, k=5):
    query = correct_spelling(query)  # ✅ ADD THIS LINE

    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, k)
    return [texts[i] for i in I[0]]


# ---------------------------
# Build prompt for Qwen
# ---------------------------
def build_prompt(query, context_chunks):
    context = "\n".join(context_chunks)

    return f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

# ---------------------------
# Ask Qwen via Ollama API
# ---------------------------
def ask_qwen(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:1.5b-instruct",  # ✅ keep YOUR installed model
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    if "response" in data:
        return data["response"]

    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    if "error" in data:
        return f"Ollama error: {data['error']}"

    return str(data)



# ---------------------------
# Run end-to-end RAG
# ---------------------------
if __name__ == "__main__":
    query = "bebefits of plersonal laon"

    # ✅ STEP 1: rewrite query using Qwen
    query = rewrite_query_with_qwen(query)

    # ✅ STEP 2: FAISS retrieval
    context_chunks = retrieve_context(query)

    # ✅ STEP 3: build answer prompt
    prompt = build_prompt(query, context_chunks)

    # ✅ STEP 4: generate answer
    answer = ask_qwen(prompt)

    print("\n🧠 Answer:\n")
    print(answer)


