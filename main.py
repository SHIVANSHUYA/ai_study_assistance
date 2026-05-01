import pdfplumber
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
OPENROUTER_KEY = "ADD_OPENROUTER_API_KEY"
MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl:free"  # You can change model here
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------
# 1. Extract Text From PDF
# ---------------------------------------------
def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ---------------------------------------------
# 2. Chunk the PDF text
# ---------------------------------------------
def chunk_text(text, chunk_size=1200):
    chunks = []
    words = text.split()
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------
# 3. Embed the chunks
# ---------------------------------------------
def embed_chunks(chunks, model):
    return model.encode(chunks)


# ---------------------------------------------
# 4. Retrieve most relevant chunks
# ---------------------------------------------
def get_top_chunks(query, chunks, chunk_embeddings, model, top_k=3):
    query_embedding = model.encode([query])[0]

    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


# ---------------------------------------------
# 5. Query OpenRouter with retrieved context
# ---------------------------------------------
def ask_openrouter(question, context):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""
You should answer questions only using the given document context and if something isn’t found there, respond with “I don’t know” and “its not in dataset." if user asks again search online and other dataset models to answer appropriately for a class 11th student, Stay polite and helpful like an exam-prep assistant, and ask for clarification whenever the user’s question is unclear. If the user wants a PDF generated, gently refuse because you can’t do that, but if they want a deeper explanation of something that is in the dataset, you may explain it in simple words using outside knowledge. And whenever the user appreciates or thanks you, respond politely.

---CONTEXT---
{context}
---END CONTEXT---

Question: {question}
"""

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )

    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------
# MASTER FUNCTION
# ---------------------------------------------
def ask_pdf(pdf_path, question):
    print("🚀 Loading PDF...")
    text = extract_pdf_text(pdf_path)

    print("📦 Chunking text...")
    chunks = chunk_text(text)

    print("🧠 Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("🔍 Embedding chunks...")
    chunk_embeddings = embed_chunks(chunks, embedder)

    print("🎯 Retrieving best chunks...")
    best_chunks = get_top_chunks(question, chunks, chunk_embeddings, embedder)

    combined_context = "\n\n".join(best_chunks)

    print("🤖 Querying OpenRouter...")
    answer = ask_openrouter(question, combined_context)

    return answer


# ---------------------------------------------
# RUN EXAMPLE
# ---------------------------------------------

user_in = True
while user_in:
    if __name__ == "__main__":
        pdf_file = "./kebo108.pdf"
        user_question = str(input("Enter your question about the PDF: "))

        if user_question.lower() in ["exit", "quit"]:
            print("Exiting...")
            user_in = False

        else:
            result = ask_pdf(pdf_file, user_question)
            print("\n\nFINAL ANSWER:\n", result)

else:
    exit()
