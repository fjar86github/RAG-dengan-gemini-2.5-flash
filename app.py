import streamlit as st
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai

# ======================
# GEMINI CLIENT
# ======================
#AIzaSyDxBoOLt-vikw1Qq5yNB9nLzEPKYEwelC0
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-2.5-flash"

# ======================
# EMBEDDING
# ======================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ======================
# SESSION STATE
# ======================
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None

# ======================
# PDF
# ======================
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)

# ======================
# CHUNK
# ======================
def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ======================
# VECTOR DB
# ======================
def build_index(chunks):
    vectors = embedder.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index

# ======================
# RETRIEVE
# ======================
def retrieve(query, index, chunks, k=3):
    q_vec = embedder.encode([query])
    _, ids = index.search(np.array(q_vec), k)
    return [chunks[i] for i in ids[0]]

# ======================
# UI
# ======================
st.title("📚 Simple RAG From Scratch")

file = st.file_uploader("Upload PDF", type=["pdf"])

if file:
    text = extract_text(file)
    chunks = chunk_text(text)

    st.session_state.chunks = chunks
    st.session_state.index = build_index(chunks)

    st.success(f"Processed {len(chunks)} chunks")

query = st.text_input("Ask question")

if query and st.session_state.index:
    docs = retrieve(query, st.session_state.index, st.session_state.chunks)
    context = "\n\n".join(docs)

    prompt = f"""
Jawab berdasarkan konteks berikut:

{context}

Pertanyaan: {query}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    st.subheader("Answer")
    st.write(response.text)

    st.subheader("Context")
    st.write(context)