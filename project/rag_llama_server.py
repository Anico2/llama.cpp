#!/usr/bin/env python3
"""
Mini RAG pipeline for llama.cpp server (stateless)
- Word document ingestion
- Token-based chunking
- FAISS vector store
- Retrieve top-k relevant chunks
- Send combined prompt to llama.cpp server
"""

from docx import Document
import tiktoken
import httpx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import json
import os

# -------------------------
# Config defaults
# -------------------------
DEFAULT_LLAMA_URL = "http://localhost:8080"
DEFAULT_CHUNK_SIZE = 15
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_TOP_K = 3
DEFAULT_SESSION_FILE = "rag_session.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# -------------------------
# Load Word document
# -------------------------
def load_docx_text(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

# -------------------------
# Chunk text by tokens
# -------------------------
def chunk_text(text: str, chunk_size: int, encoding_name: str):
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    for i in range(0, len(tokens), chunk_size):
        yield enc.decode(tokens[i:i+chunk_size])

# -------------------------
# Build or load FAISS index
# -------------------------
def build_or_load_index(chunks, session_file=DEFAULT_SESSION_FILE):
    if os.path.exists(session_file):
        data = json.load(open(session_file, "r"))
        index = faiss.read_index(data["index_file"])
        embeddings = np.array(data["embeddings"])
        return index, embeddings
    # Build index
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # Save index
    faiss.write_index(index, "faiss.index")
    json.dump({"index_file": "faiss.index", "embeddings": embeddings.tolist()}, open(session_file, "w"))
    return index, embeddings

# -------------------------
# Retrieve top-k chunks
# -------------------------
def retrieve_chunks(query, index, chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, DEFAULT_TOP_K)
    return [chunks[i] for i in I[0]]

# -------------------------
# Async llama-server query (stateless)
# -------------------------
async def query_llama(client, combined_prompt, url):
    resp = await client.post(f"{url}/completion", json={"prompt": combined_prompt})
    resp.raise_for_status()
    return resp.text

# -------------------------
# Main async workflow
# -------------------------
async def main_async(doc_path, user_prompt, llama_url, chunk_size, encoding):
    text = load_docx_text(doc_path)
    print(f"Loaded {len(text)} characters from document")

    chunks = list(chunk_text(text, chunk_size, encoding))
    print(f"Document split into {len(chunks)} chunks")

    index, embeddings = build_or_load_index(chunks)

    relevant_chunks = retrieve_chunks(user_prompt, index, chunks)
    combined_prompt = "\n".join(relevant_chunks) + "\n\nUser prompt: " + user_prompt

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        print("Querying llama.cpp server with retrieved context...\n")
        answer = await query_llama(client, combined_prompt, llama_url)
        print(answer)

# -------------------------
# Entry point
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Mini RAG llama.cpp server pipeline")
    parser.add_argument("doc_path", help="Path to Word document")
    parser.add_argument("prompt", help="Prompt/query")
    parser.add_argument("--url", default=DEFAULT_LLAMA_URL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING)
    args = parser.parse_args()

    import asyncio
    asyncio.run(main_async(
        args.doc_path,
        args.prompt,
        args.url,
        args.chunk_size,
        args.encoding
    ))

if __name__ == "__main__":
    main()
