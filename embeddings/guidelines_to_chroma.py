"""
Ingest pcs_guidelines_clean.txt into ChromaDB as a dedicated collection
with lightweight section metadata for improved retrieval during PCS coding.

Collection name: icd10pcs_guidelines
Persist dir: ./chroma_store
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple, Dict

import chromadb
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "pcs_guidelines_clean.txt")
PERSIST_DIR = os.path.join(ROOT, "chroma_store")
COLLECTION_NAME = "icd10pcs_guidelines"


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


HEADING_PAT = re.compile(r"^(Conventions|Medical and Surgical Section Guidelines|Obstetric Section Guidelines|Radiation Therapy Section Guidelines|New Technology Section Guidelines|Selection of Principal Procedure)\b", re.I)
SECTION_CODE_PAT = re.compile(r"^(?P<sec>[A-F])(\d+(?:\.\d+)?[a-z]?)\b")


def _chunk_guidelines(text: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Create simple chunks by paragraphs, carrying forward a best-effort
    section heading and guideline code marker (e.g., B3.4a) as metadata.
    Returns list of (chunk_text, metadata).
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[Tuple[str, Dict[str, str]]] = []

    current_title = ""
    current_marker = ""

    buf: List[str] = []

    def flush():
        nonlocal buf, current_title, current_marker
        if buf:
            body = "\n".join(buf).strip()
            if body:
                meta = {
                    "source": "pcs_guidelines_2024",
                    "title": current_title or "",
                    "marker": current_marker or "",
                }
                chunks.append((body, meta))
        buf = []

    for ln in lines:
        if not ln.strip():
            # paragraph boundary
            flush()
            continue

        if HEADING_PAT.match(ln):
            flush()
            current_title = ln.strip()
            continue

        m = SECTION_CODE_PAT.match(ln.strip())
        if m:
            flush()
            current_marker = m.group(0)
            # keep the line too as part of next chunk
            buf.append(ln)
            continue

        buf.append(ln)

    flush()
    return chunks


def build_and_save_index(docs: List[Document], collection_name: str, persist_dir: str):
    # Ensure consistent embedding model across collections
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    idx = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    storage_context.persist(persist_dir=persist_dir)
    return idx


def main():
    if not os.path.exists(IN_PATH):
        raise SystemExit(f"Input file not found: {IN_PATH}")

    raw = _load_text(IN_PATH)
    chunks = _chunk_guidelines(raw)
    docs: List[Document] = []
    for i, (txt, meta) in enumerate(chunks):
        # enforce small-ish chunks ~1-2k chars implicitly by paragraphing
        docs.append(Document(text=txt, metadata=meta, doc_id=f"pcs_guidelines_{i:05d}"))

    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"Building collection '{COLLECTION_NAME}' with {len(docs)} chunks…")
    build_and_save_index(docs, COLLECTION_NAME, PERSIST_DIR)
    print(f"✅ Done. Collection '{COLLECTION_NAME}' persisted in '{PERSIST_DIR}'.")


if __name__ == "__main__":
    main()
