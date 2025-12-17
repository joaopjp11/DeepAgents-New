# src/tools.py

import os
from typing import Literal, Optional, Dict, Any
from tavily import TavilyClient
from langchain_core.tools import tool
from app.config import settings
import os
from typing import List
import logging
import time
logger = logging.getLogger(__name__)

if not logger.handlers:
    # Keep it simple; inherit uvicorn level/handlers if present
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )


from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

from llama_index.llms.google_genai import GoogleGenAI  # for answer generation
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # matching the collection embeddings
from dotenv import load_dotenv
load_dotenv()
# Set up the LLM
# ------------ Eager inits (wrapped with logs so failures are clear) ------------
try:
    logger.info("Initializing GoogleGenAI LLM for LlamaIndex...")
    api_key = settings.GOOGLE_GENAI_API_KEY
    if not api_key:
        logger.warning("GOOGLE_GENAI_API_KEY is not set (env missing). LLM init may fail later.")
    Settings.llm = GoogleGenAI(
        model_name="gemini-2.5-flash",
        temperature=0.0,
        api_key=api_key,
    )
    logger.info("LLM initialized.")
except Exception as e:
    logger.exception("Failed to initialize GoogleGenAI LLM: %s", e)
    # Re-raise so you see this as the top error if it’s the cause
    raise

try:
    logger.info("Initializing HuggingFace embedding model (all-MiniLM-L6-v2) to match Chroma collections...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("Embedding model initialized (384 dimensions).")
except Exception as e:
    logger.exception("Failed to initialize HuggingFace embedding: %s", e)
    raise

_PERSIST_DIR = "chroma_store"
_COLLECTION_NAME = "icd10_tabular"
_COLLECTION_NAME_PARENTS="icd10_tabular_top_level"
_COLLECTION_NAME_PCS = "icd10pcs_tables"
_COLLECTION_NAME_PCS_GUIDELINES = "icd10pcs_guidelines"


__CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None
__INDICES: dict[str, VectorStoreIndex] = {}  # cache indices per collection

# Initialize Tavily client once
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def _get_chroma_client() -> chromadb.PersistentClient:
    global __CHROMA_CLIENT
    if __CHROMA_CLIENT is None:
        logger.info("_get_chroma_client: opening Chroma PersistentClient at %s", _PERSIST_DIR)
        if not os.path.isdir(_PERSIST_DIR):
            msg = (f"Chroma persist dir not found: {_PERSIST_DIR} "
                   f"(ensure you built the indexes first).")
            logger.error(msg)
            raise FileNotFoundError(msg)
        __CHROMA_CLIENT = chromadb.PersistentClient(
            path=_PERSIST_DIR,
            settings=ChromaSettings(allow_reset=False),
        )
        names = [c.name for c in __CHROMA_CLIENT.list_collections()]
        logger.info("_get_chroma_client: collections present: %s", names)
    return __CHROMA_CLIENT


def _get_index_for_collection(collection_name: str) -> VectorStoreIndex:
    """Open an existing Chroma collection by name (strict) and return a cached VectorStoreIndex."""
    if collection_name in __INDICES:
        return __INDICES[collection_name]

    client = _get_chroma_client()
    # strict: do NOT create
    logger.info("_get_index_for_collection: getting collection '%s'...", collection_name)
    collection = client.get_collection(collection_name)
    logger.info("_get_index_for_collection: collection found. (name=%s)", collection.name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    __INDICES[collection_name] = index
    return index


def _node_text(n) -> str:
    """
    Robustly extract the text content from a NodeWithScore or a BaseNode
    across LlamaIndex versions.
    """
    # If it's NodeWithScore, unwrap to the underlying node
    node_obj = getattr(n, "node", n)

    # Preferred: get_content (newer API)
    try:
        if hasattr(node_obj, "get_content"):
            return node_obj.get_content(metadata_mode="none")
    except Exception:
        pass

    # Fallback: get_text (older API)
    try:
        if hasattr(node_obj, "get_text"):
            return node_obj.get_text()
    except Exception:
        pass

    # Fallback: .text attr
    txt = getattr(node_obj, "text", None)
    if isinstance(txt, str):
        return txt

    return ""

@tool
def icd10_query(query: str) -> str:
    """
    Search two ICD-10 collections and return the TEXT content for each hit:
      - Top 3 from parents/top-level collection
      - Top 5 from main collection
    Deduplicates by 'code' (from metadata) while preserving order.
    """
    logger.info("icd10_query called | query=%r", query)
    t0 = time.perf_counter()

    parents_hits = []
    main_hits = []

    # Try parents/top-level (optional)
    try:
        parents_index = _get_index_for_collection(_COLLECTION_NAME_PARENTS)
        parents_hits = parents_index.as_retriever(similarity_top_k=3).retrieve(query)
        logger.info("icd10_query: parents retrieval ok; hits=%d", len(parents_hits))
    except Exception as e:
        logger.warning("icd10_query: parents retrieval skipped/failed: %s", e)

    # Main (required)
    main_index = _get_index_for_collection(_COLLECTION_NAME)
    main_hits = main_index.as_retriever(similarity_top_k=5).retrieve(query)
    logger.info("icd10_query: main retrieval ok; hits=%d", len(main_hits))

    dt = (time.perf_counter() - t0) * 1000
    logger.info("icd10_query: total retrieval time %.1f ms", dt)

    # Merge with dedup by metadata['code']
    merged = []
    seen_codes = set()

    def _add_hits(hits, label):
        for n in hits:
            meta = dict((getattr(n, "metadata", None) or getattr(getattr(n, "node", None), "metadata", None) or {}) )
            code = meta.get("code") or meta.get("name")
            if code and code in seen_codes:
                continue
            merged.append((label, n, code))
            if code:
                seen_codes.add(code)

    _add_hits(parents_hits, _COLLECTION_NAME_PARENTS)
    _add_hits(main_hits, _COLLECTION_NAME)

    out: List[str] = ["🔍 Retrieved passages (top 3 parents + top 5 main):"]
    if not merged:
        out.append(" - (no results)")
        return "\n".join(out)

    for i, (label, node, code) in enumerate(merged, start=1):
        header_bits = [f"{i:>2}. [{label}]"]
        if code:
            header_bits.append(f"Code: {code}")
        out.append(" ".join(header_bits))
        text = _node_text(node).strip()
        if text:
            out.append(text)
        else:
            out.append("(no text content found)")

    return "\n".join(out)


@tool
def icd10pcs_procedure_query(query: str) -> str:
    """
    Retrieve ICD-10-PCS procedure codes from PCS tables using semantic search.
    Returns top candidate full codes with their components.
    """
    logger.info("icd10pcs_procedure_query called | query=%r", query)
    t0 = time.perf_counter()

    index = _get_index_for_collection(_COLLECTION_NAME_PCS)

    hits = index.as_retriever(similarity_top_k=15).retrieve(query)

    dt = (time.perf_counter() - t0) * 1000
    logger.info("PCS retrieval time %.1f ms", dt)

    if not hits:
        return "❌ No ICD-10-PCS procedure codes found."

    out = ["🧠 ICD-10-PCS candidate procedure codes:\n"]

    seen = set()
    for i, n in enumerate(hits, start=1):
        meta = getattr(n.node, "metadata", {}) or {}

        full_code = meta.get("full_code")
        if not full_code or full_code in seen:
            continue
        seen.add(full_code)

        out.append(
            f"{i}. ✅ Code: {full_code}\n"
            f"   Section: {meta.get('section')}\n"
            f"   Body System: {meta.get('body_system')}\n"
            f"   Operation: {meta.get('operation')}\n"
            f"   Body Part: {meta.get('body_part')}\n"
            f"   Approach: {meta.get('approach')}\n"
            f"   Device: {meta.get('device')}\n"
            f"   Qualifier: {meta.get('qualifier')}\n"
        )

    return "\n".join(out)


@tool
def icd10pcs_guidelines_query(query: str) -> str:
    """
    Retrieve passages from the ICD-10-PCS Official Guidelines collection.
    Returns top 5 snippets with marker/title when available.
    """
    logger.info("icd10pcs_guidelines_query called | query=%r", query)
    t0 = time.perf_counter()

    try:
        index = _get_index_for_collection(_COLLECTION_NAME_PCS_GUIDELINES)
    except Exception as e:
        logger.warning("Guidelines collection not available: %s", e)
        return "❌ Guidelines collection not found. Run embeddings/guidelines_to_chroma.py first."

    hits = index.as_retriever(similarity_top_k=5).retrieve(query)
    dt = (time.perf_counter() - t0) * 1000
    logger.info("Guidelines retrieval time %.1f ms", dt)

    if not hits:
        return "(no guideline passages found)"

    out = ["📘 ICD-10-PCS Guidelines hits:"]
    for i, n in enumerate(hits, start=1):
        meta = getattr(n.node, "metadata", {}) or {}
        marker = meta.get("marker") or ""
        title = meta.get("title") or ""
        header = f"{i}. {marker} — {title}".strip(" —")
        out.append(header)
        text = _node_text(n).strip()
        if text:
            out.append(text)
    return "\n".join(out)