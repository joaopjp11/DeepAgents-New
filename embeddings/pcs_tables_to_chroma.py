# ICD-10-PCS Tables document generator for vector indexing
# This script reads the PCS tables CSV file, creates Document objects and indexes them in ChromaDB

import os
import pandas as pd
import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# LLM and embedding model configuration
Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash", temperature=0.1, api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

df = pd.read_csv('./data/icd10pcs_tables_2026.csv')

pcs_documents = []

# Convert each row of the DataFrame into a Document object
for _, row in df.iterrows():
    text = (
        f"Full Code: {row.get('full_code')}\n"
        f"Section: {row.get('section')}\n"
        f"Body System: {row.get('body_system')}\n"
        f"Operation: {row.get('operation')}\n"
        f"Operation Definition: {row.get('operation_definition')}\n"
        f"Body Part: {row.get('body_part_code')} - {row.get('body_part')}\n"
        f"Approach: {row.get('approach_code')} - {row.get('approach')}\n"
        f"Device: {row.get('device_code')} - {row.get('device')}\n"
        f"Qualifier: {row.get('qualifier_code')} - {row.get('qualifier')}"
    )

    metadata = {
        "full_code": row.get("full_code"),
        "section": row.get("section"),
        "body_system": row.get("body_system"),
        "operation": row.get("operation"),
        "operation_definition": row.get("operation_definition"),
        "body_part_code": row.get("body_part_code"),
        "body_part": row.get("body_part"),
        "approach_code": row.get("approach_code"),
        "approach": row.get("approach"),
        "device_code": row.get("device_code"),
        "device": row.get("device"),
        "qualifier_code": row.get("qualifier_code"),
        "qualifier": row.get("qualifier"),
    }

    pcs_documents.append(Document(text=text, metadata=metadata))

print(f"Total documents created: {len(pcs_documents)}")

# Initialize Chroma client for vector storage
print("\nInitializing Chroma client...")
chroma_client = chromadb.PersistentClient("./chroma_store")
chroma_collection = chroma_client.get_or_create_collection("icd10pcs_tables")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index the documents in the vector store
index = VectorStoreIndex.from_documents(
    pcs_documents,
    storage_context=storage_context,
    show_progress=True,
)

print(f"Total documents indexed: {len(pcs_documents)}")