"""In-memory Chroma RAG over data/*.md, indexed lazily on first query."""

import os

import chromadb

from insightwellness_ai.api.schema import CLASS_MAPPING

_collection = None


def _load_documents(collection, directory):
    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(directory):
        if not filename.endswith(".md"):
            continue

        with open(os.path.join(directory, filename)) as f:
            content = f.read()

        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk": i})
            ids.append(f"{filename}_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Loaded {len(documents)} chunks from {directory}")


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.Client()
        _collection = client.get_or_create_collection(name="health_knowledge")
        _load_documents(_collection, os.environ.get("KNOWLEDGE_DIR") or "data/")
    return _collection


def search_knowledge_base(query: str) -> str:
    """Semantic search over obesity knowledge"""

    if query in set(CLASS_MAPPING.values()):
        query = f"{query} BMI category definition obesity class"

    results = _get_collection().query(query_texts=[query], n_results=3)
    formatted = []
    for doc, metadata in zip(
        results["documents"][0], results["metadatas"][0], strict=True
    ):
        formatted.append(f"[Source: {metadata['source']}]\n{doc}")
    return "\n\n---\n\n".join(formatted)
