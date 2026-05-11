import os
import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="health_knowledge")


def load_documents(directory):
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


load_documents("data/")


def search_knowledge_base(query: str) -> str:
    """Semantic search over obesity knowledge"""

    obesity_classes = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ]

    if query in obesity_classes:
        query = f"{query} BMI category definition obesity class"

    results = collection.query(query_texts=[query], n_results=3)
    formatted = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append(f"[Source: {metadata['source']}]\n{doc}")
    return "\n\n---\n\n".join(formatted)
