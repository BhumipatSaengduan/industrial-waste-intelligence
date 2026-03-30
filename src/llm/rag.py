import chromadb
import uuid
from chromadb.utils import embedding_functions
from .prompts import RAG_PROMPT
from .insights import ollama_generate

CHROMA_PATH = 'data/chromadb'

def get_collection():
    """Get or create ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection('waste_documents', embedding_function=ef)

def add_documents(collection, documents, metadatas):
    """Add documents to ChromaDB collection."""
    ids = [str(uuid.uuid4()) for _ in documents]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(ids)

def rag_query(question, collection, n_results=3):
    """Full RAG pipeline: retrieve + generate."""
    results        = collection.query(query_texts=[question], n_results=n_results)
    retrieved_docs = results['documents'][0]
    retrieved_meta = results['metadatas'][0]

    context_parts = []
    for doc, meta in zip(retrieved_docs, retrieved_meta):
        context_parts.append(
            f"[{meta['type']} — {meta.get('date', meta.get('week_start', ''))}]\n{doc}")
    context = '\n\n'.join(context_parts)

    answer = ollama_generate(
        RAG_PROMPT.format(context=context, question=question),
        max_tokens=300
    )
    return {
        'question'      : question,
        'answer'        : answer,
        'retrieved_docs': len(retrieved_docs),
        'sources'       : [m['type'] for m in retrieved_meta]
    }
