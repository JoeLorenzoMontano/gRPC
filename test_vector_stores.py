import argparse
import os
from langchain_ollama import OllamaEmbeddings
from vector_store import get_vector_store, ChromaDBStore, FAISSStore
from config import ollama_config, vector_db_config, storage_config

def generate_embeddings(texts):
    """Generate embeddings for a list of texts"""
    embedding_function = OllamaEmbeddings(
        base_url=ollama_config.base_url,
        model=ollama_config.model
    )
    return embedding_function.embed_documents(texts)

def test_add_and_search(store_type="chromadb"):
    """Test adding documents and searching in the vector store"""
    print(f"\n===== Testing {store_type.upper()} vector store =====")
    
    # Initialize vector store
    vector_store = get_vector_store(store_type)
    
    # Sample documents
    documents = [
        "FAISS is an efficient similarity search library developed by Facebook Research.",
        "ChromaDB is a vector database designed for storing and searching embeddings.",
        "Retrieval-Augmented Generation (RAG) combines retrieval systems with text generation.",
        "Vector embeddings represent text as points in a high-dimensional space.",
        "Document chunking is the process of splitting large documents into smaller pieces for processing."
    ]
    
    # Generate IDs
    doc_ids = [f"test_{i}" for i in range(len(documents))]
    
    # Sample metadata
    metadatas = [
        {"source": "meta_research", "topic": "vector_search"},
        {"source": "chroma_docs", "topic": "vector_database"},
        {"source": "research_paper", "topic": "rag"},
        {"source": "lecture_notes", "topic": "embeddings"},
        {"source": "documentation", "topic": "text_processing"}
    ]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(documents)
    
    # Add documents
    print(f"Adding {len(documents)} documents to {store_type}...")
    vector_store.add_documents(doc_ids, documents, embeddings, metadatas)
    
    # Test search
    print("\nTesting search...")
    
    # Test queries
    test_queries = [
        "vector database technology",
        "document chunking for RAG",
        "retrieval augmented generation"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        # Generate query embedding
        query_embedding = generate_embeddings([query])[0]
        
        # Search
        results = vector_store.search(query_embedding, top_k=2)
        
        # Print results
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  Result {i+1} [Score: {result['score']:.4f}]")
            print(f"  ID: {result['id']}")
            if result.get('metadata'):
                print(f"  Metadata: {result['metadata']}")
            print(f"  Document: {result.get('document', '')[:100]}...")
    
    # Test document retrieval
    print("\nTesting document retrieval...")
    doc = vector_store.get_document(doc_ids[0])
    if doc:
        print(f"Retrieved document: {doc['id']}")
        print(f"Content: {doc.get('document', '')[:50]}...")
    else:
        print("Document retrieval failed")

def main():
    parser = argparse.ArgumentParser(description="Test vector stores implementation")
    parser.add_argument("--store", choices=["chromadb", "faiss", "both"], 
                       default="both", help="Vector store to test")
    args = parser.parse_args()
    
    # Ensure vector store directory exists for FAISS
    os.makedirs(os.path.dirname(vector_db_config.faiss_index_path), exist_ok=True)
    
    # Run tests
    if args.store == "both" or args.store == "chromadb":
        test_add_and_search("chromadb")
    
    if args.store == "both" or args.store == "faiss":
        test_add_and_search("faiss")
    
    print("\nTests completed.")

if __name__ == "__main__":
    main()