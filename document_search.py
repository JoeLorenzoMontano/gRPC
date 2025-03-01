import sys
import argparse
from langchain_ollama import OllamaEmbeddings
from config import ollama_config, vector_db_config
from vector_store import get_vector_store

def generate_query_embedding(query_text):
    """Generate an embedding for the query text"""
    embedding_function = OllamaEmbeddings(
        base_url=ollama_config.base_url,
        model=ollama_config.model
    )
    return embedding_function.embed_query(query_text)

def search_documents(query, top_k=5, vector_store_type=None):
    """Search for documents similar to the query"""
    # Generate embedding for the query
    query_embedding = generate_query_embedding(query)
    
    # Get the vector store
    vector_store = get_vector_store(vector_store_type)
    
    # Search for similar documents
    results = vector_store.search(query_embedding, top_k=top_k)
    
    return results

def format_results(results):
    """Format search results for display"""
    formatted = []
    for i, result in enumerate(results):
        formatted.append(f"\n===== Result {i+1} [Score: {result['score']:.4f}] =====")
        formatted.append(f"ID: {result['id']}")
        if result.get('metadata'):
            formatted.append("Metadata:")
            for k, v in result['metadata'].items():
                formatted.append(f"  {k}: {v}")
        
        # Format the document content (truncate if too long)
        doc = result.get('document', '')
        if len(doc) > 500:
            doc = doc[:500] + "..."
        formatted.append(f"\nContent:\n{doc}")
        formatted.append("=" * 50)
    
    return "\n".join(formatted)

def main():
    parser = argparse.ArgumentParser(description="Search for documents in the vector store")
    parser.add_argument("query", help="The query text to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--store", choices=["chromadb", "faiss"], 
                        default=vector_db_config.vector_store,
                        help="Vector store to use for search")
    args = parser.parse_args()
    
    # Search for documents
    print(f"Searching for: '{args.query}' using {args.store.upper()}...")
    results = search_documents(args.query, args.top_k, args.store)
    
    # Print results
    if results:
        print(f"\nFound {len(results)} results:")
        print(format_results(results))
    else:
        print("No matching documents found.")
    
if __name__ == "__main__":
    main()