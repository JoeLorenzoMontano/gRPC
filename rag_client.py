import argparse
from langchain_ollama import OllamaEmbeddings
from config import ollama_config, vector_db_config
from vector_store import get_vector_store
from ollama_client import OllamaClient

class RAGClient:
    """Retrieval-Augmented Generation client"""
    
    def __init__(self, vector_store_type=None):
        """Initialize the RAG client"""
        self.embedding_function = OllamaEmbeddings(
            base_url=ollama_config.base_url,
            model=ollama_config.model
        )
        self.vector_store = get_vector_store(vector_store_type)
        self.ollama_client = OllamaClient(
            base_url="http://localhost:3001",  # Using OpenWebUI API
            model="deepseek-r1:8b"  # You can configure this as needed
        )
    
    def generate_query_embedding(self, query_text):
        """Generate an embedding for the query text"""
        return self.embedding_function.embed_query(query_text)
    
    def retrieve_documents(self, query, top_k=5):
        """Retrieve relevant documents based on the query"""
        query_embedding = self.generate_query_embedding(query)
        return self.vector_store.search(query_embedding, top_k=top_k)
    
    def format_context(self, results):
        """Format retrieved documents as context for generation"""
        context_parts = []
        
        for i, result in enumerate(results):
            # Add document content
            context_parts.append(f"DOCUMENT {i+1}:\n{result.get('document', '')}")
            
            # Add metadata if available
            if result.get('metadata'):
                meta_str = ", ".join([f"{k}: {v}" for k, v in result['metadata'].items()
                                    if k not in ['embedding_idx', 'chunk_index']])
                context_parts.append(f"[Source: {meta_str}]\n")
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query, context):
        """Generate an answer to the query using the provided context"""
        return self.ollama_client.generate_response(context, query)
    
    def rag_query(self, query, top_k=5):
        """Perform a complete RAG query"""
        # Step 1: Retrieve relevant documents
        results = self.retrieve_documents(query, top_k=top_k)
        
        if not results:
            return "No relevant documents found to answer your query."
        
        # Step 2: Format documents as context
        context = self.format_context(results)
        
        # Step 3: Generate answer based on retrieved documents
        answer = self.generate_answer(query, context)
        
        return answer

def main():
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation using vector search")
    parser.add_argument("query", help="The query to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--store", choices=["chromadb", "faiss"], 
                        default=vector_db_config.vector_store,
                        help="Vector store to use for retrieval")
    parser.add_argument("--verbose", action="store_true", help="Show debug information")
    args = parser.parse_args()
    
    # Initialize RAG client
    client = RAGClient(args.store)
    
    if args.verbose:
        print(f"Querying with '{args.query}' using {args.store.upper()} store...")
    
    # Perform RAG query
    if args.verbose:
        # Verbose mode: show retrieved documents
        results = client.retrieve_documents(args.query, args.top_k)
        if results:
            print(f"Retrieved {len(results)} documents:")
            for i, result in enumerate(results):
                print(f"\n[Document {i+1}] Score: {result['score']:.4f}")
                if result.get('metadata'):
                    print("Metadata:", ", ".join([f"{k}: {v}" for k, v in result['metadata'].items() 
                                                if k not in ['embedding_idx', 'chunk_index']]))
                doc = result.get('document', '')
                if len(doc) > 200:
                    doc = doc[:200] + "..."
                print(f"Preview: {doc}")
            
            # Generate and display answer
            context = client.format_context(results)
            answer = client.generate_answer(args.query, context)
            print("\n" + "=" * 50)
            print("ANSWER:")
            print(answer)
        else:
            print("No relevant documents found.")
    else:
        # Simple mode: just show the answer
        answer = client.rag_query(args.query, args.top_k)
        print(answer)

if __name__ == "__main__":
    main()