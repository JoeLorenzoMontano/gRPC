# CLAUDE.md - Coding Guidelines and Commands

## Commands
- Install dependencies: `pip install -r requirements.txt`
- Run gRPC server: `python document_server.py`
- Run client: `python document_client.py`
- Run document processor: `python document_processor.py`
- Start Kafka & ChromaDB: `cd kafka && docker-compose up -d`
- Run Flink consumer: `python kafka/flink_consumer.py`
- Search documents: `python document_search.py "your query" --store chromadb`
- RAG query: `python rag_client.py "your question" --store faiss --verbose`
- Start API service: `python api_service.py` (then visit http://localhost:8080/docs for Swagger)

## Code Style Guidelines
- **Imports**: Standard libraries first, then third-party, then local modules
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use try-except blocks with specific error types
- **Dependencies**: gRPC, Kafka, ChromaDB/FAISS, Ollama, pdfplumber
- **Configuration**: Use config.py for settings, override with environment variables
- **Infrastructure**: Components communicate via Kafka topic "document_uploaded"
- **Vector Storage**: Support for both ChromaDB and FAISS via the VectorStore abstraction

## Development Best Practices
- Store documents in "documents" directory (controlled by config)
- Use config.py settings instead of hardcoding connection parameters
- When implementing new features, follow existing pattern separation (server, client, processor)
- Add proper error handling and logging for production readiness
- Document new functionality in README.md