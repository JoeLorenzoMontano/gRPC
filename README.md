# gRPC + Kafka + Vector DB + Ollama 
Joe Montanos Advanced Pipeline Repository

## Overview
This repository contains a Kafka-to-Vector Database pipeline using Apache Flink, Kafka, ChromaDB/FAISS, and gRPC. The main components of this repository include:

- A Kafka consumer that processes incoming messages and stores embeddings in vector databases.
- A Docker Compose setup to run multiple Kafka brokers with KRaft mode enabled.
- Integration of gRPC for remote procedure calls.
- Support for multiple vector stores (ChromaDB and FAISS).
- Retrieval-Augmented Generation (RAG) capabilities using Ollama models.

## Project Structure
The repository is structured as follows:

```
gRPC-Kafka-VectorDb-Ollama/
├── kafka/
│   ├── docker-compose.yml
│   ├── flink_consumer.py
│   ├── server-1.properties
│   ├── server-2.properties
│   ├── server-3.properties
├── document_server.py
├── document_pb2_grpc.py
├── document_pb2.py
├── document.proto
├── document_client.py
├── document_processor.py
├── document_search.py
├── rag_client.py
├── vector_store.py
├── config.py
├── ollama_client.py
├── requirements.txt
├── CLAUDE.md
```

- `kafka/docker-compose.yml`: Docker Compose file to set up Kafka brokers and ChromaDB.
- `kafka/flink_consumer.py`: Script to set up a Kafka consumer using Apache Flink and process messages.
- `document_server.py`: gRPC server implementation for handling document-related requests.
- `document_client.py`: Client for interacting with the gRPC document server.
- `document_processor.py`: Processes documents from Kafka and stores them in the vector database.
- `document_search.py`: CLI tool for searching documents in the vector store.
- `rag_client.py`: Implements Retrieval-Augmented Generation for answering queries.
- `vector_store.py`: Abstract interface for multiple vector stores (ChromaDB and FAISS).
- `config.py`: Centralized configuration management with environment variable support.
- `ollama_client.py`: Client for interacting with Ollama embedding and LLM models.

## Usage

### Setup and Installation
```
# Clone the repository
git clone https://github.com/JoeLorenzoMontano/gRPC-Kafka-VectorDb-Ollama.git
cd gRPC-Kafka-VectorDb-Ollama

# Install dependencies
pip install -r requirements.txt

# Start Kafka and ChromaDB containers
cd kafka && docker-compose up -d && cd ..

# Create the vector store directory for FAISS
mkdir -p vector_store
```

### Document Processing Flow
1. **Start the gRPC server**: `python document_server.py`
2. **Start the document processor**: `python document_processor.py`
3. **Upload documents using the client**: `python document_client.py`
4. **Search for documents**: `python document_search.py "your search query" --store chromadb`
5. **Use RAG for question answering**: `python rag_client.py "your question" --store faiss --verbose`

### Configuration
You can configure the system using environment variables or by modifying config.py. Key configurations:
- `VECTOR_STORE`: Choose between "chromadb" (default) or "faiss"
- `OLLAMA_MODEL`: The Ollama model to use for embeddings
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka broker addresses

## Document Ingestion Flow
1. **Kafka Producer**: Documents are uploaded to Kafka under the `document_uploaded` topic.
2. **Kafka Consumer**: The consumer reads messages from the Kafka topic.
3. **Processing**: Each message (document) is processed to generate vector embeddings using Ollama.
4. **Chunking**: Documents are split into manageable chunks for better retrieval.
5. **Storage**: The generated embeddings are stored in the selected vector database (ChromaDB or FAISS) along with metadata.

## gRPC Integration
- gRPC is used to facilitate remote procedure calls (RPC) between different components of the system.
- The `document_server.py` file sets up a gRPC server to handle document uploads, retrievals, and downloads.
- Methods implemented include:
  - `UploadDocument`: Handles uploading a document, saving it to local storage, and sending an event to Kafka.
  - `GetDocument`: Retrieves a document by its ID.
  - `DownloadDocument`: Streams a document in chunks to the client.

## RAG Implementation
The Retrieval-Augmented Generation (RAG) implementation connects vector search with LLM generation:

1. **Document Storage**: Documents are processed, chunked, and stored in vector databases.
2. **Retrieval**: When a question is asked, the system finds relevant document chunks using vector similarity.
3. **Context Preparation**: Retrieved chunks are formatted as context for the LLM.
4. **Generation**: The Ollama LLM generates answers based on the retrieved context.

Key components:
- `document_search.py`: Provides vector search functionality.
- `rag_client.py`: Implements the complete RAG pipeline.
- `vector_store.py`: Provides a unified interface for different vector databases.

## Work in Progress
This repository is a work in progress. Currently, Apache Flink integration is not fully functional. The RAG implementation is now available with both ChromaDB and FAISS as vector stores. Future enhancements may include:
1. Improved document chunking strategies
2. Support for more vector database backends
3. Enhanced metadata extraction and filtering
4. Streaming document updates

## Contributing
If you would like to contribute to this repository, please fork the repository and create a pull request with your changes.

## License
The usage of this project is restricted. For license information, please contact [Joe Lorenzo Montano](https://github.com/JoeLorenzoMontano).

## Contact
For any questions or inquiries, please contact [Joe Lorenzo Montano](https://github.com/JoeLorenzoMontano).

## Acknowledgments
- [Apache Flink](https://flink.apache.org/)
- [Kafka](https://kafka.apache.org/)
- [ChromaDB](https://chromadb.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.ai/)
- [LangChain](https://www.langchain.com/)
