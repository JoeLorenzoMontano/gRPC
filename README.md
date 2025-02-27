# gRPC + Kafka + Vector DB + Ollama 
Joe Montanos Advanced Pipeline Repository

## Overview
This repository contains a Kafka-to-ChromaDB pipeline using Apache Flink, Kafka, ChromaDB, and gRPC. The main components of this repository include:

- A Kafka consumer that processes incoming messages and stores embeddings in ChromaDB.
- A Docker Compose setup to run multiple Kafka brokers with KRaft mode enabled.
- Integration of gRPC for remote procedure calls.

## Project Structure
The repository is structured as follows:

```
gRPC/
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
```

- `kafka/docker-compose.yml`: Docker Compose file to set up Kafka brokers.
- `kafka/flink_consumer.py`: Script to set up a Kafka consumer using Apache Flink and process messages to store embeddings in ChromaDB.
- `kafka/server-1.properties`, `kafka/server-2.properties`, `kafka/server-3.properties`: Configuration files for the Kafka brokers.
- `document_server.py`: gRPC server implementation for handling document-related requests.
- `document_pb2_grpc.py`, `document_pb2.py`, `document.proto`: gRPC generated files and the proto definition.

## Usage
- The Kafka consumer listens to the `document_uploaded` topic on the Kafka brokers.
- Incoming messages are processed, and embeddings are stored in ChromaDB.

## Document Ingestion Flow
1. **Kafka Producer**: Documents are uploaded to Kafka under the `document_uploaded` topic.
2. **Kafka Consumer (Flink)**: The Flink consumer reads messages from the Kafka topic.
3. **Processing**: Each message (document) is processed to generate vector embeddings using a local Ollama embedding model.
4. **Storage**: The generated embeddings are stored in ChromaDB along with metadata.

## gRPC Integration
- gRPC is used to facilitate remote procedure calls (RPC) between different components of the system.
- The `document_server.py` file sets up a gRPC server to handle document uploads, retrievals, and downloads.
- Methods implemented include:
  - `UploadDocument`: Handles uploading a document, saving it to local storage, and sending an event to Kafka.
  - `GetDocument`: Retrieves a document by its ID.
  - `DownloadDocument`: Streams a document in chunks to the client.

## Work in Progress
This repository is a work in progress. Currently, Apache Flink integration is not fully functional. The goal of this project is to enable Large Language Model (LLM) data retrieval and/or Retrieval-Augmented Generation (RAG). We also plan to integrate Elasticsearch for enhanced search capabilities.

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
