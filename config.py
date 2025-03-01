import os
from dataclasses import dataclass

@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    document_topic: str = os.environ.get("KAFKA_DOCUMENT_TOPIC", "document_uploaded")
    consumer_group: str = os.environ.get("KAFKA_CONSUMER_GROUP", "rag_processor")
    auto_offset_reset: str = os.environ.get("KAFKA_AUTO_OFFSET_RESET", "latest")

@dataclass
class GrpcConfig:
    server_host: str = os.environ.get("GRPC_SERVER_HOST", "[::]")
    server_port: int = int(os.environ.get("GRPC_SERVER_PORT", "50051"))
    max_workers: int = int(os.environ.get("GRPC_MAX_WORKERS", "10"))
    chunk_size: int = int(os.environ.get("GRPC_CHUNK_SIZE", "4096"))

@dataclass
class StorageConfig:
    storage_dir: str = os.environ.get("STORAGE_DIR", "documents")

@dataclass
class VectorDBConfig:
    chroma_host: str = os.environ.get("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.environ.get("CHROMA_PORT", "8000"))
    collection_name: str = os.environ.get("CHROMA_COLLECTION", "documents")

@dataclass
class OllamaConfig:
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.environ.get("OLLAMA_MODEL", "llama3")

# Create singleton instances for use throughout the codebase
kafka_config = KafkaConfig()
grpc_config = GrpcConfig()
storage_config = StorageConfig()
vector_db_config = VectorDBConfig()
ollama_config = OllamaConfig()

# Ensure storage directory exists
os.makedirs(storage_config.storage_dir, exist_ok=True)