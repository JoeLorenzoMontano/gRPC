import json
import os
import pdfplumber
from confluent_kafka import Consumer, KafkaException
import chromadb
from langchain_ollama import OllamaEmbeddings
from config import kafka_config, storage_config, vector_db_config, ollama_config

# Initialize Ollama Embeddings
embedding_function = OllamaEmbeddings(base_url=ollama_config.base_url, model=ollama_config.model)

# Initialize Vector Database (ChromaDB)
chroma_client = chromadb.HttpClient(host=vector_db_config.chroma_host, port=vector_db_config.chroma_port)
collection = chroma_client.get_or_create_collection(name=vector_db_config.collection_name)

# Kafka Consumer
consumer = Consumer({
    "bootstrap.servers": kafka_config.bootstrap_servers,
    "group.id": kafka_config.consumer_group,
    "auto.offset.reset": kafka_config.auto_offset_reset
})

consumer.subscribe([kafka_config.document_topic])

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def get_embeddings(texts):
    """Retrieves embeddings using OllamaEmbeddings."""
    if texts:
        return embedding_function.embed_documents(texts)
    return None

def process_document(event):
    """Processes a document, extracts text, generates embeddings, and stores in ChromaDB."""
    print(f"Processing event: {event}")
    
    document_id = event["document_id"]
    filename = event["filename"]
    content_type = event["content_type"]
    filepath = os.path.join(storage_config.storage_dir, document_id)

    if not os.path.exists(filepath):
        print(f"File {filepath} not found, skipping...")
        return

    # Extract text
    text_content = ""
    if content_type == "pdf":
        text_content = extract_text_from_pdf(filepath)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            text_content = f.read()

    if text_content:
        # Generate embeddings
        embeddings = get_embeddings([text_content])
        
        if embeddings:
            # Store in ChromaDB with embeddings
            collection.add(
                ids=[document_id],
                documents=[text_content],
                embeddings=embeddings,
                metadatas=[{"filename": filename, "content_type": content_type}]
            )
            print(f"Document '{filename}' processed and stored in ChromaDB with embeddings.")
        else:
            print(f"Failed to generate embeddings for document '{filename}'.")

if __name__ == "__main__":
    print("Kafka Consumer listening for documents...")
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaException._PARTITION_EOF:
                continue
            else:
                print(f"Consumer error: {msg.error()}")
                continue
        
        event_data = json.loads(msg.value().decode("utf-8"))
        process_document(event_data)
