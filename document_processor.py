import json
import os
import pdfplumber
from confluent_kafka import Consumer, KafkaException
import chromadb
from langchain_ollama import OllamaEmbeddings

# Kafka Config
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "document_uploaded"
STORAGE_DIR = "documents"

# Initialize Ollama Embeddings
embedding_function = OllamaEmbeddings(base_url="http://localhost:11434", model="llama3")

# Initialize Vector Database (ChromaDB)
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="documents")

# Kafka Consumer
consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    "group.id": "rag_processor",
    "auto.offset.reset": "latest"
})

consumer.subscribe([KAFKA_TOPIC])

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
    filepath = os.path.join(STORAGE_DIR, document_id)

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
