import json
import os
import pdfplumber
from confluent_kafka import Consumer, KafkaException
import chromadb

# Kafka Config
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "document_uploaded"
STORAGE_DIR = "documents"

# Initialize Vector Database (ChromaDB)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Kafka Consumer
consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    "group.id": "rag_processor",
    "auto.offset.reset": "earliest"
})

consumer.subscribe([KAFKA_TOPIC])

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def process_document(event):
    document_id = event["document_id"]
    filename = event["filename"]
    content_type = event["content_type"]
    filepath = os.path.join(STORAGE_DIR, document_id)

    if not os.path.exists(filepath):
        print(f"File {filepath} not found, skipping...")
        return

    text_content =
