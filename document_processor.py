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
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
# Verify connection by listing existing collections
#print(chroma_client.list_collections())
collection = chroma_client.get_or_create_collection(name="documents")

# Kafka Consumer
consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    "group.id": "rag_processor",
    "auto.offset.reset": "latest"
})

consumer.subscribe([KAFKA_TOPIC])

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def process_document(event):
    print(event)
    document_id = event["document_id"]
    filename = event["filename"]
    content_type = event["content_type"]
    filepath = os.path.join(STORAGE_DIR, document_id)

    if not os.path.exists(filepath):
        print(f"File {filepath} not found, skipping...")
        return

    text_content = ""
    if content_type == "pdf":
        text_content = extract_text_from_pdf(filepath)
    else:
        with open(filepath, "r") as f:
            text_content = f.read()

    if text_content:
        # Store in ChromaDB
        collection.add(
            ids=[document_id],
            documents=[text_content],
            metadatas=[{"filename": filename}]
        )
        print(f"Document {filename} processed and stored in vector DB.")

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
