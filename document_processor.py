import json
import os
import pdfplumber
from confluent_kafka import Consumer, KafkaException
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama_client import OllamaClient

ollama_client = OllamaClient()

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

# Text Splitter (Configurable)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # Max tokens per chunk
    chunk_overlap=100,  # Overlap between chunks to preserve context
)

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def get_embeddings(text_chunks):
    """Generates embeddings for multiple text chunks."""
    if text_chunks:
        return embedding_function.embed_documents(text_chunks)
    return None

def process_document(event):
    """Processes a document, extracts text, chunks it, generates embeddings, and stores in ChromaDB."""
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
        # Split into chunks
        chunks = text_splitter.split_text(text_content)

        for i, chunk in enumerate(chunks):
            metadata = ollama_client.extract_metadata(chunk)  # Extract metadata for the current chunk

            # Generate embedding for each chunk individually
            embedding = get_embeddings([chunk])  # Ensure this returns a single embedding

            if embedding:
                chunk_id = f"{document_id}_chunk_{i}"  # Unique ID for each chunk

                # Merge extracted metadata with additional metadata
                merged_metadata = {
                    "filename": filename,
                    "content_type": content_type,
                    "chunk_index": i,
                    **metadata  # Merging extracted metadata dynamically
                }

                # Store in ChromaDB
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=embedding,  # Pass embedding as a list
                    metadatas=[merged_metadata]  # Pass merged metadata
                )
                print(f"Chunk {i} of document '{filename}' processed and stored in ChromaDB.")
            else:
                print(f"Failed to generate embedding for chunk {i} of document '{filename}'.")

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
