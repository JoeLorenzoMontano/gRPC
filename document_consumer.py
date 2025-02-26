from confluent_kafka import Consumer
import json

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "document_uploaded"
KAFKA_GROUP = "document-consumer-group"

# Kafka Consumer Setup
consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    "group.id": KAFKA_GROUP,
    "auto.offset.reset": "earliest"
})

consumer.subscribe([KAFKA_TOPIC])

print("Listening for new document uploads...")

while True:
    msg = consumer.poll(1.0)  # Poll messages from Kafka
    if msg is None:
        continue
    if msg.error():
        print(f"Consumer error: {msg.error()}")
        continue

    event = json.loads(msg.value().decode("utf-8"))
    document_id = event["document_id"]
    filename = event["filename"]
    file_path = event["file_path"]

    print(f"New document uploaded: {filename} (ID: {document_id})")
    print(f"File stored at: {file_path}")

    # Further processing (e.g., indexing into FAISS or Elasticsearch)
