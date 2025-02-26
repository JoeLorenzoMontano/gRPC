from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=["localhost:9094"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

test_doc = {
    "doc_id": "test123",
    "text": "This is a sample document about artificial intelligence.",
    "metadata": {"author": "John Doe", "timestamp": "2025-02-25"}
}

producer.send("document_ingest", test_doc)
producer.flush()
print("Message sent to Kafka!")
