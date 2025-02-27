from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource
from pyflink.common.serialization import SimpleStringSchema
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path="/chroma/db")
collection = chroma_client.get_or_create_collection("documents")

# Setup Flink
env = StreamExecutionEnvironment.get_execution_environment()

# Kafka Source (with KRaft)
kafka_source = KafkaSource.builder() \
    .set_bootstrap_servers("kafka:9092") \
    .set_topics("document_uploaded") \
    .set_value_only_deserializer(SimpleStringSchema()) \
    .build()

ds = env.from_source(kafka_source, watermark_strategy=None, source_name="KafkaSource")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_document(doc_json):
    """Processes Kafka JSON message and stores embeddings in ChromaDB."""
    try:
        doc = json.loads(doc_json)
        text = doc["text"]
        doc_id = doc["id"]

        # Generate vector embedding
        embedding = model.encode(text).tolist()

        # Store in ChromaDB
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"source": "kafka"}]
        )

        print(f"Processed and stored: {doc_id}")

    except Exception as e:
        print(f"Error processing document: {e}")

# Apply processing function to stream
ds.map(lambda msg: process_document(msg))

env.execute("Kafka to ChromaDB Pipeline")
