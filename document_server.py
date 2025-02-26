import grpc
from concurrent import futures
import os
import json
from confluent_kafka import Producer  # Use confluent_kafka instead of kafka-python
import document_pb2
import document_pb2_grpc

# Kafka Configuration
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "document_uploaded"

# Storage directory
STORAGE_DIR = "documents"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Kafka Producer Setup
producer = Producer({"bootstrap.servers": KAFKA_BROKER})

class DocumentService(document_pb2_grpc.DocumentServiceServicer):
    def UploadDocument(self, request, context):
        """Handles document uploads and publishes event to Kafka."""
        doc_id = request.filename.split('.')[0]  # Use filename without extension as doc_id
        file_path = os.path.join(STORAGE_DIR, request.filename)

        # Save the file
        with open(file_path, "wb") as f:
            f.write(request.content)

        print(f"Document {request.filename} uploaded successfully.")

        # Publish event to Kafka
        event = {
            "document_id": doc_id,
            "filename": request.filename,
            "file_path": file_path
        }
        producer.produce(KAFKA_TOPIC, key=doc_id, value=json.dumps(event))
        producer.flush()  # Ensure the message is sent

        return document_pb2.UploadResponse(document_id=doc_id, message="Upload successful")

    def GetDocument(self, request, context):
        """Retrieves small text-based documents."""
        file_path = os.path.join(STORAGE_DIR, f"{request.document_id}.txt")

        if not os.path.exists(file_path):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Document not found")
            return document_pb2.DocumentResponse()

        with open(file_path, "rb") as f:
            content = f.read()

        return document_pb2.DocumentResponse(filename=f"{request.document_id}.txt", content=content)

    def DownloadDocument(self, request, context):
        """Streams large PDF documents in chunks."""
        file_path = os.path.join(STORAGE_DIR, f"{request.document_id}.pdf")

        if not os.path.exists(file_path):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Document {request.document_id} not found")
            return

        print(f"Streaming document: {request.document_id}.pdf")
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):  # 4KB chunks
                yield document_pb2.DocumentChunk(file_data=chunk)

def serve():
    """Starts the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    document_pb2_grpc.add_DocumentServiceServicer_to_server(DocumentService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Document gRPC server is running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
