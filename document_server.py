import grpc
import os
import uuid
import json
from concurrent import futures
from confluent_kafka import Producer
import document_pb2
import document_pb2_grpc

# Kafka Config
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "document_uploaded"

# Storage directory
STORAGE_DIR = "documents"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Kafka Producer
producer = Producer({"bootstrap.servers": KAFKA_BROKER})

class DocumentService(document_pb2_grpc.DocumentServiceServicer):
    def UploadDocument(self, request, context):
        document_id = str(uuid.uuid4())
        filepath = os.path.join(STORAGE_DIR, document_id)

        try:
            with open(filepath, "wb") as f:
                f.write(request.content)

            # Send Kafka event
            event = json.dumps({
                "document_id": document_id,
                "filename": request.filename,
                "content_type": "pdf" if request.filename.endswith(".pdf") else "text"
            })
            producer.produce(KAFKA_TOPIC, key=document_id, value=event)
            producer.flush()

            return document_pb2.UploadResponse(
                document_id=document_id,
                message=f"Document '{request.filename}' uploaded successfully!"
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error storing document: {str(e)}")
            return document_pb2.UploadResponse(document_id="", message="Failed to upload document.")

    def GetDocument(self, request, context):
        filepath = os.path.join(STORAGE_DIR, request.document_id)

        if not os.path.exists(filepath):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return document_pb2.DocumentResponse()

        try:
            with open(filepath, "rb") as f:
                content = f.read()

            return document_pb2.DocumentResponse(
                filename=request.document_id,
                content=content
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            return document_pb2.DocumentResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    document_pb2_grpc.add_DocumentServiceServicer_to_server(DocumentService(), server)
    server.add_insecure_port("[::]:50051")
    print("Document gRPC server is running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
