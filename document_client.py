import grpc
import document_pb2
import document_pb2_grpc

def upload_document(filename):
    """Uploads a document (text or PDF) to the server."""
    with open(filename, "rb") as f:
        file_content = f.read()

    channel = grpc.insecure_channel("localhost:50051")
    stub = document_pb2_grpc.DocumentServiceStub(channel)

    request = document_pb2.UploadRequest(
        filename=filename.split("/")[-1],  # Only send filename, not full path
        content=file_content
    )

    response = stub.UploadDocument(request)
    print(f"Upload Response: {response.message} (ID: {response.document_id})")
    return response.document_id

def get_document(doc_id):
    """Retrieves a small text document."""
    channel = grpc.insecure_channel("localhost:50051")
    stub = document_pb2_grpc.DocumentServiceStub(channel)

    request = document_pb2.DocumentRequest(document_id=doc_id)
    response = stub.GetDocument(request)

    if response.filename:
        with open(response.filename, "wb") as f:
            f.write(response.content)
        print(f"Document {response.filename} downloaded successfully!")
    else:
        print(f"Document {doc_id} not found.")

def download_document(doc_id, save_path):
    """Streams and saves a large PDF file."""
    channel = grpc.insecure_channel("localhost:50051")
    stub = document_pb2_grpc.DocumentServiceStub(channel)

    request = document_pb2.DownloadRequest(document_id=doc_id)
    response_stream = stub.DownloadDocument(request)

    with open(save_path, "wb") as f:
        for chunk in response_stream:
            f.write(chunk.file_data)

    print(f"Document {doc_id} downloaded successfully to {save_path}")

if __name__ == "__main__":
    # Upload a document
    uploaded_doc_id = upload_document("example.pdf")

    # Download a text document
    get_document(uploaded_doc_id)

    # Stream and save a large PDF
    download_document(uploaded_doc_id, "example.pdf")
