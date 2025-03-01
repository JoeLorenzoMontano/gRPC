from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import uuid
import json
from typing import Optional, List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from vector_store import get_vector_store
from ollama_client import OllamaClient
from config import ollama_config, vector_db_config, storage_config
from document_processor import extract_text_from_pdf, text_splitter
import io
import time

# Ensure storage directories exist
os.makedirs(storage_config.storage_dir, exist_ok=True)
os.makedirs(os.path.dirname(vector_db_config.faiss_index_path), exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Initialize API
app = FastAPI(
    title="RAG API Service",
    description="API for Retrieval-Augmented Generation with ChromaDB and FAISS vector stores",
    version="1.0.0",
)

# Initialize components
ollama_client = OllamaClient(base_url="http://localhost:3001", model="deepseek-r1:8b")
embedding_function = OllamaEmbeddings(base_url=ollama_config.base_url, model=ollama_config.model)
templates = Jinja2Templates(directory="templates")

# Create a basic HTML template for a simple UI
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>RAG Service</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .container { max-width: 800px; margin-top: 2rem; }
        .results { margin-top: 2rem; }
        pre { background-color: #f5f5f5; padding: 1rem; border-radius: 0.3rem; }
        .sources { font-size: 0.8rem; margin-top: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Retrieval-Augmented Generation Service</h1>
        
        <div class="card mb-4">
            <div class="card-header">Upload Document</div>
            <div class="card-body">
                <form action="/documents/upload" method="post" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="file" class="form-label">Document File</label>
                        <input type="file" class="form-control" id="file" name="file" required>
                    </div>
                    <div class="mb-3">
                        <label for="store_type" class="form-label">Vector Store</label>
                        <select class="form-select" id="store_type" name="store_type">
                            <option value="chromadb">ChromaDB</option>
                            <option value="faiss">FAISS</option>
                        </select>
                    </div>
                    <button type="submit" class="btn btn-primary">Upload</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Ask a Question</div>
            <div class="card-body">
                <form action="/rag/query" method="post" id="queryForm">
                    <div class="mb-3">
                        <label for="query" class="form-label">Question</label>
                        <input type="text" class="form-control" id="query" name="query" required>
                    </div>
                    <div class="mb-3">
                        <label for="store_type" class="form-label">Vector Store</label>
                        <select class="form-select" id="store_type" name="store_type">
                            <option value="chromadb">ChromaDB</option>
                            <option value="faiss">FAISS</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="top_k" class="form-label">Number of Documents to Retrieve</label>
                        <input type="number" class="form-control" id="top_k" name="top_k" value="3" min="1" max="10">
                    </div>
                    <button type="submit" class="btn btn-primary">Ask</button>
                </form>
                
                <div class="results" id="results" style="display: none;">
                    <h3>Answer</h3>
                    <div id="answer" class="mb-3"></div>
                    
                    <h4>Sources</h4>
                    <div id="sources" class="sources"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('queryForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value;
            const storeType = document.getElementById('store_type').value;
            const topK = document.getElementById('top_k').value;
            
            fetch('/rag/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    'query': query,
                    'store_type': storeType,
                    'top_k': topK,
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('answer').innerHTML = data.answer.replace(/\\n/g, '<br>');
                
                let sourcesHtml = '';
                data.sources.forEach((source, i) => {
                    sourcesHtml += `<div class="mb-2">
                        <strong>Source ${i+1}:</strong> ${source.filename || 'Unknown'} (Score: ${source.score.toFixed(2)})
                        <pre>${source.document.length > 200 ? source.document.substring(0, 200) + '...' : source.document}</pre>
                    </div>`;
                });
                
                document.getElementById('sources').innerHTML = sourcesHtml;
                document.getElementById('results').style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('answer').innerHTML = 'Error processing your request. Please try again.';
                document.getElementById('results').style.display = 'block';
            });
        });
    </script>
</body>
</html>
    """)

# Caching embeddings
embeddings_cache = {}

# ----- Helper Functions -----

def generate_query_embedding(query_text):
    """Generate an embedding for the query text"""
    # Check cache first
    if query_text in embeddings_cache:
        return embeddings_cache[query_text]
    
    # Generate new embedding
    embedding = embedding_function.embed_query(query_text)
    embeddings_cache[query_text] = embedding
    return embedding

def get_embedding_for_text(text):
    """Generate embeddings for a text"""
    return embedding_function.embed_documents([text])[0]

def process_document(file_path, filename, store_type=None):
    """Process a document, extract text, generate embeddings, and store in vector DB"""
    document_id = os.path.basename(file_path)
    
    # Get appropriate vector store
    vector_store = get_vector_store(store_type)
    
    # Determine content type
    content_type = "pdf" if filename.lower().endswith(".pdf") else "text"
    
    # Extract text
    text_content = ""
    if content_type == "pdf":
        text_content = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()
    
    if not text_content:
        return False, "Failed to extract text from document"
    
    # Split into chunks
    chunks = text_splitter.split_text(text_content)
    
    for i, chunk in enumerate(chunks):
        # Generate metadata
        try:
            metadata = ollama_client.extract_metadata(chunk)
        except:
            metadata = {}  # Default if metadata extraction fails
        
        # Generate embedding
        embedding = get_embedding_for_text(chunk)
        
        if embedding:
            chunk_id = f"{document_id}_chunk_{i}"
            
            # Merge metadata
            merged_metadata = {
                "filename": filename,
                "content_type": content_type,
                "chunk_index": i,
                "document_id": document_id,
                **metadata
            }
            
            # Store in vector database
            vector_store.add_documents(
                document_ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[merged_metadata]
            )
    
    return True, f"Document processed: {len(chunks)} chunks stored in {store_type.upper()}"

# ----- API Endpoints -----

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Render the simple UI"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    store_type: str = Form("chromadb")
):
    """Upload a document and process it for the vector store"""
    try:
        # Validate store type
        if store_type not in ["chromadb", "faiss"]:
            raise HTTPException(status_code=400, detail="Invalid vector store type")
        
        # Generate a unique ID for the document
        document_id = str(uuid.uuid4())
        filename = file.filename
        
        # Save the uploaded file
        file_path = os.path.join(storage_config.storage_dir, document_id)
        with open(file_path, "wb") as f:
            file_content = await file.read()
            f.write(file_content)
        
        # Process the document
        success, message = process_document(file_path, filename, store_type)
        
        if success:
            return {
                "document_id": document_id,
                "filename": filename,
                "message": message,
                "status": "success"
            }
        else:
            return {
                "document_id": document_id,
                "filename": filename,
                "message": message,
                "status": "error"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/rag/query")
async def rag_query(
    query: str = Form(...),
    store_type: str = Form("chromadb"),
    top_k: int = Form(3),
):
    """Perform a RAG query"""
    try:
        # Validate store type
        if store_type not in ["chromadb", "faiss"]:
            raise HTTPException(status_code=400, detail="Invalid vector store type")
        
        # Validate top_k
        if top_k < 1 or top_k > 10:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
        
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        # Get vector store
        vector_store = get_vector_store(store_type)
        
        # Search for relevant documents
        results = vector_store.search(query_embedding, top_k=top_k)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "query": query
            }
        
        # Format documents as context
        context_parts = []
        for i, result in enumerate(results):
            # Add document content
            context_parts.append(f"DOCUMENT {i+1}:\n{result.get('document', '')}")
            
            # Add metadata if available
            if result.get('metadata'):
                meta_str = ", ".join([f"{k}: {v}" for k, v in result['metadata'].items()
                                     if k not in ['embedding_idx', 'chunk_index']])
                context_parts.append(f"[Source: {meta_str}]\n")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = ollama_client.generate_response(context, query)
        
        # Prepare source information for the response
        sources = []
        for result in results:
            source = {
                "document": result.get("document", ""),
                "score": result.get("score", 0),
                "id": result.get("id", ""),
            }
            
            # Add metadata
            if result.get("metadata"):
                for k, v in result.get("metadata", {}).items():
                    if k not in source and k not in ["embedding_idx", "chunk_index"]:
                        source[k] = v
            
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents/list")
async def list_documents(
    store_type: str = Query("chromadb"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List documents in the system"""
    try:
        # Create a set of unique document IDs from filenames in storage
        document_files = os.listdir(storage_config.storage_dir)
        
        documents = []
        for doc_id in document_files[offset:offset+limit]:
            # For each document, get metadata from any chunk
            vector_store = get_vector_store(store_type)
            # Try to find any chunk belonging to this document
            results = vector_store.search(
                query_embedding=generate_query_embedding(f"document_id:{doc_id}"),
                top_k=1
            )
            
            doc_info = {
                "document_id": doc_id,
                "filename": "Unknown",
                "content_type": "Unknown",
                "timestamp": time.ctime(os.path.getctime(os.path.join(storage_config.storage_dir, doc_id)))
            }
            
            # Update with metadata if found
            if results and results[0].get('metadata'):
                metadata = results[0]['metadata']
                if "filename" in metadata:
                    doc_info["filename"] = metadata["filename"]
                if "content_type" in metadata:
                    doc_info["content_type"] = metadata["content_type"]
            
            documents.append(doc_info)
        
        return {
            "documents": documents,
            "total": len(document_files),
            "limit": limit,
            "offset": offset
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/vector-stores")
async def get_vector_stores():
    """Get information about available vector stores"""
    return {
        "vector_stores": [
            {
                "id": "chromadb",
                "name": "ChromaDB",
                "description": "ChromaDB vector database",
                "type": "http",
                "host": vector_db_config.chroma_host,
                "port": vector_db_config.chroma_port
            },
            {
                "id": "faiss",
                "name": "FAISS",
                "description": "Facebook AI Similarity Search",
                "type": "file",
                "index_path": vector_db_config.faiss_index_path
            }
        ],
        "current_default": vector_db_config.vector_store
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if storage directory exists
    storage_exists = os.path.exists(storage_config.storage_dir)
    
    # Check if embeddings are working
    embeddings_working = False
    try:
        test_embedding = generate_query_embedding("test")
        embeddings_working = len(test_embedding) > 0
    except:
        embeddings_working = False
    
    return {
        "status": "healthy" if storage_exists and embeddings_working else "unhealthy",
        "version": "1.0.0",
        "storage": storage_exists,
        "embeddings": embeddings_working,
    }

# Start the server if script is run directly
if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8080, reload=True)