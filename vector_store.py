import os
import json
import numpy as np
import chromadb
import faiss
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from config import vector_db_config

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, document_ids: List[str], documents: List[str], 
                     embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using a query embedding"""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID"""
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete a document by its ID"""
        pass


class ChromaDBStore(VectorStore):
    """ChromaDB implementation of the vector store interface"""
    
    def __init__(self, collection_name: str = None):
        """Initialize ChromaDB client and collection"""
        try:
            # Try to initialize without tenant (for older versions of ChromaDB)
            self.client = chromadb.HttpClient(
                host=vector_db_config.chroma_host,
                port=vector_db_config.chroma_port
            )
            print("Initialized ChromaDB client without tenant specification")
        except Exception as e:
            print(f"Error creating basic ChromaDB client: {e}")
            # Fall back to PersistentClient if HttpClient fails
            try:
                import os
                # Create directory for persistent storage
                os.makedirs("chroma_data", exist_ok=True)
                # Use persistent client instead
                self.client = chromadb.PersistentClient(path="chroma_data")
                print("Falling back to ChromaDB PersistentClient")
            except Exception as e2:
                print(f"Error creating fallback client: {e2}")
                raise RuntimeError("Failed to initialize any ChromaDB client")
        
        self.collection_name = collection_name or vector_db_config.collection_name
        
        # Try to create or get collection with retries
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                print(f"Successfully connected to collection: {self.collection_name}")
                break
            except Exception as e:
                last_error = e
                retry_count += 1
                print(f"Attempt {retry_count}: Error getting/creating collection: {e}")
                # Sleep before retry
                import time
                time.sleep(1)
        
        if retry_count == max_retries:
            raise RuntimeError(f"Failed to create collection after {max_retries} attempts: {last_error}")
        
    def add_documents(self, document_ids: List[str], documents: List[str],
                     embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """Add documents to ChromaDB"""
        self.collection.add(
            ids=document_ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = []
        for i, doc_id in enumerate(results.get('ids', [[]])[0]):
            if i < len(results.get('documents', [[]])[0]) and i < len(results.get('metadatas', [[]])[0]):
                docs.append({
                    'id': doc_id,
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1.0 - (results['distances'][0][i] if 'distances' in results else 0)
                })
        return docs
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID from ChromaDB"""
        try:
            result = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if result and result['ids'] and len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0] if 'documents' in result and result['documents'] else None,
                    'metadata': result['metadatas'][0] if 'metadatas' in result and result['metadatas'] else None,
                    'embedding': result['embeddings'][0] if 'embeddings' in result and result['embeddings'] else None
                }
            return None
        except Exception as e:
            print(f"Error retrieving document {document_id}: {e}")
            return None
    
    def delete_document(self, document_id: str) -> None:
        """Delete a document by ID from ChromaDB"""
        self.collection.delete(ids=[document_id])


class FAISSStore(VectorStore):
    """FAISS implementation of the vector store interface"""
    
    def __init__(self, collection_name: str = None):
        """Initialize FAISS index and metadata storage"""
        self.collection_name = collection_name or vector_db_config.collection_name
        self.index_dir = os.path.dirname(vector_db_config.faiss_index_path)
        self.index_path = f"{vector_db_config.faiss_index_path}_{self.collection_name}"
        self.metadata_path = f"{vector_db_config.faiss_metadata_path.split('.')[0]}_{self.collection_name}.json"
        
        # Create directory for the index if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize or load FAISS index
        self.dim = vector_db_config.embedding_dimension
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dim)  # L2 distance, alternative: IndexFlatIP for inner product
        
        # Initialize or load metadata storage
        self.metadata = {}
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading metadata from {self.metadata_path}, initializing empty metadata store")
        
        # Track the next available ID
        self.next_id = self.index.ntotal
        
        # Maps between FAISS numeric IDs and document string IDs
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.metadata.keys())}
        
    def _save_index(self):
        """Save the FAISS index to disk"""
        faiss.write_index(self.index, self.index_path)
    
    def _save_metadata(self):
        """Save the metadata to disk"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def add_documents(self, document_ids: List[str], documents: List[str],
                     embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """Add documents to FAISS index and metadata store"""
        if not document_ids or not embeddings:
            return
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings_np)
        
        # Update metadata and ID mappings
        for i, doc_id in enumerate(document_ids):
            doc_idx = self.next_id + i
            self.metadata[doc_id] = {
                'document': documents[i] if i < len(documents) else "",
                'metadata': metadatas[i] if i < len(metadatas) else {},
                'embedding_idx': doc_idx  # Store the FAISS index
            }
            self.doc_id_to_idx[doc_id] = doc_idx
        
        # Update next available ID
        self.next_id = self.index.ntotal
        
        # Save changes to disk
        self._save_index()
        self._save_metadata()
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in FAISS"""
        if self.index.ntotal == 0:
            return []
        
        # Convert query to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_np, min(top_k, self.index.ntotal))
        
        # Map results back to documents
        results = []
        idx_to_doc_id = {idx: doc_id for doc_id, idx in self.doc_id_to_idx.items()}
        
        for i, idx in enumerate(indices[0]):
            doc_id = idx_to_doc_id.get(int(idx))
            if doc_id and doc_id in self.metadata:
                doc_data = self.metadata[doc_id]
                results.append({
                    'id': doc_id,
                    'document': doc_data.get('document', ""),
                    'metadata': doc_data.get('metadata', {}),
                    'score': 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity score
                })
        
        return results
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID from metadata store"""
        if document_id in self.metadata:
            doc_data = self.metadata[document_id]
            return {
                'id': document_id,
                'document': doc_data.get('document', ""),
                'metadata': doc_data.get('metadata', {}),
                'embedding_idx': doc_data.get('embedding_idx')
            }
        return None
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete a document by ID
        
        Note: In FAISS, we can't directly delete vectors without rebuilding the index.
        We'll mark it as deleted in metadata but it will still exist in the index.
        For production use, a periodic reindexing would be needed.
        """
        if document_id in self.metadata:
            del self.metadata[document_id]
            if document_id in self.doc_id_to_idx:
                del self.doc_id_to_idx[document_id]
            self._save_metadata()
            print(f"Document {document_id} marked as deleted in metadata, but still exists in FAISS index")


def get_vector_store(store_type: str = None, collection_name: str = None) -> VectorStore:
    """Factory function to get the appropriate vector store based on configuration"""
    store_type = store_type or vector_db_config.vector_store
    
    if store_type.lower() == "faiss":
        return FAISSStore(collection_name)
    else:  # Default to ChromaDB
        return ChromaDBStore(collection_name)