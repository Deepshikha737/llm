import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.dimension = config.EMBEDDING_DIMENSION
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        if not documents:
            return
        
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(documents)
        
        logger.info(f"Added {len(texts)} documents to vector store")
    
    def search(self, query: str, k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["similarity_score"] = float(score)
                results.append(result)
        
        return results
    
    def save(self, path: Path):
        """Save vector store to disk"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save metadata
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "model_name": self.model_name,
                "dimension": self.dimension
            }, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: Path):
        """Load vector store from disk"""
        if not path.exists():
            logger.warning(f"Vector store path does not exist: {path}")
            return
        
        # Load FAISS index
        index_path = path / "faiss.index"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
                self.model_name = data["model_name"]
                self.dimension = data["dimension"]
        
        logger.info(f"Vector store loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "model_name": self.model_name,
            "dimension": self.dimension
        }