from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseDocumentProcessor(ABC):
    """Base class for document processors"""
    
    def __init__(self):
        self.supported_formats = []
    
    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """Extract text from document"""
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from document"""
        pass
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "start_pos": start,
                "end_pos": end,
                "chunk_id": len(chunks)
            })
            
            start += chunk_size - overlap
        
        return chunks
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate if file can be processed"""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported format: {file_path.suffix}")
            return False
        
        # Check file size (50MB limit)
        if file_path.stat().st_size > 50 * 1024 * 1024:
            logger.error(f"File too large: {file_path}")
            return False
        
        return True