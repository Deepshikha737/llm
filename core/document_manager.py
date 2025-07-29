from pathlib import Path
from typing import List, Dict, Any, Union
import logging
from document_processors import PDFProcessor, DOCXProcessor, EmailProcessor
from embedding_system.vector_store import VectorStore
from llm_system.query_processor import QueryProcessor
import config

logger = logging.getLogger(__name__)

class DocumentManager:
    """Main document management system"""
    
    def __init__(self):
        # Initialize processors
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()
        self.email_processor = EmailProcessor()
        
        # Initialize vector store
        self.vector_store = VectorStore()
        
        # Initialize query processor
        self.query_processor = QueryProcessor()
        
        # Load existing vector store if available
        self._load_vector_store()
        
        # File type mapping
        self.processors = {
            '.pdf': self.pdf_processor,
            '.docx': self.docx_processor,
            '.doc': self.docx_processor,
            '.eml': self.email_processor,
            '.msg': self.email_processor,
            '.txt': self.email_processor
        }
    
    def _load_vector_store(self):
        """Load existing vector store"""
        try:
            self.vector_store.load(config.VECTOR_DB_PATH)
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}")
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single document"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File does not exist: {file_path}"}
        
        processor = self.processors.get(file_path.suffix.lower())
        if not processor:
            return {"error": f"Unsupported file type: {file_path.suffix}"}
        
        try:
            # Extract text and metadata
            text = processor.extract_text(file_path)
            metadata = processor.extract_metadata(file_path)
            
            if not text:
                return {"error": "No text extracted from document"}
            
            # Chunk the document with semantic breakpoints
            chunks = self._chunk_document_semantically(text, metadata)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Save vector store
            self._save_vector_store()
            
            return {
                "success": True,
                "file_path": str(file_path),
                "metadata": metadata,
                "chunks_created": len(chunks),
                "text_length": len(text)
            }
        
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {"error": f"Processing failed: {str(e)}"}
    
    def _chunk_document_semantically(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document based on semantic breakpoints"""
        chunks = []
        
        # Split by major sections first
        sections = self._split_by_sections(text)
        
        for section_idx, section in enumerate(sections):
            # Further chunk each section
            section_chunks = self._chunk_section(section, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            
            for chunk_idx, chunk in enumerate(section_chunks):
                chunk_data = {
                    "text": chunk["text"],
                    "metadata": {
                        **metadata,
                        "section_id": section_idx,
                        "chunk_id": chunk_idx,
                        "global_chunk_id": len(chunks),
                        "start_pos": chunk["start_pos"],
                        "end_pos": chunk["end_pos"]
                    }
                }
                chunks.append(chunk_data)
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by semantic sections"""
        import re
        
        # Common section headers
        section_patterns = [
            r'\n\s*(?:SECTION|Section|CHAPTER|Chapter)\s+\d+',
            r'\n\s*(?:ARTICLE|Article)\s+\w+',
            r'\n\s*\d+\.\s+[A-Z][^.]*\n',
            r'\n\s*[A-Z][A-Z\s]{10,}\n',  # All caps headers
            r'\n\s*---\s*Page\s+\d+\s*---\n',  # Page breaks
        ]
        
        # Find section breaks
        breaks = [0]
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                breaks.append(match.start())
        
        breaks.append(len(text))
        breaks = sorted(list(set(breaks)))
        
        # Split text at breaks
        sections = []
        for i in range(len(breaks) - 1):
            section = text[breaks[i]:breaks[i + 1]].strip()
            if section and len(section) > 50:  # Minimum section length
                sections.append(section)
        
        return sections if sections else [text]
    
    def _chunk_section(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk a section with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, start + chunk_size // 2, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end
                })
            
            start = end - overlap
        
        return chunks
    
    def _save_vector_store(self):
        """Save vector store to disk"""
        try:
            self.vector_store.save(config.VECTOR_DB_PATH)
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search for relevant documents"""
        try:
            # Process the query
            processed_query = self.query_processor.process_query(query)
            
            # Enhance query for better search
            enhanced_query = self.query_processor.enhance_query_for_search(processed_query)
            
            # Search vector store
            results = self.vector_store.search(
                enhanced_query, 
                k=k, 
                threshold=config.SIMILARITY_THRESHOLD
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "text": result["text"],
                    "similarity_score": result["similarity_score"],
                    "source": {
                        "file_name": result["metadata"]["file_name"],
                        "file_type": result["metadata"]["file_type"],
                        "section_id": result["metadata"].get("section_id"),
                        "chunk_id": result["metadata"].get("chunk_id")
                    }
                }
                formatted_results.append(formatted_result)
            
            return {
                "success": True,
                "query": query,
                "processed_query": processed_query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
        
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "vector_store": self.vector_store.get_stats(),
            "processors": {
                "pdf": len(self.pdf_processor.supported_formats),
                "docx": len(self.docx_processor.supported_formats),
                "email": len(self.email_processor.supported_formats)
            }
        }