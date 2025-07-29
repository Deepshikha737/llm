from docx import Document
from pathlib import Path
from typing import Dict, Any
import logging
from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class DOCXProcessor(BaseDocumentProcessor):
    """DOCX document processor"""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.docx', '.doc']
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        if not self.validate_file(file_path):
            return ""
        
        try:
            doc = Document(file_path)
            text_parts = []
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text)
                    if row_text:
                        text_parts.append(" | ".join(row_text))
            
            return "\n".join(text_parts)
        
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            return ""
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from DOCX"""
        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": "docx",
            "paragraphs": 0,
            "tables": 0,
            "title": "",
            "author": "",
            "subject": "",
            "created": None,
            "modified": None
        }
        
        if not self.validate_file(file_path):
            return metadata
        
        try:
            doc = Document(file_path)
            metadata["paragraphs"] = len(doc.paragraphs)
            metadata["tables"] = len(doc.tables)
            
            # Extract core properties
            core_props = doc.core_properties
            if core_props:
                metadata.update({
                    "title": core_props.title or "",
                    "author": core_props.author or "",
                    "subject": core_props.subject or "",
                    "created": core_props.created,
                    "modified": core_props.modified
                })
        
        except Exception as e:
            logger.error(f"Error extracting DOCX metadata {file_path}: {e}")
        
        return metadata