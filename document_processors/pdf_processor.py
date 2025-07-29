import PyPDF2
from pathlib import Path
from typing import Dict, Any
import logging
from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class PDFProcessor(BaseDocumentProcessor):
    """PDF document processor"""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.pdf']
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        if not self.validate_file(file_path):
            return ""
        
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return ""
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": "pdf",
            "pages": 0,
            "title": "",
            "author": "",
            "subject": "",
            "creator": "",
            "creation_date": None,
            "modification_date": None
        }
        
        if not self.validate_file(file_path):
            return metadata
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                
                if pdf_reader.metadata:
                    metadata.update({
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "subject": pdf_reader.metadata.get('/Subject', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                        "creation_date": pdf_reader.metadata.get('/CreationDate'),
                        "modification_date": pdf_reader.metadata.get('/ModDate')
                    })
        
        except Exception as e:
            logger.error(f"Error extracting PDF metadata {file_path}: {e}")
        
        return metadata