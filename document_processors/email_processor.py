import email
import email.policy
from pathlib import Path
from typing import Dict, Any
import logging
from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class EmailProcessor(BaseDocumentProcessor):
    """Email document processor"""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.eml', '.msg', '.txt']
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from email file"""
        if not self.validate_file(file_path):
            return ""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                msg = email.message_from_file(file, policy=email.policy.default)
            
            text_parts = []
            
            # Extract headers
            subject = msg.get('Subject', '')
            from_addr = msg.get('From', '')
            to_addr = msg.get('To', '')
            date = msg.get('Date', '')
            
            if subject:
                text_parts.append(f"Subject: {subject}")
            if from_addr:
                text_parts.append(f"From: {from_addr}")
            if to_addr:
                text_parts.append(f"To: {to_addr}")
            if date:
                text_parts.append(f"Date: {date}")
            
            text_parts.append("---")
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body = part.get_content()
                        if body:
                            text_parts.append(body)
                    elif content_type == "text/html":
                        # Simple HTML to text conversion
                        html_body = part.get_content()
                        if html_body:
                            # Basic HTML tag removal
                            import re
                            clean_text = re.sub('<[^<]+?>', '', html_body)
                            text_parts.append(clean_text)
            else:
                body = msg.get_content()
                if body:
                    text_parts.append(body)
            
            return "\n".join(text_parts)
        
        except Exception as e:
            logger.error(f"Error processing email {file_path}: {e}")
            return ""
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from email"""
        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": "email",
            "subject": "",
            "from": "",
            "to": "",
            "date": "",
            "message_id": "",
            "attachments": []
        }
        
        if not self.validate_file(file_path):
            return metadata
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                msg = email.message_from_file(file, policy=email.policy.default)
            
            metadata.update({
                "subject": msg.get('Subject', ''),
                "from": msg.get('From', ''),
                "to": msg.get('To', ''),
                "date": msg.get('Date', ''),
                "message_id": msg.get('Message-ID', '')
            })
            
            # Extract attachment info
            if msg.is_multipart():
                for part in msg.walk():
                    filename = part.get_filename()
                    if filename:
                        metadata["attachments"].append({
                            "filename": filename,
                            "content_type": part.get_content_type()
                        })
        
        except Exception as e:
            logger.error(f"Error extracting email metadata {file_path}: {e}")
        
        return metadata