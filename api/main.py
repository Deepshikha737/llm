from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import tempfile
import os
from typing import List
import uvicorn

from .models import (
    QueryRequest, SearchResponse, DocumentUploadResponse, 
    HealthResponse, SystemStats
)
from core.document_manager import DocumentManager
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="A system for processing documents and answering contextual queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global document manager instance
document_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize the document manager on startup"""
    global document_manager
    logger.info("Starting up the document processing system...")
    try:
        document_manager = DocumentManager()
        logger.info("Document manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document manager: {e}")
        raise

def get_document_manager() -> DocumentManager:
    """Dependency to get document manager instance"""
    if document_manager is None:
        raise HTTPException(status_code=500, detail="Document manager not initialized")
    return document_manager

@app.get("/health", response_model=HealthResponse)
async def health_check(dm: DocumentManager = Depends(get_document_manager)):
    """Health check endpoint"""
    try:
        stats = dm.get_stats()
        return HealthResponse(
            status="healthy",
            stats=SystemStats(**stats)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy")

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    dm: DocumentManager = Depends(get_document_manager)
):
    """Upload and process a document"""
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc', '.eml', '.msg', '.txt'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
        )
    
    # Check file size (50MB limit)
    if file.size and file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process the document
        result = dm.process_document(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return DocumentUploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/v1/hackrx/run", response_model=SearchResponse)
async def hackrx_run(
    request: QueryRequest,
    dm: DocumentManager = Depends(get_document_manager)
):
    """Main hackathon endpoint for query processing and retrieval"""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Search for relevant documents
        result = dm.search(request.query, k=request.k)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
        
        # Convert to response model
        search_response = SearchResponse(**result)
        
        logger.info(f"Query processed successfully. Found {len(search_response.results)} results")
        return search_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/search", response_model=SearchResponse)
async def search_documents(
    query: str,
    k: int = 5,
    dm: DocumentManager = Depends(get_document_manager)
):
    """Alternative search endpoint"""
    request = QueryRequest(query=query, k=k)
    return await hackrx_run(request, dm)

@app.get("/stats")
async def get_system_stats(dm: DocumentManager = Depends(get_document_manager)):
    """Get system statistics"""
    try:
        return dm.get_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "search": "/api/v1/hackrx/run",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG_MODE
    )