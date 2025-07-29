from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    k: int = Field(default=5, description="Number of results to return", ge=1, le=20)

class DocumentUploadResponse(BaseModel):
    success: bool
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunks_created: Optional[int] = None
    text_length: Optional[int] = None
    error: Optional[str] = None

# Response Models
class SourceInfo(BaseModel):
    file_name: str
    file_type: str
    section_id: Optional[int] = None
    chunk_id: Optional[int] = None

class SearchResult(BaseModel):
    text: str
    similarity_score: float
    source: SourceInfo

class ProcessedQuery(BaseModel):
    original_query: str
    cleaned_query: str
    domain: str
    entities: List[Dict[str, Any]]
    intent: str
    key_terms: List[str]
    query_type: str

class SearchResponse(BaseModel):
    success: bool
    query: str
    processed_query: Optional[ProcessedQuery] = None
    results: List[SearchResult]
    total_results: int
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class VectorStoreStats(BaseModel):
    total_documents: int
    index_size: int
    model_name: str
    dimension: int

class ProcessorStats(BaseModel):
    pdf: int
    docx: int
    email: int

class SystemStats(BaseModel):
    vector_store: VectorStoreStats
    processors: ProcessorStats

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    stats: Optional[SystemStats] = None