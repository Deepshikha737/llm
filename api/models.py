from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    generate_answer: bool = Field(default=True, description="Whether to generate LLM answer")

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

class LLMAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_used: int

class KeyFactors(BaseModel):
    domain: str
    intent: str
    entities_found: List[str]
    top_similarity_score: float
    sources_consulted: int

class Explanation(BaseModel):
    reasoning_process: List[str]
    key_factors: KeyFactors
    decision_basis: List[str]

class SearchResponse(BaseModel):
    success: bool
    query: str
    processed_query: Optional[ProcessedQuery] = None
    results: List[SearchResult]
    total_results: int
    llm_answer: Optional[LLMAnswer] = None
    explanation: Optional[Explanation] = None
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