# LLM-Powered Intelligent Query-Retrieval System

A sophisticated document processing and query-retrieval system designed for hackathons, specifically handling insurance, legal, HR, and compliance documents with contextual decision-making capabilities.

## 🎯 Project Overview

This system processes large documents (PDFs, DOCX, emails) and provides intelligent query responses using semantic search and natural language processing. Built with free, open-source LLMs and optimized for real-world scenarios.

## ✨ Features

### Day 1 - Base System ✅
- **Document Upload & Parsing**: PDF, DOCX, and email file processing
- **Semantic Chunking**: Intelligent document segmentation based on semantic breakpoints
- **FAISS Vector Search**: Fast similarity search with embeddings
- **Natural Language Queries**: Top-5 semantically matched chunks retrieval
- **FastAPI Endpoint**: `/api/v1/hackrx/run` as specified

### Core Capabilities
- **Multi-format Support**: PDF, DOCX, DOC, EML, MSG, TXT files
- **Intelligent Chunking**: Context-aware document segmentation
- **Domain Detection**: Automatic classification (insurance, legal, HR, compliance)
- **Entity Extraction**: Medical, financial, and legal term recognition
- **Intent Classification**: Query type understanding (coverage, eligibility, cost, etc.)
- **Explainable Results**: Structured JSON responses with confidence scores

## 🏗️ Architecture

```
📁 Project Structure
├── 📄 config.py                 # Configuration settings
├── 📄 requirements.txt          # Dependencies
├── 📄 run_server.py            # Server launcher
├── 📄 test_system.py           # System tests
├── 📁 document_processors/      # Document processing modules
│   ├── base_processor.py
│   ├── pdf_processor.py
│   ├── docx_processor.py
│   └── email_processor.py
├── 📁 embedding_system/         # Vector search system
│   └── vector_store.py
├── 📁 llm_system/              # Query processing
│   └── query_processor.py
├── 📁 core/                    # Core orchestration
│   └── document_manager.py
├── 📁 api/                     # FastAPI application
│   ├── main.py
│   └── models.py
└── 📁 sample_documents/        # Test documents
    └── insurance_policy.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-query-retrieval-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Test the system
python test_system.py
```

### 3. Start Server

```bash
# Launch the FastAPI server
python run_server.py
```

The server will be available at:
- **API**: `http://localhost:8000`
- **Documentation**: `http://localhost:8000/docs`
- **Main Endpoint**: `http://localhost:8000/api/v1/hackrx/run`

## 📡 API Endpoints

### Main Hackathon Endpoint
```http
POST /api/v1/hackrx/run
Content-Type: application/json

{
    "query": "Does this policy cover knee surgery, and what are the conditions?",
    "k": 5
}
```

### Document Upload
```http
POST /upload
Content-Type: multipart/form-data

file: <document file>
```

### Alternative Search
```http
GET /search?query=knee surgery coverage&k=5
```

### System Health
```http
GET /health
```

## 💡 Usage Examples

### Example Query-Response

**Query**: "Does this policy cover knee surgery, and what are the conditions?"

**Response**:
```json
{
    "success": true,
    "query": "Does this policy cover knee surgery, and what are the conditions?",
    "processed_query": {
        "domain": "insurance",
        "intent": "coverage_inquiry",
        "entities": [
            {"text": "knee", "label": "MEDICAL_TERM", "confidence": 1.0},
            {"text": "surgery", "label": "MEDICAL_TERM", "confidence": 1.0}
        ]
    },
    "results": [
        {
            "text": "Knee surgery is covered under this policy when medically necessary. Covered knee procedures include: - Arthroscopic knee surgery for meniscus repair - Knee replacement surgery (partial or total) - ACL/PCL reconstruction - Cartilage repair procedures",
            "similarity_score": 0.89,
            "source": {
                "file_name": "insurance_policy.txt",
                "file_type": "txt",
                "section_id": 1,
                "chunk_id": 2
            }
        }
    ],
    "total_results": 3
}
```

### Document Processing
```python
from core.document_manager import DocumentManager

# Initialize system
dm = DocumentManager()

# Process document
result = dm.process_document("path/to/document.pdf")

# Search
search_result = dm.search("What is covered under this policy?", k=5)
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Search Configuration
MAX_RETRIEVED_CHUNKS = 10
SIMILARITY_THRESHOLD = 0.7

# Domain Keywords
DOMAIN_KEYWORDS = {
    "insurance": ["coverage", "premium", "deductible", ...],
    "legal": ["contract", "clause", "liability", ...],
    # ... more domains
}
```

## 🧪 Testing

Run comprehensive tests:

```bash
python test_system.py
```

Test specific components:
```python
# Test document processing
from document_processors import PDFProcessor
processor = PDFProcessor()
text = processor.extract_text("document.pdf")

# Test vector search
from embedding_system.vector_store import VectorStore
vs = VectorStore()
results = vs.search("query", k=5)
```

## 🎯 Hackathon Features

### Day 1 Deliverables ✅
- [x] Upload and parse PDF, DOCX, and email files
- [x] Chunk documents based on semantic breakpoints
- [x] Generate embeddings and store in FAISS
- [x] Natural language query → top-5 semantically matched chunks
- [x] FastAPI endpoint: `/api/v1/hackrx/run`

### Technical Specifications ✅
- [x] Free embeddings (sentence-transformers)
- [x] FAISS vector database
- [x] Semantic search and clause retrieval
- [x] Structured JSON responses
- [x] Domain-specific processing (insurance, legal, HR, compliance)

## 🔄 Future Enhancements

- **Day 2**: LLM integration for answer generation
- **Day 3**: Advanced decision reasoning
- **Advanced Features**: Real-time processing, web interface, multi-language support

## 🛠️ Dependencies

- **FastAPI**: Web framework
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Free embeddings
- **PyPDF2**: PDF processing
- **python-docx**: DOCX processing
- **Transformers**: NLP models

## 📝 License

Open source - perfect for hackathons and educational use.

## 🏆 Hackathon Ready

This system is specifically designed for hackathon environments:
- **Fast setup**: Run in minutes
- **Free resources**: No API keys required
- **Scalable**: Handle large documents efficiently
- **Demonstrable**: Clear API responses for judges
- **Extensible**: Easy to add features

---

**Ready to revolutionize document intelligence! 🚀**