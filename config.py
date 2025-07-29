import os
from pathlib import Path
from typing import Dict, Any

# Base Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, fast embedding model
LLM_MODEL = "microsoft/DialoGPT-medium"  # Free LLM alternative

# FAISS Configuration
VECTOR_DB_PATH = CACHE_DIR / "faiss_index"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Document Processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_FILE_SIZE_MB = 50

# Query Processing
MAX_RETRIEVED_CHUNKS = 10
SIMILARITY_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.6

# Domain-specific configurations
DOMAIN_KEYWORDS = {
    "insurance": [
        "coverage", "premium", "deductible", "claim", "policy", "benefit",
        "exclusion", "copay", "coinsurance", "out-of-pocket", "network"
    ],
    "legal": [
        "contract", "clause", "liability", "obligation", "breach", "terms",
        "conditions", "jurisdiction", "dispute", "damages", "compliance"
    ],
    "hr": [
        "employee", "benefits", "leave", "compensation", "performance",
        "training", "policy", "harassment", "discrimination", "termination"
    ],
    "compliance": [
        "regulation", "requirement", "audit", "standard", "violation",
        "penalty", "certification", "disclosure", "monitoring", "reporting"
    ]
}

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG_MODE = True

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"