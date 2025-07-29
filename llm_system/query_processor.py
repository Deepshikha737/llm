import re
from typing import List, Dict, Any, Tuple
import logging
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import config

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Natural language query processor"""
    
    def __init__(self):
        self.domain_keywords = config.DOMAIN_KEYWORDS
        
        # Initialize NER pipeline for entity extraction
        try:
            self.ner_pipeline = pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.warning(f"Could not load NER model: {e}")
            self.ner_pipeline = None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query"""
        processed = {
            "original_query": query,
            "cleaned_query": self._clean_query(query),
            "domain": self._detect_domain(query),
            "entities": self._extract_entities(query),
            "intent": self._classify_intent(query),
            "key_terms": self._extract_key_terms(query),
            "query_type": self._determine_query_type(query)
        }
        
        return processed
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters but keep important punctuation
        query = re.sub(r'[^\w\s\?\!\.\,\-]', '', query)
        
        return query
    
    def _detect_domain(self, query: str) -> str:
        """Detect the domain of the query"""
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract named entities from query"""
        entities = []
        
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(query)
                for entity in ner_results:
                    entities.append({
                        "text": entity["word"],
                        "label": entity["entity_group"],
                        "confidence": entity["score"]
                    })
            except Exception as e:
                logger.warning(f"Error in entity extraction: {e}")
        
        # Add custom entity extraction for domain-specific terms
        entities.extend(self._extract_medical_terms(query))
        entities.extend(self._extract_financial_terms(query))
        
        return entities
    
    def _extract_medical_terms(self, query: str) -> List[Dict[str, Any]]:
        """Extract medical/health terms"""
        medical_terms = [
            "surgery", "treatment", "therapy", "diagnosis", "prescription",
            "knee", "hip", "shoulder", "back", "spine", "heart", "cancer",
            "diabetes", "hypertension", "mental health", "depression"
        ]
        
        entities = []
        query_lower = query.lower()
        
        for term in medical_terms:
            if term in query_lower:
                entities.append({
                    "text": term,
                    "label": "MEDICAL_TERM",
                    "confidence": 1.0
                })
        
        return entities
    
    def _extract_financial_terms(self, query: str) -> List[Dict[str, Any]]:
        """Extract financial terms"""
        financial_terms = [
            "premium", "deductible", "copay", "coinsurance", "claim",
            "coverage", "benefit", "exclusion", "liability", "damages"
        ]
        
        entities = []
        query_lower = query.lower()
        
        for term in financial_terms:
            if term in query_lower:
                entities.append({
                    "text": term,
                    "label": "FINANCIAL_TERM",
                    "confidence": 1.0
                })
        
        return entities
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the query"""
        query_lower = query.lower()
        
        # Coverage questions
        if any(word in query_lower for word in ["cover", "coverage", "covered", "include", "includes"]):
            return "coverage_inquiry"
        
        # Eligibility questions
        if any(word in query_lower for word in ["eligible", "qualify", "qualifies", "entitled"]):
            return "eligibility_check"
        
        # Cost/pricing questions
        if any(word in query_lower for word in ["cost", "price", "premium", "deductible", "copay"]):
            return "cost_inquiry"
        
        # Claim questions
        if any(word in query_lower for word in ["claim", "reimbursement", "submit", "process"]):
            return "claim_inquiry"
        
        # Procedure/process questions
        if any(word in query_lower for word in ["how", "process", "procedure", "steps", "apply"]):
            return "process_inquiry"
        
        # Comparison questions
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "better"]):
            return "comparison"
        
        return "general_inquiry"
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms for search enhancement"""
        # Simple keyword extraction
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
            "for", "of", "with", "by", "is", "are", "was", "were", "be",
            "been", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "this",
            "that", "these", "those", "what", "when", "where", "why", "how"
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query"""
        query_lower = query.lower()
        
        if query_lower.endswith('?'):
            return "question"
        elif any(word in query_lower for word in ["find", "search", "look", "show"]):
            return "search"
        elif any(word in query_lower for word in ["explain", "describe", "tell", "what"]):
            return "explanation"
        else:
            return "statement"
    
    def enhance_query_for_search(self, processed_query: Dict[str, Any]) -> str:
        """Enhance query for better semantic search"""
        enhanced_terms = []
        
        # Add original query
        enhanced_terms.append(processed_query["cleaned_query"])
        
        # Add domain-specific keywords
        domain = processed_query["domain"]
        if domain in self.domain_keywords:
            enhanced_terms.extend(self.domain_keywords[domain][:3])  # Top 3 domain keywords
        
        # Add extracted entities
        for entity in processed_query["entities"]:
            enhanced_terms.append(entity["text"])
        
        # Add key terms
        enhanced_terms.extend(processed_query["key_terms"][:5])  # Top 5 key terms
        
        return " ".join(enhanced_terms)