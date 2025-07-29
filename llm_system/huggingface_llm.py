import requests
import os
import time
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class HuggingFaceLLM:
    """Hugging Face LLM integration with proper error handling"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.api_key = os.getenv('HF_API_KEY')
        self.base_url = "https://api-inference.huggingface.co/models/"
        
        # Alternative free models to try
        self.fallback_models = [
            "microsoft/DialoGPT-medium",
            "google/flan-t5-base",
            "facebook/blenderbot-400M-distill",
            "bigscience/mt0-small"
        ]
        
        if not self.api_key:
            logger.warning("No HF_API_KEY found. Using free inference (rate limited)")
    
    def query_huggingface_llm(self, context: str, question: str, max_retries: int = 3) -> str:
        """Query Hugging Face model with proper error handling"""
        
        # Create a better prompt for instruction-tuned models
        prompt = self._create_prompt(context, question)
        
        for attempt in range(max_retries):
            try:
                # Try primary model first
                result = self._make_request(self.model_name, prompt)
                if not result.startswith("LLM error"):
                    return result
                
                # Try fallback models if primary fails
                for fallback_model in self.fallback_models:
                    if fallback_model != self.model_name:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        result = self._make_request(fallback_model, prompt)
                        if not result.startswith("LLM error"):
                            return result
                
                # Wait before retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return f"LLM error after {max_retries} attempts: {str(e)}"
        
        return "LLM error: All attempts failed"
    
    def _create_prompt(self, context: str, question: str) -> str:
        """Create an optimized prompt for better results"""
        # Truncate context if too long (models have token limits)
        max_context_length = 1500
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def _make_request(self, model_name: str, prompt: str) -> str:
        """Make API request to Hugging Face"""
        url = f"{self.base_url}{model_name}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key if available
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.3,  # Lower temperature for more consistent answers
                "do_sample": True,
                "top_p": 0.9,
                "return_full_text": False  # Only return generated text
            },
            "options": {
                "wait_for_model": True,  # Wait if model is loading
                "use_cache": False
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                return f"LLM error: Rate limited. Retry after {retry_after} seconds"
            
            # Handle model loading
            if response.status_code == 503:
                return "LLM error: Model is loading. Please try again in a few moments"
            
            if response.status_code != 200:
                error_details = response.text[:200] if response.text else "No error details"
                return f"LLM error: HTTP {response.status_code} - {error_details}"
            
            json_response = response.json()
            return self._parse_response(json_response, prompt)
            
        except requests.exceptions.Timeout:
            return "LLM error: Request timeout"
        except requests.exceptions.ConnectionError:
            return "LLM error: Connection failed"
        except Exception as e:
            return f"LLM error: {str(e)}"
    
    def _parse_response(self, json_response: Any, original_prompt: str) -> str:
        """Parse the API response"""
        try:
            # Handle different response formats
            if isinstance(json_response, list) and len(json_response) > 0:
                generated_text = json_response[0].get('generated_text', '')
            elif isinstance(json_response, dict):
                generated_text = json_response.get('generated_text', '')
            else:
                return f"LLM error: Unexpected response format: {type(json_response)}"
            
            if not generated_text:
                return "LLM error: Empty response from model"
            
            # Clean up the response
            answer = self._clean_response(generated_text, original_prompt)
            
            if not answer or len(answer.strip()) < 10:
                return "LLM error: Generated answer too short or empty"
            
            return answer
            
        except Exception as e:
            return f"LLM error parsing response: {str(e)}"
    
    def _clean_response(self, generated_text: str, original_prompt: str) -> str:
        """Clean and extract the answer from generated text"""
        # Remove the original prompt if it's included
        if original_prompt in generated_text:
            answer = generated_text.replace(original_prompt, "").strip()
        else:
            answer = generated_text.strip()
        
        # Look for answer after "Answer:" marker
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Remove common unwanted prefixes
        unwanted_prefixes = ["Based on the context", "According to the context", "The context states"]
        for prefix in unwanted_prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Clean up formatting
        answer = answer.replace('\n\n', '\n').strip()
        
        return answer

# Convenience function for backward compatibility
def query_huggingface_llm(context: str, question: str) -> str:
    """Backward compatible function"""
    llm = HuggingFaceLLM()
    return llm.query_huggingface_llm(context, question)