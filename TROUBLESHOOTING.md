# Troubleshooting Guide

## Common Hugging Face LLM Issues & Solutions

### 1. **503 Service Unavailable Error**

**Problem**: `LLM error: HTTP 503 - Model is loading`

**Cause**: Model is in a "cold start" state and needs time to load

**Solutions**:
```python
# Wait and retry with exponential backoff
time.sleep(20)  # Wait 20 seconds
result = llm.query_huggingface_llm(context, question)

# Or use wait_for_model parameter (already implemented)
payload = {
    "options": {"wait_for_model": True}
}
```

### 2. **429 Rate Limit Error**

**Problem**: `LLM error: HTTP 429 - Too Many Requests`

**Cause**: Free tier rate limiting (no API key) or exceeded API key limits

**Solutions**:
```bash
# Option 1: Get free Hugging Face API key
# 1. Sign up at https://huggingface.co/
# 2. Go to Settings > Access Tokens
# 3. Create a new token
# 4. Set environment variable:
export HF_API_KEY="your_token_here"

# Option 2: Wait and retry
# The system automatically implements exponential backoff
```

### 3. **401 Unauthorized Error**

**Problem**: `LLM error: HTTP 401 - Unauthorized`

**Cause**: Invalid API key

**Solutions**:
```bash
# Check your API key
echo $HF_API_KEY

# Set correct API key
export HF_API_KEY="hf_your_valid_token"

# Or create .env file:
echo "HF_API_KEY=hf_your_valid_token" > .env
```

### 4. **Empty or Poor Quality Responses**

**Problem**: LLM returns empty or nonsensical answers

**Cause**: Wrong model, poor prompt, or model limitations

**Solutions**:
```python
# Try different models (already implemented as fallbacks)
models_to_try = [
    "microsoft/DialoGPT-medium",      # Conversational
    "google/flan-t5-base",            # Instruction-following
    "facebook/blenderbot-400M-distill", # Dialog
    "bigscience/mt0-small"            # Multilingual
]

# Improve prompt formatting
prompt = f"""Answer the question based on the context.

Context: {context}
Question: {question}
Answer:"""
```

### 5. **Connection Timeout**

**Problem**: `LLM error: Request timeout`

**Cause**: Network issues or slow model response

**Solutions**:
```python
# Increase timeout (already set to 30s)
response = requests.post(url, json=payload, timeout=60)

# Check internet connection
ping api-inference.huggingface.co
```

## Quick Fixes for Your Original Code

### Issue 1: Response Parsing
Your original code had issues with response format handling:

```python
# ❌ Original (problematic)
if isinstance(json_response, list):
    return json_response[0]['generated_text'].strip()

# ✅ Fixed version
if isinstance(json_response, list) and len(json_response) > 0:
    return json_response[0].get('generated_text', '').strip()
```

### Issue 2: Error Handling
```python
# ❌ Original (basic error handling)
if response.status_code != 200:
    return f"LLM error: status code {response.status_code}"

# ✅ Improved error handling
if response.status_code == 503:
    return "Model loading, please retry in 30 seconds"
elif response.status_code == 429:
    return "Rate limited, please wait or add API key"
elif response.status_code != 200:
    return f"HTTP {response.status_code}: {response.text[:100]}"
```

### Issue 3: Model Parameters
```python
# ❌ Original (may cause issues)
payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 300,
        "temperature": 0.7,
    }
}

# ✅ Optimized parameters
payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 200,  # Reduced for faster response
        "temperature": 0.3,     # Lower for more consistent answers
        "do_sample": True,
        "top_p": 0.9,
        "return_full_text": False
    },
    "options": {
        "wait_for_model": True,  # Handle model loading
        "use_cache": False       # Get fresh responses
    }
}
```

## Testing Your Setup

### Test 1: Basic API Connection
```bash
python debug_hf_llm.py
```

### Test 2: Check Environment
```bash
# Check if API key is set
env | grep HF_API_KEY

# Test without API key (free tier)
unset HF_API_KEY
python debug_hf_llm.py
```

### Test 3: Manual API Test
```bash
curl -X POST \
  "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello, how are you?"}'
```

## Fallback Strategies

### 1. Use Multiple Models
The system tries multiple models if one fails:
- Primary: `microsoft/DialoGPT-medium`
- Fallback 1: `google/flan-t5-base`
- Fallback 2: `facebook/blenderbot-400M-distill`
- Fallback 3: `bigscience/mt0-small`

### 2. Graceful Degradation
```python
# If LLM fails, return search results only
search_result = dm.search(query, generate_answer=False)
```

### 3. Local Fallback (Advanced)
For hackathons, consider local models:
```bash
# Install Ollama for local models
curl https://ollama.ai/install.sh | sh
ollama run llama2:7b
```

## Monitoring & Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Model Status
```python
# Check if model is available
response = requests.get(
    "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
)
print(response.json())
```

### Performance Monitoring
```python
import time

start_time = time.time()
result = llm.query_huggingface_llm(context, question)
end_time = time.time()

print(f"LLM response time: {end_time - start_time:.2f}s")
```

## Hackathon-Specific Tips

1. **Always have a fallback**: Don't rely solely on LLM - vector search alone is valuable
2. **Cache responses**: Store successful LLM responses to avoid repeated calls
3. **Use shorter contexts**: Limit context to 1500 characters for better performance
4. **Set realistic expectations**: Free models may not be as good as GPT-4
5. **Test early**: Run the debug script before the presentation

## Getting Help

1. **Check Hugging Face Status**: https://status.huggingface.co/
2. **Model Documentation**: Check each model's page for specific requirements
3. **Community Forums**: https://discuss.huggingface.co/
4. **Debug Script**: Run `python debug_hf_llm.py` for detailed diagnostics