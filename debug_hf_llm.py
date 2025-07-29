#!/usr/bin/env python3
"""
Debug script for Hugging Face LLM integration
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from llm_system.huggingface_llm import HuggingFaceLLM, query_huggingface_llm

def test_original_code():
    """Test the original code to identify issues"""
    print("🔍 Testing Original Code Issues")
    print("=" * 50)
    
    # Test 1: Check environment variable
    hf_api_key = os.getenv('HF_API_KEY')
    print(f"1. HF_API_KEY found: {'✅' if hf_api_key else '❌'}")
    if hf_api_key:
        print(f"   Key preview: {hf_api_key[:10]}...")
    else:
        print("   ⚠️ No API key found. Using free inference (rate limited)")
    
    # Test 2: Model availability
    test_models = [
        "bigscience/mt0-small",
        "microsoft/DialoGPT-medium", 
        "google/flan-t5-base",
        "facebook/blenderbot-400M-distill"
    ]
    
    print(f"\n2. Testing model availability:")
    llm = HuggingFaceLLM()
    
    for model in test_models:
        try:
            # Test with simple prompt
            result = llm._make_request(model, "Hello, how are you?")
            status = "✅" if not result.startswith("LLM error") else "❌"
            print(f"   {model}: {status}")
            if result.startswith("LLM error"):
                print(f"      Error: {result[:100]}...")
        except Exception as e:
            print(f"   {model}: ❌ Exception: {str(e)[:50]}...")

def test_improved_integration():
    """Test the improved LLM integration"""
    print("\n🚀 Testing Improved Integration")
    print("=" * 50)
    
    # Sample context and question
    context = """
    SECTION 2: SURGICAL PROCEDURES COVERAGE
    
    2.2 Knee Surgery Coverage
    Knee surgery is covered under this policy when medically necessary. Covered knee procedures include:
    - Arthroscopic knee surgery for meniscus repair
    - Knee replacement surgery (partial or total)
    - ACL/PCL reconstruction
    - Cartilage repair procedures
    
    Conditions for knee surgery coverage:
    - Prior authorization required for elective procedures
    - Procedure must be deemed medically necessary by a qualified physician
    - Conservative treatment must be attempted first unless emergency
    - In-network provider preferred for optimal coverage
    """
    
    question = "Does this policy cover knee surgery, and what are the conditions?"
    
    print(f"Context length: {len(context)} characters")
    print(f"Question: {question}")
    print(f"\nQuerying LLM...")
    
    # Test the function
    try:
        result = query_huggingface_llm(context, question)
        print(f"\n✅ LLM Response:")
        print(f"{'=' * 30}")
        print(result)
        print(f"{'=' * 30}")
        
        # Check if it's an error
        if result.startswith("LLM error"):
            print("❌ LLM returned an error")
            return False
        else:
            print("✅ LLM returned a valid response")
            return True
            
    except Exception as e:
        print(f"❌ Exception during LLM call: {e}")
        return False

def debug_common_issues():
    """Debug common issues with Hugging Face API"""
    print("\n🔧 Common Issues & Solutions")
    print("=" * 50)
    
    issues_and_solutions = [
        {
            "issue": "503 Service Unavailable",
            "cause": "Model is loading or cold start",
            "solution": "Wait 20-60 seconds and retry, or use wait_for_model=True"
        },
        {
            "issue": "429 Too Many Requests", 
            "cause": "Rate limiting on free tier",
            "solution": "Add API key or wait before retrying"
        },
        {
            "issue": "401 Unauthorized",
            "cause": "Invalid or missing API key",
            "solution": "Check HF_API_KEY environment variable"
        },
        {
            "issue": "400 Bad Request",
            "cause": "Invalid parameters or prompt format",
            "solution": "Check model parameters and prompt structure"
        },
        {
            "issue": "Empty response",
            "cause": "Model didn't generate text or wrong response parsing",
            "solution": "Check response format and parsing logic"
        }
    ]
    
    for i, item in enumerate(issues_and_solutions, 1):
        print(f"{i}. {item['issue']}")
        print(f"   Cause: {item['cause']}")
        print(f"   Solution: {item['solution']}\n")

def test_without_api_key():
    """Test free inference without API key"""
    print("\n🆓 Testing Free Inference (No API Key)")
    print("=" * 50)
    
    # Temporarily remove API key
    original_key = os.environ.get('HF_API_KEY')
    if 'HF_API_KEY' in os.environ:
        del os.environ['HF_API_KEY']
    
    try:
        llm = HuggingFaceLLM()
        result = llm.query_huggingface_llm("The sky is blue.", "What color is the sky?")
        print(f"Free inference result: {result}")
        
        if not result.startswith("LLM error"):
            print("✅ Free inference works!")
        else:
            print("❌ Free inference failed")
            
    finally:
        # Restore API key
        if original_key:
            os.environ['HF_API_KEY'] = original_key

def main():
    """Run all debug tests"""
    print("🐛 Hugging Face LLM Debug Script")
    print("🎯 Identifying and fixing common issues")
    print("=" * 60)
    
    # Run tests
    test_original_code()
    success = test_improved_integration()
    debug_common_issues()
    test_without_api_key()
    
    print("\n📋 Summary & Recommendations")
    print("=" * 50)
    
    if success:
        print("✅ LLM integration is working!")
    else:
        print("❌ LLM integration needs fixes")
    
    print("\n💡 Recommendations:")
    print("1. For hackathons: Use free inference without API key")
    print("2. Handle rate limits with exponential backoff") 
    print("3. Use multiple fallback models")
    print("4. Implement proper error handling and timeouts")
    print("5. Cache responses to reduce API calls")
    
    print("\n🔗 Next Steps:")
    print("- Set HF_API_KEY environment variable for better rate limits")
    print("- Test with different models to find most reliable")
    print("- Integrate with the main document search system")

if __name__ == "__main__":
    main()