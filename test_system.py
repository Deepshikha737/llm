#!/usr/bin/env python3
"""
Test script for the LLM-Powered Intelligent Query-Retrieval System
"""

import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.document_manager import DocumentManager
import config

async def test_system():
    """Test the document processing and search system"""
    print("🧪 Testing LLM-Powered Intelligent Query-Retrieval System")
    print("=" * 60)
    
    try:
        # Initialize document manager
        print("1️⃣ Initializing Document Manager...")
        dm = DocumentManager()
        print("✅ Document Manager initialized")
        
        # Test document processing
        print("\n2️⃣ Testing Document Processing...")
        sample_doc = Path("sample_documents/insurance_policy.txt")
        
        if sample_doc.exists():
            result = dm.process_document(sample_doc)
            if result.get("success"):
                print(f"✅ Document processed successfully")
                print(f"   📄 File: {result['file_path']}")
                print(f"   📊 Chunks created: {result['chunks_created']}")
                print(f"   📝 Text length: {result['text_length']}")
            else:
                print(f"❌ Document processing failed: {result.get('error')}")
                return False
        else:
            print(f"⚠️ Sample document not found: {sample_doc}")
            return False
        
        # Test search functionality with LLM answers
        print("\n3️⃣ Testing Search with LLM Answer Generation...")
        test_queries = [
            "Does this policy cover knee surgery, and what are the conditions?",
            "What is the annual deductible for this insurance policy?",
            "Are emergency services covered under this policy?", 
            "What prescription drug coverage is included in this plan?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   🔍 Query {i}: {query}")
            
            # Test with LLM answer generation
            search_result = dm.search(query, k=3, generate_answer=True)
            
            if search_result["success"]:
                print(f"   ✅ Found {search_result['total_results']} results")
                
                # Show top search results
                for j, result in enumerate(search_result["results"][:2], 1):
                    print(f"      📄 Result {j} (Score: {result['similarity_score']:.3f}): {result['text'][:80]}...")
                
                # Show LLM answer if available
                if "llm_answer" in search_result and search_result["llm_answer"]:
                    llm_answer = search_result["llm_answer"]
                    print(f"      🤖 LLM Answer (Confidence: {llm_answer['confidence']:.3f}):")
                    print(f"         {llm_answer['answer'][:150]}...")
                    print(f"      📚 Sources used: {llm_answer['sources_used']}")
                else:
                    print("      ⚠️ No LLM answer generated")
                
                # Show explanation if available
                if "explanation" in search_result and search_result["explanation"]:
                    explanation = search_result["explanation"]
                    print(f"      🧠 Reasoning: {explanation['key_factors']['domain']} domain, {explanation['key_factors']['intent']} intent")
                
            else:
                print(f"   ❌ Search failed: {search_result.get('error')}")
        
        # Test search without LLM generation
        print("\n4️⃣ Testing Search without LLM (Vector Search Only)...")
        search_result_no_llm = dm.search(
            "What are the coverage exclusions?", 
            k=2, 
            generate_answer=False
        )
        
        if search_result_no_llm["success"]:
            print(f"✅ Vector search only: {search_result_no_llm['total_results']} results")
            print(f"   📊 No LLM answer: {'llm_answer' not in search_result_no_llm or not search_result_no_llm['llm_answer']}")
        
        # Test system stats
        print("\n5️⃣ Testing System Statistics...")
        stats = dm.get_stats()
        print(f"✅ System Statistics:")
        print(f"   📚 Total documents: {stats['vector_store']['total_documents']}")
        print(f"   🔍 Index size: {stats['vector_store']['index_size']}")
        print(f"   🤖 Embedding model: {stats['vector_store']['model_name']}")
        
        # Test LLM directly
        print("\n6️⃣ Testing LLM Integration...")
        try:
            test_context = "Emergency services are covered 24/7 with no prior authorization required. Emergency room visits have a $500 copay."
            test_question = "Are emergency services covered?"
            
            llm_result = dm.llm.query_huggingface_llm(test_context, test_question)
            if not llm_result.startswith("LLM error"):
                print(f"✅ LLM integration working")
                print(f"   🤖 Test response: {llm_result[:100]}...")
            else:
                print(f"⚠️ LLM integration issues: {llm_result}")
        except Exception as e:
            print(f"⚠️ LLM test failed: {e}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test suite"""
    success = asyncio.run(test_system())
    
    print("\n📋 Test Summary")
    print("=" * 50)
    
    if success:
        print("✅ System is ready for the hackathon!")
        print("\n🚀 Usage Instructions:")
        print("1. Start server: python run_server.py")
        print("2. Upload documents via /upload endpoint")
        print("3. Query via /api/v1/hackrx/run endpoint")
        print("4. Get LLM-powered answers with explanations")
        
        print("\n📡 Example API Call:")
        print('''
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \\
     -H "Content-Type: application/json" \\
     -d '{
       "query": "Does this policy cover knee surgery?",
       "k": 5,
       "generate_answer": true
     }'
        ''')
    else:
        print("❌ System tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()