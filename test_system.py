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
        
        # Test search functionality
        print("\n3️⃣ Testing Search Functionality...")
        test_queries = [
            "Does this policy cover knee surgery, and what are the conditions?",
            "What is the annual deductible?",
            "Are emergency services covered?",
            "What prescription drug coverage is included?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            search_result = dm.search(query, k=3)
            
            if search_result["success"]:
                print(f"   ✅ Found {search_result['total_results']} results")
                for j, result in enumerate(search_result["results"][:2], 1):
                    print(f"      Result {j} (Score: {result['similarity_score']:.3f}): {result['text'][:100]}...")
            else:
                print(f"   ❌ Search failed: {search_result.get('error')}")
        
        # Test system stats
        print("\n4️⃣ Testing System Statistics...")
        stats = dm.get_stats()
        print(f"✅ System Statistics:")
        print(f"   📚 Total documents: {stats['vector_store']['total_documents']}")
        print(f"   🔍 Index size: {stats['vector_store']['index_size']}")
        print(f"   🤖 Model: {stats['vector_store']['model_name']}")
        
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
    if success:
        print("\n✅ System is ready for the hackathon!")
        print("🚀 Start the server with: python run_server.py")
    else:
        print("\n❌ System tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()