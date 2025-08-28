#!/usr/bin/env python3
"""
Test script for the Wikimedia Curation Agent
"""

import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config/.env')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_basic():
    """Test basic agent functionality with a simple subject"""
    
    print("🧪 Testing Wikimedia Curation Agent")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY environment variable")
        return False
    
    try:
        from src.agent import run_curation_workflow
        
        # Test 1: Primary photos
        print("\n📸 Test 1: Primary photos of Albert Einstein")
        result = await run_curation_workflow(
            subject="Albert Einstein",
            is_primary_photos=True,
            limit=3  # Small limit for testing
        )
        
        print(f"✅ Success: {result['success']}")
        print(f"📊 Images processed: {result['images_processed']}")
        print(f"🎯 Images curated: {result['images_curated']}")
        print(f"💾 Images saved: {result['images_saved']}")
        
        if result['success']:
            print("✅ Primary photos test passed!")
        else:
            print(f"❌ Primary photos test failed: {result['message']}")
            return False
        
        # Test 2: Context photos
        print("\n🏛️ Test 2: Context photos of World War II")
        result = await run_curation_workflow(
            subject="World War II",
            is_primary_photos=False,
            limit=2  # Small limit for testing
        )
        
        print(f"✅ Success: {result['success']}")
        print(f"📊 Images processed: {result['images_processed']}")
        print(f"🎯 Images curated: {result['images_curated']}")
        print(f"💾 Images saved: {result['images_saved']}")
        
        if result['success']:
            print("✅ Context photos test passed!")
        else:
            print(f"❌ Context photos test failed: {result['message']}")
            return False
        
        print("\n🎉 All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"💥 Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_workflow_components():
    """Test individual workflow components"""
    
    print("\n🔧 Testing workflow components")
    print("=" * 50)
    
    try:
        from src.agent import WikimediaCurationAgent
        
        agent = WikimediaCurationAgent()
        
        # Test state initialization
        from src.agent import AgentState
        initial_state = AgentState(
            subject="Test Subject",
            is_primary_photos=True,
            limit=5,
            raw_images=[],
            curated_images=[],
            errors=[],
            saved_image_ids=[],
            success=False,
            message=""
        )
        
        print("✅ State initialization works")
        
        # Test workflow graph compilation
        graph = agent.graph
        print("✅ Workflow graph compilation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False

def test_imports():
    """Test that all required imports work"""
    
    print("📦 Testing imports")
    print("=" * 50)
    
    required_modules = [
        "langgraph.graph",
        "pymongo",
        "openai",
        "requests"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        print("Install with: pip install -r requirements_wikimedia_agent.txt")
        return False
    else:
        print("\n✅ All required modules available")
        return True

async def main():
    """Run all tests"""
    
    print("🚀 Starting Wikimedia Agent Tests")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed - cannot continue")
        sys.exit(1)
    
    # Test 2: Components  
    if not await test_workflow_components():
        print("\n❌ Component tests failed")
        sys.exit(1)
    
    # Test 3: Full workflow (only if we have API key)
    if os.getenv("OPENAI_API_KEY"):
        if not await test_agent_basic():
            print("\n❌ Basic agent tests failed")
            sys.exit(1)
    else:
        print("\n⚠️  Skipping full workflow test - no OPENAI_API_KEY")
    
    print("\n🎉 All tests completed successfully!")
    print("\n📝 Next steps:")
    print("1. Set OPENAI_API_KEY and MONGODB_URI in config/.env")
    print("2. Run: python src/agent.py")
    print("3. Check your MongoDB 'images' collection for results")

if __name__ == "__main__":
    asyncio.run(main())