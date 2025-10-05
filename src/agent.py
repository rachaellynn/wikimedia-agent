"""
LangGraph-based Wikimedia Image Curation Agent

This agent follows the curation workflow from biography_creator.py but as a 
standalone LangGraph workflow that can be run independently.
"""

import os
import sys
import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv


# Load environment variables from project root
load_dotenv()

# LangGraph imports
from langgraph.graph import StateGraph, END

# Import existing utilities
try:
    # When run as a module
    from .utils.mongodb_helper import save_document
    from ..workflows.image_curator import BiographyImageCurator
except ImportError:
    # When run directly, add parent to path temporarily
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils.mongodb_helper import save_document
    from workflows.image_curator import BiographyImageCurator

logger = logging.getLogger(__name__)

@dataclass
class CurationRequest:
    """Input parameters for the curation agent"""
    subject: str
    is_primary_photos: bool  # True for primary photos (of the person), False for context photos
    limit: int = 8

class AgentState(TypedDict):
    """State that flows through the LangGraph workflow"""
    # Input
    subject: str
    is_primary_photos: bool
    limit: int
    
    # Intermediate state
    raw_images: List[Dict]
    curated_images: List[Dict] 
    errors: List[str]
    
    # Output
    saved_image_ids: List[str]
    success: bool
    message: str

class WikimediaCurationAgent:
    """LangGraph agent for curating Wikimedia images"""
    
    def __init__(self):
        self.curator = BiographyImageCurator()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("search_wikimedia", self._search_wikimedia)
        workflow.add_node("curate_images", self._curate_images) 
        workflow.add_node("save_to_mongodb", self._save_to_mongodb)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define entry point
        workflow.set_entry_point("search_wikimedia")
        
        # Define edges
        workflow.add_edge("search_wikimedia", "curate_images")
        workflow.add_edge("curate_images", "save_to_mongodb")
        workflow.add_edge("save_to_mongodb", END)
        workflow.add_edge("handle_error", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "search_wikimedia",
            self._check_search_success,
            {"success": "curate_images", "error": "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "curate_images", 
            self._check_curation_success,
            {"success": "save_to_mongodb", "error": "handle_error"}
        )
        
        return workflow.compile()
    
    async def _search_wikimedia(self, state: AgentState) -> AgentState:
        """Search Wikimedia for images of the subject"""
        logger.info(f"Searching Wikimedia for: {state['subject']}")
        
        try:
            # Import the Wikimedia tool
            try:
                from .tools.wikimedia import call_wikimedia_tool
            except ImportError:
                from src.tools.wikimedia import call_wikimedia_tool
            
            # Construct search query based on photo type
            search_query = state['subject']
            if state['is_primary_photos']:
                # For primary photos, search for the person directly
                search_query = state['subject']
            else:
                # For context photos, add contextual terms
                search_query = f"{state['subject']} context historical background"
            
            # Call Wikimedia tool
            wikimedia_results = call_wikimedia_tool({
                "search": search_query,
                "limit": state['limit']
            })
            
            # Extract images from results
            raw_images = []
            if isinstance(wikimedia_results, dict) and "query" in wikimedia_results:
                if "search" in wikimedia_results["query"]:
                    raw_images = wikimedia_results["query"]["search"]
                    logger.info(f"Found {len(raw_images)} candidate images")
                else:
                    logger.warning("No search results in Wikimedia response")
            else:
                logger.warning(f"Unexpected Wikimedia response format: {wikimedia_results}")
            
            # If no results and we tried enhanced query, try simple name
            if not raw_images and search_query != state['subject']:
                logger.info(f"No results for enhanced query, trying simple: {state['subject']}")
                wikimedia_results = call_wikimedia_tool({
                    "search": state['subject'],
                    "limit": state['limit']
                })
                
                if isinstance(wikimedia_results, dict) and "query" in wikimedia_results:
                    if "search" in wikimedia_results["query"]:
                        raw_images = wikimedia_results["query"]["search"]
                        logger.info(f"Found {len(raw_images)} images with simple query")
            
            return {
                **state,
                "raw_images": raw_images,
                "errors": state.get("errors", [])
            }
            
        except Exception as e:
            error_msg = f"Wikimedia search failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "raw_images": [],
                "errors": state.get("errors", []) + [error_msg]
            }
    
    async def _curate_images(self, state: AgentState) -> AgentState:
        """Curate images using AI analysis"""
        logger.info(f"Curating {len(state['raw_images'])} images for {state['subject']}")
        
        if not state['raw_images']:
            return {
                **state,
                "curated_images": [],
                "errors": state.get("errors", []) + ["No images to curate"]
            }
        
        try:
            # Add subject to each image for curation context
            enhanced_images = []
            for img in state['raw_images']:
                enhanced_img = img.copy()
                enhanced_img["subject"] = state['subject']
                enhanced_images.append(enhanced_img)
            
            # Use the existing curator
            if state['is_primary_photos']:
                # For primary photos, prioritize portrait scoring
                curation_result = await self.curator.curate_images(
                    state['subject'], 
                    enhanced_images,
                    featured_count=1,  # One featured primary photo
                    carousel_count=min(7, len(enhanced_images) - 1)  # Rest as carousel
                )
            else:
                # For context photos, treat all as carousel (no featured)
                curation_result = await self.curator.curate_images(
                    state['subject'],
                    enhanced_images, 
                    featured_count=0,  # No featured for context photos
                    carousel_count=min(8, len(enhanced_images))  # All as context
                )
            
            # Combine all curated images
            curated_images = curation_result.get("all_images", [])
            
            # Add metadata about photo type
            for img in curated_images:
                img["photo_type"] = "primary" if state['is_primary_photos'] else "context"
                img["subject"] = state['subject']
                img["curated_at"] = datetime.utcnow()
            
            logger.info(f"Curated {len(curated_images)} images")
            
            return {
                **state,
                "curated_images": curated_images,
                "errors": state.get("errors", [])
            }
            
        except Exception as e:
            error_msg = f"Image curation failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "curated_images": [],
                "errors": state.get("errors", []) + [error_msg]
            }
    
    async def _save_to_mongodb(self, state: AgentState) -> AgentState:
        """Save curated images to MongoDB"""
        logger.info(f"Saving {len(state['curated_images'])} images to MongoDB")
        
        if not state['curated_images']:
            return {
                **state,
                "saved_image_ids": [],
                "success": False,
                "message": "No images to save",
                "errors": state.get("errors", []) + ["No curated images available"]
            }
        
        try:
            saved_ids = []
            
            # Save each image as a separate document in the images collection
            for img in state['curated_images']:
                # Create document for MongoDB
                image_doc = {
                    # Core identification
                    "subject": state['subject'],
                    "photo_type": img.get("photo_type", "primary"),
                    
                    # Image data
                    "src": img.get("src", ""),
                    "alt": img.get("alt", ""),
                    "caption": img.get("caption", ""),
                    "attribution": img.get("attribution", ""),
                    "license": img.get("license", "Unknown"),
                    
                    # Dimensions and display
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "type": img.get("type", "carousel"),  # featured or carousel
                    "featured": img.get("featured", False),
                    
                    # Source metadata
                    "source": img.get("source", "wikimedia"),
                    "original_title": img.get("title", ""),
                    "wikimedia_url": img.get("wikimediaUrl", ""),
                    
                    # AI curation data
                    "curation_data": img.get("curation_data", {}),
                    
                    # Timestamps
                    "created_at": datetime.utcnow(),
                    "curated_at": img.get("curated_at", datetime.utcnow()),
                    
                    # Additional metadata
                    "collection": "images"
                }
                
                # Save to MongoDB
                doc_id = save_document("images", image_doc)
                saved_ids.append(str(doc_id))
                logger.info(f"Saved image {img.get('caption', 'Unknown')} with ID: {doc_id}")
            
            return {
                **state,
                "saved_image_ids": saved_ids,
                "success": True,
                "message": f"Successfully saved {len(saved_ids)} images to MongoDB",
                "errors": state.get("errors", [])
            }
            
        except Exception as e:
            error_msg = f"MongoDB save failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "saved_image_ids": [],
                "success": False,
                "message": error_msg,
                "errors": state.get("errors", []) + [error_msg]
            }
    
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        errors = state.get("errors", [])
        error_summary = "; ".join(errors)
        
        logger.error(f"Workflow failed for {state['subject']}: {error_summary}")
        
        return {
            **state,
            "success": False,
            "message": f"Curation failed: {error_summary}",
            "saved_image_ids": []
        }
    
    def _check_search_success(self, state: AgentState) -> str:
        """Check if Wikimedia search was successful"""
        if state.get("raw_images") and len(state["raw_images"]) > 0:
            return "success"
        return "error"
    
    def _check_curation_success(self, state: AgentState) -> str:
        """Check if image curation was successful"""
        if state.get("curated_images") and len(state["curated_images"]) > 0:
            return "success"
        return "error"
    
    def _print_summary(self, result: Dict[str, Any], curated_images: List[Dict]) -> None:
        """Print a formatted summary of the curation results to terminal"""
        print("\n" + "="*80)
        print(f"🖼️  WIKIMEDIA CURATION RESULTS")
        print("="*80)
        
        # Basic info
        status_emoji = "✅" if result["success"] else "❌"
        photo_type_emoji = "👤" if result["photo_type"] == "primary" else "🏛️"
        
        print(f"{status_emoji} Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"{photo_type_emoji} Subject: {result['subject']}")
        print(f"📷 Photo Type: {result['photo_type'].title()} photos")
        print(f"🔍 Images Processed: {result['images_processed']}")
        print(f"🎯 Images Curated: {result['images_curated']}")
        print(f"💾 Images Saved: {result['images_saved']}")
        
        if result["errors"]:
            print(f"⚠️  Errors: {len(result['errors'])}")
        
        print(f"💬 Message: {result['message']}")
        
        # Show curated images with scores
        if curated_images:
            print("\n📋 CURATED IMAGES:")
            print("-" * 80)
            
            featured_images = [img for img in curated_images if img.get("featured", False)]
            carousel_images = [img for img in curated_images if not img.get("featured", False)]
            
            if featured_images:
                print("⭐ FEATURED:")
                for img in featured_images:
                    self._print_image_details(img)
            
            if carousel_images:
                print("🎠 CAROUSEL:")
                for img in carousel_images:
                    self._print_image_details(img)
        
        # MongoDB IDs
        if result["saved_image_ids"]:
            print(f"\n🗄️  MongoDB Document IDs:")
            for i, doc_id in enumerate(result["saved_image_ids"], 1):
                print(f"   {i}. {doc_id}")
        
        print("="*80 + "\n")
    
    def _print_image_details(self, img: Dict) -> None:
        """Print details for a single curated image"""
        caption = img.get("caption", "No caption")
        curation_data = img.get("curation_data", {})
        
        portrait_score = curation_data.get("portrait_score", 0)
        historical_score = curation_data.get("historical_score", 0)
        quality_score = curation_data.get("quality_score", 0)
        
        print(f"   📸 {caption}")
        print(f"      📊 Scores: Portrait={portrait_score}/10, Historical={historical_score}/10, Quality={quality_score}/10")
        print(f"      📐 Size: {img.get('width', 'unknown')}×{img.get('height', 'unknown')}")
        print(f"      🔗 Source: {img.get('src', 'No URL')[:80]}...")
        print()
    
    async def run(self, subject: str, is_primary_photos: bool, limit: int = 8) -> Dict[str, Any]:
        """
        Run the complete curation workflow
        
        Args:
            subject: The person/topic to search for (e.g., "taylor_swift", "ai_for_dummies")
            is_primary_photos: True for photos of the person/thing, False for context photos
            limit: Maximum number of images to process
            
        Returns:
            Dict with results including success status, saved image IDs, and message
        """
        logger.info(f"Starting curation workflow for: {subject}")
        
        # Initialize state
        initial_state = AgentState(
            subject=subject,
            is_primary_photos=is_primary_photos,
            limit=limit,
            raw_images=[],
            curated_images=[],
            errors=[],
            saved_image_ids=[],
            success=False,
            message=""
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        # Create clean result
        clean_result = {
            "success": result.get("success", False),
            "subject": result.get("subject", subject),
            "photo_type": "primary" if is_primary_photos else "context", 
            "images_processed": len(result.get("raw_images", [])),
            "images_curated": len(result.get("curated_images", [])),
            "images_saved": len(result.get("saved_image_ids", [])),
            "saved_image_ids": result.get("saved_image_ids", []),
            "message": result.get("message", ""),
            "errors": result.get("errors", [])
        }
        
        # Print summary to terminal
        self._print_summary(clean_result, result.get("curated_images", []))
        
        return clean_result

async def run_curation_workflow(subject: str, is_primary_photos: bool, limit: int = 8) -> Dict[str, Any]:
    """
    Convenience function to run the curation workflow
    
    Args:
        subject: Subject to search for (e.g., "Marie Curie", "World War II")
        is_primary_photos: True for photos OF the subject, False for context photos
        limit: Max images to process (default 8)
        
    Returns:
        Curation results with success status, counts, and MongoDB IDs
    """
    print(f"\n🚀 Starting Wikimedia curation for: {subject}")
    print(f"📷 Type: {'Primary photos' if is_primary_photos else 'Context photos'}")
    print(f"🔢 Limit: {limit} images")
    
    agent = WikimediaCurationAgent()
    return await agent.run(subject, is_primary_photos, limit)

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_agent():
        # Test with primary photos
        result = await run_curation_workflow("Albert Einstein", is_primary_photos=True, limit=5)
        print("Primary photos result:", json.dumps(result, indent=2))
        
        # Test with context photos
        result = await run_curation_workflow("Albert Einstein", is_primary_photos=False, limit=5)
        print("Context photos result:", json.dumps(result, indent=2))
    
    asyncio.run(test_agent())