"""AI-powered image curation for biographical content."""

import os
import sys
import logging
import requests
import tempfile
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.openai_helper import get_openai_helper

logger = logging.getLogger(__name__)

class BiographyImageCurator:
    def __init__(self):
        self.openai_helper = get_openai_helper()
    
    async def curate_images(self, subject: str, raw_images: List[Dict], 
                           featured_count: int = 1, carousel_count: int = 4) -> Dict[str, List[Dict]]:
        """
        Curate images for biographical content using AI analysis.
        
        Args:
            subject: The person being profiled
            raw_images: Raw image results from wikimedia
            featured_count: Number of featured images needed (usually 1)
            carousel_count: Number of carousel images needed
            
        Returns:
            Dict with 'featured' and 'carousel' image lists
        """
        logger.info(f"Curating images for {subject}: {len(raw_images)} candidates")
        
        if not raw_images:
            return {"featured": [], "carousel": []}
        
        # Step 1: Filter out video files and non-image formats
        valid_images = []
        for img in raw_images:
            title = img.get('title', '').lower()
            url = img.get('url', '').lower()
            
            # Skip video files and other non-image formats
            if any(ext in title or ext in url for ext in ['.webm', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.ogg', '.ogv']):
                logger.info(f"Skipping video file: {img.get('title', 'unknown')}")
                continue
                
            valid_images.append(img)
        
        logger.info(f"Filtered {len(raw_images)} candidates to {len(valid_images)} valid images")
        
        # Step 2: Remove duplicates before AI analysis
        deduplicated_images = self._remove_duplicates(valid_images)
        logger.info(f"Removed duplicates: {len(valid_images)} -> {len(deduplicated_images)} unique images")
        
        # Step 3: Analyze each unique image with AI
        analyzed_images = []
        for i, img in enumerate(deduplicated_images):
            try:
                logger.info(f"Analyzing image {i+1}/{len(deduplicated_images)}: {img.get('title', 'unknown')}")
                analysis = await self._analyze_image(subject, img)
                analyzed_images.append({
                    **img,
                    "analysis": analysis,
                    "portrait_score": analysis.get("portrait_score", 0),
                    "historical_score": analysis.get("historical_score", 0),
                    "quality_score": analysis.get("quality_score", 0),
                    "curated_caption": analysis.get("caption", "")
                })
                logger.info(f"✓ Analyzed {img.get('title', 'unknown')}: P:{analysis.get('portrait_score', 0)} H:{analysis.get('historical_score', 0)} Q:{analysis.get('quality_score', 0)}")
            except Exception as e:
                logger.error(f"✗ Failed to analyze image {img.get('title', 'unknown')}: {str(e)}")
                # Keep image with default scores so we don't lose it
                analyzed_images.append({
                    **img,
                    "analysis": {},
                    "portrait_score": 0,
                    "historical_score": 0, 
                    "quality_score": 0,
                    "curated_caption": ""
                })
        
        # Step 3: Sort all images by portrait score for featured selection
        portrait_sorted = sorted(
            analyzed_images,
            key=lambda x: (x["portrait_score"], x["quality_score"], x["historical_score"]),
            reverse=True
        )
        
        # Step 4: Mark the best portrait as featured, rest as carousel
        all_images = []
        total_needed = min(len(analyzed_images), featured_count + carousel_count)
        
        for i, img in enumerate(portrait_sorted[:total_needed]):
            # Mark the first (best portrait) as featured
            is_featured = i == 0
            finalized_img = self._finalize_image(img, "featured" if is_featured else "carousel")
            finalized_img["featured"] = is_featured
            all_images.append(finalized_img)
        
        featured_images = [img for img in all_images if img.get("featured")]
        carousel_images = [img for img in all_images if not img.get("featured")]
        
        logger.info(f"Curated {len(featured_images)} featured + {len(carousel_images)} carousel images for {subject}")
        
        return {
            "featured": featured_images,
            "carousel": carousel_images,
            "all_images": all_images  # For easier access to all images with featured flag
        }
    
    
    async def _analyze_image(self, subject: str, image: Dict) -> Dict:
        """
        Use AI to analyze an image for biographical content suitability.
        
        Args:
            subject: The person being profiled
            image: Image data including title, snippet, url
            
        Returns:
            Dict with analysis scores and curated caption
        """
        # Use direct vision analysis for all images
        return self._analyze_image_with_vision(subject, image)
    
    def _analyze_image_with_vision(self, subject: str, image: Dict) -> Dict:
        """
        Analyze an image using OpenAI vision API.
        
        Args:
            subject: The person being profiled
            image: Image metadata with src URL (either original or Cloudinary snapshot)
            
        Returns:
            Dict with analysis scores and curated caption
        """
        # Build analysis prompt
        title = image.get("title", "")
        snippet = image.get("snippet", "")
        width = image.get("width", 0)
        height = image.get("height", 0)
        
        prompt = f"""
        Analyze this image for a biography of {subject}:
        
        Image title: {title}
        Description: {snippet}
        Dimensions: {width}x{height}
        
        Rate this image on these criteria (0-10 scale):
        
        1. PORTRAIT SCORE: How well does this serve as a portrait/headshot?
           - 10: Clear headshot/portrait of the person
           - 5-9: Shows person clearly but not ideal headshot  
           - 1-4: Person visible but not portrait-focused
           - 0: Not a portrait or person not clearly visible
        
        2. HISTORICAL SCORE: Historical significance and interest
           - 10: Highly significant historical moment/context
           - 5-9: Notable historical context or period
           - 1-4: Some historical interest
           - 0: Little historical significance
        
        3. QUALITY SCORE: Image technical quality and clarity
           - 10: Excellent quality and clarity
           - 5-9: Good quality, minor issues acceptable for historical images
           - 1-4: Poor quality but usable
           - 0: Too poor quality to use
        
        Also create a specific, informative caption (under 60 characters) that describes what's actually shown in the image - the setting, time period, activity, or historical context. Be specific about visual details, not generic. Examples: "Speaking to Congress, 1871" not "Historical photo", "Wall Street office, 1870s" not "Business portrait".
        
        Respond in JSON format:
        {{
            "portrait_score": <number>,
            "historical_score": <number>, 
            "quality_score": <number>,
            "caption": "<short specific caption>",
            "reasoning": "<brief explanation>"
        }}
        """
        
        try:
            # Use OpenAI vision API for image analysis
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Add image URL for vision analysis
            image_url = image.get("src") or image.get("url", "")
            if image_url:
                messages[0]["content"].append({
                    "type": "image_url", 
                    "image_url": {"url": image_url}
                })
                logger.info(f"Using image URL for vision analysis: {image_url}")
            else:
                logger.error("No image URL available for vision analysis")
                raise ValueError("No image data available")
            
            # Make OpenAI vision API call
            import openai
            client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            logger.info(f"DEBUG: Raw AI response for {title}: {analysis_text}")
            
            # Parse JSON response
            import json
            try:
                analysis = json.loads(analysis_text.strip())
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for {title}. Raw response: '{analysis_text}'. Error: {str(e)}")
                # Try to extract scores from text if JSON parsing fails
                analysis = self._extract_scores_from_text(analysis_text)
                if not analysis:
                    raise e
            
            # Validate scores are in range
            for key in ["portrait_score", "historical_score", "quality_score"]:
                if key in analysis:
                    analysis[key] = max(0, min(10, int(analysis[key])))
            
            # Validate caption length
            if "caption" in analysis and len(analysis["caption"]) > 50:
                analysis["caption"] = analysis["caption"][:47] + "..."
            
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed for {title}: {str(e)}")
            return {
                "portrait_score": 0,
                "historical_score": 0,
                "quality_score": 0,
                "caption": "",
                "reasoning": f"Analysis failed: {str(e)}"
            }
    
    def _extract_scores_from_text(self, text: str) -> Optional[Dict]:
        """Extract scores from text when JSON parsing fails."""
        try:
            import re
            
            # Try to find scores in the text
            portrait_match = re.search(r'portrait[_\s]*score["\s]*:?\s*(\d+)', text, re.IGNORECASE)
            historical_match = re.search(r'historical[_\s]*score["\s]*:?\s*(\d+)', text, re.IGNORECASE)
            quality_match = re.search(r'quality[_\s]*score["\s]*:?\s*(\d+)', text, re.IGNORECASE)
            caption_match = re.search(r'caption["\s]*:?\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
            
            if portrait_match or historical_match or quality_match:
                return {
                    "portrait_score": int(portrait_match.group(1)) if portrait_match else 5,
                    "historical_score": int(historical_match.group(1)) if historical_match else 5,
                    "quality_score": int(quality_match.group(1)) if quality_match else 5,
                    "caption": caption_match.group(1) if caption_match else "",
                    "reasoning": "Extracted from text"
                }
        except Exception as e:
            logger.warning(f"Failed to extract scores from text: {str(e)}")
        
        return None
    
    def _remove_duplicates(self, images: List[Dict]) -> List[Dict]:
        """Remove duplicate and near-duplicate images based on titles and URLs."""
        if len(images) <= 1:
            return images
        
        unique_images = []
        seen_titles = set()
        seen_urls = set()
        
        for img in images:
            title = img.get('title', '').lower()
            url = img.get('url', '') or img.get('src', '')
            
            # Normalize title for comparison
            normalized_title = self._normalize_title(title)
            
            # Check for exact URL duplicates
            if url in seen_urls:
                logger.info(f"Skipping duplicate URL: {url}")
                continue
            
            # Check for similar titles (same portrait with variations)
            is_duplicate = False
            for seen_title in seen_titles:
                if self._are_titles_similar(normalized_title, seen_title):
                    logger.info(f"Skipping similar title: '{title}' (similar to existing)")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_images.append(img)
                seen_titles.add(normalized_title)
                seen_urls.add(url)
                logger.debug(f"Added unique image: {title[:50]}...")
        
        return unique_images
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for duplicate detection."""
        import re
        
        # Remove file extensions and common prefixes
        title = re.sub(r'^file:', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\.(jpg|jpeg|png|gif|svg|tiff?)$', '', title, flags=re.IGNORECASE)
        
        # Remove common variations and metadata
        title = re.sub(r'\s*-\s*(original|crop|cropped|edit|edited|restored|retouched)', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*\(.*\)\s*', '', title)  # Remove parenthetical info
        title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
        
        return title
    
    def _are_titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two normalized titles represent the same image."""
        if title1 == title2:
            return True
        
        # Split into words for comparison
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        # If they share most significant words, consider them similar
        if len(words1) > 0 and len(words2) > 0:
            common_words = words1.intersection(words2)
            similarity_ratio = len(common_words) / max(len(words1), len(words2))
            
            # High similarity threshold for portraits
            if similarity_ratio > 0.7:
                return True
        
        return False
    
    def _finalize_image(self, image: Dict, image_type: str) -> Dict:
        """
        Finalize image data with size constraints and proper formatting.
        
        Args:
            image: Analyzed image data
            image_type: 'featured' or 'carousel'
            
        Returns:
            Final image object for MongoDB storage
        """
        # Size constraints based on type
        max_width = 800 if image_type == "featured" else 640
        max_height = 600
        
        # Get original dimensions
        original_width = image.get("width", 400)
        original_height = image.get("height", 300)
        
        # Apply constraints
        if original_width > max_width or original_height > max_height:
            width_ratio = max_width / original_width
            height_ratio = max_height / original_height
            scale = min(width_ratio, height_ratio, 1)
            
            constrained_width = int(original_width * scale)
            constrained_height = int(original_height * scale)
        else:
            constrained_width = original_width
            constrained_height = original_height
        
        # Use curated caption if available, otherwise generate alt text
        caption = image.get("curated_caption", "")
        if not caption:
            # Extract meaningful info from title/snippet for alt text
            title = image.get("title", "").replace("File:", "").replace(".jpg", "").replace(".png", "")
            caption = title[:47] + "..." if len(title) > 50 else title
        
        # Get image URL
        image_url = image.get("url", "") or image.get("src", "")
        
        # Validate image URL
        if not image_url:
            logger.warning(f"Invalid or missing URL for image: {image.get('title', 'unknown')}")
            image_url = ""
        elif not image_url.startswith(("http://", "https://")):
            # Invalid URL format
            logger.warning(f"Invalid URL format for image: {image.get('title', 'unknown')}")
            image_url = ""
        
        # Determine source and set appropriate fields
        source = image.get("source", "wikimedia")
        
        # Generic structure that works for all image sources
        result = {
            "src": image_url,
            "alt": f"Image of {image.get('subject', image.get('title', 'historical figure'))}",
            "caption": caption,
            "type": image_type,
            "source": source,
            "license": image.get("license", "Unknown"),
            "width": constrained_width,
            "height": constrained_height,
        }
        
        # Add source-specific fields
        if source == "wikimedia":
            result["wikimediaUrl"] = image_url
            result["attribution"] = image.get("attribution", "Via Wikimedia Commons")
        else:
            # Generic attribution for other sources
            result["attribution"] = image.get("attribution", "Source unknown")
            
        # Store original dimensions if available
        if image.get("original_width") and image.get("original_height"):
            result["originalWidth"] = image.get("original_width")
            result["originalHeight"] = image.get("original_height")
        
        # Add analysis data
        result["curation_data"] = {
            "portrait_score": image.get("portrait_score", 0),
            "historical_score": image.get("historical_score", 0),
            "quality_score": image.get("quality_score", 0),
            "ai_reasoning": image.get("analysis", {}).get("reasoning", ""),
            "curated_at": datetime.utcnow().isoformat()
        }
        
        return result


async def curate_biographical_images(subject: str, raw_images: List[Dict], 
                                    featured_count: int = 1, carousel_count: int = 4) -> Dict[str, List[Dict]]:
    """
    Convenience function for curating biographical images.
    
    Args:
        subject: The person being profiled
        raw_images: Raw image results from wikimedia  
        featured_count: Number of featured images (default 1)
        carousel_count: Number of carousel images (default 4)
        
    Returns:
        Dict with 'featured' and 'carousel' curated images
    """
    curator = BiographyImageCurator()
    return await curator.curate_images(subject, raw_images, featured_count, carousel_count)