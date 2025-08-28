import json
import logging
import requests
from typing import Dict
# Remove unused imports - these seem to be from the previous project structure

# Get logger for this module
logger = logging.getLogger(__name__)

# Clean up unused code from previous project structure

# ------- Main Tool Functions -------

def call_wikimedia_tool(arguments: dict) -> dict:
    """
    Calls the Wikimedia API with provided arguments.

    Args:
        arguments (dict): Arguments built dynamically by Claude (or human).
            For search: {"search": "query", "limit": 10}
            For direct URL: {"filename": "File:Example.jpg"}

    Returns:
        dict: API response with direct URLs for search results.
    """
    logger.info("Calling Wikimedia tool with arguments: %s", arguments)
    
    try:
        # Log the operation
        logger.info(f"Wikimedia operation started: {arguments}")
        
        # Handle direct URL lookup
        if "filename" in arguments:
            result = get_wikimedia_metadata(arguments["filename"])
            logger.info("Wikimedia metadata lookup completed successfully")
            return result

        # Handle search
        if "search" not in arguments:
            raise Exception("For search operations, the 'search' parameter is required")

        # Build search parameters
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": arguments["search"],
            "srnamespace": 6  # File namespace
        }
        
        # Add limit if provided
        if "limit" in arguments:
            params["srlimit"] = arguments["limit"]

        # Build URL
        url = "https://commons.wikimedia.org/w/api.php"

        response = requests.get(
            url=url,
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": "Wikimedia/1.0; https://wordaroundtown.com"
            },
            timeout=30
        )

        response.raise_for_status()
        logger.info("Wikimedia API call successful")
        
        # Process the search results
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        
        # Get direct URLs for search results in smaller batches
        if search_results:
            # Get all filenames but process in smaller batches for reliability
            titles = [result.get("title", "") for result in search_results if result.get("title")]
            
            if titles:
                # Process in batches of 5 to avoid timeouts
                batch_size = 5
                all_pages = {}
                
                for i in range(0, len(titles), batch_size):
                    batch_titles = titles[i:i+batch_size]
                    
                    batch_params = {
                        "action": "query",
                        "format": "json", 
                        "titles": "|".join(batch_titles),
                        "prop": "imageinfo",
                        "iiprop": "url|size"
                    }
                
                    try:
                        batch_response = requests.get(
                            url="https://commons.wikimedia.org/w/api.php",
                            params=batch_params,
                            headers={
                                "Accept": "application/json",
                                "User-Agent": "Wikimedia/1.0; https://wordaroundtown.com"
                            },
                            timeout=8  # Shorter timeout for batches
                        )
                        batch_response.raise_for_status()
                        batch_data = batch_response.json()
                        
                        # Collect pages from this batch
                        batch_pages = batch_data.get("query", {}).get("pages", {})
                        all_pages.update(batch_pages)
                        
                    except Exception as batch_error:
                        logger.warning(f"Batch {i//batch_size + 1} failed: {str(batch_error)}")
                        continue
                
                # Process all collected results
                try:
                    
                    results_with_urls = []
                    for result in search_results:
                        filename = result.get("title", "")
                        if not filename:
                            continue
                            
                        # Find the corresponding page data
                        image_info = None
                        for page_id, page_data in all_pages.items():
                            if page_data.get("title") == filename and "imageinfo" in page_data:
                                image_info = page_data["imageinfo"][0]
                                break
                        
                        if image_info:
                            results_with_urls.append({
                                "title": filename,
                                "url": image_info.get("url", ""),
                                "snippet": result.get("snippet", ""),
                                "license": "Unknown",
                                "attribution": f"Via Wikimedia Commons: {filename}",
                                "width": image_info.get("width", 400),
                                "height": image_info.get("height", 300)
                            })
                        else:
                            logger.warning(f"No image info found for {filename}")
                            
                except Exception as e:
                    logger.error(f"Batch image info request failed: {str(e)}")
                    # Fallback: return basic results without detailed metadata
                    results_with_urls = []
                    for result in search_results[:3]:  # Limit to first 3 for safety
                        filename = result.get("title", "")
                        if filename:
                            try:
                                # Try to get individual URL as fallback
                                direct_url = get_wikimedia_url(filename)
                                results_with_urls.append({
                                    "title": filename,
                                    "url": direct_url,
                                    "snippet": result.get("snippet", ""),
                                    "license": "Unknown",
                                    "attribution": f"Via Wikimedia Commons: {filename}",
                                    "width": 640,  # Default fallback
                                    "height": 480   # Default fallback
                                })
                            except Exception as fallback_error:
                                logger.warning(f"Fallback URL fetch failed for {filename}: {str(fallback_error)}")
                                continue
            else:
                results_with_urls = []
        else:
            results_with_urls = []
        
        result = {
            "query": {
                "search": results_with_urls
            }
        }
        
        # Log successful operation result
        logger.info(f"Wikimedia search completed successfully, found {len(result.get('query', {}).get('search', []))} results")
        
        return result

    except requests.exceptions.RequestException as e:
        error_msg = f"Wikimedia API error: {str(e)}"
        logger.error(error_msg)
        # Log the failed operation
        logger.error(f"Wikimedia operation failed: {error_msg}")
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error in Wikimedia tool: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Log the failed operation
        logger.error(f"Wikimedia operation failed: {error_msg}")
        raise Exception(error_msg)

def get_wikimedia_metadata(filename: str) -> dict:
    """
    Get basic metadata for a Wikimedia Commons file including URL and dimensions.
    Simplified version for testing - skips complex license parsing.
    
    Args:
        filename (str): The filename on Wikimedia Commons (e.g., 'File:Example.jpg')
        
    Returns:
        dict: Basic metadata including URL and actual dimensions
    """
    logger.info("Getting Wikimedia metadata for file: %s", filename)
    
    # Ensure filename starts with 'File:'
    if not filename.startswith('File:'):
        filename = f'File:{filename}'
    
    # Make API request to Wikimedia for basic image info only
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": filename,
        "prop": "imageinfo",
        "iiprop": "url|size"
    }
    
    try:
        response = requests.get(
            url=api_url,
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": "Wikimedia/1.0; https://wordaroundtown.com"
            },
            timeout=10  # Shorter timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract information from response
        pages = data.get("query", {}).get("pages", {})
        for page_id in pages:
            page = pages[page_id]
            
            if "imageinfo" not in page:
                raise Exception(f"No image info found for {filename}")
                
            imageinfo = page["imageinfo"][0]
            
            # Return basic image data with actual dimensions
            return {
                "url": imageinfo["url"],
                "width": imageinfo.get("width", 400),
                "height": imageinfo.get("height", 300),
                "license": "Unknown",  # Simplified for now
                "attribution": f"Via Wikimedia Commons: {filename}"
            }
        
        raise Exception(f"No page data found for {filename}")
    except Exception as e:
        logger.error("Error getting Wikimedia metadata: %s", str(e))
        raise Exception(f"Failed to get Wikimedia metadata: {str(e)}")

def parse_wikimedia_license(content: str, filename: str) -> dict:
    """
    Parse license information from Wikimedia Commons page content.
    
    Args:
        content (str): The wiki markup content of the file page
        filename (str): The filename for fallback attribution
        
    Returns:
        dict: License and attribution information
    """
    license_info = {
        "license": "Unknown",
        "attribution": f"Image from Wikimedia Commons: {filename}"
    }
    
    try:
        content_lower = content.lower()
        
        # Common license patterns
        if "public domain" in content_lower or "{{pd" in content_lower:
            license_info["license"] = "Public Domain"
            license_info["attribution"] = f"Public domain image from Wikimedia Commons: {filename}"
        elif "cc-by-sa-4.0" in content_lower or "{{cc-by-sa-4.0" in content_lower:
            license_info["license"] = "CC BY-SA 4.0"
        elif "cc-by-sa-3.0" in content_lower or "{{cc-by-sa-3.0" in content_lower:
            license_info["license"] = "CC BY-SA 3.0"
        elif "cc-by-sa-2.5" in content_lower or "{{cc-by-sa-2.5" in content_lower:
            license_info["license"] = "CC BY-SA 2.5"
        elif "cc-by-4.0" in content_lower or "{{cc-by-4.0" in content_lower:
            license_info["license"] = "CC BY 4.0"
        elif "cc-by-3.0" in content_lower or "{{cc-by-3.0" in content_lower:
            license_info["license"] = "CC BY 3.0"
        elif "cc-by-2.5" in content_lower or "{{cc-by-2.5" in content_lower:
            license_info["license"] = "CC BY 2.5"
        elif "gfdl" in content_lower or "{{gfdl" in content_lower:
            license_info["license"] = "GFDL"
        
        # Extract author/creator information for better attribution
        author = ""
        if "|author=" in content_lower:
            # Find author field
            start = content_lower.find("|author=") + 8
            end = content_lower.find("|", start)
            if end == -1:
                end = content_lower.find("\n", start)
            if end != -1:
                author = content[start:end].strip()
                # Clean up wiki markup
                author = author.replace("[[", "").replace("]]", "").split("|")[-1]
        
        # Build proper attribution text
        if license_info["license"] != "Public Domain":
            if author:
                license_info["attribution"] = f"By {author}, {license_info['license']}, via Wikimedia Commons"
            else:
                license_info["attribution"] = f"{license_info['license']}, via Wikimedia Commons"
        
    except Exception as e:
        logger.warning(f"Error parsing license for {filename}: {str(e)}")
    
    return license_info

def get_wikimedia_url(filename: str) -> str:
    """
    Backward compatibility function - just returns the URL.
    """
    return get_wikimedia_metadata(filename)["url"]

