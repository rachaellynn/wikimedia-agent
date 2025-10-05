"""OpenAI utility functions for content generation."""

import os
import logging
import openai
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OpenAIHelper:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    
    def generate_content(self, prompt: str, model: str = "gpt-4", 
                        max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate content using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate content: {str(e)}")
    
    def categorize_content(self, subject: str, content_sample: str, 
                          available_categories: List[str], existing_categories: Optional[List[Dict]] = None) -> List[str]:
        """Categorize content using OpenAI with enhanced logic based on existing patterns."""
        if not available_categories:
            return []
        
        # Analyze existing category usage patterns
        category_context = ""
        if existing_categories:
            # Count category usage
            category_counts = {}
            for doc in existing_categories:
                for cat in doc.get('categories', []):
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Create context about popular categories
            if category_counts:
                popular_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                category_context = f"\nNote: Popular categories in existing content: {', '.join([f'{cat} ({count} uses)' for cat, count in popular_cats])}"
        
        # Enhanced prompt with subject analysis and existing patterns
        prompt = f"""
        Analyze this biography of {subject} and select 2-3 most relevant categories.
        
        Available categories: {', '.join(available_categories)}
        {category_context}
        
        Content excerpt: {content_sample[:800]}...
        
        Instructions:
        1. Focus on the person's primary field/profession
        2. Consider their major achievements and impact areas
        3. If they worked across multiple fields, select the most significant ones
        4. Prefer specific categories over general ones when both apply
        
        Return only the category names, separated by commas.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # More capable model for better categorization
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.2  # Lower temperature for more consistent results
            )
            
            categories_text = response.choices[0].message.content.strip()
            selected_categories = [cat.strip() for cat in categories_text.split(',')]
            
            # Validate categories are in our list
            valid_categories = [cat for cat in selected_categories if cat in available_categories]
            
            # Enhanced fallback logic
            if not valid_categories:
                valid_categories = self._smart_fallback_categorization(subject, available_categories)
            
            return valid_categories[:3] if valid_categories else available_categories[:1]
            
        except Exception as e:
            logger.error(f"Content categorization failed: {str(e)}")
            # Use smart fallback on error
            return self._smart_fallback_categorization(subject, available_categories)
    
    def _smart_fallback_categorization(self, subject: str, available_categories: List[str]) -> List[str]:
        """Smart fallback categorization based on subject analysis."""
        subject_lower = subject.lower()
        
        # Subject-based keyword mapping
        keyword_mappings = {
            'scientist': ['Science'],
            'physicist': ['Physics', 'Science'], 
            'mathematician': ['Mathematics', 'Science'],
            'engineer': ['Technology', 'Science'],
            'doctor': ['Healthcare'],
            'ceo': ['Business', 'Entrepreneurship'],
            'founder': ['Entrepreneurship', 'Business'],
            'journalist': ['Journalism'],
            'activist': ['Activism'],
            'investor': ['Finance', 'Business'],
            'researcher': ['Science'],
            'professor': ['Science']
        }
        
        # Check for keywords in subject name/description
        for keyword, cats in keyword_mappings.items():
            if keyword in subject_lower:
                # Return categories that exist in available list
                valid_cats = [cat for cat in cats if cat in available_categories]
                if valid_cats:
                    return valid_cats[:2]
        
        # Default fallback
        return available_categories[:1] if available_categories else []

# Global helper instance
_openai_helper = None

def get_openai_helper() -> OpenAIHelper:
    """Get singleton OpenAI helper instance."""
    global _openai_helper
    if _openai_helper is None:
        _openai_helper = OpenAIHelper()
    return _openai_helper

def generate_content(prompt: str, model: str = "gpt-4", 
                    max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """Convenience function for content generation."""
    return get_openai_helper().generate_content(prompt, model, max_tokens, temperature)

def categorize_content(subject: str, content_sample: str, 
                      available_categories: List[str], existing_categories: Optional[List[Dict]] = None) -> List[str]:
    """Convenience function for content categorization."""
    return get_openai_helper().categorize_content(subject, content_sample, available_categories, existing_categories)