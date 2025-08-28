"""MongoDB utility functions for database operations."""

import os
import logging
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class MongoDBHelper:
    def __init__(self):
        """Initialize MongoDB connection."""
        self.db = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup MongoDB connection."""
        env_path = "config/.env"
        
        if os.path.exists(env_path):
            load_dotenv(env_path)
        
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise Exception(f"MONGODB_URI not found in {env_path}")
        
        client = MongoClient(mongodb_uri)
        self.db = client.get_default_database()
        logger.info("Connected to MongoDB")
    
    def save_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Save a document to the specified collection."""
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            logger.info(f"Saved document to {collection_name}, ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to save document: {str(e)}")
            raise Exception(f"Database save failed: {str(e)}")
    
    def find_documents(self, collection_name: str, query: Dict[str, Any] = None, 
                      projection: Dict[str, Any] = None, limit: int = None) -> List[Dict]:
        """Find documents in the specified collection."""
        try:
            collection = self.db[collection_name]
            cursor = collection.find(query or {}, projection)
            
            if limit:
                cursor = cursor.limit(limit)
            
            documents = list(cursor)
            
            # Convert ObjectIds to strings for JSON serialization
            for doc in documents:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
            
            return documents
        except Exception as e:
            logger.error(f"Failed to find documents: {str(e)}")
            raise Exception(f"Database query failed: {str(e)}")
    
    def update_document(self, collection_name: str, document_id: str, 
                       update_data: Dict[str, Any]) -> bool:
        """Update a document in the specified collection."""
        try:
            collection = self.db[collection_name]
            result = collection.update_one(
                {"_id": ObjectId(document_id)}, 
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated document {document_id} in {collection_name}")
                return True
            else:
                logger.warning(f"No document updated for ID {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            raise Exception(f"Database update failed: {str(e)}")
    
    def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document from the specified collection."""
        try:
            collection = self.db[collection_name]
            result = collection.delete_one({"_id": ObjectId(document_id)})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted document {document_id} from {collection_name}")
                return True
            else:
                logger.warning(f"No document deleted for ID {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise Exception(f"Database delete failed: {str(e)}")

# Global helper instance
_mongodb_helper = None

def get_mongodb_helper() -> MongoDBHelper:
    """Get singleton MongoDB helper instance."""
    global _mongodb_helper
    if _mongodb_helper is None:
        _mongodb_helper = MongoDBHelper()
    return _mongodb_helper

def save_document(collection_name: str, document: Dict[str, Any]) -> str:
    """Convenience function to save a document."""
    return get_mongodb_helper().save_document(collection_name, document)

def find_documents(collection_name: str, query: Dict[str, Any] = None, 
                  projection: Dict[str, Any] = None, limit: int = None) -> List[Dict]:
    """Convenience function to find documents."""
    return get_mongodb_helper().find_documents(collection_name, query, projection, limit)