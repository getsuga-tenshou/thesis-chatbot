import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

from dotenv import load_dotenv

logger = logging.getLogger("api_client")
logger.setLevel(logging.INFO)

try:
    from pymongo import MongoClient
    from openai import OpenAI
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.warning(f"Failed to import one or more dependencies: {e}")
    IMPORTS_SUCCESSFUL = False

class SimpleAPIClient:
    def __init__(self, system_prompt: Optional[str] = None):
        load_dotenv()
        
        if not IMPORTS_SUCCESSFUL:
            raise ImportError("Required dependencies not available. Please install them and try again.")
            
        self._init_client(system_prompt)

    def _init_client(self, system_prompt: Optional[str] = None):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file.")

        self.mongo_db_uri = os.getenv("MONGO_DB_URI")
        self.mongodb_database_name = os.getenv("MONGODB_DATABASE_NAME", "Chatbot")
        self.mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME", "chatHistory")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        
        if not self.mongo_db_uri:
            raise ValueError("MONGO_DB_URI not set in .env file.")

        self.llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

        self.mongo_client = MongoClient(self.mongo_db_uri)
        self.mongo_client.admin.command('ping')
        logger.info("Connected to MongoDB Atlas successfully!")
        
        self.db = self.mongo_client[self.mongodb_database_name]
        self.collection = self.db[self.mongodb_collection_name]

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        
        # Create vector search index if it doesn't exist
        self._ensure_vector_index()

    def _ensure_vector_index(self):
        """Create vector search index if it doesn't exist"""
        try:
            
            indexes = list(self.collection.list_indexes())
            vector_index_exists = any("vector" in idx.get("name", "") for idx in indexes)
            
            if not vector_index_exists:
                
                index_definition = {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": 1536,     
                                "similarity": "cosine"
                            }
                        }
                    }
                }
                self.db.command({
                    "createSearchIndex": self.mongodb_collection_name,
                    "definition": index_definition,
                    "name": "vector_index"
                })
                logger.info(f"Created vector search index for collection {self.mongodb_collection_name}")
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def save_message_to_db(self, username: str, message: Dict[str, Any], session_id: Optional[str] = None):
        try:
            # Always generate a new embedding for each message
            content = message.get("content", "")
            embedding = self.generate_embedding(content)
            
            message_doc = {
                "username": username,
                "session_id": session_id or "default",
                "timestamp": datetime.utcnow(),
                "role": message.get("role"),
                "content": content,
                "embedding": embedding,
                "metadata": message.get("metadata", {})
            }
            self.collection.insert_one(message_doc)
            logger.info(f"Saved message with embedding to MongoDB for user: {username}")
        except Exception as e:
            logger.error(f"Error saving message to MongoDB: {e}")

    def get_conversation_history(self, username: str, session_id: Optional[str] = None, limit: int = 50):
        try:
            query = {"username": username}
            if session_id:
                query["session_id"] = session_id
            
            messages = list(self.collection.find(query)
                          .sort("timestamp", 1)
                          .limit(limit))
            
            return [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def generate_response(self, query: str, username: str, session_id: str, conversation_history: Optional[List[Dict]] = None) -> str:
        logger.info("Entered generate_response function.")
        if not query:
            logger.warning("Empty query received.")
            return "Query cannot be empty."
        
        logger.info("About to retrieve session summary.")
        try:
            # Check if we should update the summary
            if self.should_update_summary(session_id, username):
                logger.info("Summary needs to be updated.")
                summary = self.generate_chat_summary(session_id, username)
            else:
                summary = self.get_session_summary(session_id, username)
                
            logger.info(f"Summary retrieval complete. Summary: {summary[:100] if summary else 'None'}")
        except Exception as e:
            logger.error(f"Error retrieving summary: {e}", exc_info=True)
            summary = None

        system_prompt = self.system_prompt
        if summary:
            system_prompt += f"\n\nConversation summary so far:\n{summary}"
        logger.info("System prompt prepared.")

        try:
            logger.info(f"Generating response for query: '{query}'")
            messages = [{"role": "system", "content": system_prompt}]
            
            if conversation_history:
                logger.info("Adding conversation history to messages.")
                recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
                messages.extend(recent_history)
            
            messages.append({"role": "user", "content": query})
            logger.info("Messages prepared for OpenAI call.")

            logger.info("Calling OpenAI chat completion API...")
            response = self.openai_client.chat.completions.create(
                model=self.llm_model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            logger.info("OpenAI chat completion API call successful.")

            response_content = response.choices[0].message.content
            logger.info(f"OpenAI response content: {response_content[:100]}")  # Log first 100 chars

            self.save_message_to_db(username, {
                "role": "user",
                "content": query
            }, session_id)
            logger.info("User message saved to DB.")

            self.save_message_to_db(username, {
                "role": "assistant",
                "content": response_content
            }, session_id)
            logger.info("Assistant message saved to DB.")

            return response_content
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Apologies, an error occurred: {str(e)}"

    def find_similar_messages_by_embedding(self, query_embedding: List[float], username: str, limit: int = 5) -> List[Dict]:
        """
        Find similar messages using an existing embedding
        """
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "vector_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": limit
                        }
                    }
                },
                {
                    "$match": {
                        "username": username
                    }
                },
                {
                    "$project": {
                        "role": 1,
                        "content": 1,
                        "timestamp": 1,
                        "score": { "$meta": "searchScore" }
                    }
                }
            ]
            
            similar_messages = list(self.collection.aggregate(pipeline))
            return similar_messages
        except Exception as e:
            logger.error(f"Error finding similar messages: {e}")
            return []

    def generate_chat_summary(self, session_id: str, username: str, num_messages: int = 10) -> Optional[str]:
        logger.info(f"Entered generate_chat_summary with session_id: {session_id}, username: {username}")
        if not session_id or session_id == "default" or not username:
            logger.warning("Invalid session_id or username for summary generation.")
            return None

        try:
            logger.info("Fetching recent messages for summary.")
            messages = list(self.collection.find(
                {"session_id": session_id, "username": username}
            ).sort("timestamp", -1).limit(num_messages))
            
            if not messages:
                logger.info("No messages found for summary generation.")
                return None

            messages.reverse()
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            summary_prompt = """Please provide a concise summary of the following conversation. 
            Focus on the main topics discussed and any important decisions or conclusions reached. 
            Keep the summary under 200 words.
            
            Conversation:
            {text}"""
            
            logger.info("Calling OpenAI for summary generation.")
            summary = self.openai_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "system", "content": summary_prompt.format(text=text)}],
                temperature=0.3,
                max_tokens=200
            ).choices[0].message.content
            
            logger.info(f"Summary generated: {summary[:100]}")  # Log first 100 chars

            # Store summary in a separate collection
            summary_collection = self.db["Summaries"]
            summary_collection.update_one(
                {"session_id": session_id, "username": username},
                {
                    "$set": {
                        "summary": summary,
                        "updated_at": datetime.utcnow(),
                        "message_count": len(messages)
                    }
                },
                upsert=True
            )
            logger.info("Summary stored successfully.")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating or storing summary: {e}", exc_info=True)
            return None

    def get_session_summary(self, session_id: str, username: str) -> str:
        logger.info(f"Entered get_session_summary with session_id: {session_id}, username: {username}")
        if not session_id or session_id == "default" or not username:
            logger.warning("Invalid session_id or username for summary retrieval.")
            return ""
            
        try:
            summary_collection = self.db["Summaries"]
            doc = summary_collection.find_one({"session_id": session_id, "username": username})
            
            if doc and "summary" in doc:
                logger.info("Summary found in database.")
                return doc["summary"]
            else:
                logger.info("No summary found in database, generating new summary.")
                return self.generate_chat_summary(session_id, username) or ""
                
        except Exception as e:
            logger.error(f"Error retrieving summary from database: {e}", exc_info=True)
            return ""

    def should_update_summary(self, session_id: str, username: str) -> bool:
        try:
            summary_collection = self.db["Summaries"]
            doc = summary_collection.find_one({"session_id": session_id, "username": username})
            
            if not doc:
                return True
                
            # Get current message count
            current_count = self.collection.count_documents({
                "session_id": session_id,
                "username": username
            })
            
            # Update if message count has increased by more than 5
            return current_count - doc.get("message_count", 0) > 5
            
        except Exception as e:
            logger.error(f"Error checking if summary should be updated: {e}")
            return False

if __name__ == '__main__':
    print("Initializing SimpleAPIClient...")
    print("Ensure .env file has: OPENAI_API_KEY, MONGO_DB_URI")
    
    try:
        client = SimpleAPIClient()
        print("SimpleAPIClient initialized.")

        print("\nTesting response generation...")
        test_query = "Hello, how are you?"
        response = client.generate_response(test_query, "test_user", "default")
        print(f"User Query: {test_query}")
        print(f"LLM Response: {response}")

    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
        print("Please check your .env configuration and MongoDB connection.")
