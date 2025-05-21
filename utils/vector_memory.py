import uuid
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from pymongo.errors import PyMongoError
    from utils.auth_setup import USE_MONGODB, conversations_collection, db
    try:
        from utils.auth_setup import threads_collection
    except ImportError:
        threads_collection = None
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    threads_collection = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_memory")

class Thread:
    def __init__(self, username: str, thread_id: Optional[str] = None, thread_name: Optional[str] = None):
        """Initialize a thread to manage multiple conversations
        
        Args:
            username: Username who owns this thread
            thread_id: Optional thread ID (will be generated if not provided)
            thread_name: Optional name for the thread
        """
        self.username = username
        self.thread_id = thread_id or str(uuid.uuid4())
        self.thread_name = thread_name or f"Chat {datetime.now().strftime('%Y-%m-%d')}"
        self.created_at = datetime.now()
        self.summary = ""
        
        self.thread_store_dir = "chat_history"
        self.user_dir = os.path.join(self.thread_store_dir, username or "anonymous")
        self.thread_file_path = os.path.join(self.user_dir, f"{self.thread_id}_thread.json")
        
        if not os.path.exists(self.user_dir):
            os.makedirs(self.user_dir, exist_ok=True)
            
        self.use_mongodb = PYMONGO_AVAILABLE and USE_MONGODB and threads_collection is not None
                
        if thread_id:
            self.load_thread(thread_id)
        else:
            self.save_thread()
    
    def load_thread(self, thread_id: str) -> bool:
        """Load thread metadata from storage
        
        Args:
            thread_id: ID of thread to load
            
        Returns:
            bool: Success status
        """
        if self.use_mongodb:
            try:
                thread_data = threads_collection.find_one(
                    {"username": self.username, "thread_id": thread_id}
                )
                
                if thread_data:
                    self.thread_id = thread_data["thread_id"]
                    self.thread_name = thread_data.get("thread_name", self.thread_name)
                    self.created_at = thread_data.get("created_at", self.created_at)
                    self.summary = thread_data.get("summary", "")
                    logger.info(f"Loaded thread {thread_id} from MongoDB")
                    return True
            except Exception as e:
                logger.error(f"Failed to load thread {thread_id} from MongoDB: {e}")
        
        try:
            thread_file = os.path.join(self.user_dir, f"{thread_id}_thread.json")
            if os.path.exists(thread_file):
                with open(thread_file, 'r') as f:
                    thread_data = json.load(f)
                
                self.thread_id = thread_data["thread_id"]
                self.thread_name = thread_data.get("thread_name", self.thread_name)
                self.created_at = datetime.fromisoformat(
                    thread_data.get("created_at", datetime.now().isoformat())
                )
                self.summary = thread_data.get("summary", "")
                self.thread_file_path = thread_file
                logger.info(f"Loaded thread {thread_id} from local storage")
                return True
        except Exception as e:
            logger.error(f"Failed to load thread {thread_id} from local storage: {e}")
        
        return False
    
    def save_thread(self) -> bool:
        """Save thread metadata to storage
        
        Returns:
            bool: Success status
        """
        if self.use_mongodb:
            try:
                thread_data = {
                    "thread_id": self.thread_id,
                    "username": self.username,
                    "thread_name": self.thread_name,
                    "created_at": self.created_at,
                    "summary": self.summary,
                    "updated_at": datetime.now()
                }
                
                threads_collection.update_one(
                    {"thread_id": self.thread_id, "username": self.username},
                    {"$set": thread_data},
                    upsert=True
                )
                logger.debug(f"Saved thread {self.thread_id} to MongoDB")
            except Exception as e:
                logger.error(f"Failed to save thread {self.thread_id} to MongoDB: {e}")
                
        try:
            thread_data = {
                "thread_id": self.thread_id,
                "username": self.username,
                "thread_name": self.thread_name,
                "created_at": self.created_at.isoformat(),
                "summary": self.summary,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.thread_file_path, 'w') as f:
                json.dump(thread_data, f, indent=2)
            
            logger.debug(f"Saved thread {self.thread_id} to local file")
            return True
        except Exception as e:
            logger.error(f"Failed to save thread {self.thread_id} to local file: {e}")
            return False
    
    def update_summary(self, summary: str) -> bool:
        """Update the thread summary
        
        Args:
            summary: New summary text
            
        Returns:
            bool: Success status
        """
        self.summary = summary
        return self.save_thread()
    
    def rename(self, new_name: str) -> bool:
        """Rename the thread
        
        Args:
            new_name: New thread name
            
        Returns:
            bool: Success status
        """
        self.thread_name = new_name
        return self.save_thread()

class VectorMemory:
    def __init__(self, username: Optional[str] = None, thread_id: Optional[str] = None):
        """Initialize conversation memory with MongoDB or file-based storage

        Args:
            username: Optional username for associating conversations with users
            thread_id: Optional thread ID to associate with conversations
        """
        self.messages: List[Dict[str, Any]] = []
        self.session_id: str = str(uuid.uuid4())
        self.summary: str = "No conversation has occurred yet."
        self.username = username
        self.thread_id = thread_id
        self.current_thread = None
        
        self.message_store_dir = "chat_history"
        self.user_dir = os.path.join(self.message_store_dir, username or "anonymous")
        self.message_file_path = os.path.join(self.user_dir, f"{self.session_id}_messages.json")

        if not os.path.exists(self.user_dir):
            os.makedirs(self.user_dir, exist_ok=True)
        
        self.use_mongodb = PYMONGO_AVAILABLE and USE_MONGODB
        
        if username:
            if thread_id:
                self.current_thread = Thread(username, thread_id)
            else:
                self.current_thread = self._get_or_create_default_thread()
                if self.current_thread:
                    self.thread_id = self.current_thread.thread_id
            
            if self.current_thread:
                self.load_most_recent_session()
                
    def _get_or_create_default_thread(self) -> Thread:
        """Get the default thread for the current user or create it if it doesn't exist

        Returns:
            Thread: Default thread
        """
        if not self.username:
            return None
            
        if self.use_mongodb and threads_collection is not None:
            try:
                thread_data = threads_collection.find_one(
                    {"username": self.username, "is_default": True}
                )
                
                if thread_data:
                    return Thread(self.username, thread_data["thread_id"])
            except Exception as e:
                logger.error(f"Failed to get default thread from MongoDB: {e}")
                
        try:
            thread_dirs = [f for f in os.listdir(self.user_dir) if f.endswith("_thread.json")]
            for thread_file in thread_dirs:
                try:
                    with open(os.path.join(self.user_dir, thread_file), 'r') as f:
                        thread_data = json.load(f)
                    
                    if thread_data.get("is_default", False):
                        return Thread(self.username, thread_data["thread_id"])
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Failed to get default thread from local storage: {e}")
            
        default_thread = Thread(self.username, thread_name="Chat 1")
        
        if self.use_mongodb and threads_collection is not None:
            try:
                threads_collection.update_one(
                    {"thread_id": default_thread.thread_id},
                    {"$set": {"is_default": True}}
                )
            except Exception as e:
                logger.error(f"Failed to mark thread as default in MongoDB: {e}")
                
        try:
            with open(default_thread.thread_file_path, 'r') as f:
                thread_data = json.load(f)
                
            thread_data["is_default"] = True
            
            with open(default_thread.thread_file_path, 'w') as f:
                json.dump(thread_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to mark thread as default in local file: {e}")
            
        return default_thread
        
    def get_all_threads(self) -> List[Dict[str, Any]]:
        """Get all threads for the current user

        Returns:
            List of thread metadata dictionaries
        """
        if not self.username:
            return []
            
        threads = []
        
        if self.use_mongodb and threads_collection is not None:
            try:
                cursor = threads_collection.find({"username": self.username})
                for thread_data in cursor:
                    thread_data.pop("_id", None)  # Remove MongoDB ID
                    threads.append(thread_data)
            except Exception as e:
                logger.error(f"Failed to get threads from MongoDB: {e}")
                
        try:
            thread_files = [f for f in os.listdir(self.user_dir) if f.endswith("_thread.json")]
            for file_name in thread_files:
                try:
                    with open(os.path.join(self.user_dir, file_name), 'r') as f:
                        thread_data = json.load(f)
                    
                    if not any(t["thread_id"] == thread_data["thread_id"] for t in threads):
                        threads.append(thread_data)
                except Exception as e:
                    logger.error(f"Failed to read thread file {file_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to get threads from local storage: {e}")
            
        return threads
        
    def create_new_thread(self, thread_name: Optional[str] = None) -> Thread:
        """Create a new thread

        Args:
            thread_name: Optional name for the thread

        Returns:
            Thread: Newly created thread
        """
        if not self.username:
            return None
            
        new_thread = Thread(self.username, thread_name=thread_name)
        
        self.current_thread = new_thread
        self.thread_id = new_thread.thread_id
        
        self.clear()
        

        self.save_messages()
        
        logger.info(f"Created new thread: {thread_name} with ID: {new_thread.thread_id}")
        
        return new_thread
        
    def switch_thread(self, thread_id: str) -> bool:
        """Switch to a different thread

        Args:
            thread_id: ID of thread to switch to

        Returns:
            bool: Success status
        """
        if not self.username:
            return False
            
        new_thread = Thread(self.username, thread_id)
        if not new_thread.thread_id:
            return False
            
        self.current_thread = new_thread
        self.thread_id = new_thread.thread_id
        
        return self.load_most_recent_session()
    
    def load_most_recent_session(self) -> bool:
        """Load the most recent conversation session for the current user/thread

        Returns:
            bool: True if a session was loaded, False otherwise
        """
        if not self.username:
            return False
            
        
        if self.use_mongodb:
            try:
                
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    
                    filter_query = {"username": self.username}
                    if self.thread_id:
                        filter_query["thread_id"] = self.thread_id
                        
                    
                    pipeline = [
                        {"$match": filter_query},
                        {"$sort": {"timestamp": -1}},
                        {"$limit": 1},
                        {"$group": {"_id": "$session_id"}}
                    ]
                    result = list(conversations_collection.aggregate(pipeline))
                    
                    if result:
                        session_id = result[0]["_id"]
                        return self.load_session(session_id)
                    else:
                        logger.info(f"No previous sessions found for user {self.username} in MongoDB")
                else:
                    logger.warning("MongoDB conversations_collection not available")
            except PyMongoError as e:
                logger.error(f"Failed to load recent session from MongoDB: {e}")
                
        try:
            if not os.path.exists(self.user_dir):
                return False
                
            session_files = [f for f in os.listdir(self.user_dir) if f.endswith("_messages.json")]
            if not session_files:
                return False
                
            if self.thread_id:
                thread_session_files = []
                for file_name in session_files:
                    try:
                        with open(os.path.join(self.user_dir, file_name), 'r') as f:
                            data = json.load(f)
                            if data and len(data) > 0 and data[0].get("thread_id") == self.thread_id:
                                thread_session_files.append(file_name)
                    except Exception:
                        continue
                        
                if thread_session_files:
                    session_files = thread_session_files
            
            if not session_files:
                return False
                
            session_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.user_dir, f)), reverse=True)
            
            most_recent_file = session_files[0]
            session_id = most_recent_file.split('_')[0]
            
            return self.load_session(session_id)
        except Exception as e:
            logger.error(f"Failed to load recent session from local storage: {e}")
            
        return False
            
    def new_conversation(self) -> bool:
        """Start a new conversation in the current thread

        Returns:
            bool: Success status
        """
        return self.clear()

    # Update other methods to integrate with Thread
    def load_session(self, session_id: str) -> bool:
        """Load a specific conversation session by ID

        Args:
            session_id: The session ID to load

        Returns:
            bool: True if session was loaded successfully, False otherwise
        """
        if not session_id:
            return False
            
        if self.use_mongodb:
            try:
                
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    
                    filter_query = {"username": self.username, "session_id": session_id}
                    if self.thread_id:
                        filter_query["thread_id"] = self.thread_id
                        
                    cursor = conversations_collection.find(
                        filter_query,
                        {"_id": 0, "role": 1, "content": 1, "timestamp": 1, "thread_id": 1}
                    ).sort("timestamp", 1)
                    
                    messages = list(cursor)
                    if messages:
                        self.session_id = session_id
                        self.messages = messages
                        self.summary = self._get_or_create_summary(messages)
                        
                        
                        if not self.thread_id and "thread_id" in messages[0]:
                            self.thread_id = messages[0]["thread_id"]
                            
                        logger.info(f"Loaded session {session_id} with {len(messages)} messages from MongoDB")
                        return True
                else:
                    logger.warning("MongoDB conversations_collection not available")
            except PyMongoError as e:
                logger.error(f"Failed to load session {session_id} from MongoDB: {e}")
        
        try:
            session_file = os.path.join(self.user_dir, f"{session_id}_messages.json")
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    messages = json.load(f)
                
                
                for msg in messages:
                    if "timestamp" in msg and isinstance(msg["timestamp"], str):
                        try:
                            msg["timestamp"] = datetime.fromisoformat(msg["timestamp"].replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            msg["timestamp"] = datetime.now()
                
                
                if self.thread_id and messages and messages[0].get("thread_id") != self.thread_id:
                    logger.info(f"Session {session_id} belongs to different thread, not loading")
                    return False
                    
                
                if not self.thread_id and messages and "thread_id" in messages[0]:
                    self.thread_id = messages[0]["thread_id"]
                
                self.session_id = session_id
                self.messages = messages
                self.message_file_path = session_file
                self.summary = self._get_or_create_summary(self.messages)
                logger.info(f"Loaded session {session_id} with {len(self.messages)} messages from local storage")
                return True
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from local storage: {e}")
            
        return False

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get a list of all sessions for the current user or thread

        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        
        if self.use_mongodb and self.username:
            try:
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    base_query = {"username": self.username}
                    if self.thread_id:
                        base_query["thread_id"] = self.thread_id
                        
                    pipeline = [
                        {"$match": base_query},
                        {"$group": {
                            "_id": "$session_id",
                            "message_count": {"$sum": 1},
                            "first_message": {"$min": "$timestamp"},
                            "last_message": {"$max": "$timestamp"},
                            "thread_id": {"$first": "$thread_id"}
                        }}
                    ]
                    
                    cursor = conversations_collection.aggregate(pipeline)
                    mongodb_sessions = list(cursor)
                    
                    for session in mongodb_sessions:
                        sessions.append({
                            "session_id": session["_id"],
                            "message_count": session["message_count"],
                            "first_message": session["first_message"],
                            "last_message": session["last_message"],
                            "thread_id": session.get("thread_id", self.thread_id),
                            "source": "mongodb"
                        })
                else:
                    logger.warning("MongoDB conversations_collection not available")
            except PyMongoError as e:
                logger.error(f"Failed to list sessions from MongoDB: {e}")
        
        if os.path.exists(self.user_dir):
            try:
                session_files = [f for f in os.listdir(self.user_dir) if f.endswith("_messages.json")]
                
                for file in session_files:
                    session_id = file.split('_')[0]
                    file_path = os.path.join(self.user_dir, file)
                    
                    
                    if any(s["session_id"] == session_id for s in sessions):
                        continue
                    
                    try:
                        with open(file_path, 'r') as f:
                            messages = json.load(f)
                            
                        if self.thread_id and messages and messages[0].get("thread_id") != self.thread_id:
                            continue
                            
                        session_thread_id = None
                        if messages:
                            session_thread_id = messages[0].get("thread_id", None)
                        
                        
                        first_message_time = None
                        last_message_time = None
                        
                        if messages:
                            if "timestamp" in messages[0]:
                                timestamp = messages[0]["timestamp"]
                                if isinstance(timestamp, str):
                                    try:
                                        first_message_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    except (ValueError, TypeError):
                                        first_message_time = datetime(2025, 1, 1)
                                elif isinstance(timestamp, datetime):
                                    first_message_time = timestamp
                            
                            if "timestamp" in messages[-1]:
                                timestamp = messages[-1]["timestamp"]
                                if isinstance(timestamp, str):
                                    try:
                                        last_message_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    except (ValueError, TypeError):
                                        last_message_time = datetime(2025, 1, 1)
                                elif isinstance(timestamp, datetime):
                                    last_message_time = timestamp
                        
                        sessions.append({
                            "session_id": session_id,
                            "message_count": len(messages),
                            "first_message": first_message_time,
                            "last_message": last_message_time,
                            "thread_id": session_thread_id,
                            "source": "local"
                        })
                    except Exception as e:
                        logger.error(f"Error reading session file {file}: {e}")
                        
                        try:
                            
                            corrupted_path = os.path.join(self.user_dir, f"{file}.corrupted")
                            os.rename(file_path, corrupted_path)
                            logger.info(f"Renamed corrupted file {file} to {file}.corrupted")
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Failed to list local sessions: {e}")
        
        
        def safe_sort_key(session):
            last_msg = session.get("last_message")
            if last_msg is None:
                return datetime(2025, 1, 1)
            elif isinstance(last_msg, datetime):
                return last_msg  # Already a datetime object
            elif isinstance(last_msg, str):
                try:
                    return datetime.fromisoformat(last_msg.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    
                    return datetime(2025, 1, 1)
            else:
                
                return datetime(2025, 1, 1)
                
        sessions.sort(key=safe_sort_key, reverse=True)
        return sessions

    def add_message(self, role: str, content: str) -> Dict[str, Any]:
        """Add a message to the conversation history
        
        Args:
            role: The role of the message sender (user, assistant)
            content: The message content
            
        Returns:
            Dict containing the added message
        """
        
        timestamp = datetime.now()
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "session_id": self.session_id
        }
        
        
        if self.username:
            message["username"] = self.username
            
        if self.thread_id:
            message["thread_id"] = self.thread_id
        
        
        self.messages.append(message)
        
        
        if self.use_mongodb and self.username:
            try:
                
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    
                    mongo_message = message.copy()
                    
                    conversations_collection.insert_one(mongo_message)
                    logger.debug(f"Saved message to MongoDB: {message['id']}")
                else:
                    logger.warning("MongoDB conversations_collection not available")
            except PyMongoError as e:
                logger.error(f"Failed to save message to MongoDB: {e}")
        
        
        self.save_messages()
        
        return message

    def _get_or_create_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Extract or create a summary from message history
        
        Args:
            messages: List of message objects
            
        Returns:
            str: Summary text or default message
        """
        
        for msg in messages:
            if msg.get("is_summary", False):
                return msg["content"]
                
        return "No summary available for this conversation."

    def save_messages(self) -> None:
        """Save messages to storage (MongoDB and/or local file)"""
        try:
            json_messages = []
            for msg in self.messages:
                json_msg = msg.copy()
                
                if isinstance(json_msg.get("timestamp"), datetime):
                    json_msg["timestamp"] = json_msg["timestamp"].isoformat()
                json_messages.append(json_msg)
                
            with open(self.message_file_path, 'w') as f:
                json.dump(json_messages, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving messages to local file: {str(e)}")

    def store_summary(self, summary_text: str) -> None:
        """Store a summary of the conversation
        
        Args:
            summary_text: The summary text
        """
        
        self.summary = summary_text
        
        
        if self.use_mongodb and self.username:
            try:
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    
                    summary_message = {
                        "id": str(uuid.uuid4()),
                        "username": self.username,
                        "session_id": self.session_id,
                        "role": "system",
                        "content": summary_text,
                        "timestamp": datetime.now(),
                        "is_summary": True
                    }
                    
                    
                    if self.thread_id:
                        summary_message["thread_id"] = self.thread_id
                    
                    
                    conversations_collection.insert_one(summary_message)
                else:
                    logger.warning("MongoDB conversations_collection not available")
            except PyMongoError as e:
                logger.error(f"Failed to store summary in MongoDB: {e}")
    
    def get_recent_messages_text(self, k: int = 3) -> str:
        """Returns the last k messages as a formatted string
        
        Args:
            k: Number of recent messages to include
            
        Returns:
            Formatted string of messages
        """
        if not self.messages:
            return ""
        
        recent_k_messages = self.messages[-k:]
        formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_k_messages])
        return formatted_history

    def update_summary(self, new_summary: str) -> None:
        """Update the conversation summary
        
        Args:
            new_summary: New summary text
        """
        self.summary = new_summary
        self.store_summary(new_summary)
        
        
        if self.current_thread:
            self.current_thread.update_summary(new_summary)

    def update_thread_summary(self, summary_text: str) -> bool:
        """Update the summary for the current thread
        
        Args:
            summary_text: New summary text
            
        Returns:
            bool: Success status
        """
        if not self.current_thread:
            return False
            
        return self.current_thread.update_summary(summary_text)
        
    def generate_thread_summary(self, api_client=None) -> str:
        """Generate a summary of the entire thread using the LLM
        
        Args:
            api_client: Optional API client for LLM access
            
        Returns:
            str: Generated summary
        """
        if not self.current_thread or not api_client or not self.thread_id:
            return self.current_thread.summary if self.current_thread else ""
            

        all_sampled_messages = []
        
        if self.use_mongodb:
            try:
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    
                    cursor = conversations_collection.find(
                        {"thread_id": self.thread_id, "role": {"$in": ["user", "assistant"]}},
                        {"role": 1, "content": 1, "_id": 0}
                    ).sort("timestamp", -1).limit(10)  # Get 10 most recent messages
                    
                    thread_messages = list(cursor)
                    all_sampled_messages.extend(thread_messages)
            except Exception as e:
                logger.error(f"Error retrieving thread messages from MongoDB: {e}")
                
        if not all_sampled_messages:
            try:
                session_files = [f for f in os.listdir(self.user_dir) if f.endswith("_messages.json")]
                thread_messages = []
                
                for file in session_files[:5]:  # Limit to 5 files for performance
                    try:
                        with open(os.path.join(self.user_dir, file), 'r') as f:
                            messages = json.load(f)
                            
                        if messages and messages[0].get("thread_id") == self.thread_id:
                            sample_size = min(3, len(messages))
                            sampled = [{"role": msg["role"], "content": msg["content"]} 
                                      for msg in messages[-sample_size:]] 
                            thread_messages.extend(sampled)
                    except Exception as e:
                        logger.error(f"Error reading session file {file}: {e}")
                
                all_sampled_messages.extend(thread_messages)
            except Exception as e:
                logger.error(f"Error processing local files for thread summary: {e}")
                
        if all_sampled_messages:
            from utils.config import THREAD_SUMMARY_TEMPLATE
            
            
            messages_text = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content'][:100]}..." 
                for msg in all_sampled_messages
            ])
            
            
            try:
                summary_prompt = THREAD_SUMMARY_TEMPLATE.format(
                    thread_name=self.current_thread.thread_name,
                    messages=messages_text
                )
                
                summary = api_client.llm.invoke(summary_prompt)
                
                if isinstance(summary, str):
                    
                    if "Summary:" in summary:
                        summary = summary.split("Summary:")[-1].strip()
                    
                    
                    self.current_thread.update_summary(summary)
                    return summary
                
            except Exception as e:
                logger.error(f"Error generating thread summary: {e}")
        
        
        return self.current_thread.summary if self.current_thread else ""

    def get_all_messages(self) -> List[Dict[str, Any]]:
        """Get all messages for the current session
        
        Returns:
            List of message dictionaries
        """
        return self.messages
    
    def clear(self) -> bool:
        """Clear the current conversation and start a new session"""
        try:

            old_session_id = self.session_id
            

            self.session_id = str(uuid.uuid4())
            self.messages = []
            self.summary = "No conversation has occurred yet."
            
            
            self.message_file_path = os.path.join(self.user_dir, f"{self.session_id}_messages.json")
            
            
            self.save_messages()
            
            logger.info(f"Cleared session {old_session_id}, started new session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False

if __name__ == '__main__':
    
    memory = VectorMemory(username="test_user")
    print(f"Session ID: {memory.session_id}")

    memory.add_message("user", "Hello, assistant!")
    memory.add_message("assistant", "Hello! How can I help you today?")
    memory.add_message("user", "Tell me about ethics.")

    print("\nAll Messages:")
    for msg in memory.get_all_messages():
        print(msg)

    print(f"\nSummary: {memory.summary}")
    memory.update_summary("User and assistant exchanged greetings. User asked about ethics.")
    print(f"Updated Summary: {memory.summary}")
    
    print(f"\nRecent messages (text):\n{memory.get_recent_messages_text(k=2)}")

    all_sessions = memory.get_all_sessions()
    print(f"\nAll sessions for {memory.username}: {len(all_sessions)}")
    for session in all_sessions:
        print(f"- {session['session_id']} ({session['message_count']} messages, source: {session['source']})") 