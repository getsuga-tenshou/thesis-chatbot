import uuid
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import MongoDB dependencies
try:
    from pymongo.errors import PyMongoError
    from utils.auth_setup import USE_MONGODB, conversations_collection, db
    # Try to import threads_collection, but don't fail if it's not there yet
    try:
        from utils.auth_setup import threads_collection
    except ImportError:
        threads_collection = None
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    threads_collection = None

# Configure logging
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
        
        # Local storage paths
        self.thread_store_dir = "chat_history"
        self.user_dir = os.path.join(self.thread_store_dir, username or "anonymous")
        self.thread_file_path = os.path.join(self.user_dir, f"{self.thread_id}_thread.json")
        
        # Ensure storage directory exists
        if not os.path.exists(self.user_dir):
            os.makedirs(self.user_dir, exist_ok=True)
            
        # Configure storage mode
        self.use_mongodb = PYMONGO_AVAILABLE and USE_MONGODB and threads_collection is not None
                
        # Load or create thread
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
        # Try MongoDB first
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
        
        # Fallback to local file
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
        # Save to MongoDB
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
                
                # Upsert the thread data
                threads_collection.update_one(
                    {"thread_id": self.thread_id, "username": self.username},
                    {"$set": thread_data},
                    upsert=True
                )
                logger.debug(f"Saved thread {self.thread_id} to MongoDB")
            except Exception as e:
                logger.error(f"Failed to save thread {self.thread_id} to MongoDB: {e}")
                
        # Save to local file
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
        
        # Path for local file storage (fallback)
        self.message_store_dir = "chat_history"
        self.user_dir = os.path.join(self.message_store_dir, username or "anonymous")
        self.message_file_path = os.path.join(self.user_dir, f"{self.session_id}_messages.json")

        # Ensure storage directory exists
        if not os.path.exists(self.user_dir):
            os.makedirs(self.user_dir, exist_ok=True)
        
        # Configure storage mode
        self.use_mongodb = PYMONGO_AVAILABLE and USE_MONGODB
        
        # Initialize or load thread if username provided
        if username:
            # If thread_id provided, load that thread, otherwise create or load default thread
            if thread_id:
                self.current_thread = Thread(username, thread_id)
            else:
                # Try to find or create default thread
                self.current_thread = self._get_or_create_default_thread()
                if self.current_thread:
                    self.thread_id = self.current_thread.thread_id
            
            # Load most recent session if we have a thread
            if self.current_thread:
                self.load_most_recent_session()
                
    def _get_or_create_default_thread(self) -> Thread:
        """Get the default thread for the current user or create it if it doesn't exist

        Returns:
            Thread: Default thread
        """
        if not self.username:
            return None
            
        # Try to find default thread in MongoDB first
        if self.use_mongodb and threads_collection is not None:
            try:
                thread_data = threads_collection.find_one(
                    {"username": self.username, "is_default": True}
                )
                
                if thread_data:
                    return Thread(self.username, thread_data["thread_id"])
            except Exception as e:
                logger.error(f"Failed to get default thread from MongoDB: {e}")
                
        # Try local storage
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
            
        # Create default thread if not found
        default_thread = Thread(self.username, thread_name="Chat 1")
        
        # Mark as default in storage
        if self.use_mongodb and threads_collection is not None:
            try:
                threads_collection.update_one(
                    {"thread_id": default_thread.thread_id},
                    {"$set": {"is_default": True}}
                )
            except Exception as e:
                logger.error(f"Failed to mark thread as default in MongoDB: {e}")
                
        # Also mark in local file
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
        
        # Get threads from MongoDB
        if self.use_mongodb and threads_collection is not None:
            try:
                cursor = threads_collection.find({"username": self.username})
                for thread_data in cursor:
                    thread_data.pop("_id", None)  # Remove MongoDB ID
                    threads.append(thread_data)
            except Exception as e:
                logger.error(f"Failed to get threads from MongoDB: {e}")
                
        # Get threads from local storage
        try:
            thread_files = [f for f in os.listdir(self.user_dir) if f.endswith("_thread.json")]
            for file_name in thread_files:
                try:
                    with open(os.path.join(self.user_dir, file_name), 'r') as f:
                        thread_data = json.load(f)
                    
                    # Check if thread is already in the list (from MongoDB)
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
        
        # Update current thread
        self.current_thread = new_thread
        self.thread_id = new_thread.thread_id
        
        # Clear session to start fresh
        self.clear()
        
        # Save an empty message to ensure the session exists
        self.save_messages()
        
        # Explicitly load the new session
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
            
        # Load thread
        new_thread = Thread(self.username, thread_id)
        if not new_thread.thread_id:
            return False
            
        # Update current thread
        self.current_thread = new_thread
        self.thread_id = new_thread.thread_id
        
        # Load most recent session for this thread
        return self.load_most_recent_session()
    
    def load_most_recent_session(self) -> bool:
        """Load the most recent conversation session for the current user/thread

        Returns:
            bool: True if a session was loaded, False otherwise
        """
        if not self.username:
            return False
            
        # Try MongoDB first (if available)
        if self.use_mongodb:
            try:
                # Check if conversations_collection is available
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    # Create a filter based on username and thread_id
                    filter_query = {"username": self.username}
                    if self.thread_id:
                        filter_query["thread_id"] = self.thread_id
                        
                    # Find most recent session for this user/thread
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
                
        # Fallback to local storage
        try:
            # Find most recent session file
            if not os.path.exists(self.user_dir):
                return False
                
            session_files = [f for f in os.listdir(self.user_dir) if f.endswith("_messages.json")]
            if not session_files:
                return False
                
            # If thread_id specified, filter to only include files with that thread
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
                
            # Sort by file modification time (most recent first)
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
            
        # Try MongoDB first (if available)
        if self.use_mongodb:
            try:
                # Check if conversations_collection is available
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    # Create filter query
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
                        
                        # Update thread_id if it was in the message but not set in memory
                        if not self.thread_id and "thread_id" in messages[0]:
                            self.thread_id = messages[0]["thread_id"]
                            
                        logger.info(f"Loaded session {session_id} with {len(messages)} messages from MongoDB")
                        return True
                else:
                    logger.warning("MongoDB conversations_collection not available")
            except PyMongoError as e:
                logger.error(f"Failed to load session {session_id} from MongoDB: {e}")
        
        # Fallback to local storage
        try:
            session_file = os.path.join(self.user_dir, f"{session_id}_messages.json")
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    messages = json.load(f)
                
                # Convert string timestamps to datetime objects
                for msg in messages:
                    if "timestamp" in msg and isinstance(msg["timestamp"], str):
                        try:
                            # Handle ISO format strings
                            msg["timestamp"] = datetime.fromisoformat(msg["timestamp"].replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            # If parsing fails, use current time
                            msg["timestamp"] = datetime.now()
                
                # If thread_id specified, check that it matches
                if self.thread_id and messages and messages[0].get("thread_id") != self.thread_id:
                    logger.info(f"Session {session_id} belongs to different thread, not loading")
                    return False
                    
                # Update thread_id if it was in the message but not set in memory
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
        
        # Try MongoDB first (if available)
        if self.use_mongodb and self.username:
            try:
                # Check if conversations_collection is available
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    # Base query: username and optional thread_id
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
        
        # Add local sessions (if user directory exists)
        if os.path.exists(self.user_dir):
            try:
                session_files = [f for f in os.listdir(self.user_dir) if f.endswith("_messages.json")]
                
                for file in session_files:
                    session_id = file.split('_')[0]
                    file_path = os.path.join(self.user_dir, file)
                    
                    # Skip sessions already found in MongoDB
                    if any(s["session_id"] == session_id for s in sessions):
                        continue
                    
                    try:
                        with open(file_path, 'r') as f:
                            messages = json.load(f)
                            
                        # Skip if thread_id filter is applied and doesn't match
                        if self.thread_id and messages and messages[0].get("thread_id") != self.thread_id:
                            continue
                            
                        # Determine thread ID
                        session_thread_id = None
                        if messages:
                            # Get thread_id from first message if available
                            session_thread_id = messages[0].get("thread_id", None)
                        
                        # Determine timestamps for sorting
                        first_message_time = None
                        last_message_time = None
                        
                        if messages:
                            # Handle first message timestamp
                            if "timestamp" in messages[0]:
                                timestamp = messages[0]["timestamp"]
                                if isinstance(timestamp, str):
                                    try:
                                        first_message_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    except (ValueError, TypeError):
                                        first_message_time = datetime(2025, 1, 1)
                                elif isinstance(timestamp, datetime):
                                    first_message_time = timestamp
                            
                            # Handle last message timestamp
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
                        # Skip this corrupted file but continue processing others
                        try:
                            # Optionally delete or rename corrupted files
                            corrupted_path = os.path.join(self.user_dir, f"{file}.corrupted")
                            os.rename(file_path, corrupted_path)
                            logger.info(f"Renamed corrupted file {file} to {file}.corrupted")
                        except Exception:
                            # If rename fails, just continue
                            pass
            except Exception as e:
                logger.error(f"Failed to list local sessions: {e}")
        
        # Sort by last message timestamp, newest first
        def safe_sort_key(session):
            last_msg = session.get("last_message")
            # Handle different types of timestamps by converting to datetime object or using a fallback date
            if last_msg is None:
                # Use a recent future date for entries with no timestamp (sorts to beginning when reverse=True)
                return datetime(2025, 1, 1)
            elif isinstance(last_msg, datetime):
                return last_msg  # Already a datetime object
            elif isinstance(last_msg, str):
                # Try to parse string to datetime
                try:
                    return datetime.fromisoformat(last_msg.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    # If parsing fails, use a default date
                    return datetime(2025, 1, 1)
            else:
                # For any other type, use default date
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
        # Always use datetime objects for timestamps
        timestamp = datetime.now()
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "session_id": self.session_id
        }
        
        # Add username if available
        if self.username:
            message["username"] = self.username
            
        # Add thread_id if available
        if self.thread_id:
            message["thread_id"] = self.thread_id
        
        # Add to local memory first
        self.messages.append(message)
        
        # Try to save to MongoDB
        if self.use_mongodb and self.username:
            try:
                # Make sure conversations_collection is defined
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    # Create a copy of the message for MongoDB
                    mongo_message = message.copy()
                    # MongoDB already expects a datetime object, so no conversion needed
                    conversations_collection.insert_one(mongo_message)
                    logger.debug(f"Saved message to MongoDB: {message['id']}")
                else:
                    logger.warning("MongoDB conversations_collection not available")
            except PyMongoError as e:
                logger.error(f"Failed to save message to MongoDB: {e}")
        
        # For local storage, we'll convert datetime to string in save_messages
        self.save_messages()
        
        return message

    def _get_or_create_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Extract or create a summary from message history
        
        Args:
            messages: List of message objects
            
        Returns:
            str: Summary text or default message
        """
        # Check if there's a summary message
        for msg in messages:
            if msg.get("is_summary", False):
                return msg["content"]
                
        return "No summary available for this conversation."

    def save_messages(self) -> None:
        """Save messages to storage (MongoDB and/or local file)"""
        # Always save to local file as backup
        try:
            # Convert messages to JSON-serializable format
            json_messages = []
            for msg in self.messages:
                json_msg = msg.copy()
                # Convert datetime objects to ISO format strings for JSON serialization
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
        # Update local summary
        self.summary = summary_text
        
        # Check if we should add as a message
        if self.use_mongodb and self.username:
            try:
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    # Create a special summary message
                    summary_message = {
                        "id": str(uuid.uuid4()),
                        "username": self.username,
                        "session_id": self.session_id,
                        "role": "system",
                        "content": summary_text,
                        "timestamp": datetime.now(),
                        "is_summary": True
                    }
                    
                    # Add thread_id if available
                    if self.thread_id:
                        summary_message["thread_id"] = self.thread_id
                    
                    # Save to MongoDB
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
        
        # Also update thread summary if available
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
            
        # Build a sample of messages from this thread
        all_sampled_messages = []
        
        # Try MongoDB first
        if self.use_mongodb:
            try:
                if 'conversations_collection' in globals() and conversations_collection is not None:
                    # Sample messages from this thread
                    cursor = conversations_collection.find(
                        {"thread_id": self.thread_id, "role": {"$in": ["user", "assistant"]}},
                        {"role": 1, "content": 1, "_id": 0}
                    ).sort("timestamp", -1).limit(10)  # Get 10 most recent messages
                    
                    thread_messages = list(cursor)
                    all_sampled_messages.extend(thread_messages)
            except Exception as e:
                logger.error(f"Error retrieving thread messages from MongoDB: {e}")
                
        # Use local files if needed
        if not all_sampled_messages:
            try:
                session_files = [f for f in os.listdir(self.user_dir) if f.endswith("_messages.json")]
                thread_messages = []
                
                for file in session_files[:5]:  # Limit to 5 files for performance
                    try:
                        with open(os.path.join(self.user_dir, file), 'r') as f:
                            messages = json.load(f)
                            
                        # Check if these messages belong to our thread
                        if messages and messages[0].get("thread_id") == self.thread_id:
                            # Add a sample of messages from this file
                            sample_size = min(3, len(messages))
                            sampled = [{"role": msg["role"], "content": msg["content"]} 
                                      for msg in messages[-sample_size:]]  # Take most recent
                            thread_messages.extend(sampled)
                    except Exception as e:
                        logger.error(f"Error reading session file {file}: {e}")
                
                all_sampled_messages.extend(thread_messages)
            except Exception as e:
                logger.error(f"Error processing local files for thread summary: {e}")
                
        # If we have messages, generate a summary
        if all_sampled_messages:
            from utils.config import THREAD_SUMMARY_TEMPLATE
            
            # Format messages for the prompt
            messages_text = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content'][:100]}..." 
                for msg in all_sampled_messages
            ])
            
            # Generate summary
            try:
                summary_prompt = THREAD_SUMMARY_TEMPLATE.format(
                    thread_name=self.current_thread.thread_name,
                    messages=messages_text
                )
                
                summary = api_client.llm.invoke(summary_prompt)
                
                if isinstance(summary, str):
                    # Extract summary if in expected format
                    if "Summary:" in summary:
                        summary = summary.split("Summary:")[-1].strip()
                    
                    # Update thread summary
                    self.current_thread.update_summary(summary)
                    return summary
                
            except Exception as e:
                logger.error(f"Error generating thread summary: {e}")
        
        # Return existing summary if we couldn't generate a new one
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
            # Save the old session ID for reference
            old_session_id = self.session_id
            
            # Generate new session ID
            self.session_id = str(uuid.uuid4())
            self.messages = []
            self.summary = "No conversation has occurred yet."
            
            # Update file path for local storage
            self.message_file_path = os.path.join(self.user_dir, f"{self.session_id}_messages.json")
            
            # Save the empty state
            self.save_messages()
            
            logger.info(f"Cleared session {old_session_id}, started new session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False

if __name__ == '__main__':
    # Example Usage
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