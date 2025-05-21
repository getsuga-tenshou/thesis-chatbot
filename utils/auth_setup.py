import os
import streamlit as st
from dotenv import load_dotenv
import extra_streamlit_components as stx
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
import hashlib
import sqlite3
from pathlib import Path
import logging
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auth_setup")

load_dotenv()

USERNAME = "username"
PASSWORD = "password"
NAME = "name"
SU = "su"
BLANK = ""

db_path = os.getenv('SQLITE_DB_PATH', './db')
db_name = os.getenv('SQLITE_DB', 'auth.db')
os.makedirs(db_path, exist_ok=True)
db_file = os.path.join(db_path, db_name)

MONGO_DB_URI = os.getenv('MONGO_DB_URI')
MONGODB_DATABASE_NAME = os.getenv('MONGODB_DATABASE_NAME', 'Chatbot')
MONGODB_USERS_COLLECTION = os.getenv('MONGODB_USERS_COLLECTION', 'userData')
MONGODB_CONVERSATIONS_COLLECTION = os.getenv('MONGODB_CONVERSATIONS_COLLECTION', 'chatHistory')
threads_collection = None

USE_MONGODB = bool(MONGO_DB_URI)

mongo_client = None
db = None
users_collection = None
conversations_collection = None

@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection and collections"""
    global mongo_client, db, users_collection, conversations_collection, threads_collection
    
    if not MONGO_DB_URI:
        logger.warning("MONGO_DB_URI not set. Using SQLite for authentication.")
        return False
    
    try:
        mongo_client = MongoClient(MONGO_DB_URI)
        mongo_client.admin.command('ping')
        logger.info("Connected to MongoDB successfully!")
        
        db = mongo_client[MONGODB_DATABASE_NAME]
        users_collection = db[MONGODB_USERS_COLLECTION]
        conversations_collection = db[MONGODB_CONVERSATIONS_COLLECTION]
        threads_collection = db['threads']
        
        users_collection.create_index([("username", ASCENDING)], unique=True)
        conversations_collection.create_index([("session_id", ASCENDING)])
        conversations_collection.create_index([("username", ASCENDING)])
        conversations_collection.create_index([("username", 1), ("session_id", 1)])
        conversations_collection.create_index([("username", 1), ("thread_id", 1)])
        threads_collection.create_index([("username", 1), ("thread_id", 1)])
        threads_collection.create_index([("username", 1), ("is_default", 1)])
        
        return True
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return False

class AES256CBC:
    def __init__(self, key, nonce):
        self.key = hashlib.sha256(key.encode()).digest()
        self.nonce = nonce.encode()
    
    def encrypt(self, plaintext):
        try:
            cipher = AES.new(self.key, AES.MODE_CBC)
            ct_bytes = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
            iv = base64.b64encode(cipher.iv).decode('utf-8')
            ct = base64.b64encode(ct_bytes).decode('utf-8')
            return f"{iv}:{ct}"
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return None
    
    def decrypt(self, ciphertext):
        try:
            iv, ct = ciphertext.split(':')
            iv = base64.b64decode(iv)
            ct = base64.b64decode(ct)
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            pt = unpad(cipher.decrypt(ct), AES.block_size)
            return pt.decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return None

class SafeCookieManager:
    def __init__(self):
        self._cookie_manager = None
        self._cookies = {}
        self._initialized = False
    
    @property
    def cookie_manager(self):
        if self._cookie_manager is None:
            try:
                self._cookie_manager = stx.CookieManager()
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize cookie manager: {e}")
        return self._cookie_manager
    
    def get(self, cookie_name):
        """Safely get a cookie value"""
        if not self._initialized:
            return None
            
        try:
            if cookie_name in self._cookies:
                return self._cookies[cookie_name]
                
            result = self.cookie_manager.get(cookie=cookie_name)
            if result:
                self._cookies[cookie_name] = result
            return result
        except Exception as e:
            logger.error(f"Error getting cookie '{cookie_name}': {e}")
            return None
    
    def set(self, cookie_name, value):
        """Safely set a cookie value"""
        if not self._initialized:
            return False
            
        try:
            self.cookie_manager.set(cookie_name, value)
            self._cookies[cookie_name] = value
            return True
        except Exception as e:
            logger.error(f"Error setting cookie '{cookie_name}': {e}")
            return False
    
    def delete(self, cookie_name):
        """Safely delete a cookie"""
        if not self._initialized:
            return False
            
        try:
            if cookie_name in self._cookies:
                del self._cookies[cookie_name]
                
            try:
                self.cookie_manager.delete(cookie_name)
            except KeyError:
                pass
            except Exception as e:
                logger.error(f"Error deleting cookie '{cookie_name}': {e}")
                
            return True
        except Exception as e:
            logger.error(f"General error in delete cookie '{cookie_name}': {e}")
            return False
    
    def get_all(self):
        """Safely get all cookies"""
        if not self._initialized:
            return {}
            
        try:
            all_cookies = self.cookie_manager.get_all()
            self._cookies.update(all_cookies)
            return all_cookies
        except Exception as e:
            logger.error(f"Error getting all cookies: {e}")
            return {}

def get_cookie_manager():
    return SafeCookieManager()

def mongodb_get_user(username):
    """Get a user from MongoDB by username"""
    if users_collection is None:
        return None
        
    try:
        user = users_collection.find_one({"username": username})
        return user
    except Exception as e:
        logger.error(f"Error getting user from MongoDB: {e}")
        return None

def mongodb_create_user(username, password, name="", is_su=0):
    """Create a new user in MongoDB"""
    if users_collection is None:
        return False
        
    try:
        enc_password = os.getenv('ENC_PASSWORD', 'YouWillNeverGuessThisSecretKey32')
        enc_nonce = os.getenv('ENC_NONCE', 'nonsensical')
        encrypted_password = AES256CBC(enc_password, enc_nonce).encrypt(password)
        
        if not encrypted_password:
            logger.error("Failed to encrypt password")
            return False
            
        now = datetime.now().isoformat()
        user_doc = {
            "username": username,
            "password": encrypted_password,
            "name": name,
            "su": is_su,
            "created_at": now,
            "last_login": now,
            "login_count": 0,
        }
        
        result = users_collection.insert_one(user_doc)
        return bool(result.inserted_id)
    except DuplicateKeyError:
        logger.warning(f"User '{username}' already exists in MongoDB")
        return False
    except Exception as e:
        logger.error(f"Error creating user in MongoDB: {e}")
        return False

def mongodb_get_all_users():
    """Get all users from MongoDB"""
    if users_collection is None:
        return []
        
    try:
        return list(users_collection.find({}, {"_id": 0}))
    except Exception as e:
        logger.error(f"Error getting all users from MongoDB: {e}")
        return []

def mongodb_delete_user(username):
    """Delete a user from MongoDB"""
    if users_collection is None:
        return False
        
    try:
        result = users_collection.delete_one({"username": username})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Error deleting user from MongoDB: {e}")
        return None

def mongodb_update_login(username):
    """Update the last login time and login count for a user"""
    if users_collection is None:
        return False
        
    try:
        now = datetime.now().isoformat()
        result = users_collection.update_one(
            {"username": username},
            {
                "$set": {"last_login": now},
                "$inc": {"login_count": 1}
            }
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error updating login info in MongoDB: {e}")
        return None

def init_db():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT,
        su INTEGER DEFAULT 0
    )
    ''')
    conn.commit()
    conn.close()

def get_user(username):
    """Get a user by username - uses MongoDB if available, otherwise SQLite"""
    if USE_MONGODB:
        user = mongodb_get_user(username)
        if user:
            return (user["username"], user["password"], user["su"])
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT username, password, su FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        return user
    except Exception as e:
        logger.error(f"Error getting user '{username}': {e}")
        return None

def create_user(username, password, is_su=0, name=""):
    """Create a new user - uses MongoDB if available, otherwise SQLite"""
    if not username or not password:
        return False
    
    if USE_MONGODB:
        return mongodb_create_user(username, password, name, is_su)
    
    try:
        enc_password = os.getenv('ENC_PASSWORD', 'YouWillNeverGuessThisSecretKey32')
        enc_nonce = os.getenv('ENC_NONCE', 'nonsensical')
        encrypted_password = AES256CBC(enc_password, enc_nonce).encrypt(password)
        
        if not encrypted_password:
            logger.error("Failed to encrypt password")
            return False
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password, name, su) VALUES (?, ?, ?, ?)", 
                        (username, encrypted_password, name, is_su))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"User '{username}' already exists")
            conn.close()
            return False
    except Exception as e:
        logger.error(f"Error creating user '{username}': {e}")
        return False

def get_all_users():
    """Get all users - uses MongoDB if available, otherwise SQLite"""
    if USE_MONGODB:
        mongo_users = mongodb_get_all_users()
        if mongo_users:
            return [(user["username"], user["password"], user["su"]) for user in mongo_users]
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT username, password, su FROM users")
        users = cursor.fetchall()
        conn.close()
        return users
    except Exception as e:
        logger.error(f"Error getting all users: {e}")
        return []

def delete_user(username):
    """Delete a user - uses MongoDB if available, otherwise SQLite"""
    if USE_MONGODB:
        if mongodb_delete_user(username):
            return True
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting user '{username}': {e}")
        return False

def login(username, password):
    user = get_user(username)
    if not user:
        return None
    
    try:
        stored_username, stored_password, is_su = user
        
        enc_password = os.getenv('ENC_PASSWORD', 'YouWillNeverGuessThisSecretKey32')
        enc_nonce = os.getenv('ENC_NONCE', 'nonsensical')
        decrypted_password = AES256CBC(enc_password, enc_nonce).decrypt(stored_password)
        
        if decrypted_password is None:
            logger.error(f"Failed to decrypt password for user '{username}'")
            return None
            
        if password == decrypted_password:
            if USE_MONGODB:
                mongodb_update_login(username)
                
            return {"username": stored_username, "password": stored_password, "su": is_su}
        return None
    except Exception as e:
        logger.error(f"Error during login for user '{username}': {e}")
        return None

def handle_auth_cookie():
    cookie_manager = get_cookie_manager()
    cookie_name = os.getenv('COOKIE_NAME', 'ethosbot-auth')
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if st.session_state.user is None:
        stored_cookie = cookie_manager.get(cookie_name)
        if stored_cookie:
            user = get_user(stored_cookie.get('username'))
            if user and user[1] == stored_cookie.get('password'):
                st.session_state.user = {
                    'username': user[0],
                    'password': user[1],
                    'su': user[2]
                }
    
    
    if st.session_state.user:
        if st.session_state.get('remember_me', False):
            cookie_manager.set(
                cookie_name, 
                {"username": st.session_state.user["username"], "password": st.session_state.user["password"]}
            )
        else:
            cookie_manager.delete(cookie_name)

def admin_panel():
    if not st.session_state.user or st.session_state.user['su'] != 1:
        st.error("You need to be a super user to access this panel")
        return
    
    st.title("Admin Panel")
    
    st.subheader("All Users")
    users = get_all_users()
    user_data = []
    for user in users:
        user_data.append({"Username": user[0], "Is Admin": "Yes" if user[2] == 1 else "No"})
    st.table(user_data)
    
    st.subheader("Create User")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    new_name = st.text_input("Display Name")
    is_admin = st.checkbox("Is Admin")
    
    if st.button("Create User"):
        if create_user(new_username, new_password, 1 if is_admin else 0, new_name):
            st.success(f"User {new_username} created successfully!")
            st.rerun()
        else:
            st.error("Failed to create user. Username may already exist.")
    
    st.subheader("Delete User")
    users = get_all_users()
    usernames = [user[0] for user in users]
    username_to_delete = st.selectbox("Select user to delete", [""] + usernames)
    
    if username_to_delete and st.button("Delete User"):
        delete_user(username_to_delete)
        st.success(f"User {username_to_delete} deleted successfully!")
        st.rerun()

def setup_auth():
    global USE_MONGODB
    USE_MONGODB = init_mongodb()
    
    init_db()
    
    admin = get_user("admin")
    if not admin:
        create_user("admin", "admin", 1, "Administrator")
        st.success("Created default admin user (username: admin, password: admin)")
        st.warning("Please change the default admin password!")
    
    handle_auth_cookie()

def is_authenticated():
    return st.session_state.user is not None

def requires_auth(func):
    def wrapper(*args, **kwargs):
        if is_authenticated():
            return func(*args, **kwargs)
        else:
            st.error("You must be logged in to view this content")
    return wrapper 