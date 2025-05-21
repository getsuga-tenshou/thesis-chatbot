import os
import logging
from typing import List, Optional, Dict, Any
import uuid

from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger("api_client")
logger.setLevel(logging.INFO)

# Import MongoDB and OpenAI libraries
try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    from langchain_mongodb import MongoDBAtlasVectorSearch
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.messages import SystemMessage, HumanMessage
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.warning(f"Failed to import one or more dependencies: {e}")
    IMPORTS_SUCCESSFUL = False

class LangchainMongoDBClient:
    def __init__(self, system_prompt: Optional[str] = None):
        load_dotenv()
        
        if not IMPORTS_SUCCESSFUL:
            raise ImportError("Required dependencies not available. Please install them and try again.")
            
        self._init_client(system_prompt)

    def _init_client(self, system_prompt: Optional[str] = None):
        # Initialize OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file.")

        # Initialize MongoDB connection
        self.mongo_db_uri = os.getenv("MONGO_DB_URI")
        self.mongodb_database_name = os.getenv("MONGODB_DATABASE_NAME", "Chatbot")
        self.mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME", "Embeddings")
        
        if not self.mongo_db_uri:
            raise ValueError("MONGO_DB_URI not set in .env file.")

        # Set model names
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini-2024-07-18")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

        # Initialize MongoDB client
        self.mongo_client = MongoClient(self.mongo_db_uri)
        # Ping to confirm connection
        self.mongo_client.admin.command('ping')
        logger.info("Connected to MongoDB Atlas successfully!")
        
        # Set up database and collection
        self.db = self.mongo_client[self.mongodb_database_name]
        self.collection = self.db[self.mongodb_collection_name]
        
        # Create vector search index if it doesn't exist
        try:
            # Check if collection has documents
            if self.collection.count_documents({}) == 0:
                logger.info(f"Collection {self.mongodb_collection_name} is empty")
                
            # Check if collection has a vector search index
            indexes = list(self.collection.list_indexes())
            vector_index_exists = any("vector" in idx.get("name", "") for idx in indexes)
            
            if not vector_index_exists:
                # Create vector search index
                try:
                    index_definition = {
                        "mappings": {
                            "dynamic": True,
                            "fields": {
                                "embedding": {
                                    "type": "knnVector",
                                    "dimensions": 1536,  # For text-embedding-3-small
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
                    logger.error(f"Failed to create vector search index: {e}")
                    logger.warning("Proceeding with direct LLM queries without vector search")
        except Exception as e:
            logger.error(f"Error checking vector index: {e}")

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=self.openai_api_key
        )

        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model_name=self.llm_model_name,
            temperature=0.5,
            openai_api_key=self.openai_api_key
        )
    
        # Set up vector store
        try:
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name="vector_index",
                text_key="text",
                embedding_key="embedding"
            )
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
            )
            logger.info("Vector search retriever initialized successfully")
            self.use_rag = True
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            logger.warning("Falling back to direct LLM queries without RAG")
            self.use_rag = False
        
        # Set up system prompt
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
    
        # Set up prompt templates
        # RAG prompt template (with document context)
        self.rag_chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            SystemMessagePromptTemplate.from_template(
                "Context information from knowledge base:\n{context}\n\n"
                "Conversation summary: {summary}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
        # Direct LLM prompt template (fallback, no document context)
        self.direct_chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            SystemMessagePromptTemplate.from_template("Conversation summary: {summary}"),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

        # Set up RAG chain
        if self.use_rag:
            self.rag_chain = (
                {
                    "context": lambda x: self.get_relevant_context(x["question"]),
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: self.format_chat_history(x["chat_history"]),
                    "summary": lambda x: x.get("summary", "No summary available.")
                }
                | self.rag_chat_prompt
                | self.llm
                | StrOutputParser()
            )
        
        # Set up direct LLM chain (fallback or when no relevant docs found)
        self.direct_llm_chain = (
            self.direct_chat_prompt
            | self.llm
            | StrOutputParser()
        )

    def format_chat_history(self, chat_history_str: str) -> List:
        if not chat_history_str:
            return []
            
        messages = []
        lines = chat_history_str.strip().split("\n")
        
        for line in lines:
            if line.startswith("User:"):
                content = line[5:].strip()
                messages.append(HumanMessage(content=content))
            elif line.startswith("Assistant:"):
                content = line[10:].strip()
                messages.append(SystemMessage(content=content))
        
        return messages

    def get_relevant_context(self, query: str) -> str:
        try:
            if not self.use_rag:
                return ""
                
            docs = self.retriever.get_relevant_documents(query)
            if not docs:
                return ""
                
            context_texts = [doc.page_content for doc in docs]
            return "\n\n".join(context_texts)
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """Add texts to vector store with embeddings
        
        Args:
            texts: List of text documents to add
            metadatas: Optional metadata for each document
            
        Returns:
            List of document IDs
        """
        if not texts:
            logger.warning("No texts provided to add.")
            return []
            
        try:
            logger.info(f"Adding {len(texts)} texts to MongoDB collection '{self.mongodb_collection_name}'...")
            
            # Split long texts into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            all_chunks = []
            all_metadatas = []
            
            # Process each text document
            for i, text in enumerate(texts):
                # Split text into smaller chunks
                chunks = text_splitter.split_text(text)
                all_chunks.extend(chunks)
                
                # Create metadata for each chunk
                if metadatas and i < len(metadatas):
                    doc_metadata = metadatas[i]
                    # Apply the same metadata to all chunks from this document
                    all_metadatas.extend([doc_metadata.copy() for _ in range(len(chunks))])
                else:
                    # Create default metadata
                    all_metadatas.extend([{"source": f"document_{i}"} for _ in range(len(chunks))])
            
            # Add document chunks to vector store
            if self.use_rag:
                try:
                    ids = self.vector_store.add_texts(
                        texts=all_chunks,
                        metadatas=all_metadatas
                    )
                    logger.info(f"Added {len(ids)} text chunks with embeddings")
                    return ids
                except Exception as e:
                    logger.error(f"Error adding texts to vector store: {e}")
                    # Fall back to simple storage without embeddings
                    logger.warning("Falling back to simple document storage")
            
            # Simple document storage without vector embeddings (fallback)
            document_ids = []
            for i, chunk in enumerate(all_chunks):
                doc_id = str(uuid.uuid4())
                document_ids.append(doc_id)
                
                doc = {
                    "_id": doc_id,
                    "text": chunk,
                    "metadata": all_metadatas[i] if i < len(all_metadatas) else {}
                }
                self.collection.insert_one(doc)
            
            logger.info(f"Added {len(document_ids)} text chunks without embeddings")
            return document_ids
        except Exception as e:
            logger.error(f"Error adding texts: {e}")
            raise

    def generate_rag_response(self, query: str, conversation_history=None) -> str:
        """Generate a response using RAG or direct LLM based on availability
        
        Args:
            query: User's query
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response text
        """
        if not query:
            return "Query cannot be empty."
            
        try:
            logger.info(f"Generating response for query: '{query}'")
            
            # Format conversation history if provided
            chat_history = ""
            summary = "No summary available."
            if conversation_history:
                # Limit to the most recent messages (prevent context length issues)
                recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
                
                # Format history as a string with clear role indicators
                history_messages = []
                for msg in recent_history:
                    # Skip system messages
                    if msg.get('role') == 'system':
                        if msg.get('is_summary', False):
                            summary = msg.get('content', summary)
                        continue
                    role = msg.get('role', '').capitalize()
                    content = msg.get('content', '')
                    history_messages.append(f"{role}: {content}")
                
                chat_history = "\n".join(history_messages)
                logger.debug(f"Using conversation history with {len(recent_history)} messages")
            
            # Prepare input for chain
            chain_input = {
                "question": query,
                "chat_history": chat_history,
                "summary": summary
            }
            
            # Try RAG approach first
            if self.use_rag:
                try:
                    # Check if we have documents in the collection
                    doc_count = self.collection.count_documents({})
                    if doc_count > 0:
                        logger.info(f"Using RAG with {doc_count} documents available")
                        response = self.rag_chain.invoke(chain_input)
                        return response
                    else:
                        logger.warning("No documents in collection, falling back to direct LLM")
                except Exception as e:
                    logger.error(f"RAG retrieval failed: {e}")
                    logger.warning("Falling back to direct LLM")
            
            # Fallback to direct LLM
            response = self.direct_llm_chain.invoke(chain_input)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Apologies, an error occurred: {str(e)}"

if __name__ == '__main__':
    print("Initializing LangchainMongoDBClient...")
    print("Ensure .env file has: OPENAI_API_KEY, MONGO_DB_URI")
    
    try:
        client = LangchainMongoDBClient()
        print("LangchainMongoDBClient initialized.")

        print("\nTesting response generation...")
        questions = [
            "What is Langchain?",
            "What color is the sky based on the context provided?"
        ]
        
        for q_idx, test_query in enumerate(questions):
            print(f"\n--- Query {q_idx+1} ---")
            print(f"User Query: {test_query}")
            response = client.generate_rag_response(test_query)
            print(f"LLM Response: {response}")

    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
        print("Please check your .env configuration, MongoDB connection, and model names.") 