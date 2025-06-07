import streamlit as st
import os
from dotenv import load_dotenv
from utils.auth_setup import setup_auth, is_authenticated, login
from utils.api_client import SimpleAPIClient
import uuid

load_dotenv()

st.set_page_config(
    page_title="EthosBot - Ethical Teaching Assistant",
    page_icon="🤖",
    layout="wide"
)

setup_auth()

if not is_authenticated():
    st.write("## Welcome to EthosBot")
    st.write("An Ethical Teaching Assistant")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("Please log in to access the application.")
        
        with st.form("login_form"):
            st.write("### Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember me")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                user = login(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.remember_me = remember_me
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sign Up", use_container_width=True):
                st.switch_page("pages/Signup.py")
        with col2:
            if st.button("Admin Panel", use_container_width=True):
                st.switch_page("pages/Admin.py")
else:
    st.title("EthosBot: Ethical Teaching Assistant")
    
    st.info("""
    ### How it works:

    Today's Topic: **Fine-Tuning** 🎯

    1. **Learning Phase** 📚
       - You'll first go through a learning phase where you'll explore and understand fine-tuning
       - The assistant will guide you through key concepts and principles of fine-tuning
       - Take your time to absorb the information and ask questions about fine-tuning techniques

    2. **Discussion Phase** 💭
       - After learning about fine-tuning, you'll enter a discussion phase
       - Share your thoughts on how you would apply fine-tuning in real-life scenarios
       - Discuss your decision-making process and reactions when implementing fine-tuning
       - The assistant will help you reflect on your choices and their implications

    Ready to begin? Type your message below to start the conversation!
    """)
    
    if "api_client" not in st.session_state:
        try:
            st.session_state.api_client = SimpleAPIClient(
                system_prompt="""You are an educational AI assistant focused on teaching fine-tuning in a structured manner. Follow these guidelines strictly:

1. Learning Phase:
   - Start with basic concepts and gradually progress to advanced topics
   - Provide clear explanations with examples
   - Only answer questions related to fine-tuning
   - If asked about unrelated topics, politely redirect to fine-tuning
   - Use a step-by-step teaching approach

2. Discussion Phase:
   - Use the Socratic method to guide ethical discussions
   - Ask thought-provoking questions about fine-tuning applications
   - Encourage critical thinking about implementation decisions
   - Focus on ethical considerations and real-world implications
   - Maintain a respectful and constructive dialogue

3. General Rules:
   - Stay strictly on topic (fine-tuning)
   - If asked about other topics, politely explain that this session is focused on fine-tuning
   - Maintain a professional and educational tone
   - Encourage active participation and questions
   - Provide constructive feedback and guidance"""
            )
            st.toast(f"Connected to LLM: {st.session_state.api_client.llm_model_name}", icon="✅")
        except Exception as e:
            st.error(f"Error initializing API client: {str(e)}")
            st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    with st.sidebar:
        st.subheader("Chat Options")
        
        if st.button("Load Chat History", use_container_width=True):
            try:
                history = st.session_state.api_client.get_conversation_history(
                    st.session_state.user["username"]
                )
                if history:
                    st.session_state.messages = history
                    st.success(f"Loaded {len(history)} messages")
                    st.rerun()
                else:
                    st.info("No chat history found")
            except Exception as e:
                st.error(f"Error loading history: {e}")
        
        st.write("---")
        
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.success("Logged out successfully!")
            st.rerun()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        user_message = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."):
                try:
                    response = st.session_state.api_client.generate_response(
                        user_input,
                        st.session_state.user["username"],
                        st.session_state.session_id,
                        st.session_state.messages[:-1]
                    )
                    
                    st.write(response)
                    
                    assistant_message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.rerun()

if __name__ == "__main__":
    pass
