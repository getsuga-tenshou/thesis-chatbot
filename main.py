import streamlit as st
import os
from dotenv import load_dotenv
from utils.auth_setup import setup_auth, is_authenticated, login
from utils.api_client import LangchainMongoDBClient

load_dotenv()

st.set_page_config(
    page_title="EthosBot - Socratic Ethics Assistant",
    page_icon="🧠",
    layout="wide"
)

setup_auth()

if not is_authenticated():
    st.write("## Welcome to EthosBot")
    st.write("This application provides an interactive Socratic Ethics Assistant.")
    
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
                st.switch_page("pages/2_👤_Signup.py")
        with col2:
            if st.button("Admin Panel", use_container_width=True):
                st.switch_page("pages/3_🔐_Admin.py")
else:
    st.title("EthosBot: Socratic Ethics Assistant")
    
    if "api_client" not in st.session_state:
        try:
            from utils.config import SOCRATIC_PROMPT
            st.session_state.api_client = LangchainMongoDBClient(system_prompt=SOCRATIC_PROMPT)
            st.toast(f"Connected to LLM: {st.session_state.api_client.llm_model_name}", icon="✅")
        except Exception as e:
            st.error(f"Error initializing API client: {str(e)}")
            st.write("Using mock client for demonstration purposes.")
    
    if "memory" not in st.session_state:
        from utils.vector_memory import VectorMemory
        st.session_state.memory = VectorMemory(username=st.session_state.user["username"])
        if len(st.session_state.memory.get_all_messages()) > 0:
            st.toast(f"Loaded previous conversation with {len(st.session_state.memory.get_all_messages())} messages", icon="📝")
    
    with st.sidebar:
        st.subheader("Chat Instances")
        
        all_chats = st.session_state.memory.get_all_threads()
        
        if st.button("New Chat", use_container_width=True):
            new_chat = st.session_state.memory.create_new_thread(f"Chat {len(all_chats) + 1}")
            st.session_state.new_thread_created = True
            st.session_state.current_thread_id = new_chat.thread_id
            st.rerun()
        
        st.write("---")
        
        for chat in all_chats:
            chat_name = chat.get("thread_name", f"Chat {chat.get('thread_id')[:6]}")
            is_current = chat.get("thread_id") == st.session_state.memory.thread_id
            button_text = f"► {chat_name}" if is_current else chat_name
            
            if st.button(button_text, key=chat["thread_id"], use_container_width=True):
                st.session_state.memory.switch_thread(chat["thread_id"])
                st.rerun()
        
        st.write("---")
        
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.success("Logged out successfully!")
            st.rerun()

    chat_col = st.container()
    
    with chat_col:
        for message in st.session_state.memory.get_all_messages():
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    user_prompt = st.chat_input("Share your thoughts or ask about an ethical scenario...")
    
    if user_prompt:
        previous_messages = st.session_state.memory.get_all_messages()
        
        st.session_state.memory.add_message("user", user_prompt)
        
        with st.chat_message("user"):
            st.write(user_prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."):
                try:
                    response_text = st.session_state.api_client.generate_rag_response(user_prompt, previous_messages)
                    
                    st.write(response_text)
                    
                    st.session_state.memory.add_message("assistant", response_text)
                    
                    if len(st.session_state.memory.get_all_messages()) > 1:
                        from utils.config import SUMMARY_TEMPLATE
                        conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" 
                                                    for msg in st.session_state.memory.get_all_messages()[-8:]])
                        summary_prompt_text = SUMMARY_TEMPLATE.format(conversation=conversation_text)
                        summary_response = st.session_state.api_client.llm.invoke(summary_prompt_text)
                        if isinstance(summary_response, str):
                            summary = summary_response
                            if "Summary:" in summary:
                                summary = summary.split("Summary:")[-1].strip()
                            st.session_state.memory.update_summary(summary)
                    
                    st.rerun()
                except Exception as e:
                    error_msg = f"Apologies, an error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.memory.add_message("assistant", error_msg)
                    st.rerun()

if __name__ == "__main__":
    pass