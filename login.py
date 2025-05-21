import streamlit as st
from utils.auth_setup import setup_auth, login, is_authenticated

st.set_page_config(page_title="EthosBot Login", page_icon="🔑", layout="wide")

setup_auth()

if is_authenticated():
    st.success("You are already logged in!")
    st.write("You can now access the application.")
    st.markdown("[Go to Main Application](http://localhost:8501/app)")
    st.stop()

st.title("Log in to EthosBot")
st.markdown("Access the Socratic Ethics Assistant")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        remember_me = st.checkbox("Remember me")
        
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if not username or not password:
                st.error("Username and password are required.")
            else:
                user = login(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.remember_me = remember_me
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    
    st.markdown("Don't have an account? [Sign Up](http://localhost:8501/signup)")
    st.markdown("[Go to Admin Panel](http://localhost:8501/admin)")
    st.markdown("[Back to Main App](http://localhost:8501/app)")