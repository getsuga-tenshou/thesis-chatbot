import streamlit as st
from utils.auth_setup import setup_auth, login, is_authenticated

# Page configuration
st.set_page_config(page_title="Login - EthosBot", page_icon="🔑", layout="wide")

# Initialize authentication system
setup_auth()

# Check if already logged in
if is_authenticated():
    st.success("You are already logged in!")
    st.write("You can now access the application.")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Main Application", use_container_width=True):
            st.switch_page("main.py")
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.success("Logged out successfully!")
            st.rerun()
    
    st.stop()

# Main content
st.title("Log in to EthosBot")
st.markdown("Access the Socratic Ethics Assistant")

# Center the login form
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
                    # Store in session state
                    st.session_state.user = user
                    st.session_state.remember_me = remember_me
                    st.success("Logged in successfully!")
                    st.switch_page("main.py")
                else:
                    st.error("Invalid username or password.")
    
    # Navigation links
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign Up", use_container_width=True):
            st.switch_page("pages/2_👤_Signup.py")
    with col2:
        if st.button("Admin Panel", use_container_width=True):
            st.switch_page("pages/3_🔐_Admin.py")
    
    if st.button("Back to Main App", use_container_width=True):
        st.switch_page("main.py") 