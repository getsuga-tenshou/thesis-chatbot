import streamlit as st
from utils.auth_setup import setup_auth, is_authenticated, create_user, get_user


st.set_page_config(page_title="Sign Up - EthosBot", page_icon="👤", layout="wide")


setup_auth()


if is_authenticated():
    st.success("You are already logged in!")
    st.write("You can now access the application.")
    
    
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


st.title("Create an Account")
st.markdown("Sign up to access the Socratic Ethics Assistant")


col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    signup_success = False
    
    with st.form("signup_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submitted = st.form_submit_button("Sign Up")
        
        if submitted:
            if not new_username or not new_password:
                st.error("Username and password are required.")
            elif new_password != confirm_password:
                st.error("Passwords don't match.")
            elif get_user(new_username):
                st.error("Username already exists.")
            else:
                if create_user(new_username, new_password, 0):
                    st.success(f"User {new_username} created successfully!")
                    signup_success = True
                else:
                    st.error("Failed to create user.")
    
    if signup_success:
        st.info("Please log in with your new account")
        if st.button("Go to Login"):
            st.switch_page("pages/1_🔑_Login.py")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            st.switch_page("pages/1_🔑_Login.py")
    with col2:
        if st.button("Admin Panel", use_container_width=True):
            st.switch_page("pages/3_🔐_Admin.py")
    
    if st.button("Back to Main App", use_container_width=True):
        st.switch_page("main.py") 