import streamlit as st
from utils.auth_setup import setup_auth, is_authenticated, create_user, get_user, login

st.set_page_config(page_title="EthosBot Signup", page_icon="👤", layout="wide")

setup_auth()

if is_authenticated():
    st.success("You are already logged in!")
    st.write("You can now access the application.")
    st.markdown("[Go to Main Application](http://localhost:8501/app)")
    st.stop()

st.title("Create an Account")
st.markdown("Sign up to access the Socratic Ethics Assistant")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
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
                    st.info("Please log in with your new account.")
                    st.markdown("[Go to Login](http://localhost:8501/login)")
                else:
                    st.error("Failed to create user.")
    
    st.markdown("Already have an account? [Login](http://localhost:8501/login)")
    st.markdown("[Go to Admin Panel](http://localhost:8501/admin)")
    st.markdown("[Back to Main App](http://localhost:8501/app)")