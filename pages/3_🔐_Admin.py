import streamlit as st
from utils.auth_setup import setup_auth, is_authenticated, admin_panel


st.set_page_config(page_title="Admin Panel - EthosBot", page_icon="🔐", layout="wide")


setup_auth()


st.title("EthosBot Admin Panel")


if not is_authenticated():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Please log in with an administrator account to access the admin panel.")
        
        with st.form("admin_login_form"):
            st.write("### Admin Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            submitted = st.form_submit_button("Login")
            
            if submitted:
                from utils.auth_setup import login
                user = login(username, password)
                if user:
                    if user['su'] == 1:
                        st.session_state.user = user
                        st.success("Admin login successful!")
                        st.rerun()
                    else:
                        st.error("This account does not have administrator privileges.")
                else:
                    st.error("Invalid username or password.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", use_container_width=True):
                st.switch_page("pages/1_🔑_Login.py")
        with col2:
            if st.button("Sign Up", use_container_width=True):
                st.switch_page("pages/2_👤_Signup.py")
        
        if st.button("Back to Main App", use_container_width=True):
            st.switch_page("main.py")
else:
    user = st.session_state.user
    
    if user['su'] == 1:
        admin_panel()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Main Application", use_container_width=True):
                st.switch_page("main.py")
        with col2:
            if st.button("Logout", use_container_width=True):
                st.session_state.user = None
                st.success("Logged out successfully!")
                st.rerun()
    else:
        st.error("You need administrator privileges to access this page.")
        st.markdown("Please log in with an administrator account or contact the system administrator.")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to Main App", use_container_width=True):
                st.switch_page("main.py")
        with col2:
            if st.button("Logout", use_container_width=True):
                st.session_state.user = None
                st.success("Logged out successfully!")
                st.rerun() 