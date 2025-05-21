import streamlit as st
from utils.auth_setup import setup_auth, is_authenticated, admin_panel

st.set_page_config(page_title="EthosBot Admin", page_icon="🔐", layout="wide")

setup_auth()

st.title("EthosBot Admin Panel")

if not is_authenticated():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Please log in with an administrator account to access the admin panel.")
        st.markdown("""
        ### Access Options
        - [Login](http://localhost:8501/login) if you have an admin account
        - [Back to Main App](http://localhost:8501/app)
        """)
else:
    user = st.session_state.user
    
    if user['su'] == 1:
        admin_panel()
        
        st.markdown("---")
        st.markdown("""
        ### Navigation
        - [Main Application](http://localhost:8501/app)
        - [Logout](http://localhost:8501/login)
        """)
    else:
        st.error("You need administrator privileges to access this page.")
        st.markdown("Please log in with an administrator account or contact the system administrator.")
        
        st.markdown("---")
        st.markdown("""
        ### Navigation
        - [Back to Main App](http://localhost:8501/app)
        - [Logout](http://localhost:8501/login)
        """)
        
        if st.button("Logout"):
            st.session_state.user = None
            st.success("Logged out successfully!")
            st.rerun()