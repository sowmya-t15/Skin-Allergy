"""
Authentication Module for Streamlit App
Handles user login, registration, and session management
"""

import streamlit as st
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from database_manager import DatabaseManager

class AuthManager:
    """Manages authentication and user sessions in Streamlit"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
        # Initialize session state
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'session_id' not in st.session_state:
            st.session_state.session_id = secrets.token_urlsafe(32)
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current logged-in user"""
        return st.session_state.get('user')
    
    def get_session_id(self) -> str:
        """Get current session ID"""
        return st.session_state.get('session_id', '')
    
    def login_form(self) -> bool:
        """Display login form and handle authentication"""
        st.markdown("## 🔐 Login to Your Account")
        
        with st.form("login_form"):
            st.markdown("### Please enter your credentials")
            username = st.text_input("Username or Email", placeholder="Enter username or email")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            remember_me = st.checkbox("Remember me")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🔑 Login", use_container_width=True)
            with col2:
                if st.form_submit_button("📝 Register", use_container_width=True):
                    st.session_state.show_register = True
                    st.rerun()
        
        if login_button:
            if not username or not password:
                st.error("Please fill in all fields")
                return False
            
            with st.spinner("Authenticating..."):
                result = self.db.authenticate_user(username, password)
                
                if result['success']:
                    # Set session state
                    st.session_state.authenticated = True
                    st.session_state.user = result['user']
                    
                    st.success(f"Welcome back, {result['user']['username']}! 🎉")
                    time.sleep(1)  # Brief pause for user experience
                    st.rerun()
                    return True
                else:
                    st.error(f"❌ {result['message']}")
                    return False
        
        return False
    
    def register_form(self) -> bool:
        """Display registration form and handle user creation"""
        st.markdown("## 📝 Create New Account")
        
        with st.form("register_form"):
            st.markdown("### Please fill in your information")
            
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                username = st.text_input("Username", placeholder="Choose a username")
            with col2:
                email = st.text_input("Email", placeholder="Enter your email")
                
            password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            col1, col2 = st.columns(2)
            with col1:
                register_button = st.form_submit_button("✅ Create Account", use_container_width=True)
            with col2:
                if st.form_submit_button("🔙 Back to Login", use_container_width=True):
                    st.session_state.show_register = False
                    st.rerun()
        
        if register_button:
            # Validation
            errors = []
            if not all([full_name, username, email, password, confirm_password]):
                errors.append("Please fill in all fields")
            if len(username) < 3:
                errors.append("Username must be at least 3 characters")
            if len(password) < 6:
                errors.append("Password must be at least 6 characters")
            if password != confirm_password:
                errors.append("Passwords do not match")
            if not terms:
                errors.append("Please accept the terms and conditions")
            if "@" not in email or "." not in email:
                errors.append("Please enter a valid email address")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
                return False
            
            with st.spinner("Creating account..."):
                result = self.db.create_user(username, email, password, full_name)
                
                if result['success']:
                    st.success("🎉 Account created successfully!")
                    st.info("Please log in with your new credentials")
                    time.sleep(1)
                    st.session_state.show_register = False
                    st.rerun()
                    return True
                else:
                    st.error(f"❌ {result['message']}")
                    return False
        
        return False
    
    def logout(self):
        """Log out the current user"""
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_id = secrets.token_urlsafe(32)
        st.success("👋 Logged out successfully!")
        time.sleep(1)
        st.rerun()
    
    def require_authentication(self, show_login: bool = True) -> bool:
        """Require authentication for a page/function"""
        if not self.is_authenticated():
            if show_login:
                st.warning("🔒 Please log in to access this feature")
                self.show_auth_ui()
            return False
        return True
    
    def show_auth_ui(self):
        """Show authentication UI (login/register)"""
        # Check if we should show register form
        show_register = st.session_state.get('show_register', False)
        
        if show_register:
            self.register_form()
        else:
            self.login_form()
        
        # Add some helpful links
        st.markdown("---")
        st.markdown("### Need Help?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Forgot Password?"):
                st.info("Password reset functionality coming soon!")
        with col2:
            if st.button("📧 Contact Support"):
                st.info("Email: support@skinallergy.app")
        with col3:
            if st.button("📚 User Guide"):
                st.info("Visit our help documentation")
    
    def user_profile_sidebar(self):
        """Show user profile in sidebar"""
        if self.is_authenticated():
            user = self.get_current_user()
            
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 👤 User Profile")
                st.markdown(f"**{user['full_name']}**")
                st.markdown(f"@{user['username']}")
                st.markdown(f"📧 {user['email']}")
                
                # User actions
                if st.button("📊 View History", use_container_width=True):
                    st.session_state.show_history = True
                
                if st.button("⚙️ Settings", use_container_width=True):
                    st.session_state.show_settings = True
                
                if st.button("🚪 Logout", use_container_width=True):
                    self.logout()
    
    def get_user_context(self) -> Dict[str, Any]:
        """Get user context for database operations"""
        if self.is_authenticated():
            user = self.get_current_user()
            return {
                'user_id': user['id'],
                'username': user['username'],
                'session_id': self.get_session_id()
            }
        else:
            return {
                'user_id': None,
                'username': 'anonymous',
                'session_id': self.get_session_id()
            }

def init_auth(db_manager: DatabaseManager = None) -> AuthManager:
    """Initialize authentication manager"""
    if 'auth_manager' not in st.session_state:
        if db_manager is None:
            db_manager = DatabaseManager()
        st.session_state.auth_manager = AuthManager(db_manager)
    
    return st.session_state.auth_manager

# Decorators for authentication
def require_auth(func):
    """Decorator to require authentication for a function"""
    def wrapper(*args, **kwargs):
        auth = init_auth()
        if auth.require_authentication():
            return func(*args, **kwargs)
        return None
    return wrapper

def get_user_context():
    """Helper function to get user context"""
    auth = init_auth()
    return auth.get_user_context()
