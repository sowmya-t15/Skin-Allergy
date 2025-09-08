#!/usr/bin/env python3
"""
Quick debug script to test authentication and database flow
"""

import streamlit as st
from auth_manager import AuthManager
from database_manager import DatabaseManager

def main():
    st.title("🔍 Debug Streamlit Auth & DB")
    
    # Initialize components
    try:
        db = DatabaseManager()
        auth = AuthManager(db)
        
        st.success("✅ Components initialized successfully")
        
        # Check authentication state
        st.subheader("Authentication State")
        is_auth = auth.is_authenticated()
        st.write(f"Is authenticated: {is_auth}")
        
        if is_auth:
            user = auth.get_current_user()
            st.write(f"Current user: {user}")
            
            # Test database query
            st.subheader("Database Test")
            predictions = db.get_user_predictions(user['id'], limit=10)
            st.write(f"Found {len(predictions)} predictions for user {user['id']}")
            
            if predictions:
                st.write("First prediction:")
                st.json(predictions[0])
        else:
            st.write("Not authenticated - showing login form")
            auth.show_auth_ui()
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
