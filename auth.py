import streamlit as st
import json
import os
from pathlib import Path
import database

def initialize_auth():
    """Initialize authentication system by creating necessary database tables and default admin user."""
    if not hasattr(st.session_state, "auth_initialized"):
        # Initialize database tables
        db_initialized = database.initialize_database()
        
        # Create default admin account if it doesn't exist
        if db_initialized:
            success, message = database.create_user("admin", "admin123", "admin@example.com")
            if success:
                st.success("Default admin account created")
            # If it already exists, that's fine (message will indicate username exists)
        
        st.session_state.auth_initialized = True

def create_user_accounts_file():
    """Create user accounts file if it doesn't exist (legacy method)."""
    user_dir = Path("user_accounts")
    user_dir.mkdir(exist_ok=True)
    
    user_file = user_dir / "users.json"
    
    if not user_file.exists():
        # Create a default admin user
        default_users = {
            "admin": {
                "password": "admin123",
                "email": "admin@example.com",
                "created_at": "2025-01-01 00:00:00"
            }
        }
        
        with open(user_file, "w") as f:
            json.dump(default_users, f)

def authenticate(username, password):
    """Authenticate a user with username and password."""
    if not username or not password:
        return False
    
    # Try database authentication first
    if hasattr(st.session_state, "db_connected") and st.session_state.db_connected:
        # Use database authentication
        return database.verify_user(username, password)
    
    # Fall back to file-based authentication
    user_file = Path("user_accounts") / "users.json"
    
    if not user_file.exists():
        create_user_accounts_file()
    
    try:
        with open(user_file, "r") as f:
            users = json.load(f)
        
        if username in users and users[username]["password"] == password:
            return True
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
    
    return False

def get_user_info(username):
    """Get user information."""
    # Try database method first
    if hasattr(st.session_state, "db_connected") and st.session_state.db_connected:
        user = database.get_user(username)
        if user:
            # Convert to dictionary format
            return {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at.strftime("%Y-%m-%d %H:%M:%S") if user.created_at else None,
                "last_login": user.last_login.strftime("%Y-%m-%d %H:%M:%S") if user.last_login else None,
                "settings": json.loads(user.settings) if user.settings else {}
            }
    
    # Fall back to file-based method
    user_file = Path("user_accounts") / "users.json"
    
    if not user_file.exists():
        return None
    
    try:
        with open(user_file, "r") as f:
            users = json.load(f)
        
        if username in users:
            # Return a copy without the password
            user_info = users[username].copy()
            user_info.pop("password", None)
            return user_info
    except Exception as e:
        st.error(f"Error retrieving user info: {str(e)}")
    
    return None

def get_user_settings(username):
    """Get user settings from database or file."""
    # Try database method first
    if hasattr(st.session_state, "db_connected") and st.session_state.db_connected:
        return database.get_user_settings(username)
    
    # Fall back to file-based method
    user_info = get_user_info(username)
    if user_info and "settings" in user_info:
        return user_info["settings"]
    
    return {}
