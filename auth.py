import streamlit as st
import json
import os
from pathlib import Path

def create_user_accounts_file():
    """Create user accounts file if it doesn't exist."""
    user_dir = Path("user_accounts")
    user_dir.mkdir(exist_ok=True)
    
    user_file = user_dir / "users.json"
    
    if not user_file.exists():
        # Create a default admin user
        default_users = {
            "admin": {
                "password": "admin123",
                "email": "admin@example.com",
                "created_at": "2025-01-01 00:00:00",
                "settings": {
                    "anomaly_threshold": 0.5,
                    "selected_algorithms": ["isolation_forest", "autoencoder", "kmeans"],
                    "theme": "dark"
                }
            }
        }
        
        with open(user_file, "w") as f:
            json.dump(default_users, f)

def authenticate(username, password):
    """Authenticate a user with username and password."""
    if not username or not password:
        return False
    
    user_file = Path("user_accounts") / "users.json"
    
    if not user_file.exists():
        create_user_accounts_file()
    
    try:
        with open(user_file, "r") as f:
            users = json.load(f)
        
        if username in users and users[username]["password"] == password:
            # Update last login time if needed in a real application
            return True
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
    
    return False

def get_user_info(username):
    """Get user information."""
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
    """Get user settings from file."""
    user_info = get_user_info(username)
    
    if user_info and "settings" in user_info:
        return user_info["settings"]
    
    # Default settings if none are found
    return {
        "anomaly_threshold": 0.5,
        "selected_algorithms": ["isolation_forest", "autoencoder", "kmeans"],
        "theme": "dark"
    }

def save_user_settings(username, settings):
    """Save user settings to file."""
    user_file = Path("user_accounts") / "users.json"
    
    if not user_file.exists():
        return False
    
    try:
        with open(user_file, "r") as f:
            users = json.load(f)
        
        if username in users:
            users[username]["settings"] = settings
            
            with open(user_file, "w") as f:
                json.dump(users, f)
            
            return True
    except Exception as e:
        st.error(f"Error saving user settings: {str(e)}")
    
    return False
