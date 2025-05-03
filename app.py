import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import time
from pathlib import Path

# Import modules
from auth import authenticate, create_user_accounts_file, get_user_settings
from utils import get_icon, footer
import visualization as viz
from data_processing import load_data, preprocess_data

# Set page config
st.set_page_config(
    page_title="Energy Anomaly Detection System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False
if "data" not in st.session_state:
    st.session_state.data = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "anomalies" not in st.session_state:
    st.session_state.anomalies = None
if "model_results" not in st.session_state:
    st.session_state.model_results = {}
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
if "settings" not in st.session_state:
    st.session_state.settings = {
        "anomaly_threshold": 0.5,
        "selected_algorithms": ["isolation_forest", "autoencoder", "kmeans"],
        "theme": "dark"
    }

# Ensure user accounts directory exists
create_user_accounts_file()

# App starts here
if not st.session_state.authenticated:
    # Show Login/Sign-up page
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center;'>Energy Anomaly Detection System</h1>", unsafe_allow_html=True)
        
        # Animation/visualization for the login page
        viz.display_energy_animation()
        
        # Login/Sign-up tabs
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_button"):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    
                    # Load user settings
                    user_settings = get_user_settings(username)
                    if user_settings:
                        st.session_state.settings = user_settings
                    
                    st.session_state.current_page = "home"
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            new_username = st.text_input("Username", key="signup_username")
            email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            
            if st.button("Sign Up", key="signup_button"):
                if not new_username or not email or not new_password:
                    st.error("All fields are required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # File-based user storage
                    user_dir = Path("user_accounts")
                    user_dir.mkdir(exist_ok=True)
                    
                    user_file = user_dir / "users.json"
                    
                    if user_file.exists():
                        with open(user_file, "r") as f:
                            users = json.load(f)
                    else:
                        users = {}
                    
                    if new_username in users:
                        st.error("Username already exists")
                    else:
                        users[new_username] = {
                            "password": new_password,
                            "email": email,
                            "created_at": str(datetime.now())
                        }
                        
                        with open(user_file, "w") as f:
                            json.dump(users, f)
                        
                        st.success("Account created successfully! Please login.")
    
    # Display footer
    footer()

else:
    # Main application interface for authenticated users
    # Sidebar navigation
    with st.sidebar:
        st.image(get_icon("bolt"), width=50)
        st.title(f"Welcome, {st.session_state.username}")
        
        # Ordered dictionary to ensure sidebar menu appears in the specified order
        menu_options = {
            "app": {"label": "App", "icon": "bolt"},
            "home": {"label": "Home", "icon": "home"},
            "dashboard": {"label": "Dashboard", "icon": "bar-chart-2"},
            "upload": {"label": "Upload Data", "icon": "upload"},
            "detection": {"label": "Run Detection", "icon": "search"},
            "results": {"label": "Results", "icon": "activity"},
            "insights": {"label": "Model Insights", "icon": "pie-chart"},
            "recommendations": {"label": "Recommendations", "icon": "award"},
            "settings": {"label": "Settings", "icon": "settings"}
        }
        
        for page_id, page_info in menu_options.items():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(get_icon(page_info["icon"]), width=25)
            with col2:
                if st.button(page_info["label"], key=f"nav_{page_id}"):
                    st.session_state.current_page = page_id
                    st.rerun()
        
        if st.button("Logout", key="logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.current_page = "login"
            st.rerun()
    
    # Main content area based on current page
    if st.session_state.current_page == "app":
        # App overview page
        st.title("Energy Anomaly Detection App")
        st.markdown("""
        ### Overview
        This application helps you detect anomalies in energy consumption data using advanced machine learning algorithms.
        
        ### Features
        - **Multiple Detection Algorithms**: Isolation Forest, Autoencoder, and K-Means clustering
        - **Interactive Visualizations**: Explore your data with dynamic charts
        - **Actionable Insights**: Get recommendations based on detected anomalies
        - **Comprehensive Reports**: Export your findings as CSV or PDF
        
        ### Getting Started
        1. Upload your energy consumption data
        2. Run anomaly detection with your preferred algorithm
        3. Explore results and recommendations
        
        Use the sidebar navigation to move between different sections of the app.
        """)
        
        st.image("assets/background.svg", use_column_width=True)
    
    elif st.session_state.current_page == "home":
        from pages.home import show_home
        show_home()
    
    elif st.session_state.current_page == "dashboard":
        from pages.dashboard import show_dashboard
        show_dashboard()
    
    elif st.session_state.current_page == "upload":
        from pages.upload import show_upload
        show_upload()
    
    elif st.session_state.current_page == "detection":
        from pages.detection import show_detection
        show_detection()
    
    elif st.session_state.current_page == "results":
        from pages.results import show_results
        show_results()
    
    elif st.session_state.current_page == "insights":
        from pages.insights import show_insights
        show_insights()
    
    elif st.session_state.current_page == "recommendations":
        from pages.recommendations import show_recommendations
        show_recommendations()
    
    elif st.session_state.current_page == "settings":
        from pages.settings import show_settings
        show_settings()
    
    # Display footer on every page
    footer()
