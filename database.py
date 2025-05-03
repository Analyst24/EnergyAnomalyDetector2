"""
File-based database module for Energy Anomaly Detection System.
This is a 100% offline version that uses JSON files for data storage.
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import uuid
import time
import pickle

# Initialize file-based storage directories
DATA_DIR = Path("data_storage")
USERS_DIR = DATA_DIR / "users"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, USERS_DIR, DATASETS_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set session state for database connection (always true for file-based)
if "db_connected" not in st.session_state:
    st.session_state.db_connected = True

# File-based database operations
def initialize_database():
    """Initialize the file-based database by creating necessary directories."""
    try:
        # Create a default admin user if it doesn't exist
        users_file = USERS_DIR / "users.json"
        if not users_file.exists():
            default_users = {
                "admin": {
                    "password": "admin123",
                    "email": "admin@example.com",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "settings": {
                        "anomaly_threshold": 0.5,
                        "selected_algorithms": ["isolation_forest", "autoencoder", "kmeans"],
                        "theme": "dark"
                    }
                }
            }
            with open(users_file, "w") as f:
                json.dump(default_users, f, indent=2)
        
        st.success("File-based database initialized successfully")
        return True
    except Exception as e:
        st.error(f"Failed to initialize file-based database: {str(e)}")
        return False

# User management functions
def create_user(username, password, email=None):
    """Create a new user in the database."""
    try:
        users_file = USERS_DIR / "users.json"
        if users_file.exists():
            with open(users_file, "r") as f:
                users = json.load(f)
        else:
            users = {}
        
        # Check if user already exists
        if username in users:
            return False, "Username already exists"
        
        # Create new user
        users[username] = {
            "password": password,  # In production, this should be hashed
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "settings": {}
        }
        
        with open(users_file, "w") as f:
            json.dump(users, f, indent=2)
        
        return True, "User created successfully"
    except Exception as e:
        return False, str(e)

def get_user(username):
    """Get user by username."""
    try:
        users_file = USERS_DIR / "users.json"
        if not users_file.exists():
            return None
        
        with open(users_file, "r") as f:
            users = json.load(f)
        
        if username in users:
            return users[username]
        
        return None
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

def verify_user(username, password):
    """Verify user credentials."""
    try:
        users_file = USERS_DIR / "users.json"
        if not users_file.exists():
            return False
        
        with open(users_file, "r") as f:
            users = json.load(f)
        
        if username in users and users[username]["password"] == password:
            # Update last login
            users[username]["last_login"] = datetime.now().isoformat()
            
            with open(users_file, "w") as f:
                json.dump(users, f, indent=2)
            
            return True
        
        return False
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False

def update_user_settings(username, settings):
    """Update user settings."""
    try:
        users_file = USERS_DIR / "users.json"
        if not users_file.exists():
            return False, "User not found"
        
        with open(users_file, "r") as f:
            users = json.load(f)
        
        if username in users:
            users[username]["settings"] = settings
            
            with open(users_file, "w") as f:
                json.dump(users, f, indent=2)
            
            return True, "Settings updated successfully"
        
        return False, "User not found"
    except Exception as e:
        return False, str(e)

def get_user_settings(username):
    """Get user settings."""
    try:
        users_file = USERS_DIR / "users.json"
        if not users_file.exists():
            return {}
        
        with open(users_file, "r") as f:
            users = json.load(f)
        
        if username in users and "settings" in users[username]:
            return users[username]["settings"]
        
        return {}
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return {}

# Dataset management functions
def save_dataset(username, name, description, dataframe, file_path=None):
    """Save dataset metadata and records to file."""
    try:
        # Generate a unique ID for the dataset
        dataset_id = str(uuid.uuid4())
        
        # Save metadata
        dataset_dir = DATASETS_DIR / username
        dataset_dir.mkdir(exist_ok=True)
        
        metadata = {
            "id": dataset_id,
            "name": name,
            "description": description,
            "upload_date": datetime.now().isoformat(),
            "file_path": str(file_path) if file_path else None,
            "row_count": len(dataframe),
            "column_count": len(dataframe.columns),
            "columns": list(dataframe.columns),
            "dtypes": {col: str(dataframe[col].dtype) for col in dataframe.columns}
        }
        
        metadata_file = dataset_dir / f"{dataset_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save data
        data_file = dataset_dir / f"{dataset_id}_data.csv"
        dataframe.to_csv(data_file, index=False)
        
        # Keep track of datasets per user
        user_datasets_file = dataset_dir / "datasets.json"
        if user_datasets_file.exists():
            with open(user_datasets_file, "r") as f:
                user_datasets = json.load(f)
        else:
            user_datasets = []
        
        user_datasets.append({
            "id": dataset_id,
            "name": name,
            "description": description,
            "upload_date": datetime.now().isoformat()
        })
        
        with open(user_datasets_file, "w") as f:
            json.dump(user_datasets, f, indent=2)
        
        return True, f"Dataset '{name}' saved with {len(dataframe)} records", dataset_id
    except Exception as e:
        return False, str(e), None

def get_user_datasets(username):
    """Get all datasets for a user."""
    try:
        dataset_dir = DATASETS_DIR / username
        user_datasets_file = dataset_dir / "datasets.json"
        
        if not user_datasets_file.exists():
            return []
        
        with open(user_datasets_file, "r") as f:
            user_datasets = json.load(f)
        
        return user_datasets
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []

def get_dataset(dataset_id, username):
    """Get dataset by ID including all data."""
    try:
        dataset_dir = DATASETS_DIR / username
        metadata_file = dataset_dir / f"{dataset_id}_metadata.json"
        data_file = dataset_dir / f"{dataset_id}_data.csv"
        
        if not metadata_file.exists() or not data_file.exists():
            return None, None
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        data = pd.read_csv(data_file)
        
        return metadata, data
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None, None

# Model management functions
def save_model_metadata(username, name, model_type, file_path, parameters, metrics):
    """Save model metadata to file."""
    try:
        # Generate a unique ID for the model
        model_id = str(uuid.uuid4())
        
        # Save metadata
        model_dir = MODELS_DIR / username
        model_dir.mkdir(exist_ok=True)
        
        metadata = {
            "id": model_id,
            "name": name,
            "model_type": model_type,
            "created_at": datetime.now().isoformat(),
            "file_path": str(file_path),
            "parameters": parameters,
            "metrics": metrics
        }
        
        metadata_file = model_dir / f"{model_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Keep track of models per user
        user_models_file = model_dir / "models.json"
        if user_models_file.exists():
            with open(user_models_file, "r") as f:
                user_models = json.load(f)
        else:
            user_models = []
        
        user_models.append({
            "id": model_id,
            "name": name,
            "model_type": model_type,
            "created_at": datetime.now().isoformat()
        })
        
        with open(user_models_file, "w") as f:
            json.dump(user_models, f, indent=2)
        
        return True, f"Model '{name}' metadata saved", model_id
    except Exception as e:
        return False, str(e), None

def get_user_models(username):
    """Get all models for a user."""
    try:
        model_dir = MODELS_DIR / username
        user_models_file = model_dir / "models.json"
        
        if not user_models_file.exists():
            return []
        
        with open(user_models_file, "r") as f:
            user_models = json.load(f)
        
        return user_models
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []

# Detection results functions
def save_detection_result(dataset_id, model_id, anomaly_indices, scores, username):
    """Save anomaly detection results to file."""
    try:
        # Generate a unique ID for the result
        result_id = str(uuid.uuid4())
        
        # Save result
        result_dir = RESULTS_DIR / username
        result_dir.mkdir(exist_ok=True)
        
        # Calculate anomaly stats
        anomaly_count = len(anomaly_indices)
        total_count = len(scores)
        anomaly_percentage = (anomaly_count / total_count) * 100 if total_count > 0 else 0
        
        result = {
            "id": result_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_percentage,
            "anomaly_indices": anomaly_indices.tolist() if hasattr(anomaly_indices, 'tolist') else list(anomaly_indices),
            "scores": scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        }
        
        result_file = result_dir / f"{result_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Keep track of results per user
        user_results_file = result_dir / "results.json"
        if user_results_file.exists():
            with open(user_results_file, "r") as f:
                user_results = json.load(f)
        else:
            user_results = []
        
        user_results.append({
            "id": result_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_percentage
        })
        
        with open(user_results_file, "w") as f:
            json.dump(user_results, f, indent=2)
        
        return True, f"Detection results saved with {anomaly_count} anomalies detected", result_id
    except Exception as e:
        return False, str(e), None

def get_detection_result(result_id, username):
    """Get detection result by ID."""
    try:
        result_dir = RESULTS_DIR / username
        result_file = result_dir / f"{result_id}_result.json"
        
        if not result_file.exists():
            return None
        
        with open(result_file, "r") as f:
            result = json.load(f)
        
        return result
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

# Recommendations functions
def save_recommendations(detection_result_id, recommendations, username):
    """Save recommendations based on anomaly detection."""
    try:
        recommendations_dir = RESULTS_DIR / username / "recommendations"
        recommendations_dir.mkdir(exist_ok=True)
        
        recommendations_data = {
            "detection_result_id": detection_result_id,
            "created_at": datetime.now().isoformat(),
            "recommendations": recommendations
        }
        
        recommendations_file = recommendations_dir / f"{detection_result_id}_recommendations.json"
        with open(recommendations_file, "w") as f:
            json.dump(recommendations_data, f, indent=2)
        
        return True, f"{len(recommendations)} recommendations saved"
    except Exception as e:
        return False, str(e)

def get_recommendations(detection_result_id, username):
    """Get recommendations for a detection result."""
    try:
        recommendations_dir = RESULTS_DIR / username / "recommendations"
        recommendations_file = recommendations_dir / f"{detection_result_id}_recommendations.json"
        
        if not recommendations_file.exists():
            return []
        
        with open(recommendations_file, "r") as f:
            recommendations_data = json.load(f)
        
        return recommendations_data["recommendations"]
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []