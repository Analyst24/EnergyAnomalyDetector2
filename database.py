"""
Database module for Energy Anomaly Detection System.
Handles database connections and operations using SQLAlchemy.
"""

import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    st.error("DATABASE_URL environment variable not set. Database functionality will not work.")

# Create SQLAlchemy engine
try:
    engine = create_engine(DATABASE_URL)
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    st.session_state.db_connected = True
except Exception as e:
    st.error(f"Failed to connect to database: {str(e)}")
    st.session_state.db_connected = False


# Define database models
class User(Base):
    """User model for authentication and settings."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)  # Store hashed password
    email = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    last_login = Column(DateTime, nullable=True)
    settings = Column(Text, nullable=True)  # JSON string of user settings
    
    # Relationships
    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="user", cascade="all, delete-orphan")


class Dataset(Base):
    """Dataset model for storing uploaded energy data."""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String(200), nullable=True)
    upload_date = Column(DateTime, default=datetime.now)
    file_path = Column(String(200), nullable=True)  # Path to file if stored on disk
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    data_metadata = Column(Text, nullable=True)  # JSON string with dataset metadata
    
    # Relationships
    user = relationship("User", back_populates="datasets")
    energy_records = relationship("EnergyRecord", back_populates="dataset", cascade="all, delete-orphan")
    detection_results = relationship("DetectionResult", back_populates="dataset", cascade="all, delete-orphan")


class EnergyRecord(Base):
    """Energy consumption record model."""
    __tablename__ = "energy_records"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    consumption = Column(Float, nullable=False)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    occupancy = Column(Integer, nullable=True)
    day_of_week = Column(Integer, nullable=True)
    hour_of_day = Column(Integer, nullable=True)
    is_weekend = Column(Boolean, nullable=True)
    is_holiday = Column(Boolean, nullable=True)
    additional_features = Column(Text, nullable=True)  # JSON string with additional features
    
    # Relationships
    dataset = relationship("Dataset", back_populates="energy_records")


class Model(Base):
    """Machine learning model metadata."""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # isolation_forest, autoencoder, kmeans
    created_at = Column(DateTime, default=datetime.now)
    file_path = Column(String(200), nullable=True)  # Path to saved model file
    parameters = Column(Text, nullable=True)  # JSON string with model parameters
    metrics = Column(Text, nullable=True)  # JSON string with model performance metrics
    
    # Relationships
    user = relationship("User", back_populates="models")
    detection_results = relationship("DetectionResult", back_populates="model", cascade="all, delete-orphan")


class DetectionResult(Base):
    """Anomaly detection results."""
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    anomaly_count = Column(Integer, nullable=False, default=0)
    anomaly_percentage = Column(Float, nullable=False, default=0.0)
    result_data = Column(Text, nullable=True)  # JSON string with detailed results
    
    # Relationships
    dataset = relationship("Dataset", back_populates="detection_results")
    model = relationship("Model", back_populates="detection_results")
    recommendations = relationship("Recommendation", back_populates="detection_result", cascade="all, delete-orphan")


class Recommendation(Base):
    """Recommendations based on anomaly detection."""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True)
    detection_result_id = Column(Integer, ForeignKey("detection_results.id"), nullable=False)
    category = Column(String(50), nullable=False)  # e.g., "energy_saving", "maintenance", "operation"
    priority = Column(Integer, nullable=False, default=1)  # 1 (highest) to 5 (lowest)
    title = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    potential_savings = Column(Float, nullable=True)  # Estimated savings in kWh or currency
    implementation_difficulty = Column(Integer, nullable=True)  # 1 (easy) to 5 (hard)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    detection_result = relationship("DetectionResult", back_populates="recommendations")


# Create tables if they don't exist
def initialize_database():
    """Initialize the database by creating all defined tables if they don't exist."""
    try:
        Base.metadata.create_all(engine)
        st.success("Database initialized successfully")
        return True
    except Exception as e:
        st.error(f"Failed to initialize database: {str(e)}")
        return False


# User management functions
def create_user(username, password, email=None):
    """Create a new user in the database."""
    session = Session()
    try:
        # Check if user already exists
        existing_user = session.query(User).filter(User.username == username).first()
        if existing_user:
            session.close()
            return False, "Username already exists"
        
        # Create new user
        new_user = User(
            username=username,
            password=password,  # In production, this should be hashed
            email=email,
            created_at=datetime.now(),
            settings=json.dumps({})  # Default empty settings
        )
        session.add(new_user)
        session.commit()
        session.close()
        return True, "User created successfully"
    except Exception as e:
        session.rollback()
        session.close()
        return False, str(e)


def get_user(username):
    """Get user by username."""
    session = Session()
    try:
        user = session.query(User).filter(User.username == username).first()
        session.close()
        return user
    except Exception as e:
        session.close()
        st.error(f"Database error: {str(e)}")
        return None


def verify_user(username, password):
    """Verify user credentials."""
    session = Session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if user and user.password == password:  # In production, compare hashed passwords
            # Update last login
            user.last_login = datetime.now()
            session.commit()
            session.close()
            return True
        session.close()
        return False
    except Exception as e:
        session.close()
        st.error(f"Database error: {str(e)}")
        return False


def update_user_settings(username, settings):
    """Update user settings."""
    session = Session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if user:
            user.settings = json.dumps(settings)
            session.commit()
            session.close()
            return True, "Settings updated successfully"
        session.close()
        return False, "User not found"
    except Exception as e:
        session.rollback()
        session.close()
        return False, str(e)


def get_user_settings(username):
    """Get user settings."""
    session = Session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if user and user.settings:
            settings = json.loads(user.settings)
            session.close()
            return settings
        session.close()
        return {}
    except Exception as e:
        session.close()
        st.error(f"Database error: {str(e)}")
        return {}


# Dataset management functions
def save_dataset(username, name, description, dataframe, file_path=None):
    """Save dataset metadata and records to database."""
    session = Session()
    try:
        # Get user
        user = session.query(User).filter(User.username == username).first()
        if not user:
            session.close()
            return False, "User not found"
        
        # Create dataset
        new_dataset = Dataset(
            user_id=user.id,
            name=name,
            description=description,
            upload_date=datetime.now(),
            file_path=file_path,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            data_metadata=json.dumps({
                "columns": list(dataframe.columns),
                "dtypes": {col: str(dataframe[col].dtype) for col in dataframe.columns}
            })
        )
        session.add(new_dataset)
        session.flush()  # Get the dataset_id
        
        # Extract records from dataframe
        records = []
        for _, row in dataframe.iterrows():
            # Extract standard fields
            record = {
                "dataset_id": new_dataset.id,
                "timestamp": row.get("timestamp", datetime.now()),
                "consumption": float(row.get("consumption", 0.0))
            }
            
            # Optional fields
            if "temperature" in row:
                record["temperature"] = float(row.get("temperature"))
            if "humidity" in row:
                record["humidity"] = float(row.get("humidity"))
            if "occupancy" in row:
                record["occupancy"] = int(row.get("occupancy"))
            if "day_of_week" in row:
                record["day_of_week"] = int(row.get("day_of_week"))
            if "hour_of_day" in row:
                record["hour_of_day"] = int(row.get("hour_of_day"))
            if "is_weekend" in row:
                record["is_weekend"] = bool(row.get("is_weekend"))
            if "is_holiday" in row:
                record["is_holiday"] = bool(row.get("is_holiday"))
            
            # Additional features as JSON
            additional_features = {
                col: row[col] for col in row.index 
                if col not in ["timestamp", "consumption", "temperature", "humidity", 
                             "occupancy", "day_of_week", "hour_of_day", "is_weekend", "is_holiday"]
            }
            if additional_features:
                record["additional_features"] = json.dumps(additional_features)
            
            records.append(EnergyRecord(**record))
        
        # Batch insert records
        session.bulk_save_objects(records)
        session.commit()
        session.close()
        return True, f"Dataset '{name}' saved with {len(records)} records"
    except Exception as e:
        session.rollback()
        session.close()
        return False, str(e)


def get_user_datasets(username):
    """Get all datasets for a user."""
    session = Session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user:
            session.close()
            return []
        
        datasets = session.query(Dataset).filter(Dataset.user_id == user.id).all()
        result = []
        for dataset in datasets:
            result.append({
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "upload_date": dataset.upload_date,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count
            })
        
        session.close()
        return result
    except Exception as e:
        session.close()
        st.error(f"Database error: {str(e)}")
        return []


def get_dataset(dataset_id):
    """Get dataset by ID including all energy records."""
    session = Session()
    try:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            session.close()
            return None, []
        
        # Get all energy records for this dataset
        records = session.query(EnergyRecord).filter(EnergyRecord.dataset_id == dataset_id).all()
        
        # Convert records to dataframe
        data = []
        for record in records:
            row = {
                "timestamp": record.timestamp,
                "consumption": record.consumption,
            }
            
            # Add optional fields if they exist
            if record.temperature is not None:
                row["temperature"] = record.temperature
            if record.humidity is not None:
                row["humidity"] = record.humidity
            if record.occupancy is not None:
                row["occupancy"] = record.occupancy
            if record.day_of_week is not None:
                row["day_of_week"] = record.day_of_week
            if record.hour_of_day is not None:
                row["hour_of_day"] = record.hour_of_day
            if record.is_weekend is not None:
                row["is_weekend"] = record.is_weekend
            if record.is_holiday is not None:
                row["is_holiday"] = record.is_holiday
            
            # Add additional features
            if record.additional_features:
                additional = json.loads(record.additional_features)
                for key, value in additional.items():
                    row[key] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Dataset metadata
        metadata = {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "upload_date": dataset.upload_date,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count
        }
        
        session.close()
        return metadata, df
    except Exception as e:
        session.close()
        st.error(f"Database error: {str(e)}")
        return None, None


# Model management functions
def save_model_metadata(username, name, model_type, file_path, parameters, metrics):
    """Save model metadata to database."""
    session = Session()
    try:
        # Get user
        user = session.query(User).filter(User.username == username).first()
        if not user:
            session.close()
            return False, "User not found", None
        
        # Create model entry
        new_model = Model(
            user_id=user.id,
            name=name,
            model_type=model_type,
            created_at=datetime.now(),
            file_path=file_path,
            parameters=json.dumps(parameters),
            metrics=json.dumps(metrics)
        )
        session.add(new_model)
        session.commit()
        model_id = new_model.id
        session.close()
        return True, f"Model '{name}' metadata saved", model_id
    except Exception as e:
        session.rollback()
        session.close()
        return False, str(e), None


def get_user_models(username):
    """Get all models for a user."""
    session = Session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user:
            session.close()
            return []
        
        models = session.query(Model).filter(Model.user_id == user.id).all()
        result = []
        for model in models:
            result.append({
                "id": model.id,
                "name": model.name,
                "model_type": model.model_type,
                "created_at": model.created_at,
                "parameters": json.loads(model.parameters) if model.parameters else {},
                "metrics": json.loads(model.metrics) if model.metrics else {}
            })
        
        session.close()
        return result
    except Exception as e:
        session.close()
        st.error(f"Database error: {str(e)}")
        return []


# Detection result functions
def save_detection_result(dataset_id, model_id, anomaly_indices, scores):
    """Save anomaly detection results to database."""
    session = Session()
    try:
        # Get row count from dataset
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            session.close()
            return False, "Dataset not found", None
        
        row_count = dataset.row_count
        anomaly_count = len(anomaly_indices)
        anomaly_percentage = (anomaly_count / row_count) * 100 if row_count > 0 else 0
        
        # Create result entry
        result_data = {
            "anomaly_indices": anomaly_indices.tolist() if hasattr(anomaly_indices, 'tolist') else list(anomaly_indices),
            "scores": scores.tolist() if hasattr(scores, 'tolist') else list(scores),
        }
        
        new_result = DetectionResult(
            dataset_id=dataset_id,
            model_id=model_id,
            created_at=datetime.now(),
            anomaly_count=anomaly_count,
            anomaly_percentage=anomaly_percentage,
            result_data=json.dumps(result_data)
        )
        session.add(new_result)
        session.commit()
        result_id = new_result.id
        session.close()
        return True, f"Detection results saved with {anomaly_count} anomalies ({anomaly_percentage:.2f}%)", result_id
    except Exception as e:
        session.rollback()
        session.close()
        return False, str(e), None


def get_detection_result(result_id):
    """Get detection result by ID."""
    session = Session()
    try:
        result = session.query(DetectionResult).filter(DetectionResult.id == result_id).first()
        if not result:
            session.close()
            return None
        
        # Get related dataset and model
        dataset = session.query(Dataset).filter(Dataset.id == result.dataset_id).first()
        model = session.query(Model).filter(Model.id == result.model_id).first()
        
        # Prepare result
        detection_result = {
            "id": result.id,
            "created_at": result.created_at,
            "anomaly_count": result.anomaly_count,
            "anomaly_percentage": result.anomaly_percentage,
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "row_count": dataset.row_count
            },
            "model": {
                "id": model.id,
                "name": model.name,
                "model_type": model.model_type
            }
        }
        
        # Parse result data
        if result.result_data:
            result_data = json.loads(result.result_data)
            detection_result["anomaly_indices"] = result_data.get("anomaly_indices", [])
            detection_result["scores"] = result_data.get("scores", [])
        
        session.close()
        return detection_result
    except Exception as e:
        session.close()
        st.error(f"Database error: {str(e)}")
        return None


# Recommendation functions
def save_recommendations(detection_result_id, recommendations):
    """Save recommendations based on anomaly detection."""
    session = Session()
    try:
        # Verify detection result exists
        result = session.query(DetectionResult).filter(DetectionResult.id == detection_result_id).first()
        if not result:
            session.close()
            return False, "Detection result not found"
        
        # Save each recommendation
        for rec in recommendations:
            new_rec = Recommendation(
                detection_result_id=detection_result_id,
                category=rec.get("category", "general"),
                priority=rec.get("priority", 3),
                title=rec.get("title", "Untitled recommendation"),
                description=rec.get("description", ""),
                potential_savings=rec.get("potential_savings"),
                implementation_difficulty=rec.get("implementation_difficulty", 3),
                created_at=datetime.now()
            )
            session.add(new_rec)
        
        session.commit()
        session.close()
        return True, f"{len(recommendations)} recommendations saved"
    except Exception as e:
        session.rollback()
        session.close()
        return False, str(e)


def get_recommendations(detection_result_id):
    """Get recommendations for a detection result."""
    session = Session()
    try:
        recs = session.query(Recommendation).filter(
            Recommendation.detection_result_id == detection_result_id
        ).order_by(Recommendation.priority).all()
        
        result = []
        for rec in recs:
            result.append({
                "id": rec.id,
                "category": rec.category,
                "priority": rec.priority,
                "title": rec.title,
                "description": rec.description,
                "potential_savings": rec.potential_savings,
                "implementation_difficulty": rec.implementation_difficulty,
                "created_at": rec.created_at
            })
        
        session.close()
        return result
    except Exception as e:
        session.close()
        st.error(f"Database error: {str(e)}")
        return []