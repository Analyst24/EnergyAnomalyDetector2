import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import time
import streamlit as st
import pickle
import os
from pathlib import Path

# TensorFlow imports are conditionally loaded to prevent startup issues
tensorflow_available = False
try:
    # Use an offline approach with lazy importing
    # This will only attempt to import if the modules are already installed locally
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping
    tensorflow_available = True
except (ImportError, TypeError) as e:
    st.warning("TensorFlow not available in offline mode. Using alternative models.")
    # Create placeholder variables to prevent namespace errors
    tf = None
    Sequential = None
    Dense = None
    Input = None
    Model = None
    EarlyStopping = None

def isolation_forest_model(data, contamination=0.05):
    """
    Trains an Isolation Forest model for anomaly detection.
    
    Args:
        data: DataFrame containing features
        contamination: The expected proportion of anomalies
    
    Returns:
        Dictionary with model, anomalies, and scores
    """
    st.info("Training Isolation Forest model...")
    start_time = time.time()
    
    # Prepare the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Create and train the model
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled)
    
    # Predict anomalies
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)
    anomalies = np.where(predictions == -1)[0]
    
    training_time = time.time() - start_time
    
    return {
        "model": model,
        "scaler": scaler,
        "anomalies": anomalies,
        "scores": scores,
        "training_time": training_time,
        "predictions": predictions
    }

def autoencoder_model(data, epochs=50, batch_size=32, contamination=0.05):
    """
    Trains an Autoencoder model for anomaly detection.
    
    Args:
        data: DataFrame containing features
        epochs: Number of training epochs
        batch_size: Training batch size
        contamination: The expected proportion of anomalies
    
    Returns:
        Dictionary with model, anomalies, and scores
    """
    # Check if TensorFlow is available
    if not tensorflow_available:
        st.warning("TensorFlow is not available in offline mode.")
        st.info("Falling back to Isolation Forest for anomaly detection.")
        # Fall back to Isolation Forest
        return isolation_forest_model(data, contamination=contamination)
        
    st.info("Training Autoencoder model...")
    start_time = time.time()
    
    # Prepare the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Get feature dimension
    input_dim = X_scaled.shape[1]
    
    try:
        # Create the autoencoder model
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(int(input_dim * 0.75), activation='relu')(input_layer)
        encoded = Dense(int(input_dim * 0.5), activation='relu')(encoded)
        encoded = Dense(int(input_dim * 0.33), activation='relu')(encoded)
        
        # Bottleneck
        bottleneck = Dense(int(input_dim * 0.25), activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(int(input_dim * 0.33), activation='relu')(bottleneck)
        decoded = Dense(int(input_dim * 0.5), activation='relu')(decoded)
        decoded = Dense(int(input_dim * 0.75), activation='relu')(decoded)
        
        # Output layer
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Create and compile the model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        history = autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
            callbacks=[early_stopping]
        )
        
        # Get reconstruction error
        reconstructions = autoencoder.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Determine threshold based on contamination parameter
        threshold = np.percentile(reconstruction_errors, 100 * (1 - contamination))
        
        # Find anomalies
        predictions = (reconstruction_errors > threshold).astype(int)
        anomalies = np.where(predictions == 1)[0]
        
        # Normalize scores to be consistent with isolation forest (higher = more normal)
        max_error = max(reconstruction_errors)
        scores = 1 - (reconstruction_errors / max_error)
        
        training_time = time.time() - start_time
        
        return {
            "model": autoencoder,
            "scaler": scaler,
            "anomalies": anomalies,
            "scores": scores,
            "training_time": training_time,
            "reconstruction_errors": reconstruction_errors,
            "threshold": threshold,
            "predictions": predictions
        }
    except Exception as e:
        st.error(f"Error training Autoencoder model: {str(e)}")
        st.info("Falling back to Isolation Forest for anomaly detection.")
        # Fall back to Isolation Forest
        return isolation_forest_model(data, contamination=contamination)

def kmeans_model(data, n_clusters=2, contamination=0.05):
    """
    Uses K-Means clustering for anomaly detection.
    Points furthest from cluster centers are considered anomalies.
    
    Args:
        data: DataFrame containing features
        n_clusters: Number of clusters
        contamination: The expected proportion of anomalies
    
    Returns:
        Dictionary with model, anomalies, and scores
    """
    st.info("Training K-Means model...")
    start_time = time.time()
    
    # Prepare the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Create and train the model
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)
    
    # Get cluster centers and calculate distances
    centers = model.cluster_centers_
    labels = model.labels_
    
    # Calculate distance of each point to its cluster center
    distances = np.zeros(len(X_scaled))
    for i in range(len(X_scaled)):
        cluster_center = centers[labels[i]]
        distances[i] = np.linalg.norm(X_scaled[i] - cluster_center)
    
    # Determine threshold based on contamination parameter
    threshold = np.percentile(distances, 100 * (1 - contamination))
    
    # Find anomalies
    predictions = (distances > threshold).astype(int)
    anomalies = np.where(predictions == 1)[0]
    
    # Normalize distances to be consistent with other methods (higher = more normal)
    max_distance = max(distances)
    scores = 1 - (distances / max_distance)
    
    training_time = time.time() - start_time
    
    return {
        "model": model,
        "scaler": scaler,
        "anomalies": anomalies,
        "scores": scores,
        "training_time": training_time,
        "distances": distances,
        "threshold": threshold,
        "predictions": predictions
    }

def evaluate_model_with_synthetic_labels(data, anomalies, original_data=None):
    """
    Evaluate model performance using synthetic labels for demonstration.
    This is for the case where we don't have ground truth labels.
    
    Args:
        data: DataFrame containing features
        anomalies: Indices of detected anomalies
        original_data: Original data with potential labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create synthetic "ground truth" for evaluation
    # In real scenarios, you would use actual labeled data
    synthetic_labels = np.zeros(len(data))
    
    # If original_data has a 'label' column, use it (rare in real scenarios)
    has_labels = False
    if original_data is not None and 'label' in original_data.columns:
        has_labels = True
        synthetic_labels = original_data['label'].values
    
    # If no labels provided, use the extreme values as synthetic anomalies
    if not has_labels:
        if 'consumption' in data.columns:
            # Mark highest and lowest 2.5% as anomalies (5% total)
            consumption = data['consumption'].values
            lower_threshold = np.percentile(consumption, 2.5)
            upper_threshold = np.percentile(consumption, 97.5)
            synthetic_labels[(consumption <= lower_threshold) | (consumption >= upper_threshold)] = 1
        else:
            # Use statistical methods to create synthetic labels
            # Mark 5% of data as anomalies based on their distance from the mean
            X_scaled = StandardScaler().fit_transform(data)
            distances = np.linalg.norm(X_scaled, axis=1)
            threshold = np.percentile(distances, 95)
            synthetic_labels[distances >= threshold] = 1
    
    # Model predictions (1 for anomalies, 0 for normal)
    predictions = np.zeros(len(data))
    predictions[anomalies] = 1
    
    # Calculate metrics
    try:
        accuracy = accuracy_score(synthetic_labels, predictions)
        precision = precision_score(synthetic_labels, predictions, zero_division=0)
        recall = recall_score(synthetic_labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(synthetic_labels, predictions)
        
        # If confusion matrix is smaller than 2x2, expand it
        if cm.shape != (2, 2):
            expanded_cm = np.zeros((2, 2))
            for i in range(min(2, cm.shape[0])):
                for j in range(min(2, cm.shape[1])):
                    expanded_cm[i, j] = cm[i, j]
            cm = expanded_cm
            
        tn, fp, fn, tp = cm.ravel()
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "synthetic_labels": synthetic_labels,
            "predictions": predictions
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {
            "error": str(e)
        }

def save_model(model_results, model_type, username):
    """
    Save a trained model for future use.
    
    Args:
        model_results: Dictionary with model and other information
        model_type: String identifying the model type
        username: Username for organization
    
    Returns:
        Path to saved model
    """
    # Create models directory if it doesn't exist
    models_dir = Path(f"user_data/{username}/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save the model
    model_path = models_dir / f"{model_type}_{timestamp}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_results, f)
    
    return model_path

def load_model(model_path):
    """
    Load a trained model.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded model results dictionary
    """
    with open(model_path, 'rb') as f:
        model_results = pickle.load(f)
    
    return model_results
