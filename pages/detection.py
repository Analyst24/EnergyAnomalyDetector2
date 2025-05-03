import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from models import isolation_forest_model, autoencoder_model, kmeans_model, evaluate_model_with_synthetic_labels
import visualization as viz

def show_detection():
    """Display the anomaly detection page of the application."""
    st.title("Run Anomaly Detection")
    
    # Check if data is available
    if not st.session_state.data_uploaded or st.session_state.ml_data is None:
        st.warning("Please upload and process data first.")
        
        if st.button("Go to Upload Page", key="goto_upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        
        return
    
    # Get the data
    ml_data = st.session_state.ml_data
    original_data = st.session_state.processed_data
    
    # Detection settings
    st.markdown("### Detection Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Algorithm selection
        algorithm_options = {
            "isolation_forest": "Isolation Forest",
            "autoencoder": "Autoencoder",
            "kmeans": "K-Means Clustering"
        }
        
        selected_algorithms = []
        st.markdown("#### Select Algorithms")
        
        for algo_id, algo_name in algorithm_options.items():
            if algo_id in st.session_state.settings["selected_algorithms"]:
                default = True
            else:
                default = False
            
            if st.checkbox(algo_name, value=default, key=f"algo_{algo_id}"):
                selected_algorithms.append(algo_id)
        
        if not selected_algorithms:
            st.warning("Please select at least one algorithm.")
    
    with col2:
        # Anomaly threshold (contamination parameter)
        st.markdown("#### Anomaly Sensitivity")
        anomaly_threshold = st.slider(
            "Threshold", 
            min_value=0.01, 
            max_value=0.2, 
            value=st.session_state.settings["anomaly_threshold"],
            step=0.01,
            help="Higher values will detect more anomalies but increase false positives"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            if "isolation_forest" in selected_algorithms:
                n_estimators = st.slider(
                    "Isolation Forest: Number of Trees", 
                    min_value=50, 
                    max_value=200, 
                    value=100, 
                    step=10
                )
            
            if "autoencoder" in selected_algorithms:
                epochs = st.slider(
                    "Autoencoder: Training Epochs", 
                    min_value=10, 
                    max_value=100, 
                    value=50, 
                    step=5
                )
            
            if "kmeans" in selected_algorithms:
                n_clusters = st.slider(
                    "K-Means: Number of Clusters", 
                    min_value=2, 
                    max_value=10, 
                    value=2, 
                    step=1
                )
    
    # Run detection button
    if st.button("Run Anomaly Detection", key="run_detection") and selected_algorithms:
        with st.spinner("Running anomaly detection..."):
            # Prepare to store results
            model_results = {}
            all_anomalies = []
            start_time = time.time()
            
            # Run selected algorithms
            if "isolation_forest" in selected_algorithms:
                st.markdown("#### Running Isolation Forest")
                iso_forest_results = isolation_forest_model(
                    ml_data, 
                    contamination=anomaly_threshold
                )
                
                # Evaluate with synthetic labels
                iso_forest_metrics = evaluate_model_with_synthetic_labels(
                    ml_data, 
                    iso_forest_results["anomalies"],
                    original_data
                )
                
                # Store results
                model_results["isolation_forest"] = {
                    "anomalies": iso_forest_results["anomalies"],
                    "scores": iso_forest_results["scores"],
                    "training_time": iso_forest_results["training_time"],
                    "metrics": iso_forest_metrics
                }
                
                all_anomalies.extend(iso_forest_results["anomalies"])
            
            if "autoencoder" in selected_algorithms:
                st.markdown("#### Running Autoencoder")
                autoencoder_results = autoencoder_model(
                    ml_data,
                    epochs=epochs if "epochs" in locals() else 50,
                    contamination=anomaly_threshold
                )
                
                # Evaluate with synthetic labels
                autoencoder_metrics = evaluate_model_with_synthetic_labels(
                    ml_data, 
                    autoencoder_results["anomalies"],
                    original_data
                )
                
                # Store results
                model_results["autoencoder"] = {
                    "anomalies": autoencoder_results["anomalies"],
                    "scores": autoencoder_results["scores"],
                    "training_time": autoencoder_results["training_time"],
                    "metrics": autoencoder_metrics
                }
                
                all_anomalies.extend(autoencoder_results["anomalies"])
            
            if "kmeans" in selected_algorithms:
                st.markdown("#### Running K-Means Clustering")
                kmeans_results = kmeans_model(
                    ml_data,
                    n_clusters=n_clusters if "n_clusters" in locals() else 2,
                    contamination=anomaly_threshold
                )
                
                # Evaluate with synthetic labels
                kmeans_metrics = evaluate_model_with_synthetic_labels(
                    ml_data, 
                    kmeans_results["anomalies"],
                    original_data
                )
                
                # Store results
                model_results["kmeans"] = {
                    "anomalies": kmeans_results["anomalies"],
                    "scores": kmeans_results["scores"],
                    "training_time": kmeans_results["training_time"],
                    "metrics": kmeans_metrics
                }
                
                all_anomalies.extend(kmeans_results["anomalies"])
            
            # Get unique anomalies
            unique_anomalies = list(set(all_anomalies))
            total_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.anomalies = unique_anomalies
            st.session_state.model_results = model_results
            
            # Display success message
            st.success(f"Detection complete! Found {len(unique_anomalies)} anomalies in {total_time:.2f} seconds.")
            
            # Show detection summary
            st.markdown("### Detection Summary")
            
            # Metrics by model
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Performance")
                for model_name, results in model_results.items():
                    display_name = {
                        "isolation_forest": "Isolation Forest",
                        "autoencoder": "Autoencoder",
                        "kmeans": "K-Means"
                    }.get(model_name, model_name)
                    
                    st.markdown(f"**{display_name}:**")
                    metrics = results["metrics"]
                    st.markdown(f"- Accuracy: {metrics['accuracy']:.4f}")
                    st.markdown(f"- Precision: {metrics['precision']:.4f}")
                    st.markdown(f"- Recall: {metrics['recall']:.4f}")
                    st.markdown(f"- Training Time: {results['training_time']:.2f}s")
            
            with col2:
                st.markdown("#### Anomaly Summary")
                st.markdown(f"- Total Records: {len(ml_data)}")
                st.markdown(f"- Total Anomalies: {len(unique_anomalies)}")
                st.markdown(f"- Anomaly Percentage: {(len(unique_anomalies) / len(ml_data) * 100):.2f}%")
                st.markdown(f"- Total Detection Time: {total_time:.2f}s")
            
            # Plot model comparison
            st.markdown("#### Model Comparison")
            viz.plot_model_comparison(model_results)
            
            # Navigation buttons for next steps
            st.markdown("### Next Steps")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Detailed Results", key="goto_results"):
                    st.session_state.current_page = "results"
                    st.rerun()
            
            with col2:
                if st.button("View Model Insights", key="goto_insights"):
                    st.session_state.current_page = "insights"
                    st.rerun()
