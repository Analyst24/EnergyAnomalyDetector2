import streamlit as st
import json
import pandas as pd
from auth import get_user_settings, save_user_settings

def show_settings():
    """Display the settings page of the application."""
    st.title("Settings")
    
    # Load current settings
    current_settings = st.session_state.settings
    
    # Create form for settings
    with st.form("settings_form"):
        st.markdown("### Anomaly Detection Settings")
        
        # Anomaly threshold slider
        anomaly_threshold = st.slider(
            "Anomaly Detection Sensitivity",
            min_value=0.01,
            max_value=0.2,
            value=current_settings.get("anomaly_threshold", 0.05),
            step=0.01,
            help="Higher values will detect more anomalies but may increase false positives"
        )
        
        st.markdown("### Algorithm Selection")
        
        # Algorithm selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            isolation_forest = st.checkbox(
                "Isolation Forest",
                value="isolation_forest" in current_settings.get("selected_algorithms", []),
                help="Effective for detecting outliers in data"
            )
        
        with col2:
            autoencoder = st.checkbox(
                "Autoencoder",
                value="autoencoder" in current_settings.get("selected_algorithms", []),
                help="Neural network-based anomaly detection"
            )
        
        with col3:
            kmeans = st.checkbox(
                "K-Means Clustering",
                value="kmeans" in current_settings.get("selected_algorithms", []),
                help="Cluster-based anomaly detection"
            )
        
        st.markdown("### Theme Settings")
        
        # Theme selection (currently only dark theme is available)
        theme = st.selectbox(
            "Application Theme",
            options=["dark"],
            index=0,
            disabled=True,
            help="Currently only dark theme is available"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            # Isolation Forest settings
            st.markdown("#### Isolation Forest Settings")
            isolation_forest_estimators = st.slider(
                "Number of Trees",
                min_value=50,
                max_value=200,
                value=current_settings.get("isolation_forest_estimators", 100),
                step=10,
                help="More trees can improve accuracy but increase computational cost"
            )
            
            # Autoencoder settings
            st.markdown("#### Autoencoder Settings")
            autoencoder_epochs = st.slider(
                "Training Epochs",
                min_value=10,
                max_value=100,
                value=current_settings.get("autoencoder_epochs", 50),
                step=5,
                help="Number of training epochs for the autoencoder model"
            )
            
            # K-Means settings
            st.markdown("#### K-Means Settings")
            kmeans_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=current_settings.get("kmeans_clusters", 2),
                step=1,
                help="Number of clusters for K-Means algorithm"
            )
        
        # Submit button
        submitted = st.form_submit_button("Save Settings")
        
        if submitted:
            # Collect selected algorithms
            selected_algorithms = []
            if isolation_forest:
                selected_algorithms.append("isolation_forest")
            if autoencoder:
                selected_algorithms.append("autoencoder")
            if kmeans:
                selected_algorithms.append("kmeans")
            
            # Show error if no algorithms selected
            if not selected_algorithms:
                st.error("Please select at least one algorithm.")
                return
            
            # Update settings
            new_settings = {
                "anomaly_threshold": anomaly_threshold,
                "selected_algorithms": selected_algorithms,
                "theme": theme,
                "isolation_forest_estimators": isolation_forest_estimators,
                "autoencoder_epochs": autoencoder_epochs,
                "kmeans_clusters": kmeans_clusters
            }
            
            # Save settings
            st.session_state.settings = new_settings
            
            # Save to user file if authenticated
            if st.session_state.authenticated and st.session_state.username:
                save_user_settings(st.session_state.username, new_settings)
            
            st.success("Settings saved successfully!")
    
    # Reset to defaults button
    if st.button("Reset to Defaults"):
        # Default settings
        default_settings = {
            "anomaly_threshold": 0.05,
            "selected_algorithms": ["isolation_forest", "autoencoder", "kmeans"],
            "theme": "dark",
            "isolation_forest_estimators": 100,
            "autoencoder_epochs": 50,
            "kmeans_clusters": 2
        }
        
        # Update session state
        st.session_state.settings = default_settings
        
        # Save to user file if authenticated
        if st.session_state.authenticated and st.session_state.username:
            save_user_settings(st.session_state.username, default_settings)
        
        st.success("Settings reset to defaults. Refresh the page to see changes.")
        st.rerun()
    
    # Display current settings
    st.markdown("### Current Settings")
    
    # Format settings for display
    display_settings = {
        "Anomaly Threshold": f"{current_settings.get('anomaly_threshold', 0.05):.2f}",
        "Selected Algorithms": ", ".join([algo.replace('_', ' ').title() for algo in current_settings.get("selected_algorithms", [])]),
        "Theme": current_settings.get("theme", "dark").title()
    }
    
    # Display as a table
    st.table(pd.DataFrame({"Setting": list(display_settings.keys()), "Value": list(display_settings.values())}))
    
    # Session management
    st.markdown("### Session Management")
    
    if st.button("Logout"):
        # Clear session state
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.current_page = "login"
        st.rerun()

