import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from utils import get_icon
import visualization as viz
from data_processing import generate_sample_data

def show_dashboard():
    """Display the dashboard page of the application."""
    st.title("Energy Consumption Dashboard")
    
    # Check if data is available in session state
    if st.session_state.data is not None and not st.session_state.data.empty:
        data = st.session_state.data
        processed_data = st.session_state.processed_data if st.session_state.processed_data is not None else data
        
        # Show metrics based on actual data
        display_dashboard_with_data(processed_data, st.session_state.anomalies)
    else:
        # Show sample dashboard if no data is uploaded
        display_sample_dashboard()

def display_dashboard_with_data(data, anomalies=None):
    """
    Display dashboard with actual uploaded data.
    
    Args:
        data: DataFrame with processed data
        anomalies: Indices of detected anomalies (if any)
    """
    # Create dashboard summary with metrics (shown at the top always)
    viz.create_dashboard_summary(data, anomalies if anomalies is not None else [])
    
    # Add tab-based navigation for different dashboard views
    dashboard_tabs = st.tabs(["📋 Data Table", "📊 Metrics", "🌍 Energy Impact", "📈 Analysis"])
    
    with dashboard_tabs[0]:
        # Key metrics table in tabular format - first tab
        viz.create_key_metrics_table(data, anomalies)
    
    with dashboard_tabs[1]:
        # Main metrics section in the second tab
        st.markdown("### Energy Consumption Metrics")
        
        # Consumption overview chart
        viz.plot_consumption_overview(data)
        
        # Show location breakdown if available
        if 'location' in data.columns:
            st.markdown("### Consumption by Location")
            
            # Group by location
            location_data = data.groupby('location')['consumption'].mean().reset_index()
            
            fig = px.bar(
                location_data,
                x='location',
                y='consumption',
                title='Average Consumption by Location',
                labels={'location': 'Location', 'consumption': 'Avg. Consumption'}
            )
            
            # Update layout for dark theme
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(20,20,20,0.8)',
                font=dict(color='white'),
                height=400,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with dashboard_tabs[2]:
        # Emoji-based energy impact visualization in the third tab
        viz.create_emoji_energy_impact(data, anomalies if anomalies is not None else [])
    
    with dashboard_tabs[3]:
        # Detailed analysis charts in the fourth tab
        st.markdown("### Energy Consumption Analysis")
        
        # If anomalies have been detected, show anomaly distribution
        if anomalies is not None and len(anomalies) > 0:
            st.markdown("### Anomaly Distribution")
            viz.plot_anomaly_distribution(data, anomalies)
            
            # Show anomaly types and time analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Anomaly Types")
                viz.plot_anomaly_types(data, anomalies)
            
            with col2:
                st.markdown("### Time of Day Analysis")
                viz.plot_time_analysis(data, anomalies)
        
        # Display additional insights if available
        if 'temperature' in data.columns and 'consumption' in data.columns:
            st.markdown("### Temperature vs. Consumption")
            fig = px.scatter(
                data,
                x='temperature',
                y='consumption',
                title='Consumption vs. Temperature',
                labels={'temperature': 'Temperature', 'consumption': 'Energy Consumption'},
                color='temperature' if 'temperature' in data.columns else None,
                color_continuous_scale='Viridis'
            )
            
            if anomalies is not None and len(anomalies) > 0:
                # Add anomaly points
                anomaly_data = data.iloc[anomalies]
                fig.add_scatter(
                    x=anomaly_data['temperature'],
                    y=anomaly_data['consumption'],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='x'),
                    name='Anomalies'
                )
            
            # Update layout for dark theme
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(20,20,20,0.8)',
                font=dict(color='white'),
                height=400,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_sample_dashboard():
    """Display a sample dashboard when no data is available."""
    st.info("No data has been uploaded yet. Displaying a sample dashboard.")
    
    # Generate sample data
    sample_data = generate_sample_data()
    
    # Create sample anomalies (about 5% of the data)
    n_samples = len(sample_data)
    n_anomalies = int(0.05 * n_samples)
    sample_anomalies = np.random.choice(range(n_samples), size=n_anomalies, replace=False)
    
    # Display dashboard with sample data
    display_dashboard_with_data(sample_data, sample_anomalies)
    
    # Add button to upload real data
    st.markdown("### Upload Your Data")
    if st.button("Upload Data", key="dashboard_upload_btn"):
        st.session_state.current_page = "upload"
        st.rerun()
