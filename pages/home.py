import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime, timedelta

def show_home():
    """Display the home page of the application."""
    st.title("Energy Anomaly Detection System")
    
    # Create two columns with different widths
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Welcome message
        st.markdown("""
        ### Welcome to the Energy Efficiency Monitoring System
        
        This platform helps you identify anomalies in energy consumption data using advanced machine learning algorithms.
        """)
        
        # Animated energy consumption visual
        create_animated_energy_visual()
    
    with col2:
        # System overview
        st.markdown("### System Overview")
        
        # Some key metrics with animations
        key_metrics = [
            {"label": "Algorithms", "value": "3", "delta": "+New AutoEncoder"},
            {"label": "Detection Accuracy", "value": "95%", "delta": "+2.3%"},
            {"label": "Processing Speed", "value": "500k", "delta": "rows/min"}
        ]
        
        for metric in key_metrics:
            st.metric(
                label=metric["label"],
                value=metric["value"],
                delta=metric["delta"]
            )
        
        # Quick access buttons
        st.markdown("### Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload Data", key="home_upload"):
                st.session_state.current_page = "upload"
                st.rerun()
        
        with col2:
            if st.button("View Dashboard", key="home_dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()
    
    # System features
    st.markdown("### Key Features")
    
    # Create three columns for features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        🔍 **Anomaly Detection**
        - Isolation Forest Algorithm
        - Auto-Encoder Neural Networks
        - K-Means Clustering
        """)
    
    with col2:
        st.markdown("""
        📊 **Interactive Visualizations**
        - Consumption patterns
        - Anomaly distribution
        - Time of day analysis
        """)
    
    with col3:
        st.markdown("""
        💡 **Energy Efficiency Insights**
        - Consumption recommendations
        - Model performance metrics
        - Exportable reports
        """)
    
    # Recent system activity (simulated)
    st.markdown("### Recent Activity")
    
    # Generate some fake recent activity data
    activity_data = generate_activity_data()
    
    # Display as a table
    st.table(activity_data)

def create_animated_energy_visual():
    """Create an animated energy consumption visual."""
    # Generate data for visualization
    np.random.seed(42)  # For reproducibility
    
    # Create time series data
    days = 30
    hours_per_day = 24
    n_points = days * hours_per_day
    
    # Time values
    times = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=n_points,
        freq='H'
    )
    
    # Generate a realistic consumption pattern with daily cycles
    hours = np.array([t.hour for t in times])
    is_weekend = np.array([1 if t.dayofweek >= 5 else 0 for t in times])
    
    # Base pattern: higher during work hours on weekdays
    base = 10 + 15 * np.sin(np.pi * (hours - 6) / 12) * (hours >= 6) * (hours <= 18)
    # Weekend adjustment: different pattern on weekends
    weekend_adj = (1 - 0.3 * is_weekend) * base
    # Add random noise
    noise = np.random.normal(0, 1, n_points)
    # Introduce a few anomalies
    anomalies = np.zeros(n_points)
    anomaly_indices = np.random.choice(n_points, size=int(0.05 * n_points), replace=False)
    anomalies[anomaly_indices] = np.random.uniform(5, 15, size=len(anomaly_indices))
    
    # Final consumption values
    consumption = weekend_adj + noise + anomalies
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': times,
        'consumption': consumption,
        'is_anomaly': [i in anomaly_indices for i in range(n_points)]
    })
    
    # Create interactive plotly chart
    fig = px.line(
        df, 
        x='timestamp', 
        y='consumption',
        title='Energy Consumption Pattern with Anomalies',
        labels={'timestamp': 'Time', 'consumption': 'Energy Consumption'}
    )
    
    # Add markers for anomalies
    anomaly_data = df[df['is_anomaly']]
    fig.add_scatter(
        x=anomaly_data['timestamp'],
        y=anomaly_data['consumption'],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Anomalies'
    )
    
    # Update layout for dark theme
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.8)',
        font=dict(color='white'),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode='closest'
    )
    
    # Display
    st.plotly_chart(fig, use_container_width=True)

def generate_activity_data():
    """Generate simulated recent activity data."""
    # Current time
    now = datetime.now()
    
    # Create activity entries
    activities = [
        {
            "Time": (now - timedelta(minutes=random.randint(5, 60))).strftime("%H:%M"),
            "User": random.choice(["admin", "analyst", "manager"]),
            "Activity": random.choice([
                "Data upload (56MB)",
                "Anomaly detection run",
                "Report export",
                "Model training",
                "Dashboard view"
            ])
        }
        for _ in range(5)
    ]
    
    # Sort by time (most recent first)
    activities = sorted(activities, key=lambda x: x["Time"], reverse=True)
    
    return pd.DataFrame(activities)
