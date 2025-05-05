import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from datetime import datetime, timedelta
import io
import json
import random
import os
import plotly.io as pio
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def display_energy_animation():
    """Display an energy-themed animation on the login page with original colors and visual style."""
    # Create a placeholder for the animation
    animation_placeholder = st.empty()
    
    # Define key frames for the animation
    frames = 30
    
    # Original color palette
    energy_color = '#4CAF50'  # Original green color
    particle_color_scale = 'Viridis'  # Original color scale
    bg_color = 'rgba(0,0,0,0)'  # Transparent background
    
    # Display animation - with original visuals
    for i in range(frames):
        progress = i / frames
        
        # Create a simple energy wave visualization
        fig = go.Figure()
        
        # Generate wave data with original pattern
        x = np.linspace(0, 10, 100)
        y = np.sin(x + progress * 2 * np.pi) * np.exp(-0.1 * x)
        
        # Add the main wave with original color and styling
        fig.add_trace(go.Scatter(
            x=x, 
            y=y, 
            mode='lines',
            line=dict(color=energy_color, width=3),
            name='Energy'
        ))
        
        # Add a second wave for visual effect (part of original design)
        y2 = 0.5 * np.sin(x + progress * 4 * np.pi + 1) * np.exp(-0.05 * x)
        fig.add_trace(go.Scatter(
            x=x, 
            y=y2, 
            mode='lines',
            line=dict(color='rgba(76, 175, 80, 0.4)', width=2),
            name='Energy Harmonic'
        ))
        
        # Add the original "energy particles" effect
        particles_x = np.random.uniform(0, 10, 20)
        particles_y = np.random.uniform(-1, 1, 20)
        
        fig.add_trace(go.Scatter(
            x=particles_x,
            y=particles_y,
            mode='markers',
            marker=dict(
                size=8,
                color=np.random.uniform(0, 1, 20),
                colorscale=particle_color_scale,
                opacity=0.8
            ),
            name='Particles'
        ))
        
        # Set layout with original styling
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        # Update the animation placeholder with unique key for each frame
        animation_placeholder.plotly_chart(fig, use_container_width=True, key=f"animation_frame_{i}")
        
        # Control animation speed
        time.sleep(0.05)
    
    # Keep the last frame with a unique key
    animation_placeholder.plotly_chart(fig, use_container_width=True, key="animation_final_frame")

def plot_consumption_overview(data):
    """
    Plot energy consumption overview.
    
    Args:
        data: DataFrame with timestamp and consumption columns
    """
    if 'timestamp' not in data.columns or 'consumption' not in data.columns:
        st.error("Required columns (timestamp, consumption) missing from data")
        return
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create time series plot
    fig = px.line(
        data, 
        x='timestamp', 
        y='consumption',
        title='Energy Consumption Overview',
        labels={'timestamp': 'Time', 'consumption': 'Energy Consumption'}
    )
    
    # Add daily average as a smoothed line
    daily_avg = data.set_index('timestamp').resample('D')['consumption'].mean().reset_index()
    fig.add_scatter(
        x=daily_avg['timestamp'],
        y=daily_avg['consumption'],
        mode='lines',
        line=dict(width=3, color='yellow'),
        name='Daily Average'
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
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Create summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate and display daily pattern
        data['hour'] = data['timestamp'].dt.hour
        hourly_avg = data.groupby('hour')['consumption'].mean()
        peak_hour = hourly_avg.idxmax()
        low_hour = hourly_avg.idxmin()
        
        st.markdown(f"""
        **Daily Pattern:**
        - Peak consumption: {peak_hour}:00 hours
        - Lowest consumption: {low_hour}:00 hours
        """)
    
    with col2:
        # Calculate and display weekly pattern
        data['day'] = data['timestamp'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = data.groupby('day')['consumption'].mean()
        daily_avg = daily_avg.reindex(day_order)
        peak_day = daily_avg.idxmax()
        low_day = daily_avg.idxmin()
        
        st.markdown(f"""
        **Weekly Pattern:**
        - Peak day: {peak_day}
        - Lowest day: {low_day}
        """)
    
    with col3:
        # Calculate and display key statistics
        avg = data['consumption'].mean()
        max_val = data['consumption'].max()
        min_val = data['consumption'].min()
        std_dev = data['consumption'].std()
        
        st.markdown(f"""
        **Key Statistics:**
        - Average: {avg:.2f} units
        - Maximum: {max_val:.2f} units
        - Minimum: {min_val:.2f} units
        - Std Dev: {std_dev:.2f} units
        """)

def plot_anomaly_distribution(data, anomalies):
    """
    Plot energy consumption with anomalies highlighted.
    
    Args:
        data: DataFrame with timestamp and consumption columns
        anomalies: Indices of anomalies detected
    """
    if 'consumption' not in data.columns:
        st.error("Required column (consumption) missing from data")
        return
    
    # Create a copy for plotting
    plot_data = data.copy()
    
    # Determine x-axis
    if 'timestamp' in plot_data.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(plot_data['timestamp']):
            plot_data['timestamp'] = pd.to_datetime(plot_data['timestamp'])
        x_col = 'timestamp'
        x_label = 'Time'
    else:
        # Use index if no timestamp
        plot_data['index'] = plot_data.index
        x_col = 'index'
        x_label = 'Data Point Index'
    
    # Create figure
    fig = go.Figure()
    
    # Add normal data points
    normal_data = plot_data.drop(anomalies)
    
    fig.add_trace(go.Scatter(
        x=normal_data[x_col],
        y=normal_data['consumption'],
        mode='lines',
        name='Normal',
        line=dict(color='#2E86C1')
    ))
    
    # Add anomaly points if any
    if len(anomalies) > 0:
        anomaly_data = plot_data.iloc[anomalies]
        
        fig.add_trace(go.Scatter(
            x=anomaly_data[x_col],
            y=anomaly_data['consumption'],
            mode='markers',
            name='Anomalies',
            marker=dict(
                size=10,
                color='red',
                symbol='x'
            )
        ))
    
    # Update layout
    fig.update_layout(
        title='Energy Consumption with Anomalies',
        xaxis_title=x_label,
        yaxis_title='Energy Consumption',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.8)',
        font=dict(color='white'),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def plot_anomaly_types(data, anomalies):
    """
    Create a bar chart showing types or patterns of anomalies.
    
    Args:
        data: DataFrame with additional feature columns
        anomalies: Indices of anomalies detected
    """
    if len(anomalies) == 0:
        st.info("No anomalies detected for type analysis.")
        return
    
    # Get anomaly data
    anomaly_data = data.iloc[anomalies].copy()
    
    # Attempt to classify anomalies by pattern
    anomaly_types = {}
    
    # Check if time information is available
    if 'timestamp' in anomaly_data.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(anomaly_data['timestamp']):
            anomaly_data['timestamp'] = pd.to_datetime(anomaly_data['timestamp'])
        
        # Add time-based features
        anomaly_data['hour'] = anomaly_data['timestamp'].dt.hour
        anomaly_data['day_of_week'] = anomaly_data['timestamp'].dt.dayofweek
        
        # Count by hour of day (time of day pattern)
        hour_counts = anomaly_data['hour'].value_counts()
        anomaly_types['Time of Day'] = hour_counts.to_dict()
    
    # Check for consumption-based patterns if consumption exists
    if 'consumption' in anomaly_data.columns:
        # Calculate statistics
        data_mean = data['consumption'].mean()
        data_std = data['consumption'].std()
        
        # Classify by deviation
        high_threshold = data_mean + 2 * data_std
        low_threshold = data_mean - 2 * data_std
        
        high_anomalies = anomaly_data[anomaly_data['consumption'] > high_threshold]
        low_anomalies = anomaly_data[anomaly_data['consumption'] < low_threshold]
        mid_anomalies = anomaly_data[
            (anomaly_data['consumption'] <= high_threshold) & 
            (anomaly_data['consumption'] >= low_threshold)
        ]
        
        consumption_types = {
            'High Consumption': len(high_anomalies),
            'Low Consumption': len(low_anomalies),
            'Pattern Deviation': len(mid_anomalies)
        }
        
        anomaly_types['Consumption Pattern'] = consumption_types
    
    # Create visualizations based on available classifications
    if 'Consumption Pattern' in anomaly_types:
        # Create bar chart for consumption patterns
        pattern_data = pd.DataFrame({
            'Pattern': list(anomaly_types['Consumption Pattern'].keys()),
            'Count': list(anomaly_types['Consumption Pattern'].values())
        })
        
        if len(pattern_data) > 0:
            fig1 = px.bar(
                pattern_data,
                x='Pattern',
                y='Count',
                title='Anomalies by Consumption Pattern',
                color='Pattern',
                color_discrete_map={
                    'High Consumption': 'red',
                    'Low Consumption': 'blue',
                    'Pattern Deviation': 'orange'
                }
            )
            
            # Update layout for dark theme
            fig1.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(20,20,20,0.8)',
                font=dict(color='white'),
                height=400,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            # Display the figure
            st.plotly_chart(fig1, use_container_width=True)
    
    # If there are less than 3 anomalies, show details
    if len(anomaly_data) < 10:
        st.markdown("### Anomaly Details")
        
        for i, idx in enumerate(anomalies):
            anomaly = data.iloc[idx]
            
            with st.expander(f"Anomaly {i+1} (Index {idx})"):
                st.write(anomaly)

def plot_time_analysis(data, anomalies):
    """
    Create a time-of-day analysis for anomalies.
    
    Args:
        data: DataFrame with timestamp column
        anomalies: Indices of anomalies detected
    """
    if 'timestamp' not in data.columns:
        st.info("No timestamp data available for time analysis.")
        return
    
    if len(anomalies) == 0:
        st.info("No anomalies detected for time analysis.")
        return
    
    # Get anomaly data
    anomaly_data = data.iloc[anomalies].copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(anomaly_data['timestamp']):
        anomaly_data['timestamp'] = pd.to_datetime(anomaly_data['timestamp'])
    
    # Extract time features
    anomaly_data['hour'] = anomaly_data['timestamp'].dt.hour
    anomaly_data['day'] = anomaly_data['timestamp'].dt.day_name()
    anomaly_data['month'] = anomaly_data['timestamp'].dt.month_name()
    
    # Create time of day distribution
    hour_counts = anomaly_data['hour'].value_counts().sort_index()
    day_counts = anomaly_data['day'].value_counts()
    
    # Order days correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = day_counts.reindex(day_order).fillna(0)
    
    # Create layout with two charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Hour of day chart
        fig1 = px.bar(
            x=hour_counts.index,
            y=hour_counts.values,
            labels={'x': 'Hour of Day', 'y': 'Number of Anomalies'},
            title='Anomalies by Hour of Day'
        )
        
        # Update layout for dark theme
        fig1.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Display the figure
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Day of week chart
        fig2 = px.bar(
            x=day_counts.index,
            y=day_counts.values,
            labels={'x': 'Day of Week', 'y': 'Number of Anomalies'},
            title='Anomalies by Day of Week'
        )
        
        # Update layout for dark theme
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(categoryorder='array', categoryarray=day_order)
        )
        
        # Display the figure
        st.plotly_chart(fig2, use_container_width=True)

def plot_confusion_matrix(cm, model_name="Model"):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: numpy array, confusion matrix
        model_name: string, name of the model
    """
    # Create labels
    labels = ['Normal', 'Anomaly']
    
    # Create figure
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale='Viridis',
        title=f"{model_name} Confusion Matrix"
    )
    
    # Update layout for dark theme
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.8)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def plot_model_comparison(results):
    """
    Plot comparison of model performance metrics.
    
    Args:
        results: Dictionary with model results
    """
    if not results:
        st.info("No model results to compare.")
        return
    
    # Extract metrics for each model
    models = []
    accuracy = []
    precision = []
    recall = []
    training_time = []
    
    for model_name, result in results.items():
        if 'metrics' in result and 'accuracy' in result['metrics']:
            models.append(model_name)
            accuracy.append(result['metrics']['accuracy'])
            precision.append(result['metrics']['precision'])
            recall.append(result['metrics']['recall'])
            training_time.append(result['training_time'])
    
    if not models:
        st.info("No complete metrics found for comparison.")
        return
    
    # Create DataFrame for metrics
    df_metrics = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    })
    
    # Create DataFrame for training time
    df_time = pd.DataFrame({
        'Model': models,
        'Training Time (s)': training_time
    })
    
    # Create layout with two charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create bar chart for metrics
        fig1 = px.bar(
            df_metrics.melt(id_vars=['Model'], var_name='Metric', value_name='Value'),
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            labels={'Value': 'Score (0-1)'}
        )
        
        # Update layout for dark theme
        fig1.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Display the figure
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Create bar chart for training time
        fig2 = px.bar(
            df_time,
            x='Model',
            y='Training Time (s)',
            title='Training Time Comparison',
            color='Model'
        )
        
        # Update layout for dark theme
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Display the figure
        st.plotly_chart(fig2, use_container_width=True)

def plot_recommendations(data, anomalies):
    """
    Create visualizations for recommendations based on anomaly analysis.
    Simplified for clarity.
    
    Args:
        data: DataFrame with consumption and other columns
        anomalies: Indices of anomalies detected
    """
    if 'consumption' not in data.columns:
        st.error("Data missing required column: consumption")
        return
    
    if len(anomalies) == 0:
        st.info("No anomalies detected for recommendations.")
        return
    
    # Create a copy of the data and mark anomalies
    plot_data = data.copy()
    plot_data['is_anomaly'] = 0
    plot_data.loc[anomalies, 'is_anomaly'] = 1
    
    # Get anomaly data
    anomaly_data = plot_data[plot_data['is_anomaly'] == 1]
    normal_data = plot_data[plot_data['is_anomaly'] == 0]
    
    # Calculate basic statistics
    normal_avg = normal_data['consumption'].mean()
    anomaly_avg = anomaly_data['consumption'].mean()
    percent_difference = ((anomaly_avg / normal_avg) - 1) * 100 if normal_avg > 0 else 0
    
    # Display simple metrics
    st.markdown("### Potential Savings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Normal vs. Anomaly Consumption", 
            f"{normal_avg:.2f} vs {anomaly_avg:.2f}",
            delta=f"{percent_difference:.1f}%" if percent_difference > 0 else f"{percent_difference:.1f}%",
            delta_color="inverse"
        )
    
    with col2:
        # Calculate potential savings
        if anomaly_avg > normal_avg:
            excess_per_point = anomaly_avg - normal_avg
            total_excess = excess_per_point * len(anomalies)
            savings_percent = (total_excess / plot_data['consumption'].sum()) * 100
            st.metric(
                "Potential Energy Savings", 
                f"{total_excess:.2f} units",
                delta=f"{savings_percent:.1f}% of total",
                delta_color="normal"
            )
        else:
            st.metric(
                "Potential Energy Savings", 
                "0.00 units",
                delta="0.0%",
                delta_color="off"
            )
    
    # Create a pie chart for data distribution
    pie_data = pd.DataFrame({
        'Category': ['Normal Data', 'Anomalous Data'],
        'Count': [len(data) - len(anomalies), len(anomalies)]
    })
    
    # Create simple pie chart
    fig1 = px.pie(
        pie_data, 
        values='Count', 
        names='Category',
        title='Distribution of Normal vs Anomalous Data Points',
        color='Category',
        color_discrete_map={'Normal Data': 'green', 'Anomalous Data': 'red'}
    )
    
    # Update layout for dark theme
    fig1.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Display pie chart
    st.plotly_chart(fig1, use_container_width=True)
    
    # Simple bar chart comparing normal vs anomaly consumption
    comparison_data = pd.DataFrame({
        'Category': ['Normal', 'Anomaly'],
        'Average Consumption': [normal_avg, anomaly_avg]
    })
    
    fig2 = px.bar(
        comparison_data,
        x='Category',
        y='Average Consumption',
        color='Category',
        title='Normal vs Anomaly Consumption Comparison',
        color_discrete_map={'Normal': 'green', 'Anomaly': 'red'}
    )
    
    # Update layout for dark theme
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.8)',
        font=dict(color='white'),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Display bar chart
    st.plotly_chart(fig2, use_container_width=True)
    
    # Create feature correlation if possible
    if 'temperature' in data.columns:
        # Create scatter plot
        fig3 = px.scatter(
            plot_data,
            x='temperature',
            y='consumption',
            color='is_anomaly',
            title='Consumption vs. Temperature',
            labels={'temperature': 'Temperature', 'consumption': 'Energy Consumption', 'is_anomaly': 'Anomaly'},
            color_discrete_map={0: 'blue', 1: 'red'},
            category_orders={'is_anomaly': [0, 1]}
        )
        
        # Update layout for dark theme
        fig3.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(title='', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Display scatter plot
        st.plotly_chart(fig3, use_container_width=True)

def create_key_metrics_table(data, anomalies=None):
    """
    Create a table of key performance metrics for easy scanning and export.
    
    Args:
        data: DataFrame with consumption data
        anomalies: Indices of anomalies detected (optional)
    """
    st.markdown("## 📊 Key Performance Metrics")
    
    # Check if data is available
    if data is None or len(data) == 0:
        st.warning("No data available for metrics calculation.")
        return
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate basic consumption metrics
    if 'consumption' in data.columns:
        # Calculate general statistics
        metrics["Total Data Points"] = len(data)
        metrics["Total Consumption"] = f"{data['consumption'].sum():.2f} units"
        metrics["Average Consumption"] = f"{data['consumption'].mean():.2f} units"
        metrics["Minimum Consumption"] = f"{data['consumption'].min():.2f} units"
        metrics["Maximum Consumption"] = f"{data['consumption'].max():.2f} units"
        metrics["Standard Deviation"] = f"{data['consumption'].std():.2f} units"
        
        # Calculate coefficient of variation (CV)
        cv = data['consumption'].std() / data['consumption'].mean() if data['consumption'].mean() > 0 else 0
        metrics["Coefficient of Variation"] = f"{cv:.4f}"
        
        # Percentiles for distribution understanding
        metrics["25th Percentile"] = f"{data['consumption'].quantile(0.25):.2f} units"
        metrics["Median (50th Percentile)"] = f"{data['consumption'].quantile(0.5):.2f} units"
        metrics["75th Percentile"] = f"{data['consumption'].quantile(0.75):.2f} units"
        metrics["90th Percentile"] = f"{data['consumption'].quantile(0.9):.2f} units"
        
        # Calculate anomaly-related metrics if anomalies are provided
        if anomalies is not None and len(anomalies) > 0:
            anomaly_data = data.iloc[anomalies]
            normal_data = data.drop(index=anomalies)
            
            metrics["Total Anomalies"] = len(anomalies)
            metrics["Anomaly Percentage"] = f"{(len(anomalies) / len(data) * 100):.2f}%"
            
            if not anomaly_data.empty:
                metrics["Average Anomaly Consumption"] = f"{anomaly_data['consumption'].mean():.2f} units"
                metrics["Max Anomaly Consumption"] = f"{anomaly_data['consumption'].max():.2f} units"
                
                # Calculate anomaly deviation
                if not normal_data.empty:
                    normal_avg = normal_data['consumption'].mean()
                    anomaly_avg = anomaly_data['consumption'].mean()
                    if normal_avg > 0:
                        deviation = ((anomaly_avg / normal_avg) - 1) * 100
                        metrics["Anomaly Deviation"] = f"{deviation:.2f}%"
                        
                        # Estimate potential savings
                        excess = anomaly_avg - normal_avg
                        if excess > 0:
                            potential_savings = excess * len(anomalies)
                            savings_pct = (potential_savings / data['consumption'].sum()) * 100
                            metrics["Potential Savings"] = f"{potential_savings:.2f} units ({savings_pct:.2f}%)"
    
    # Time-based metrics if timestamp is available
    if 'timestamp' in data.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        date_range = data['timestamp'].max() - data['timestamp'].min()
        days_span = date_range.days + (date_range.seconds / 86400)
        metrics["Date Range"] = f"{days_span:.1f} days"
        metrics["Start Date"] = data['timestamp'].min().strftime('%Y-%m-%d %H:%M')
        metrics["End Date"] = data['timestamp'].max().strftime('%Y-%m-%d %H:%M')
        
        # Time density
        if days_span > 0:
            readings_per_day = len(data) / days_span
            metrics["Readings Per Day"] = f"{readings_per_day:.2f}"
        
        # Calculate time period distribution for anomalies
        if anomalies is not None and len(anomalies) > 0:
            anomaly_data = data.iloc[anomalies].copy()
            
            # Hour of day analysis
            if len(anomaly_data) > 0:
                anomaly_data['hour'] = anomaly_data['timestamp'].dt.hour
                peak_hours = anomaly_data['hour'].value_counts().nlargest(3).index.tolist()
                peak_hours_str = ', '.join([f"{h:02d}:00" for h in peak_hours])
                metrics["Peak Anomaly Hours"] = peak_hours_str
                
                # Day of week analysis
                anomaly_data['day'] = anomaly_data['timestamp'].dt.day_name()
                peak_day = anomaly_data['day'].value_counts().idxmax()
                metrics["Peak Anomaly Day"] = peak_day
    
    # Location-based metrics if available
    if 'location' in data.columns:
        # Count unique locations
        unique_locations = data['location'].nunique()
        metrics["Unique Locations"] = unique_locations
        
        # Find highest consumption location
        location_consumption = data.groupby('location')['consumption'].mean()
        highest_location = location_consumption.idxmax()
        highest_value = location_consumption.max()
        metrics["Highest Consumption Location"] = f"{highest_location} ({highest_value:.2f} units)"
        
        # If anomalies are available, check location distribution
        if anomalies is not None and len(anomalies) > 0:
            anomaly_data = data.iloc[anomalies]
            if 'location' in anomaly_data.columns and len(anomaly_data) > 0:
                anomaly_locations = anomaly_data['location'].value_counts()
                top_anomaly_location = anomaly_locations.idxmax()
                top_anomaly_pct = (anomaly_locations[top_anomaly_location] / len(anomaly_data)) * 100
                metrics["Top Anomaly Location"] = f"{top_anomaly_location} ({top_anomaly_pct:.2f}%)"
    
    # Environmental metrics if temperature is available
    if 'temperature' in data.columns:
        metrics["Average Temperature"] = f"{data['temperature'].mean():.2f}°"
        
        # Calculate correlation between temperature and consumption
        if 'consumption' in data.columns:
            temp_corr = data['temperature'].corr(data['consumption'])
            metrics["Temperature-Consumption Correlation"] = f"{temp_corr:.4f}"
    
    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    
    # Display as an interactive table with download option
    st.dataframe(metrics_df, use_container_width=True)
    
    # Add download button
    csv = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Metrics CSV",
        csv,
        "energy_metrics.csv",
        "text/csv",
        key="download-metrics-csv"
    )

def create_emoji_energy_impact(data, anomalies=None, baseline=None):
    """
    Create an emoji-based visualization of energy impact and efficiency levels.
    
    Args:
        data: DataFrame with consumption data
        anomalies: Indices of anomalies detected (optional)
        baseline: Optional baseline for comparison (e.g., industry average)
    """
    st.markdown("## 🌍 Energy Impact Visualization")
    st.markdown("### Visualizing your energy efficiency with emojis")
    
    # No data handling
    if data is None or len(data) == 0:
        st.warning("No data available for emoji visualization.")
        cols = st.columns(5)
        for i in range(5):
            cols[i].markdown(f"<h1 style='text-align: center; color: gray;'>❓</h1>", unsafe_allow_html=True)
        st.markdown("Upload data to see your energy impact visualization.")
        return
    
    # Energy consumption analysis
    if 'consumption' in data.columns:
        # Calculate metrics
        total_consumption = data['consumption'].sum()
        average_consumption = data['consumption'].mean()
        
        # Anomaly impact if available
        anomaly_impact = 0
        if anomalies is not None and len(anomalies) > 0:
            anomaly_data = data.iloc[anomalies]
            normal_data = data.drop(index=anomalies)
            
            if not normal_data.empty and not anomaly_data.empty:
                anomaly_avg = anomaly_data['consumption'].mean()
                normal_avg = normal_data['consumption'].mean()
                anomaly_impact = (anomaly_avg - normal_avg) / (anomaly_avg if anomaly_avg > 0 else 1) * 100
        
        # Determine energy efficiency category based on metrics
        # Higher values mean less efficient
        efficiency_score = 0
        
        # Add anomaly impact to score (0-40 points)
        if anomaly_impact > 50:
            efficiency_score += 40  # Very high anomaly impact
        elif anomaly_impact > 30:
            efficiency_score += 30
        elif anomaly_impact > 15:
            efficiency_score += 20
        elif anomaly_impact > 5:
            efficiency_score += 10
            
        # Check for spikes (0-30 points)
        consumption_max = data['consumption'].max()
        consumption_min = data['consumption'].min()
        consumption_range = consumption_max - consumption_min
        
        if consumption_range > 0:
            variation_coefficient = consumption_range / average_consumption
            if variation_coefficient > 4:
                efficiency_score += 30  # Very high variation
            elif variation_coefficient > 3:
                efficiency_score += 20
            elif variation_coefficient > 2:
                efficiency_score += 10
        
        # Check baseline comparison if available (0-30 points)
        if baseline is not None and baseline > 0:
            baseline_ratio = average_consumption / baseline
            if baseline_ratio > 1.5:
                efficiency_score += 30  # Much worse than baseline
            elif baseline_ratio > 1.2:
                efficiency_score += 20
            elif baseline_ratio > 1:
                efficiency_score += 10
        else:
            # Without baseline, assign points based on coefficient of variation
            cv = data['consumption'].std() / average_consumption if average_consumption > 0 else 0
            if cv > 0.5:
                efficiency_score += 30  # High variation often indicates inefficiency
            elif cv > 0.3:
                efficiency_score += 20
            elif cv > 0.1:
                efficiency_score += 10
                
        # Map score to emoji categories (0-100 scale)
        # 5 categories from excellent to poor
        emoji_categories = [
            {"range": (0, 20), "emoji": "🌱", "color": "green", "label": "Excellent", "description": "Very energy efficient with minimal waste"},
            {"range": (20, 40), "emoji": "🍃", "color": "lightgreen", "label": "Good", "description": "Good energy efficiency with some room for improvement"},
            {"range": (40, 60), "emoji": "🌤️", "color": "yellow", "label": "Average", "description": "Average energy efficiency with notable improvement potential"},
            {"range": (60, 80), "emoji": "🌥️", "color": "orange", "label": "Below Average", "description": "Below average efficiency with significant waste"},
            {"range": (80, 100), "emoji": "🔥", "color": "red", "label": "Poor", "description": "Poor energy efficiency with critical issues to address"}
        ]
        
        # Find the appropriate category
        category = None
        for cat in emoji_categories:
            if cat["range"][0] <= efficiency_score <= cat["range"][1]:
                category = cat
                break
        
        if category is None:
            category = emoji_categories[-1]  # Use worst category if score exceeds range
            
        # Display emoji visualization
        cols = st.columns(5)
        
        for i, cat in enumerate(emoji_categories):
            with cols[i]:
                if cat == category:
                    st.markdown(f"<h1 style='text-align: center; color: {cat['color']};'>{cat['emoji']}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; font-weight: bold; color: {cat['color']};'>{cat['label']}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h1 style='text-align: center; color: gray;'>{cat['emoji']}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: gray;'>{cat['label']}</p>", unsafe_allow_html=True)
        
        # Display description and recommendations
        st.markdown(f"### Energy Efficiency Rating: {category['label']}")
        st.markdown(f"{category['description']}")
        
        # Create gauge chart for visual representation
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=efficiency_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Efficiency Score (Lower is Better)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': category['color']},
                'steps': [
                    {'range': [0, 20], 'color': 'green'},
                    {'range': [20, 40], 'color': 'lightgreen'},
                    {'range': [40, 60], 'color': 'yellow'},
                    {'range': [60, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'red'}
                ]
            }
        ))
        
        # Update layout for dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Display gauge chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recommendations based on category
        st.markdown("### Recommendations")
        
        if category['label'] == "Excellent":
            st.markdown("""
            - Continue your current energy management practices
            - Consider sharing your successful strategies with others
            - Monitor for any changes to maintain performance
            """)
        elif category['label'] == "Good":
            st.markdown("""
            - Address anomalies during peak hours to further improve efficiency
            - Optimize temperature settings for energy savings
            - Implement a regular maintenance schedule for energy systems
            """)
        elif category['label'] == "Average":
            st.markdown("""
            - Investigate and fix the recurring anomalies in your energy consumption
            - Consider energy auditing to identify specific improvement areas
            - Upgrade inefficient equipment and systems
            - Implement better controls for energy management
            """)
        elif category['label'] == "Below Average":
            st.markdown("""
            - Prioritize addressing high-consumption anomalies
            - Implement an energy management system with real-time monitoring
            - Conduct a comprehensive energy audit immediately
            - Consider retrofitting major energy-consuming systems
            - Train staff on energy-efficient practices
            """)
        else:  # Poor
            st.markdown("""
            - Critical action needed: Address all identified anomalies immediately
            - Implement strict energy management protocols with daily monitoring
            - Consider overhauling major energy systems
            - Engage energy efficiency experts for comprehensive solutions
            - Set up alerts for abnormal consumption patterns
            """)
    else:
        st.error("Data missing required column: consumption")

def create_dashboard_summary(data, anomalies):
    """
    Create a simplified summary dashboard.
    
    Args:
        data: DataFrame with consumption data
        anomalies: Indices of anomalies detected
    """
    if data is None or len(data) == 0:
        st.info("No data available for dashboard summary.")
        return
    
    # Only show minimal information at the top
    total_records = len(data)
    anomaly_count = len(anomalies)
    
    # Create a simple pie chart for normal vs anomalous data
    if anomaly_count > 0:
        normal_count = total_records - anomaly_count
        
        # Create pie chart data
        pie_data = pd.DataFrame({
            'Category': ['Normal', 'Anomalous'],
            'Count': [normal_count, anomaly_count]
        })
        
        # Create pie chart
        fig = px.pie(
            pie_data,
            values='Count',
            names='Category',
            title='Data Distribution',
            color='Category',
            color_discrete_map={'Normal': 'green', 'Anomalous': 'red'},
            hole=0.4
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False
        )
        
        # Add annotations in the center
        fig.update_traces(
            textinfo='percent+label', 
            textposition='inside'
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)