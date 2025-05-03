import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import base64
from io import BytesIO
import os

# Enforce offline mode
os.environ['PLOTLY_OFFLINE'] = 'True'

def display_energy_animation():
    """Display an energy-themed animation on the login page."""
    # Create a sample sine wave to simulate energy consumption
    t = np.linspace(0, 10, 100)
    y = np.sin(t) + np.sin(t * 2.5) / 2 + np.sin(t * 7) / 4
    
    # Add some random variation to simulate anomalies
    np.random.seed(42)
    anomaly_indices = np.random.choice(range(len(y)), size=5, replace=False)
    y[anomaly_indices] += np.random.uniform(1, 2, size=len(anomaly_indices))
    
    # Create a DataFrame
    df = pd.DataFrame({
        'time': t,
        'value': y,
        'anomaly': [i in anomaly_indices for i in range(len(y))]
    })
    
    # Create Plotly figure
    fig = px.line(df, x='time', y='value', title='Energy Consumption Patterns')
    
    # Highlight anomalies
    anomalies_df = df[df['anomaly']]
    fig.add_scatter(
        x=anomalies_df['time'],
        y=anomalies_df['value'],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Anomalies'
    )
    
    # Update layout for dark theme
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title='Time',
        yaxis_title='Energy Consumption',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def plot_consumption_overview(data):
    """
    Plot energy consumption overview.
    
    Args:
        data: DataFrame with timestamp and consumption columns
    """
    if 'timestamp' not in data.columns or 'consumption' not in data.columns:
        st.error("Data missing required columns: timestamp and consumption")
        return
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create the line plot
    fig = px.line(
        data, 
        x='timestamp', 
        y='consumption',
        title='Energy Consumption Overview',
        labels={'timestamp': 'Time', 'consumption': 'Energy Consumption'}
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
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def plot_anomaly_distribution(data, anomalies):
    """
    Plot energy consumption with anomalies highlighted.
    
    Args:
        data: DataFrame with timestamp and consumption columns
        anomalies: Indices of anomalies detected
    """
    if 'timestamp' not in data.columns or 'consumption' not in data.columns:
        st.error("Data missing required columns: timestamp and consumption")
        return
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create a copy of the data
    plot_data = data.copy()
    plot_data['is_anomaly'] = 0
    plot_data.loc[anomalies, 'is_anomaly'] = 1
    
    # Create the line plot
    fig = px.line(
        plot_data, 
        x='timestamp', 
        y='consumption',
        title='Anomaly Distribution',
        labels={'timestamp': 'Time', 'consumption': 'Energy Consumption'}
    )
    
    # Add scatter plot for anomalies
    anomaly_data = plot_data[plot_data['is_anomaly'] == 1]
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
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
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
        st.info("No anomalies detected to analyze.")
        return
    
    # Create a copy of the anomaly data
    anomaly_data = data.iloc[anomalies].copy()
    
    # Try to categorize anomalies by available features
    features_to_check = ['location', 'meter_id', 'season', 'time_of_day']
    available_features = [f for f in features_to_check if f in anomaly_data.columns]
    
    if not available_features:
        # If no categorical features, categorize by consumption level
        anomaly_data['anomaly_type'] = pd.cut(
            anomaly_data['consumption'], 
            bins=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        available_features = ['anomaly_type']
    
    # Create plots for each available feature
    for feature in available_features:
        # Count occurrences of each category
        counts = anomaly_data[feature].value_counts().reset_index()
        counts.columns = [feature, 'count']
        
        # Create bar chart
        fig = px.bar(
            counts,
            x=feature,
            y='count',
            title=f'Anomaly Distribution by {feature}',
            labels={feature: feature.replace('_', ' ').title(), 'count': 'Number of Anomalies'}
        )
        
        # Update layout for dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=350,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)

def plot_time_analysis(data, anomalies):
    """
    Create a time-of-day analysis for anomalies.
    
    Args:
        data: DataFrame with timestamp column
        anomalies: Indices of anomalies detected
    """
    if 'timestamp' not in data.columns:
        st.error("Data missing required column: timestamp")
        return
    
    if len(anomalies) == 0:
        st.info("No anomalies detected to analyze.")
        return
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create a copy of the data
    plot_data = data.copy()
    
    # Extract hour and create time categories
    plot_data['hour'] = plot_data['timestamp'].dt.hour
    plot_data['time_category'] = pd.cut(
        plot_data['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
        include_lowest=True
    )
    
    # Mark anomalies
    plot_data['is_anomaly'] = 0
    plot_data.loc[anomalies, 'is_anomaly'] = 1
    
    # Create anomaly data
    anomaly_data = plot_data[plot_data['is_anomaly'] == 1]
    
    # Count by time category
    time_counts = anomaly_data['time_category'].value_counts().reset_index()
    time_counts.columns = ['time_category', 'count']
    
    # Count by hour for heatmap
    hour_counts = anomaly_data['hour'].value_counts().reset_index()
    hour_counts.columns = ['hour', 'count']
    hour_counts = hour_counts.sort_values('hour')
    
    # Create bar chart for time categories
    fig1 = px.bar(
        time_counts,
        x='time_category',
        y='count',
        title='Anomalies by Time of Day',
        labels={'time_category': 'Time of Day', 'count': 'Number of Anomalies'}
    )
    
    # Update layout for dark theme
    fig1.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.8)',
        font=dict(color='white'),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Create heatmap by hour
    heatmap_data = pd.DataFrame(index=range(24), columns=['count'])
    heatmap_data['count'] = 0
    
    for _, row in hour_counts.iterrows():
        heatmap_data.loc[row['hour'], 'count'] = row['count']
    
    fig2 = px.imshow(
        heatmap_data.values.reshape(24, 1),
        labels=dict(x='', y='Hour of Day', color='Count'),
        y=[str(h) for h in range(24)],
        x=['Anomalies'],
        color_continuous_scale='Viridis',
        title='Hourly Anomaly Heatmap'
    )
    
    # Update layout for dark theme
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.8)',
        font=dict(color='white'),
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Display figures
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

def plot_confusion_matrix(cm, model_name="Model"):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: numpy array, confusion matrix
        model_name: string, name of the model
    """
    # Create a DataFrame for the confusion matrix
    df_cm = pd.DataFrame(
        cm, 
        index=['Normal', 'Anomaly'],
        columns=['Predicted Normal', 'Predicted Anomaly']
    )
    
    # Create heatmap using Plotly
    fig = px.imshow(
        df_cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title=f'Confusion Matrix - {model_name}'
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
    
    # Create bar chart for metrics
    fig1 = px.bar(
        df_metrics.melt(id_vars=['Model'], var_name='Metric', value_name='Value'),
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        labels={'Value': 'Score (0-1)', 'Model': ''}
    )
    
    # Create bar chart for training time
    fig2 = px.bar(
        df_time,
        x='Model',
        y='Training Time (s)',
        title='Model Training Time Comparison',
        labels={'Training Time (s)': 'Seconds', 'Model': ''}
    )
    
    # Update layouts for dark theme
    for fig in [fig1, fig2]:
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
    
    # Display figures
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

def plot_recommendations(data, anomalies):
    """
    Create visualizations for recommendations based on anomaly analysis.
    
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
    
    # Create consumption distribution
    fig1 = px.histogram(
        plot_data,
        x='consumption',
        color='is_anomaly',
        marginal='box',
        title='Consumption Distribution with Anomalies',
        labels={'consumption': 'Energy Consumption', 'is_anomaly': 'Anomaly'},
        color_discrete_map={0: 'blue', 1: 'red'},
        category_orders={'is_anomaly': [0, 1]}
    )
    
    # Update layout for dark theme
    fig1.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.8)',
        font=dict(color='white'),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(title='', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Create feature correlation if possible
    if 'temperature' in data.columns:
        # Calculate savings potential
        # Assuming high consumption at comfortable temperatures is inefficient
        has_temp_feature = True
        
        # Create scatter plot
        fig2 = px.scatter(
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
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(title='', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
    else:
        has_temp_feature = False
    
    # Display figures
    st.plotly_chart(fig1, use_container_width=True)
    
    if has_temp_feature:
        st.plotly_chart(fig2, use_container_width=True)
    
    # Calculate potential savings
    normal_mean = plot_data[plot_data['is_anomaly'] == 0]['consumption'].mean()
    anomaly_mean = anomaly_data['consumption'].mean()
    
    if anomaly_mean > normal_mean:
        excess = anomaly_mean - normal_mean
        percentage = (excess / anomaly_mean) * 100
        
        # Create a gauge chart for potential savings
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Potential Energy Savings (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "orange"}
                ]
            }
        ))
        
        # Update layout for dark theme
        fig3.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,0.8)',
            font=dict(color='white'),
            height=300,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Display gauge chart
        st.plotly_chart(fig3, use_container_width=True)

def create_dashboard_summary(data, anomalies):
    """
    Create a summary dashboard with key metrics.
    
    Args:
        data: DataFrame with consumption data
        anomalies: Indices of anomalies detected
    """
    if data is None or len(data) == 0:
        st.info("No data available for dashboard summary.")
        return
    
    # Calculate key metrics
    total_records = len(data)
    anomaly_count = len(anomalies)
    anomaly_percentage = (anomaly_count / total_records) * 100 if total_records > 0 else 0
    
    # Calculate consumption metrics if available
    if 'consumption' in data.columns:
        avg_consumption = data['consumption'].mean()
        max_consumption = data['consumption'].max()
        
        # Calculate metrics for anomalies
        if anomaly_count > 0:
            anomaly_data = data.iloc[anomalies]
            anomaly_avg = anomaly_data['consumption'].mean()
            normal_avg = data.drop(anomalies)['consumption'].mean()
            deviation = ((anomaly_avg / normal_avg) - 1) * 100 if normal_avg > 0 else 0
        else:
            anomaly_avg = 0
            deviation = 0
    else:
        avg_consumption = "N/A"
        max_consumption = "N/A"
        anomaly_avg = "N/A"
        deviation = "N/A"
    
    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        st.metric("Anomalies", f"{anomaly_count:,} ({anomaly_percentage:.1f}%)")
    
    with col3:
        if isinstance(avg_consumption, (int, float)):
            st.metric("Avg. Consumption", f"{avg_consumption:.2f}")
        else:
            st.metric("Avg. Consumption", avg_consumption)
    
    with col4:
        if isinstance(deviation, (int, float)):
            st.metric("Anomaly Deviation", f"{deviation:.1f}%", delta=f"{deviation:.1f}%")
        else:
            st.metric("Anomaly Deviation", deviation)
    
    # Create a second row of metrics if time information is available
    if 'timestamp' in data.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Calculate time metrics
        date_range = data['timestamp'].max() - data['timestamp'].min()
        days_span = date_range.days + (date_range.seconds / 86400)
        
        # Extract time of day for anomalies if any
        if anomaly_count > 0:
            anomaly_data = data.iloc[anomalies].copy()
            anomaly_data['hour'] = anomaly_data['timestamp'].dt.hour
            peak_hour = anomaly_data['hour'].mode().iloc[0]
            peak_hour_str = f"{peak_hour:02d}:00-{(peak_hour+1) % 24:02d}:00"
            
            # Calculate day of week
            anomaly_data['day'] = anomaly_data['timestamp'].dt.day_name()
            peak_day = anomaly_data['day'].mode().iloc[0]
        else:
            peak_hour_str = "N/A"
            peak_day = "N/A"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Date Range", f"{days_span:.1f} days")
        
        with col2:
            st.metric("Peak Anomaly Hour", peak_hour_str)
        
        with col3:
            st.metric("Peak Anomaly Day", peak_day)
        
        # Additional location metric if available
        if 'location' in data.columns and anomaly_count > 0:
            anomaly_data = data.iloc[anomalies]
            if 'location' in anomaly_data.columns:
                top_location = anomaly_data['location'].mode().iloc[0]
                with col4:
                    st.metric("Top Anomaly Location", top_location)
