import streamlit as st
import pandas as pd
import numpy as np
import os

# Enforce offline mode
os.environ['PLOTLY_OFFLINE'] = 'True'
os.environ['OFFLINE_MODE'] = '1'
os.environ['BROWSER_GATHER_USAGE_STATS'] = '0'

# Initialize plotly in offline mode
import plotly.io as pio
pio.templates.default = "plotly_dark"

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Use non-interactive backend
import seaborn as sns
from datetime import datetime, timedelta
import base64
from io import BytesIO

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
            {"range": (20, 40), "emoji": "🌿", "color": "lightgreen", "label": "Good", "description": "Good efficiency with some room for improvement"},
            {"range": (40, 60), "emoji": "🌤️", "color": "yellow", "label": "Moderate", "description": "Average efficiency with significant improvement potential"},
            {"range": (60, 80), "emoji": "🔆", "color": "orange", "label": "Below Average", "description": "Below average efficiency with concerning waste"},
            {"range": (80, 100), "emoji": "🔥", "color": "red", "label": "Poor", "description": "Poor efficiency with critical waste issues"}
        ]
        
        # Find the appropriate category
        category = next((cat for cat in emoji_categories if cat["range"][0] <= efficiency_score <= cat["range"][1]), 
                        emoji_categories[-1])  # Default to the last category if none match
        
        # Display the emoji visualization
        cols = st.columns(5)
        
        # Calculate how many of each emoji to show
        filled_emojis = min(5, max(1, int(round(efficiency_score / 20))))
        empty_spaces = 5 - filled_emojis
        
        # Display filled emojis (representing inefficiency)
        for i in range(filled_emojis):
            cols[i].markdown(f"<h1 style='text-align: center; color: {category['color']};'>{category['emoji']}</h1>", 
                            unsafe_allow_html=True)
            
        # Display empty spaces (representing efficiency potential)
        for i in range(filled_emojis, 5):
            cols[i].markdown("<h1 style='text-align: center; color: gray;'>⚪</h1>", unsafe_allow_html=True)
        
        # Display category and description
        st.markdown(f"### {category['label']} Energy Efficiency")
        st.markdown(f"**{category['description']}**")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Efficiency Score", f"{100-efficiency_score}/100", 
                     delta=f"{-efficiency_score}" if efficiency_score > 50 else f"{100-efficiency_score}")
            
        with col2:
            anomaly_percentage = len(anomalies) / len(data) * 100 if anomalies is not None and len(data) > 0 else 0
            st.metric("Anomaly Percentage", f"{anomaly_percentage:.1f}%", 
                     delta=f"-{anomaly_percentage:.1f}%" if anomaly_percentage > 0 else None)
            
        with col3:
            variation_text = f"{data['consumption'].std() / data['consumption'].mean() * 100:.1f}%" if data['consumption'].mean() > 0 else "N/A"
            st.metric("Consumption Variation", variation_text)
        
        # Environmental impact visualization
        st.markdown("### 🌍 Environmental Impact")
        
        # Calculate a simplified environmental impact score (demonstration purposes)
        # In a real system, this would use actual conversion factors for carbon, etc.
        emissions_score = min(100, efficiency_score * 1.2)  # Higher inefficiency = higher emissions
        
        # Environmental impact emojis
        impact_emojis = {
            "forest": {"emoji": "🌳", "count": max(1, int(5 - (emissions_score / 20)))},
            "warming": {"emoji": "🌡️", "count": max(1, int(emissions_score / 20))}
        }
        
        impact_cols = st.columns(5)
        
        # Show warming indicators
        for i in range(impact_emojis["warming"]["count"]):
            impact_cols[i].markdown(f"<h1 style='text-align: center;'>🌡️</h1>", unsafe_allow_html=True)
            
        # Show forest/trees (positive)
        for i in range(impact_emojis["warming"]["count"], 5):
            impact_cols[i].markdown(f"<h1 style='text-align: center;'>🌳</h1>", unsafe_allow_html=True)
        
        # Tips based on score
        st.markdown("### 💡 Quick Efficiency Tips")
        
        if efficiency_score < 20:
            st.success("Continue your excellent energy management practices!")
        elif efficiency_score < 40:
            st.info("Consider scheduling regular energy audits to maintain your good performance.")
        elif efficiency_score < 60:
            st.info("Look into optimizing your peak consumption hours and addressing the anomalies.")
        elif efficiency_score < 80:
            st.warning("Implement an energy management system and address the identified anomalies promptly.")
        else:
            st.error("Urgent action needed: Consider a comprehensive energy audit and immediate anomaly remediation.")
    else:
        st.warning("No consumption data available for emoji visualization.")
        cols = st.columns(5)
        for i in range(5):
            cols[i].markdown(f"<h1 style='text-align: center; color: gray;'>❓</h1>", unsafe_allow_html=True)
        st.markdown("Please make sure your data includes a 'consumption' column.")

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
