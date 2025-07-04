import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import visualization as viz

def show_recommendations():
    """Display the recommendations page of the application."""
    st.title("Energy Efficiency Recommendations")
    
    # Check if detection has been run
    if not hasattr(st.session_state, 'anomalies') or st.session_state.anomalies is None:
        st.warning("Please run anomaly detection first to generate recommendations.")
        
        if st.button("Go to Detection Page", key="goto_detection"):
            st.session_state.current_page = "detection"
            st.rerun()
        
        return
    
    # Get the data and results
    data = st.session_state.processed_data
    anomalies = st.session_state.anomalies
    
    if len(anomalies) == 0:
        st.info("No anomalies detected. Your energy consumption appears to be within normal patterns.")
        return
    
    # Calculate metrics for recommendations
    anomaly_data = data.iloc[anomalies].copy()
    normal_data = data.drop(index=anomalies).copy()
    
    # Create visualizations for recommendations
    viz.plot_recommendations(data, anomalies)
    
    # Show main recommendations based on anomaly analysis
    st.markdown("### Key Recommendations")
    
    # Create columns for recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate savings potential from fixing anomalies
        if 'consumption' in data.columns:
            anomaly_avg = anomaly_data['consumption'].mean() if not anomaly_data.empty else 0
            normal_avg = normal_data['consumption'].mean() if not normal_data.empty else 0
            
            if anomaly_avg > normal_avg and normal_avg > 0:
                excess = anomaly_avg - normal_avg
                percentage = (excess / anomaly_avg) * 100
                potential_savings = percentage * len(anomalies) / len(data) * 100
                
                # Create a gauge chart for potential savings
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=potential_savings,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Potential Energy Savings (%)"},
                    gauge={
                        'axis': {'range': [0, 30]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 10], 'color': "lightgreen"},
                            {'range': [10, 20], 'color': "yellow"},
                            {'range': [20, 30], 'color': "orange"}
                        ]
                    },
                    delta={'reference': 0, 'increasing': {'color': "green"}}
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                savings_text = f"""
                By addressing the identified anomalies, you could potentially reduce overall energy consumption by
                approximately **{potential_savings:.1f}%**. This represents the difference between anomalous and normal
                consumption patterns in your dataset.
                """
                st.markdown(savings_text)
    
    with col2:
        # Analyze patterns in anomalies
        st.markdown("#### Anomaly Patterns")
        
        # Time-based patterns
        if 'timestamp' in anomaly_data.columns:
            # Extract hour of day
            anomaly_data['hour'] = anomaly_data['timestamp'].dt.hour
            hour_counts = anomaly_data['hour'].value_counts().sort_index()
            
            # Find peak hours (top 3)
            peak_hours = hour_counts.sort_values(ascending=False).head(3).index.tolist()
            peak_hours_str = ', '.join([f"{h:02d}:00-{(h+1) % 24:02d}:00" for h in peak_hours])
            
            st.markdown(f"**Peak Anomaly Hours:** {peak_hours_str}")
            
            # Extract day of week
            anomaly_data['day'] = anomaly_data['timestamp'].dt.day_name()
            day_counts = anomaly_data['day'].value_counts()
            peak_day = day_counts.idxmax()
            
            st.markdown(f"**Peak Anomaly Day:** {peak_day}")
        
        # Location-based patterns if available
        if 'location' in anomaly_data.columns:
            location_counts = anomaly_data['location'].value_counts()
            top_location = location_counts.idxmax()
            location_pct = (location_counts[top_location] / len(anomaly_data)) * 100
            
            st.markdown(f"**Top Anomaly Location:** {top_location} ({location_pct:.1f}%)")
    
    # Generate specific recommendations based on the data
    st.markdown("### Specific Recommendations")
    
    recommendations = generate_recommendations(data, anomalies)
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}. {rec['title']}**")
        st.markdown(rec['description'])
        
        # Add horizontal divider except after the last recommendation
        if i < len(recommendations):
            st.markdown("---")
    
    # Implementation timeline
    st.markdown("### Implementation Timeline")
    
    # Create timeline data
    timeline_data = create_recommendation_timeline(recommendations)
    
    # Simple table view (always works)
    show_as_table = st.checkbox("Show as table instead of chart", value=False)
    
    if show_as_table or timeline_data.empty:
        # Display as a simple table
        if not timeline_data.empty:
            st.write("Recommendation Implementation Timeline:")
            st.dataframe(timeline_data[["Recommendation", "Start", "End", "Priority"]])
        else:
            st.info("No timeline data available.")
    else:
        try:
            # Ensure all date values are strings to avoid JSON serialization issues
            for col in ["Start", "End"]:
                if col in timeline_data.columns:
                    timeline_data[col] = timeline_data[col].astype(str)
            
            # Display as a Gantt chart
            try:
                fig = px.timeline(
                    timeline_data, 
                    x_start="Start", 
                    x_end="End", 
                    y="Recommendation",
                    color="Priority",
                    color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"},
                    title="Recommended Implementation Timeline"
                )
                
                # Add date formatting
                fig.update_xaxes(
                    type='category',  # Treat dates as categories to avoid date parsing issues
                    tickangle=-45     # Angle the dates for better readability
                )
                
                # Update layout for dark theme
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(20,20,20,0.8)',
                    font=dict(color='white'),
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                    xaxis=dict(title="")
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create timeline chart (using table view instead): {str(e)}")
                st.dataframe(timeline_data[["Recommendation", "Start", "End", "Priority"]])
        except Exception as e:
            st.error(f"Could not display implementation timeline: {str(e)}")
            # If all else fails, just show priority info in text form
            st.write("Priority implementation order:")
            for priority in ["High", "Medium", "Low"]:
                recs = [r["title"] for r in recommendations if r.get("title") in timeline_data["Recommendation"].values]
                if recs:
                    st.write(f"**{priority} Priority:** {', '.join(recs)}")
    
    # Expected outcomes
    st.markdown("### Expected Outcomes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Energy savings
        if 'consumption' in data.columns:
            anomaly_consumption = anomaly_data['consumption'].sum()
            normal_avg = normal_data['consumption'].mean() if not normal_data.empty else 0
            potential_reduction = anomaly_consumption - (normal_avg * len(anomalies))
            
            if potential_reduction > 0:
                st.metric(
                    "Energy Reduction",
                    f"{potential_reduction:.1f} units",
                    delta=f"-{(potential_reduction / anomaly_consumption * 100):.1f}%"
                )
            else:
                st.metric("Energy Reduction", "N/A")
    
    with col2:
        # Efficiency improvement
        if 'consumption' in data.columns:
            current_efficiency = data['consumption'].mean()
            target_efficiency = normal_data['consumption'].mean() if not normal_data.empty else current_efficiency
            
            if target_efficiency < current_efficiency:
                improvement = ((current_efficiency - target_efficiency) / current_efficiency) * 100
                st.metric(
                    "Efficiency Improvement",
                    f"{improvement:.1f}%",
                    delta=f"{improvement:.1f}%"
                )
            else:
                st.metric("Efficiency Improvement", "N/A")
    
    with col3:
        # Anomaly reduction
        st.metric(
            "Anomaly Reduction",
            f"{len(anomalies)}",
            delta=f"-{len(anomalies)}"
        )

def generate_recommendations(data, anomalies):
    """
    Generate specific recommendations based on anomaly analysis.
    
    Args:
        data: DataFrame with processed data
        anomalies: Indices of detected anomalies
    
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    
    # Get anomaly data
    anomaly_data = data.iloc[anomalies].copy()
    
    # Check if we have timestamp data for time-based recommendations
    if 'timestamp' in anomaly_data.columns:
        # Extract hour of day
        anomaly_data['hour'] = anomaly_data['timestamp'].dt.hour
        hour_counts = anomaly_data['hour'].value_counts()
        
        # Check for night-time anomalies
        night_hours = [h for h in range(24) if h < 6 or h >= 22]
        night_anomalies = sum(hour_counts.get(h, 0) for h in night_hours)
        
        if night_anomalies > len(anomaly_data) * 0.2:  # If more than 20% are at night
            recommendations.append({
                'title': "Reduce Night-Time Energy Consumption",
                'description': """
                A significant portion of anomalies occur during night hours (10PM-6AM) when facilities should be minimally operational.
                Consider:
                - Implementing automated shutdown procedures for non-essential equipment
                - Reviewing HVAC settings during unoccupied hours
                - Checking for equipment that may be unnecessarily running 24/7
                """
            })
        
        # Check for weekend anomalies
        anomaly_data['is_weekend'] = anomaly_data['timestamp'].dt.dayofweek >= 5
        weekend_anomalies = anomaly_data['is_weekend'].sum()
        
        if weekend_anomalies > len(anomaly_data) * 0.3:  # If more than 30% are on weekends
            recommendations.append({
                'title': "Optimize Weekend Operations",
                'description': """
                Analysis shows anomalous consumption patterns occurring on weekends when activity should be reduced.
                Recommended actions:
                - Review weekend operational schedules
                - Implement weekend setback temperatures
                - Ensure proper equipment shutdown protocols for non-business days
                """
            })
    
    # Temperature-related recommendations if temperature data is available
    if 'temperature' in anomaly_data.columns and 'consumption' in anomaly_data.columns:
        # Check correlation between temperature and consumption
        temp_corr = anomaly_data['temperature'].corr(anomaly_data['consumption'])
        
        if abs(temp_corr) > 0.6:  # Strong correlation
            if temp_corr > 0:  # Positive correlation (consumption increases with temperature)
                recommendations.append({
                    'title': "Improve Cooling System Efficiency",
                    'description': """
                    High energy consumption strongly correlates with higher temperatures, indicating cooling inefficiencies.
                    Consider:
                    - Checking HVAC system efficiency and maintenance status
                    - Reviewing building insulation and potential air leaks
                    - Implementing smart cooling strategies with optimal setpoints
                    - Exploring shade solutions or reflective materials to reduce heat gain
                    """
                })
            else:  # Negative correlation (consumption increases with lower temperature)
                recommendations.append({
                    'title': "Optimize Heating System Performance",
                    'description': """
                    Energy consumption increases significantly at lower temperatures, suggesting heating inefficiencies.
                    Recommendations:
                    - Inspect heating system for maintenance needs
                    - Consider programmable or smart thermostats to optimize heating cycles
                    - Review building insulation, particularly around windows and doors
                    - Implement zoned heating to reduce energy waste in unused areas
                    """
                })
    
    # Location-based recommendations if location data is available
    if 'location' in anomaly_data.columns:
        location_counts = anomaly_data['location'].value_counts()
        top_locations = location_counts.nlargest(2).index.tolist()
        
        if location_counts[top_locations[0]] > len(anomaly_data) * 0.4:  # If a single location has >40% of anomalies
            recommendations.append({
                'title': f"Audit High-Risk Location: {top_locations[0]}",
                'description': f"""
                The {top_locations[0]} location accounts for a disproportionate number of energy anomalies.
                Recommended actions:
                - Conduct a comprehensive energy audit of this location
                - Check for aging or inefficient equipment
                - Review operational protocols and schedules
                - Consider prioritizing this location for equipment upgrades
                """
            })
    
    # Consumption pattern recommendations
    if 'consumption' in anomaly_data.columns:
        # Check for sudden spikes
        if anomaly_data['consumption'].max() > data['consumption'].mean() * 3:
            recommendations.append({
                'title': "Manage Consumption Spikes",
                'description': """
                Analysis detected extreme consumption spikes significantly above normal operation.
                Consider:
                - Implementing load shedding during peak demand periods
                - Checking for equipment short-cycling or malfunction
                - Staggering startup times for major equipment to reduce peak demand
                - Installing power factor correction if appropriate
                """
            })
        
        # Check for consistently high baseline
        baseline_anomalies = anomaly_data[anomaly_data['consumption'] > data['consumption'].quantile(0.75)]
        if len(baseline_anomalies) > len(anomaly_data) * 0.5:  # If more than 50% are high baseline
            recommendations.append({
                'title': "Reduce Base Load Consumption",
                'description': """
                Many anomalies show elevated baseline consumption, indicating inefficient idle operations.
                Recommendations:
                - Identify and address vampire/phantom loads
                - Implement automated power management for workstations and equipment
                - Review always-on systems for optimization opportunities
                - Consider energy-efficient equipment upgrades for frequently used devices
                """
            })
    
    # If we don't have enough specific recommendations, add general ones
    if len(recommendations) < 3:
        general_recommendations = [
            {
                'title': "Implement Energy Monitoring System",
                'description': """
                Installing a real-time energy monitoring system will help identify anomalies as they occur rather than
                after the fact. This enables immediate corrective action and helps develop a more comprehensive
                understanding of energy usage patterns over time.
                """
            },
            {
                'title': "Conduct Staff Energy Awareness Training",
                'description': """
                Employee behavior can significantly impact energy consumption. Develop and implement a training program
                that raises awareness about energy consumption and provides practical tips for conservation. Simple actions
                like turning off unused equipment and lights can have a substantial cumulative effect.
                """
            },
            {
                'title': "Develop Scheduled Maintenance Protocols",
                'description': """
                Regular maintenance of energy-consuming equipment ensures optimal efficiency. Develop a comprehensive
                maintenance schedule for all major systems, with special attention to HVAC, refrigeration, and production
                equipment. Well-maintained equipment consumes less energy and has a longer operational life.
                """
            },
            {
                'title': "Invest in Energy Efficient Lighting",
                'description': """
                If not already implemented, transitioning to LED lighting can reduce lighting energy consumption by up to 75%
                compared to traditional lighting. Additionally, installing occupancy sensors and daylight harvesting systems
                can further reduce unnecessary lighting usage.
                """
            }
        ]
        
        # Add general recommendations until we have at least 4 total
        for rec in general_recommendations:
            if len(recommendations) >= 4:
                break
            if not any(r['title'] == rec['title'] for r in recommendations):
                recommendations.append(rec)
    
    return recommendations

def create_recommendation_timeline(recommendations):
    """
    Create a timeline for implementing recommendations.
    
    Args:
        recommendations: List of recommendation dictionaries
    
    Returns:
        DataFrame with timeline data
    """
    try:
        # Start date (today)
        start_date = datetime.now().date()
        
        # Create timeline data
        timeline_data = []
        
        for i, rec in enumerate(recommendations):
            try:
                # Assign priority (higher index = lower priority)
                if i < len(recommendations) / 3:
                    priority = "High"
                    duration = 30  # 30 days for high priority
                elif i < len(recommendations) * 2 / 3:
                    priority = "Medium"
                    duration = 60  # 60 days for medium priority
                else:
                    priority = "Low"
                    duration = 90  # 90 days for low priority
                
                # Calculate start and end dates
                # Stagger start dates to avoid everything starting at once
                rec_start = start_date + timedelta(days=i * 15)
                rec_end = rec_start + timedelta(days=duration)
                
                # Convert dates to strings to avoid JSON serialization issues
                # Use simple string format that works across all systems
                rec_start_str = rec_start.strftime('%Y-%m-%d')
                rec_end_str = rec_end.strftime('%Y-%m-%d')
                
                timeline_data.append({
                    "Recommendation": rec['title'],
                    "Start": rec_start_str,  # String format
                    "End": rec_end_str,      # String format
                    "Priority": priority
                })
            except Exception as e:
                # Skip this recommendation if there's an error
                print(f"Error processing recommendation {i}: {str(e)}")
                continue
        
        # Create DataFrame and ensure all date columns are strings
        df = pd.DataFrame(timeline_data)
        
        # Extra protection - convert any remaining date objects to strings
        if not df.empty:
            for col in ["Start", "End"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        
        return df
    except Exception as e:
        # Return empty DataFrame if there's a critical error
        print(f"Error creating timeline: {str(e)}")
        return pd.DataFrame(columns=["Recommendation", "Start", "End", "Priority"])
