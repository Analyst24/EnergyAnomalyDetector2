import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import base64
from utils import create_pdf_report, get_download_link, save_plot_as_image
import visualization as viz
from data_processing import save_results

def show_results():
    """Display the results page of the application."""
    st.title("Anomaly Detection Results")
    
    # Check if detection has been run
    if not hasattr(st.session_state, 'anomalies') or st.session_state.anomalies is None:
        st.warning("Please run anomaly detection first.")
        
        if st.button("Go to Detection Page", key="goto_detection"):
            st.session_state.current_page = "detection"
            st.rerun()
        
        return
    
    # Get the data and results
    data = st.session_state.processed_data
    anomalies = st.session_state.anomalies
    model_results = st.session_state.model_results if hasattr(st.session_state, 'model_results') else {}
    
    # Create anomaly distribution visualization
    if len(anomalies) > 0:
        # Create pie chart data for normal vs anomaly points
        total_points = len(data)
        normal_points = total_points - len(anomalies)
        anomaly_points = len(anomalies)
        anomaly_percentage = (anomaly_points / total_points) * 100 if total_points > 0 else 0
        
        # Display a simple header with the percentage
        st.markdown(f"### {anomaly_percentage:.1f}% of data points identified as anomalous")
        
        # Create columns for side-by-side visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Create pie chart for visualization
            pie_data = pd.DataFrame({
                'Category': ['Normal Data', 'Anomalous Data'],
                'Count': [normal_points, anomaly_points]
            })
            
            fig = px.pie(
                pie_data, 
                values='Count', 
                names='Category',
                title='Distribution of Normal vs Anomalous Data Points',
                color='Category',
                color_discrete_map={'Normal Data': 'green', 'Anomalous Data': 'red'}
            )
            
            # Update layout for dark theme
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Add anomaly distribution plot
            viz.plot_anomaly_distribution(data, anomalies)
    else:
        st.success("No anomalies detected in the dataset. All data points appear to be normal.")
    
    # Add tab-based navigation for different views of results
    results_tabs = st.tabs(["📊 Visualizations", "📥 Export Options"])
    
    # Add content for Visualizations tab
    with results_tabs[0]:
        # Show time analysis if timestamp is available
        if 'timestamp' in data.columns and len(anomalies) > 0:
            st.markdown("### Time-based Analysis")
            viz.plot_time_analysis(data, anomalies)
        
        # Show anomaly types
        if len(anomalies) > 0:
            st.markdown("### Anomaly Types")
            viz.plot_anomaly_types(data, anomalies)
            
            # Show anomaly data table with simple formatting
            st.markdown("### Anomalous Data Points")
            
            # Get anomaly data
            anomaly_data = data.iloc[anomalies].copy()
            
            # Add anomaly score if available from any model
            if model_results and any('scores' in result for result in model_results.values()):
                # Use scores from first available model
                for model_name, result in model_results.items():
                    if 'scores' in result:
                        scores = result['scores']
                        model_anomalies = result['anomalies']
                        
                        # Create a scores column
                        anomaly_data['anomaly_score'] = np.nan
                        
                        # Map scores to the anomaly data
                        for idx in anomalies:
                            if idx < len(scores):
                                anomaly_data.loc[idx, 'anomaly_score'] = scores[idx]
                        
                        break
            
            # Select only the most relevant columns to display
            display_columns = []
            
            # Always include these columns if they exist
            for col in ['timestamp', 'consumption', 'anomaly_score', 'location', 'temperature']:
                if col in anomaly_data.columns:
                    display_columns.append(col)
            
            # Include other important columns if not too many are selected yet
            if len(display_columns) < 6:
                for col in anomaly_data.columns:
                    if col not in display_columns and not col.startswith('_') and len(display_columns) < 6:
                        display_columns.append(col)
            
            # Display the data
            st.dataframe(anomaly_data[display_columns])
    
    # Add content for Export Options tab
    with results_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export button
            if st.button("Export Results to CSV", key="export_csv"):
                # Create CSV file
                csv_data = save_results(data, anomalies)
                
                # Generate download link
                st.markdown(
                    get_download_link(csv_data, "anomaly_results.csv", "Download CSV"),
                    unsafe_allow_html=True
                )
        
        with col2:
            # PDF report option
            if st.button("Generate PDF Report", key="generate_pdf"):
                with st.spinner("Generating PDF report..."):
                    # Create PDF report
                    pdf_data = create_pdf_report(data, anomalies, model_results)
                    
                    # Generate download link
                    st.markdown(
                        get_download_link(pdf_data, "anomaly_detection_report.pdf", "Download PDF Report"),
                        unsafe_allow_html=True
                    )
    
    # Add model comparison if multiple models are available
    if len(model_results) > 1:
        st.markdown("### Model Comparison")
        viz.plot_model_comparison(model_results)