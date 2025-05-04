import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime
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
    
    # Show results summary
    st.markdown("### Results Summary")
    
    # Create dashboard summary
    viz.create_dashboard_summary(data, anomalies)
    
    # Add tab-based navigation for different views of results
    results_tabs = st.tabs(["📋 Key Metrics", "📊 Visualizations", "📥 Export Options"])
    
    with results_tabs[0]:
        # Key metrics table in tabular format
        viz.create_key_metrics_table(data, anomalies)
    
    with results_tabs[1]:
        # Main results visualization
        st.markdown("### Anomaly Distribution")
        viz.plot_anomaly_distribution(data, anomalies)
    
    # Add content for Export Options tab
    with results_tabs[2]:
        st.markdown("### Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Data")
            
            # CSV export
            if st.button("Export to CSV", key="export_csv_tab"):
                # Create CSV file
                csv_data = save_results(data, anomalies)
                
                # Generate download link
                st.markdown(
                    get_download_link(csv_data, "anomaly_results.csv", "Download CSV"),
                    unsafe_allow_html=True
                )
            
            # Add option to export metrics as CSV
            metrics_btn = st.button("Export Metrics as CSV", key="export_metrics_csv")
            if metrics_btn:
                # Create metrics dictionary
                metrics = {}
                
                # Basic metrics
                metrics["Total Data Points"] = len(data)
                metrics["Total Anomalies"] = len(anomalies)
                metrics["Anomaly Percentage"] = f"{(len(anomalies) / len(data) * 100):.2f}%"
                
                if 'consumption' in data.columns:
                    metrics["Average Consumption"] = f"{data['consumption'].mean():.2f}"
                    
                    if len(anomalies) > 0:
                        anomaly_data = data.iloc[anomalies]
                        normal_data = data.drop(index=anomalies)
                        
                        metrics["Average Anomaly Consumption"] = f"{anomaly_data['consumption'].mean():.2f}"
                        metrics["Average Normal Consumption"] = f"{normal_data['consumption'].mean():.2f}"
                
                # Create a DataFrame from the metrics dictionary
                metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                
                # Generate CSV
                csv = metrics_df.to_csv(index=False).encode('utf-8')
                
                # Create download link
                st.markdown(
                    get_download_link(io.BytesIO(csv), "energy_metrics.csv", "Download Metrics CSV"),
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown("#### Generate Report")
            
            # PDF report option
            if st.button("Generate PDF Report", key="generate_pdf_tab"):
                with st.spinner("Generating PDF report..."):
                    # Create PDF report
                    pdf_data = create_pdf_report(data, anomalies, model_results)
                    
                    # Generate download link
                    st.markdown(
                        get_download_link(pdf_data, "anomaly_detection_report.pdf", "Download PDF Report"),
                        unsafe_allow_html=True
                    )
            
            # Export charts as images
            if st.button("Export Charts as Images", key="export_images_tab"):
                # Create plots for export
                with st.spinner("Generating chart images..."):
                    # Anomaly distribution
                    fig1 = px.scatter(
                        data, 
                        x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                        y='consumption' if 'consumption' in data.columns else data.iloc[:, 0],
                        title="Anomaly Distribution",
                        labels={"x": "Time", "y": "Consumption"}
                    )
                    
                    # Add anomaly points
                    if len(anomalies) > 0:
                        anomaly_data = data.iloc[anomalies]
                        fig1.add_scatter(
                            x=anomaly_data.index if 'timestamp' not in anomaly_data.columns else anomaly_data['timestamp'],
                            y=anomaly_data['consumption'] if 'consumption' in anomaly_data.columns else anomaly_data.iloc[:, 0],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='x'),
                            name='Anomalies'
                        )
                    
                    img1 = save_plot_as_image(fig1)
                    st.markdown(
                        get_download_link(img1, "anomaly_distribution.png", "Download Anomaly Distribution Chart"),
                        unsafe_allow_html=True
                    )
                    
                    # If there are anomalies, create more charts
                    if len(anomalies) > 0:
                        # Time analysis
                        if 'timestamp' in data.columns:
                            anomaly_data = data.iloc[anomalies].copy()
                            anomaly_data['hour'] = pd.to_datetime(anomaly_data['timestamp']).dt.hour
                            
                            hour_counts = anomaly_data['hour'].value_counts().sort_index()
                            
                            fig2 = px.bar(
                                x=hour_counts.index, 
                                y=hour_counts.values,
                                title="Anomalies by Hour of Day",
                                labels={"x": "Hour of Day", "y": "Number of Anomalies"}
                            )
                            
                            img2 = save_plot_as_image(fig2)
                            st.markdown(
                                get_download_link(img2, "anomalies_by_hour.png", "Download Time Analysis Chart"),
                                unsafe_allow_html=True
                            )
    
    # Original tabs now used for visualization details
    tab1, tab2 = st.tabs(["Anomaly Analysis", "Model Comparison"])
    
    with tab1:
        st.markdown("#### Anomaly Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly types chart
            viz.plot_anomaly_types(data, anomalies)
        
        with col2:
            # Time of day analysis
            viz.plot_time_analysis(data, anomalies)
        
        st.markdown("#### Detailed Anomaly Data")
        
        # Display table of anomalies
        if len(anomalies) > 0:
            anomaly_data = data.iloc[anomalies].copy()
            
            # Add anomaly score if available from any model
            if model_results and any('scores' in result for result in model_results.values()):
                # Use scores from first available model
                for model_name, result in model_results.items():
                    if 'scores' in result:
                        scores = result['scores']
                        model_anomalies = result['anomalies']
                        
                        # Create a scores column
                        anomaly_data['anomaly_score'] = 0.0
                        
                        # Map scores to the anomaly data
                        for i, idx in enumerate(model_anomalies):
                            if idx in anomaly_data.index:
                                anomaly_data.loc[idx, 'anomaly_score'] = scores[idx]
                        
                        break
            
            # Select columns to display
            display_columns = []
            
            # Always include these columns if they exist
            for col in ['timestamp', 'consumption', 'anomaly_score', 'location', 'meter_id']:
                if col in anomaly_data.columns:
                    display_columns.append(col)
            
            # Include other important columns
            for col in anomaly_data.columns:
                if col not in display_columns and col not in ['hour', 'day_of_week', 'month', 'contradiction_flag', 'contradiction_reason']:
                    display_columns.append(col)
            
            # Show contradiction columns at the end if they exist
            for col in ['contradiction_flag', 'contradiction_reason']:
                if col in anomaly_data.columns:
                    display_columns.append(col)
            
            # Display the data
            st.dataframe(anomaly_data[display_columns])
        else:
            st.info("No anomalies detected.")
    
    with tab2:
        st.markdown("#### Model Performance Comparison")
        
        if model_results:
            # Plot model comparison
            viz.plot_model_comparison(model_results)
            
            # Detailed metrics by model
            st.markdown("#### Detailed Model Metrics")
            
            # Create tabs for each model
            model_tabs = st.tabs([model_name.replace('_', ' ').title() for model_name in model_results.keys()])
            
            for i, (model_name, results) in enumerate(model_results.items()):
                with model_tabs[i]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Performance Metrics")
                        metrics = results["metrics"]
                        
                        metrics_df = pd.DataFrame({
                            "Metric": ["Accuracy", "Precision", "Recall"],
                            "Value": [
                                metrics['accuracy'],
                                metrics['precision'],
                                metrics['recall']
                            ]
                        })
                        
                        # Create bar chart for metrics
                        fig = px.bar(
                            metrics_df,
                            x="Metric",
                            y="Value",
                            title=f"{model_name.replace('_', ' ').title()} Performance",
                            labels={"Value": "Score (0-1)"}
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("##### Confusion Matrix")
                        if 'confusion_matrix' in metrics:
                            viz.plot_confusion_matrix(metrics['confusion_matrix'], model_name.replace('_', ' ').title())
                        else:
                            st.info("Confusion matrix not available.")
                    
                    # Additional model-specific visualizations
                    if model_name == "isolation_forest":
                        # Display histogram of anomaly scores
                        st.markdown("##### Anomaly Score Distribution")
                        
                        scores = results["scores"]
                        scores_df = pd.DataFrame({
                            "score": scores,
                            "is_anomaly": ["Anomaly" if i in results["anomalies"] else "Normal" for i in range(len(scores))]
                        })
                        
                        fig = px.histogram(
                            scores_df,
                            x="score",
                            color="is_anomaly",
                            title="Isolation Forest Score Distribution",
                            labels={"score": "Anomaly Score (lower is more anomalous)"},
                            opacity=0.7
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif model_name == "autoencoder":
                        # Display reconstruction error distribution
                        if 'reconstruction_errors' in results:
                            st.markdown("##### Reconstruction Error Distribution")
                            
                            errors = results["reconstruction_errors"]
                            threshold = results.get("threshold", np.percentile(errors, 95))
                            
                            errors_df = pd.DataFrame({
                                "error": errors,
                                "is_anomaly": ["Anomaly" if i in results["anomalies"] else "Normal" for i in range(len(errors))]
                            })
                            
                            fig = px.histogram(
                                errors_df,
                                x="error",
                                color="is_anomaly",
                                title="Autoencoder Reconstruction Error",
                                labels={"error": "Reconstruction Error (higher is more anomalous)"},
                                opacity=0.7
                            )
                            
                            # Add threshold line
                            fig.add_vline(
                                x=threshold,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Threshold",
                                annotation_position="top right"
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
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif model_name == "kmeans":
                        # Display distance distribution
                        if 'distances' in results:
                            st.markdown("##### Distance to Cluster Center")
                            
                            distances = results["distances"]
                            threshold = results.get("threshold", np.percentile(distances, 95))
                            
                            distances_df = pd.DataFrame({
                                "distance": distances,
                                "is_anomaly": ["Anomaly" if i in results["anomalies"] else "Normal" for i in range(len(distances))]
                            })
                            
                            fig = px.histogram(
                                distances_df,
                                x="distance",
                                color="is_anomaly",
                                title="K-Means Distance to Cluster Center",
                                labels={"distance": "Distance (higher is more anomalous)"},
                                opacity=0.7
                            )
                            
                            # Add threshold line
                            fig.add_vline(
                                x=threshold,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Threshold",
                                annotation_position="top right"
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
                            
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model results available for comparison.")
    
    with tab3:
        st.markdown("#### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Export Data")
            
            # CSV export
            if st.button("Export to CSV", key="export_csv"):
                # Create CSV file
                csv_data = save_results(data, anomalies)
                
                # Generate download link
                st.markdown(
                    get_download_link(
                        csv_data, 
                        "anomaly_results.csv", 
                        "Download CSV"
                    ),
                    unsafe_allow_html=True
                )
            
            # PDF export
            if st.button("Export to PDF", key="export_pdf"):
                # Create PDF report
                pdf_data = create_pdf_report(data, anomalies, model_results)
                
                # Generate download link
                st.markdown(
                    get_download_link(
                        pdf_data, 
                        "anomaly_report.pdf", 
                        "Download PDF Report"
                    ),
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown("##### Export Visualizations")
            
            # Create a plot for export
            if st.button("Export Main Visualization", key="export_viz"):
                # Create figure
                fig = px.line(
                    data, 
                    x='timestamp', 
                    y='consumption',
                    title='Energy Consumption with Anomalies'
                )
                
                # Add anomaly points
                anomaly_data = data.iloc[anomalies]
                fig.add_scatter(
                    x=anomaly_data['timestamp'],
                    y=anomaly_data['consumption'],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Anomalies'
                )
                
                # Update layout
                fig.update_layout(
                    height=600,
                    width=1000
                )
                
                # Save image
                img_bytes = save_plot_as_image(fig)
                
                # Generate download link
                st.markdown(
                    get_download_link(
                        img_bytes, 
                        "anomaly_visualization.png", 
                        "Download Visualization"
                    ),
                    unsafe_allow_html=True
                )
