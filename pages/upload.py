import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import os
from data_processing import load_data, preprocess_data, generate_sample_data, detect_contradictions

def show_upload():
    """Display the data upload page of the application."""
    st.title("Upload Energy Consumption Data")
    
    # Create tabs for upload or use sample data
    tab1, tab2 = st.tabs(["Upload Your Data", "Use Sample Data"])
    
    with tab1:
        st.markdown("""
        ### Upload CSV File
        
        Upload your energy consumption data in CSV format. The file should ideally contain:
        
        - `timestamp`: Date and time of the reading
        - `consumption`: Energy consumption value
        - Other optional columns: `meter_id`, `location`, `temperature`, `humidity`, etc.
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            with st.spinner("Loading and validating data..."):
                # Load and validate the data
                raw_data = load_data(uploaded_file)
                
                if raw_data is not None and not raw_data.empty:
                    # Show data preview
                    st.success(f"Successfully loaded {len(raw_data)} records.")
                    
                    # Display data summary
                    display_data_summary(raw_data)
                    
                    # Process the data if requested
                    if st.button("Process Data", key="process_uploaded_data"):
                        with st.spinner("Processing data..."):
                            # Preprocess the data
                            ml_data, processed_data = preprocess_data(raw_data)
                            
                            # Detect contradictions
                            flagged_data = detect_contradictions(processed_data)
                            
                            # Save to session state
                            st.session_state.data = raw_data
                            st.session_state.processed_data = flagged_data if flagged_data is not None else processed_data
                            st.session_state.ml_data = ml_data
                            st.session_state.data_uploaded = True
                            
                            st.success("Data processed successfully!")
                            
                            # Show contradiction summary if any were found
                            if flagged_data is not None and 'contradiction_flag' in flagged_data.columns:
                                contradiction_count = flagged_data['contradiction_flag'].sum()
                                if contradiction_count > 0:
                                    st.warning(f"Detected {contradiction_count} potential data contradictions or inconsistencies.")
                                    
                                    # Show sample of contradictory data
                                    contradictions = flagged_data[flagged_data['contradiction_flag']]
                                    st.dataframe(contradictions.head(10))
                            
                            # Suggest next steps
                            st.markdown("### Next Steps")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Run Anomaly Detection", key="goto_detection"):
                                    st.session_state.current_page = "detection"
                                    st.rerun()
                            
                            with col2:
                                if st.button("View Dashboard", key="goto_dashboard"):
                                    st.session_state.current_page = "dashboard"
                                    st.rerun()
    
    with tab2:
        st.markdown("""
        ### Use Sample Data
        
        You can use our sample energy consumption dataset to explore the system without uploading your own data.
        """)
        
        if st.button("Generate Sample Data", key="generate_sample"):
            with st.spinner("Generating sample data..."):
                # Generate sample data
                sample_data = generate_sample_data()
                
                # Display data summary
                display_data_summary(sample_data)
                
                # Process the sample data
                ml_data, processed_data = preprocess_data(sample_data)
                
                # Save to session state
                st.session_state.data = sample_data
                st.session_state.processed_data = processed_data
                st.session_state.ml_data = ml_data
                st.session_state.data_uploaded = True
                
                st.success("Sample data generated and processed successfully!")
                
                # Suggest next steps
                st.markdown("### Next Steps")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Run Anomaly Detection", key="sample_goto_detection"):
                        st.session_state.current_page = "detection"
                        st.rerun()
                
                with col2:
                    if st.button("View Dashboard", key="sample_goto_dashboard"):
                        st.session_state.current_page = "dashboard"
                        st.rerun()

def display_data_summary(data):
    """
    Display a summary of the uploaded data.
    
    Args:
        data: DataFrame with the uploaded data
    """
    # Create expanders for different views of the data
    with st.expander("Data Preview"):
        st.dataframe(data.head(10))
    
    with st.expander("Data Statistics"):
        # Get basic statistics for numeric columns
        numeric_stats = data.describe().T
        
        # Format the statistics
        st.dataframe(numeric_stats)
    
    with st.expander("Column Information"):
        # Get column information
        column_info = []
        
        for col in data.columns:
            col_type = str(data[col].dtype)
            non_null = data[col].count()
            null_count = data[col].isna().sum()
            unique_count = data[col].nunique()
            
            # For categorical columns, show most frequent values
            if data[col].dtype == 'object' or unique_count < 10:
                if unique_count <= 20:  # Only for columns with reasonable number of categories
                    top_values = data[col].value_counts().head(3).to_dict()
                    top_values_str = ", ".join([f"{k}: {v}" for k, v in top_values.items()])
                else:
                    top_values_str = f"{unique_count} unique values"
            else:
                # For numeric columns, show range
                if pd.api.types.is_numeric_dtype(data[col]):
                    min_val = data[col].min()
                    max_val = data[col].max()
                    top_values_str = f"Range: {min_val} to {max_val}"
                else:
                    top_values_str = "Non-numeric data"
            
            column_info.append({
                "Column": col,
                "Type": col_type,
                "Non-Null Count": non_null,
                "Null Count": null_count,
                "Unique Values": unique_count,
                "Top Values/Range": top_values_str
            })
        
        # Display as DataFrame
        st.dataframe(pd.DataFrame(column_info))
