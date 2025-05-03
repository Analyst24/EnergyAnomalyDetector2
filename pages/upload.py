import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import os
import plotly.express as px
from data_processing import load_data, preprocess_data, generate_sample_data, detect_contradictions

def show_upload():
    """Display the data upload page of the application."""
    st.title("Upload Energy Consumption Data")
    
    # Create tabs for upload or use sample data
    tab1, tab2 = st.tabs(["Upload Your Data", "Use Sample Data"])
    
    with tab1:
        st.markdown("""
        ### Upload Your Energy Data
        
        Upload your energy consumption data. The system supports multiple file formats and will automatically adapt to various dataset structures.
        
        #### Supported Data Formats:
        - **CSV files**: Common format with comma-separated values
        - **Excel files**: .xlsx or .xls spreadsheets
        - **JSON files**: For structured data
        - **Parquet files**: For column-oriented data
        
        #### The system will automatically detect:
        - Time-related columns (timestamps, dates)
        - Energy consumption measurements
        - Temperature, humidity, and other environmental factors
        - Location and device identifiers
        
        Don't worry if your column names differ from the standard - the system will intelligently map them!
        """)
        
        # File uploader with expanded format support
        uploaded_file = st.file_uploader("Choose your energy data file", type=['csv', 'xlsx', 'xls', 'json', 'parquet'])
        
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
    Display a comprehensive summary of the uploaded data.
    
    Args:
        data: DataFrame with the uploaded data
    """
    # Create expanders for different views of the data
    with st.expander("Data Preview", expanded=True):
        st.dataframe(data.head(10))
        
        # Basic dataset information
        st.markdown(f"""
        ### Dataset Summary
        - **Records**: {len(data)} rows
        - **Columns**: {len(data.columns)} columns
        - **Memory Usage**: {data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB
        """)
        
        # Show time span information if timestamp column exists
        if 'timestamp' in data.columns:
            time_min = data['timestamp'].min()
            time_max = data['timestamp'].max()
            time_span = time_max - time_min
            
            # Calculate typical sampling interval
            if len(data) > 1:
                time_diffs = data['timestamp'].diff().dropna()
                median_interval = time_diffs.median()
                
                # Determine if dataset has regular intervals
                std_interval = time_diffs.std()
                is_regular = std_interval < median_interval * 0.5
                
                st.markdown(f"""
                ### Time Series Information
                - **Time Span**: {time_span} ({time_min} to {time_max})
                - **Typical Sampling Interval**: {median_interval}
                - **Regular Sampling**: {"Yes" if is_regular else "No"}
                """)
    
    with st.expander("Data Statistics"):
        # Show tabs for different statistical views
        stat_tab1, stat_tab2 = st.tabs(["Numeric Statistics", "Category Distributions"])
        
        with stat_tab1:
            # Enhanced numeric statistics
            if not data.select_dtypes(include=['number']).empty:
                numeric_stats = data.describe(percentiles=[.05, .25, .5, .75, .95]).T
                
                # Add additional statistics
                if len(data) > 1:
                    numeric_stats['skew'] = data.select_dtypes(include=['number']).skew()
                    numeric_stats['kurtosis'] = data.select_dtypes(include=['number']).kurtosis()
                    
                    # Add coefficient of variation (CV) - normalized measure of dispersion
                    # Filter to only numeric columns to avoid timestamp issues
                    numeric_only = numeric_stats.index.isin(data.select_dtypes(include=['number']).columns)
                    if any(numeric_only):
                        numeric_stats.loc[numeric_only, 'cv'] = (
                            numeric_stats.loc[numeric_only, 'std'] / 
                            numeric_stats.loc[numeric_only, 'mean'].abs().replace(0, float('nan'))
                        )
                
                # Format the statistics
                st.dataframe(numeric_stats)
                
                # If we have consumption data, show quick visualization
                if 'consumption' in data.columns:
                    st.subheader("Consumption Distribution")
                    try:
                        fig = px.histogram(data, x='consumption', nbins=50)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.write("Could not generate histogram for consumption data.")
            else:
                st.info("No numeric columns found in the dataset.")
        
        with stat_tab2:
            # Show category distributions for categorical columns
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                for col in cat_cols:
                    if data[col].nunique() <= 20:  # Only show for columns with reasonable number of categories
                        st.subheader(f"{col} Distribution")
                        try:
                            # Get value counts and convert to DataFrame for better display
                            val_counts = data[col].value_counts()
                            # Create pie chart
                            fig = px.pie(names=val_counts.index, values=val_counts.values, title=f"{col} Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.write(f"Could not generate pie chart for {col}.")
                            st.write(data[col].value_counts())
                    else:
                        st.write(f"Column '{col}' has {data[col].nunique()} unique values (too many to show distribution).")
            else:
                st.info("No categorical columns found in the dataset.")
    
    with st.expander("Column Information", expanded=True):
        # Get column information
        column_info = []
        
        st.markdown("### Column Mappings")
        st.markdown("The system automatically detected these column mappings:")
        
        # Create mapping table
        mapping_info = []
        
        # Check for standard energy-related columns and what they map to
        key_columns = {
            'timestamp': 'Time information',
            'consumption': 'Energy consumption data',
            'temperature': 'Environmental temperature',
            'humidity': 'Environmental humidity',
            'location': 'Location/zone information',
            'meter_id': 'Meter or device identifier'
        }
        
        for target_col, description in key_columns.items():
            if target_col in data.columns:
                mapping_info.append({
                    "Standardized Column": target_col,
                    "Original Column": target_col,
                    "Description": description,
                    "Status": "✅ Found",
                })
            else:
                # Try to find a column that might have been mapped to this standard name
                possible_sources = []
                for col in data.columns:
                    col_lower = col.lower()
                    if target_col == 'timestamp' and any(time_word in col_lower for time_word in ['time', 'date']):
                        possible_sources.append(col)
                    elif target_col == 'consumption' and any(keyword in col_lower for keyword in ['consumption', 'energy', 'power', 'kwh']):
                        possible_sources.append(col)
                    elif target_col == 'temperature' and 'temp' in col_lower:
                        possible_sources.append(col)
                    elif target_col == 'humidity' and 'humid' in col_lower:
                        possible_sources.append(col)
                    elif target_col == 'location' and any(keyword in col_lower for keyword in ['location', 'area', 'zone', 'room']):
                        possible_sources.append(col)
                    elif target_col == 'meter_id' and any(keyword in col_lower for keyword in ['meter', 'id', 'device']):
                        possible_sources.append(col)
                
                if possible_sources:
                    mapping_info.append({
                        "Standardized Column": target_col,
                        "Original Column": ", ".join(possible_sources),
                        "Description": description,
                        "Status": "🔄 Mapped",
                    })
                else:
                    mapping_info.append({
                        "Standardized Column": target_col,
                        "Original Column": "N/A",
                        "Description": description,
                        "Status": "❌ Not found",
                    })
        
        st.table(pd.DataFrame(mapping_info))
        
        st.markdown("### Detailed Column Information")
        
        # More detailed column information
        for col in data.columns:
            col_type = str(data[col].dtype)
            non_null = data[col].count()
            null_count = data[col].isna().sum()
            unique_count = data[col].nunique()
            
            # Determine column category for better organization
            if col == 'timestamp' or 'date' in col.lower() or 'time' in col.lower():
                col_category = 'Time'
            elif col == 'consumption' or any(keyword in col.lower() for keyword in ['energy', 'power', 'kwh', 'watt']):
                col_category = 'Energy'
            elif col == 'temperature' or 'temp' in col.lower():
                col_category = 'Environment'
            elif col == 'humidity' or 'humid' in col.lower():
                col_category = 'Environment'
            elif col == 'location' or any(keyword in col.lower() for keyword in ['area', 'zone', 'room', 'region']):
                col_category = 'Location'
            elif col == 'meter_id' or any(keyword in col.lower() for keyword in ['meter', 'device', 'sensor']):
                col_category = 'Device'
            else:
                col_category = 'Other'
            
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
                "Category": col_category,
                "Type": col_type,
                "Non-Null Count": non_null,
                "Null Count": null_count,
                "Unique Values": unique_count,
                "Top Values/Range": top_values_str
            })
        
        # Display as DataFrame
        st.dataframe(pd.DataFrame(column_info))
    
    # If it's a time series (has timestamp column), offer a time series preview
    if 'timestamp' in data.columns and 'consumption' in data.columns:
        with st.expander("Time Series Preview"):
            try:
                # Check if dataset is too large for plotting
                if len(data) > 10000:
                    # Downsample for visualization
                    st.info(f"Dataset is large ({len(data)} rows). Showing downsampled visualization.")
                    
                    # Convert to datetime index for resampling
                    temp_df = data.set_index('timestamp')
                    
                    # Determine appropriate sampling frequency
                    time_range = (temp_df.index.max() - temp_df.index.min()).total_seconds()
                    if time_range > 86400 * 365:  # More than a year
                        freq = 'W'  # Weekly
                    elif time_range > 86400 * 30:  # More than a month
                        freq = 'D'  # Daily
                    else:
                        freq = 'H'  # Hourly
                        
                    # Resample
                    plot_data = temp_df['consumption'].resample(freq).mean().reset_index()
                else:
                    plot_data = data[['timestamp', 'consumption']]
                
                # Create interactive time series plot
                fig = px.line(plot_data, x='timestamp', y='consumption', title='Energy Consumption Over Time')
                fig.update_layout(xaxis_title='Time', yaxis_title='Consumption')
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a daily/hourly pattern analysis if we have enough data
                if len(data) > 24:
                    st.subheader("Consumption Patterns")
                    
                    # Create hourly patterns
                    if 'hour' not in data.columns and 'timestamp' in data.columns:
                        hour_data = data.copy()
                        hour_data['hour'] = hour_data['timestamp'].dt.hour
                    else:
                        hour_data = data
                        
                    if 'hour' in hour_data.columns:
                        hourly_avg = hour_data.groupby('hour')['consumption'].mean().reset_index()
                        fig_hourly = px.bar(hourly_avg, x='hour', y='consumption', title='Average Consumption by Hour of Day')
                        fig_hourly.update_layout(xaxis_title='Hour of Day', yaxis_title='Average Consumption')
                        st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    # Create day of week patterns if we have sufficient data
                    if len(data) > 168:  # More than a week of hourly data
                        if 'day_of_week' not in data.columns and 'timestamp' in data.columns:
                            day_data = data.copy()
                            day_data['day_of_week'] = day_data['timestamp'].dt.dayofweek
                        else:
                            day_data = data
                            
                        if 'day_of_week' in day_data.columns:
                            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            daily_avg = day_data.groupby('day_of_week')['consumption'].mean().reset_index()
                            daily_avg['day_name'] = daily_avg['day_of_week'].apply(lambda x: day_names[x] if x < len(day_names) else f"Day {x}")
                            fig_daily = px.bar(daily_avg, x='day_name', y='consumption', title='Average Consumption by Day of Week')
                            fig_daily.update_layout(xaxis_title='Day of Week', yaxis_title='Average Consumption')
                            st.plotly_chart(fig_daily, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate time series preview: {str(e)}")
