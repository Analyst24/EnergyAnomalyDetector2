import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import io
import os
from pathlib import Path
import datetime
import time

def load_data(uploaded_file):
    """
    Load and validate uploaded CSV data.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        DataFrame with loaded data or None if invalid
    """
    try:
        # Check file extension
        if not uploaded_file.name.endswith('.csv'):
            st.error("Please upload a CSV file.")
            return None
        
        # Load the CSV file
        data = pd.read_csv(uploaded_file)
        
        # Basic validation
        if data.empty:
            st.error("The uploaded file is empty.")
            return None
        
        # Check for minimum required columns
        required_columns = ['timestamp', 'consumption']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.warning(f"Warning: Missing recommended columns: {', '.join(missing_columns)}")
            
            # If timestamp is missing, try to find alternatives or create a dummy
            if 'timestamp' in missing_columns:
                time_columns = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
                
                if time_columns:
                    # Use the first time-related column as timestamp
                    data['timestamp'] = data[time_columns[0]]
                    st.info(f"Using '{time_columns[0]}' as timestamp.")
                else:
                    # Create a dummy timestamp starting from today
                    st.warning("No time-related columns found. Creating a dummy timestamp sequence.")
                    start_date = datetime.datetime.now()
                    data['timestamp'] = [start_date + datetime.timedelta(hours=i) for i in range(len(data))]
            
            # If consumption is missing, try to find alternatives
            if 'consumption' in missing_columns:
                # Look for numeric columns that might represent consumption
                numeric_cols = data.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) > 0:
                    # Use the first numeric column as consumption
                    data['consumption'] = data[numeric_cols[0]]
                    st.info(f"Using '{numeric_cols[0]}' as consumption.")
                else:
                    st.error("No numeric columns found to use as consumption data.")
                    return None
        
        # Convert timestamp to datetime if it's not already
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except Exception as e:
            st.error(f"Error converting timestamp column to datetime: {str(e)}")
            # Create a dummy timestamp as fallback
            start_date = datetime.datetime.now()
            data['timestamp'] = [start_date + datetime.timedelta(hours=i) for i in range(len(data))]
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        return data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data):
    """
    Preprocess data for anomaly detection.
    
    Args:
        data: DataFrame with raw data
    
    Returns:
        Tuple of (preprocessed_data_for_ml, processed_full_data)
    """
    if data is None or data.empty:
        return None, None
    
    # Create a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Handle missing values
    st.info("Handling missing values...")
    
    # For consumption (numeric columns), use linear interpolation
    numeric_columns = processed_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        missing_count = processed_data[col].isna().sum()
        if missing_count > 0:
            st.warning(f"Found {missing_count} missing values in {col}. Using interpolation.")
            processed_data[col] = processed_data[col].interpolate(method='linear', limit_direction='both')
    
    # For categorical columns, fill with most frequent value
    categorical_columns = processed_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        missing_count = processed_data[col].isna().sum()
        if missing_count > 0:
            most_frequent = processed_data[col].mode()[0]
            st.warning(f"Found {missing_count} missing values in {col}. Filling with most frequent value: '{most_frequent}'.")
            processed_data[col] = processed_data[col].fillna(most_frequent)
    
    # Feature engineering
    st.info("Performing feature engineering...")
    
    # Extract time features from timestamp
    if 'timestamp' in processed_data.columns:
        processed_data['hour'] = processed_data['timestamp'].dt.hour
        processed_data['day_of_week'] = processed_data['timestamp'].dt.dayofweek
        processed_data['month'] = processed_data['timestamp'].dt.month
        
        # Create time of day category
        hours = processed_data['hour']
        processed_data['time_of_day'] = pd.cut(
            hours, 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
            include_lowest=True
        )
        
        # Create season (Northern Hemisphere)
        month = processed_data['month']
        processed_data['season'] = pd.cut(
            month, 
            bins=[0, 3, 6, 9, 12], 
            labels=['Winter', 'Spring', 'Summer', 'Fall'], 
            include_lowest=True
        )
    
    # Calculate rolling features for time series
    if 'consumption' in processed_data.columns:
        processed_data['consumption_rolling_mean'] = processed_data['consumption'].rolling(window=24, min_periods=1).mean()
        processed_data['consumption_rolling_std'] = processed_data['consumption'].rolling(window=24, min_periods=1).std().fillna(0)
    
    # Calculate temperature-related features if available
    if 'temperature' in processed_data.columns and 'consumption' in processed_data.columns:
        # Calculate consumption per degree (efficiency metric)
        processed_data['consumption_per_degree'] = processed_data['consumption'] / processed_data['temperature'].abs().clip(lower=1)
    
    # Prepare data for ML (select and encode features)
    ml_data = processed_data.copy()
    
    # Keep only numeric features and encode categorical ones
    feature_columns = []
    
    # Include basic numeric features
    for col in ml_data.select_dtypes(include=['float64', 'int64']).columns:
        if col not in ['timestamp']:  # Exclude timestamp from ML features
            feature_columns.append(col)
    
    # Encode categorical features if any left
    for col in ml_data.select_dtypes(include=['object', 'category']).columns:
        if col not in ['timestamp']:  # Exclude timestamp from ML features
            # Use label encoding for categorical variables
            le = LabelEncoder()
            ml_data[f'{col}_encoded'] = le.fit_transform(ml_data[col].astype(str))
            feature_columns.append(f'{col}_encoded')
    
    # Final dataset for ML
    ml_features = ml_data[feature_columns]
    
    # Check for constant columns and remove them
    constant_columns = [col for col in ml_features.columns if ml_features[col].nunique() == 1]
    if constant_columns:
        st.warning(f"Removing constant columns: {', '.join(constant_columns)}")
        ml_features = ml_features.drop(columns=constant_columns)
    
    # Fill any remaining NaN with 0
    ml_features = ml_features.fillna(0)
    
    st.success("Data preprocessing complete!")
    
    return ml_features, processed_data

def detect_contradictions(data):
    """
    Detect contradictory or inconsistent data.
    
    Args:
        data: DataFrame with processed data
    
    Returns:
        DataFrame with contradictions flagged
    """
    if data is None or data.empty:
        return None
    
    # Create a copy of the data
    flagged_data = data.copy()
    flagged_data['contradiction_flag'] = False
    flagged_data['contradiction_reason'] = ''
    
    # Check for physical impossibilities
    # 1. Negative consumption values
    if 'consumption' in flagged_data.columns:
        neg_consumption = flagged_data['consumption'] < 0
        if neg_consumption.any():
            flagged_data.loc[neg_consumption, 'contradiction_flag'] = True
            flagged_data.loc[neg_consumption, 'contradiction_reason'] += 'Negative consumption; '
    
    # 2. Extreme temperature variations if temperature column exists
    if 'temperature' in flagged_data.columns:
        # Check for rapid temperature changes (more than 15 degrees between consecutive readings)
        temp_diff = abs(flagged_data['temperature'].diff())
        extreme_temp_change = temp_diff > 15
        if extreme_temp_change.any():
            flagged_data.loc[extreme_temp_change, 'contradiction_flag'] = True
            flagged_data.loc[extreme_temp_change, 'contradiction_reason'] += 'Extreme temperature change; '
    
    # 3. Zero consumption with active usage indicators
    if all(col in flagged_data.columns for col in ['consumption', 'meter_id']):
        zero_with_meter = (flagged_data['consumption'] == 0) & (flagged_data['meter_id'].notna())
        if zero_with_meter.any():
            flagged_data.loc[zero_with_meter, 'contradiction_flag'] = True
            flagged_data.loc[zero_with_meter, 'contradiction_reason'] += 'Zero consumption with active meter; '
    
    # 4. Impossible time patterns
    if 'timestamp' in flagged_data.columns:
        # Check for future timestamps
        future_dates = flagged_data['timestamp'] > pd.Timestamp.now()
        if future_dates.any():
            flagged_data.loc[future_dates, 'contradiction_flag'] = True
            flagged_data.loc[future_dates, 'contradiction_reason'] += 'Future timestamp; '
        
        # Check for duplicated timestamps
        dup_timestamps = flagged_data.duplicated(subset=['timestamp'])
        if dup_timestamps.any():
            flagged_data.loc[dup_timestamps, 'contradiction_flag'] = True
            flagged_data.loc[dup_timestamps, 'contradiction_reason'] += 'Duplicated timestamp; '
    
    # Trim trailing separators from reason string
    flagged_data['contradiction_reason'] = flagged_data['contradiction_reason'].str.rstrip('; ')
    
    # Summary of contradictions
    contradiction_count = flagged_data['contradiction_flag'].sum()
    if contradiction_count > 0:
        st.warning(f"Detected {contradiction_count} potential data contradictions or inconsistencies.")
    
    return flagged_data

def save_results(data, anomalies, filename="anomaly_results.csv"):
    """
    Save the anomaly detection results to a CSV file.
    
    Args:
        data: Original processed DataFrame
        anomalies: Indices of detected anomalies
        filename: Output filename
    
    Returns:
        BytesIO object with the CSV data
    """
    # Create a copy of the data
    results_data = data.copy()
    
    # Mark anomalies
    results_data['is_anomaly'] = 0
    results_data.loc[anomalies, 'is_anomaly'] = 1
    
    # Select columns to include
    output_columns = ['timestamp', 'consumption', 'is_anomaly']
    
    # Add additional columns if they exist
    for col in ['location', 'meter_id', 'temperature', 'humidity', 'season', 'time_of_day']:
        if col in results_data.columns:
            output_columns.append(col)
    
    # Add any contradiction flags if they exist
    if 'contradiction_flag' in results_data.columns:
        output_columns.extend(['contradiction_flag', 'contradiction_reason'])
    
    # Create the CSV in memory
    output = io.BytesIO()
    results_data[output_columns].to_csv(output, index=False)
    output.seek(0)
    
    return output

def generate_sample_data():
    """
    Generate sample energy consumption data for testing.
    
    Returns:
        DataFrame with synthetic data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of data points (1 month of hourly data)
    n_points = 24 * 30
    
    # Generate timestamps (hourly for the last month)
    end_date = pd.Timestamp.now().floor('H')
    start_date = end_date - pd.Timedelta(hours=n_points-1)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=n_points)
    
    # Base consumption pattern (daily cycle with weekday/weekend difference)
    hours = np.array([h.hour for h in timestamps])
    weekdays = np.array([d.weekday() for d in timestamps])
    is_weekend = (weekdays >= 5).astype(int)
    
    # Create daily pattern (higher during day, lower at night)
    daily_pattern = 10 + 8 * np.sin(np.pi * (hours - 5) / 12) * (hours >= 5) * (hours <= 22)
    
    # Weekend pattern (different from weekdays)
    weekend_factor = 1.0 + 0.3 * is_weekend
    
    # Base consumption with patterns
    base_consumption = daily_pattern * weekend_factor
    
    # Add temperature effect (inverse correlation with consumption for heating)
    temps = 15 + 10 * np.sin(np.pi * (timestamps.dayofyear - 30) / 180)  # Yearly cycle
    temp_effect = 5 * (1 / (1 + np.exp(0.5 * (temps - 15))))  # Logistic function
    
    # Add humidity (partially correlated with temperature)
    humidity = 40 + 20 * np.sin(np.pi * (timestamps.dayofyear - 90) / 180) + 5 * np.random.randn(n_points)
    humidity = np.clip(humidity, 0, 100)
    
    # Final consumption with noise and occasional spikes (anomalies)
    noise = 1.5 * np.random.randn(n_points)
    consumption = base_consumption + temp_effect + noise
    
    # Add occasional anomalies (about 5% of the data)
    n_anomalies = int(0.05 * n_points)
    anomaly_idx = np.random.choice(n_points, n_anomalies, replace=False)
    # Convert to numpy array first, modify, then convert back
    consumption_array = consumption.to_numpy()
    consumption_array[anomaly_idx] += np.random.uniform(5, 15, size=n_anomalies)
    consumption = pd.Series(consumption_array)
    
    # Create meter IDs (3 different meters)
    meter_ids = np.random.choice(['M001', 'M002', 'M003'], size=n_points)
    
    # Create locations (5 different locations)
    locations = np.random.choice(['Office', 'Warehouse', 'Store', 'Factory', 'Datacenter'], size=n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'consumption': consumption,
        'temperature': temps,
        'humidity': humidity,
        'meter_id': meter_ids,
        'location': locations
    })
    
    # Add time of day and season based on timestamp
    data['hour'] = data['timestamp'].dt.hour
    data['time_of_day'] = pd.cut(
        data['hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
        include_lowest=True
    )
    
    data['month'] = data['timestamp'].dt.month
    data['season'] = pd.cut(
        data['month'], 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Fall'], 
        include_lowest=True
    )
    
    return data
