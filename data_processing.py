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
    Load and validate uploaded data files.
    Supports CSV, Excel, and other formats.
    Automatically adapts to various dataset structures.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        DataFrame with loaded data or None if invalid
    """
    try:
        # Check file extension and load accordingly
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            # Try different encodings and delimiters for CSV
            try:
                data = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # Try with different encoding if default fails
                data = pd.read_csv(uploaded_file, encoding='latin1')
            except pd.errors.ParserError:
                # Try with different delimiter if comma fails
                try:
                    data = pd.read_csv(uploaded_file, sep=';')
                except:
                    data = pd.read_csv(uploaded_file, sep='\t')
        elif file_ext in ['xlsx', 'xls']:
            data = pd.read_excel(uploaded_file)
        elif file_ext == 'json':
            data = pd.read_json(uploaded_file)
        elif file_ext == 'parquet':
            data = pd.read_parquet(uploaded_file)
        else:
            st.error(f"Unsupported file format: .{file_ext}. Please upload CSV, Excel, JSON, or Parquet file.")
            return None
        
        # Basic validation
        if data.empty:
            st.error("The uploaded file is empty.")
            return None
        
        # Analyze the dataset structure
        st.info("Analyzing dataset structure...")
        
        # 1. Identify time-related columns
        time_columns = []
        for col in data.columns:
            col_lower = col.lower()
            # Check for common time-related column names
            if any(time_word in col_lower for time_word in ['time', 'date', 'year', 'month', 'day', 'hour']):
                time_columns.append(col)
            # Try to check if column data looks like dates
            elif data[col].dtype == 'object':
                try:
                    pd.to_datetime(data[col].iloc[0])
                    time_columns.append(col)
                except:
                    pass
        
        # 2. Identify energy consumption related columns
        consumption_columns = []
        energy_keywords = ['consumption', 'energy', 'usage', 'power', 'kwh', 'kw', 'watt', 'load', 'demand']
        
        for col in data.columns:
            col_lower = col.lower()
            if pd.api.types.is_numeric_dtype(data[col]):
                # Check for common energy-related column names
                if any(keyword in col_lower for keyword in energy_keywords):
                    consumption_columns.append(col)
        
        # If no energy columns found, use numeric columns with reasonable variance
        if not consumption_columns:
            numeric_cols = data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                # Only consider numeric columns with reasonable variance (not binary/categorical)
                if data[col].nunique() > 10 and data[col].std() > 0:
                    consumption_columns.append(col)
        
        # Sort consumption columns by relevance (prefer columns with energy keywords)
        consumption_columns = sorted(consumption_columns, 
                                    key=lambda x: sum(keyword in x.lower() for keyword in energy_keywords), 
                                    reverse=True)
        
        # 3. Handle timestamp column
        if not time_columns:
            st.warning("No time-related columns found. Creating a timestamp sequence.")
            start_date = datetime.datetime.now() - datetime.timedelta(days=30)
            data['timestamp'] = [start_date + datetime.timedelta(hours=i) for i in range(len(data))]
        else:
            # Use the most likely timestamp column
            primary_time_col = time_columns[0]
            
            # Try to convert to datetime with various formats
            try:
                data['timestamp'] = pd.to_datetime(data[primary_time_col], errors='coerce')
                st.info(f"Using '{primary_time_col}' as timestamp.")
            except:
                # Try different formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M', '%d-%m-%Y %H:%M', 
                           '%m/%d/%Y %H:%M', '%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        data['timestamp'] = pd.to_datetime(data[primary_time_col], format=fmt, errors='coerce')
                        if not data['timestamp'].isna().all():
                            st.info(f"Using '{primary_time_col}' as timestamp with format {fmt}.")
                            break
                    except:
                        continue
            
            # If still having NaT values, try to fix them
            if data['timestamp'].isna().any():
                st.warning(f"Some timestamp values couldn't be parsed. Filling with estimated values.")
                # Interpolate timestamps if possible
                valid_idx = data['timestamp'].dropna().index
                if len(valid_idx) > 1:
                    # Calculate average time difference between valid timestamps
                    avg_timedelta = (data.loc[valid_idx[-1], 'timestamp'] - data.loc[valid_idx[0], 'timestamp']) / (len(valid_idx) - 1)
                    
                    # Fill missing values
                    for i in data.index:
                        if pd.isna(data.loc[i, 'timestamp']):
                            # Find closest previous valid timestamp
                            prev_valid = valid_idx[valid_idx < i]
                            if len(prev_valid) > 0:
                                prev_idx = prev_valid[-1]
                                steps = i - prev_idx
                                data.loc[i, 'timestamp'] = data.loc[prev_idx, 'timestamp'] + avg_timedelta * steps
                            else:
                                # No previous valid timestamp, use the first valid one minus steps
                                next_idx = valid_idx[0]
                                steps = next_idx - i
                                data.loc[i, 'timestamp'] = data.loc[next_idx, 'timestamp'] - avg_timedelta * steps
        
        # Handle consumption data
        if not consumption_columns:
            st.error("No suitable consumption data columns found in the dataset.")
            return None
        else:
            primary_consumption_col = consumption_columns[0]
            data['consumption'] = data[primary_consumption_col]
            st.info(f"Using '{primary_consumption_col}' as consumption data.")
            
            # Identify other potentially useful columns
            # Temperature
            temp_columns = [col for col in data.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
            if temp_columns:
                data['temperature'] = data[temp_columns[0]]
                st.info(f"Using '{temp_columns[0]}' as temperature data.")
            
            # Humidity
            humidity_columns = [col for col in data.columns if 'humid' in col.lower() or 'humidity' in col.lower()]
            if humidity_columns:
                data['humidity'] = data[humidity_columns[0]]
                st.info(f"Using '{humidity_columns[0]}' as humidity data.")
            
            # Location or area information
            location_columns = [col for col in data.columns if any(loc in col.lower() for loc in ['location', 'area', 'zone', 'region', 'room'])]
            if location_columns:
                data['location'] = data[location_columns[0]]
                st.info(f"Using '{location_columns[0]}' as location data.")
            
            # Device or meter ID
            id_columns = [col for col in data.columns if any(id_word in col.lower() for id_word in ['id', 'meter', 'device', 'sensor'])]
            if id_columns:
                data['meter_id'] = data[id_columns[0]]
                st.info(f"Using '{id_columns[0]}' as meter ID.")
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Report dataset summary
        st.success(f"Successfully loaded {len(data)} records spanning from {data['timestamp'].min()} to {data['timestamp'].max()}.")
        
        return data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data):
    """
    Preprocess data for anomaly detection.
    Adapts to various data formats and automatically derives features.
    
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
    
    # For consumption (numeric columns), use interpolation based on time series characteristics
    numeric_columns = processed_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        missing_count = processed_data[col].isna().sum()
        if missing_count > 0:
            # Check if we can use time-based interpolation for time series data
            if 'timestamp' in processed_data.columns and missing_count < len(processed_data) * 0.3:  # Less than 30% missing
                st.warning(f"Found {missing_count} missing values in {col}. Using time-based interpolation.")
                # Set timestamp as index for time-based interpolation
                temp_df = processed_data.set_index('timestamp')
                temp_df[col] = temp_df[col].interpolate(method='time')
                processed_data[col] = temp_df[col].values
            else:
                # Fall back to linear interpolation
                st.warning(f"Found {missing_count} missing values in {col}. Using linear interpolation.")
                processed_data[col] = processed_data[col].interpolate(method='linear', limit_direction='both')
                
            # If still have missing values at the edges, use forward/backward fill
            if processed_data[col].isna().any():
                processed_data[col] = processed_data[col].fillna(method='ffill').fillna(method='bfill')
    
    # For categorical columns, fill with most frequent value
    categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        missing_count = processed_data[col].isna().sum()
        if missing_count > 0:
            # If it's a high cardinality categorical variable, use a special "Unknown" category
            if processed_data[col].nunique() > 10:
                st.warning(f"Found {missing_count} missing values in high-cardinality column {col}. Using 'Unknown' category.")
                processed_data[col] = processed_data[col].fillna('Unknown')
            else:
                most_frequent = processed_data[col].mode()[0]
                st.warning(f"Found {missing_count} missing values in {col}. Filling with most frequent value: '{most_frequent}'.")
                processed_data[col] = processed_data[col].fillna(most_frequent)
    
    # Advanced feature engineering
    st.info("Performing feature engineering...")
    
    # Extract time-based features from timestamp
    if 'timestamp' in processed_data.columns:
        # Basic time features
        processed_data['hour'] = processed_data['timestamp'].dt.hour
        processed_data['day_of_week'] = processed_data['timestamp'].dt.dayofweek
        processed_data['day'] = processed_data['timestamp'].dt.day
        processed_data['month'] = processed_data['timestamp'].dt.month
        processed_data['year'] = processed_data['timestamp'].dt.year
        processed_data['is_weekend'] = processed_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Create cyclic time features to preserve continuity
        processed_data['hour_sin'] = np.sin(2 * np.pi * processed_data['hour'] / 24)
        processed_data['hour_cos'] = np.cos(2 * np.pi * processed_data['hour'] / 24)
        processed_data['month_sin'] = np.sin(2 * np.pi * processed_data['month'] / 12)
        processed_data['month_cos'] = np.cos(2 * np.pi * processed_data['month'] / 12)
        processed_data['day_of_week_sin'] = np.sin(2 * np.pi * processed_data['day_of_week'] / 7)
        processed_data['day_of_week_cos'] = np.cos(2 * np.pi * processed_data['day_of_week'] / 7)
        
        # Create time of day category for easy interpretation
        hours = processed_data['hour']
        processed_data['time_of_day'] = pd.cut(
            hours, 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
            include_lowest=True
        )
        
        # Create season (handles both hemispheres)
        month = processed_data['month']
        if 'location' in processed_data.columns:
            # Very simplistic hemisphere detection based on location names
            southern_hemisphere_keywords = ['australia', 'brazil', 'argentina', 'chile', 'south africa', 'new zealand']
            southern_hemisphere = any(k in str(processed_data['location'].iloc[0]).lower() for k in southern_hemisphere_keywords)
            
            if southern_hemisphere:
                # Southern hemisphere seasons
                processed_data['season'] = pd.cut(
                    month, 
                    bins=[0, 3, 6, 9, 12], 
                    labels=['Summer', 'Fall', 'Winter', 'Spring'], 
                    include_lowest=True
                )
                st.info("Detected Southern Hemisphere location - adjusted seasons accordingly.")
            else:
                # Northern hemisphere seasons (default)
                processed_data['season'] = pd.cut(
                    month, 
                    bins=[0, 3, 6, 9, 12], 
                    labels=['Winter', 'Spring', 'Summer', 'Fall'], 
                    include_lowest=True
                )
        else:
            # Default to Northern hemisphere
            processed_data['season'] = pd.cut(
                month, 
                bins=[0, 3, 6, 9, 12], 
                labels=['Winter', 'Spring', 'Summer', 'Fall'], 
                include_lowest=True
            )
        
        # Detect data frequency and granularity
        if len(processed_data) > 1:
            # Calculate time differences
            time_diffs = processed_data['timestamp'].diff().dropna()
            median_diff = time_diffs.median()
            
            # Identify granularity of data (hourly, daily, etc.)
            if median_diff <= pd.Timedelta(minutes=5):
                granularity = 'high_frequency'
                window_size = 12  # 1 hour for 5-min data
            elif median_diff <= pd.Timedelta(hours=1):
                granularity = 'hourly'
                window_size = 24  # 24 hours
            elif median_diff <= pd.Timedelta(days=1):
                granularity = 'daily'
                window_size = 7  # 7 days
            else:
                granularity = 'low_frequency'
                window_size = 4  # 4 data points
                
            st.info(f"Detected data granularity: {granularity} (median time difference: {median_diff})")
        else:
            # Default for very small datasets
            granularity = 'unknown'
            window_size = 3
    
    # Consumption-based features
    if 'consumption' in processed_data.columns:
        # Determine appropriate window sizes based on dataset
        # Default values first
        short_window = max(3, min(7, len(processed_data) // 100))
        medium_window = max(7, min(14, len(processed_data) // 50))
        long_window = max(14, min(30, len(processed_data) // 20))
        
        # Adjust based on detected granularity if available
        if 'timestamp' in processed_data.columns and len(processed_data) > 1:
            # Calculate time differences again (to avoid scope issues)
            time_diffs = processed_data['timestamp'].diff().dropna()
            if not time_diffs.empty:
                median_diff = time_diffs.median()
                
                # Adjust windows based on data frequency
                if median_diff <= pd.Timedelta(minutes=5):
                    # High frequency data (5-min intervals or less)
                    short_window = max(6, min(12, len(processed_data) // 100))
                    medium_window = max(12, min(24, len(processed_data) // 50))
                    long_window = max(24, min(72, len(processed_data) // 20))
                elif median_diff <= pd.Timedelta(hours=1):
                    # Hourly data
                    short_window = max(4, min(8, len(processed_data) // 100))
                    medium_window = max(12, min(24, len(processed_data) // 50))
                    long_window = max(24, min(48, len(processed_data) // 20))
                elif median_diff <= pd.Timedelta(days=1):
                    # Daily data
                    short_window = max(3, min(7, len(processed_data) // 100))
                    medium_window = max(7, min(14, len(processed_data) // 50))
                    long_window = max(14, min(30, len(processed_data) // 20))
        
        # Ensure windows aren't larger than the dataset
        max_window = len(processed_data) // 4
        short_window = min(short_window, max_window)
        medium_window = min(medium_window, max_window)
        long_window = min(long_window, max_window)
        
        # Create rolling features if there's enough data
        if len(processed_data) > short_window * 2:
            processed_data['consumption_rolling_mean_short'] = processed_data['consumption'].rolling(
                window=short_window, min_periods=1).mean()
            processed_data['consumption_rolling_std_short'] = processed_data['consumption'].rolling(
                window=short_window, min_periods=1).std().fillna(0)
            
        if len(processed_data) > medium_window * 2:
            processed_data['consumption_rolling_mean_medium'] = processed_data['consumption'].rolling(
                window=medium_window, min_periods=1).mean()
            processed_data['consumption_rolling_std_medium'] = processed_data['consumption'].rolling(
                window=medium_window, min_periods=1).std().fillna(0)
            
        if len(processed_data) > long_window * 2:
            processed_data['consumption_rolling_mean_long'] = processed_data['consumption'].rolling(
                window=long_window, min_periods=1).mean()
            processed_data['consumption_rolling_std_long'] = processed_data['consumption'].rolling(
                window=long_window, min_periods=1).std().fillna(0)
        
        # Add rate of change (percentage change) features
        processed_data['consumption_pct_change'] = processed_data['consumption'].pct_change().fillna(0)
        
        # Add lag features if enough data
        if len(processed_data) > 10:
            shift_values = [1, 2, 3]  # Small shifts for any dataset
            
            # Add more lags for larger datasets with higher frequency
            if len(processed_data) > 100 and 'timestamp' in processed_data.columns:
                # Calculate time differences again to determine additional lag values
                time_diffs = processed_data['timestamp'].diff().dropna()
                if not time_diffs.empty:
                    median_diff = time_diffs.median()
                    
                    # Add larger lags for higher frequency data
                    if median_diff <= pd.Timedelta(hours=1):
                        shift_values.extend([6, 12, 24])
            
            for shift in shift_values:
                if shift < len(processed_data) // 4:  # Only add if shift is reasonable for dataset size
                    processed_data[f'consumption_lag_{shift}'] = processed_data['consumption'].shift(shift).fillna(
                        processed_data['consumption'].mean())
                    
                    # Add diff features
                    processed_data[f'consumption_diff_{shift}'] = processed_data['consumption'].diff(shift).fillna(0)
    
    # Temperature and environmental feature engineering
    if 'temperature' in processed_data.columns and 'consumption' in processed_data.columns:
        # Check if temperature and consumption both have enough variation
        if processed_data['temperature'].std() > 0 and processed_data['consumption'].std() > 0:
            # Nonlinear temperature features (to capture nonlinear relationships with energy)
            processed_data['temperature_squared'] = processed_data['temperature'] ** 2
            
            # Temperature bands for heating/cooling thresholds
            # Generally, heating is needed below 65°F (18°C) and cooling above 75°F (24°C)
            processed_data['heating_degree'] = (18 - processed_data['temperature']).clip(lower=0)
            processed_data['cooling_degree'] = (processed_data['temperature'] - 24).clip(lower=0)
            
            # Consumption efficiency metrics
            processed_data['consumption_per_degree'] = processed_data['consumption'] / processed_data['temperature'].abs().clip(lower=1)
            
            # Interaction with time of day
            if 'hour' in processed_data.columns:
                # Daytime/nighttime temperature interaction
                processed_data['temp_day_interaction'] = processed_data['temperature'] * (
                    (processed_data['hour'] >= 6) & (processed_data['hour'] <= 18)).astype(int)
    
    # Humidity interactions if available
    if all(col in processed_data.columns for col in ['humidity', 'temperature', 'consumption']):
        # Heat index (feels-like temperature) - simplified approximation
        processed_data['heat_index'] = processed_data['temperature'] + 0.05 * processed_data['humidity']
        
        # Interaction with consumption
        processed_data['humidity_consumption_interaction'] = processed_data['humidity'] * processed_data['consumption']
    
    # Location-based features if available
    if 'location' in processed_data.columns and 'consumption' in processed_data.columns:
        # Calculate location-specific consumption statistics
        location_mean = processed_data.groupby('location')['consumption'].transform('mean')
        location_std = processed_data.groupby('location')['consumption'].transform('std')
        
        # Z-score within location (how unusual is the consumption for this specific location)
        processed_data['consumption_zscore_by_location'] = (processed_data['consumption'] - location_mean) / location_std.clip(lower=0.0001)
    
    # Prepare data for ML (select and encode features)
    ml_data = processed_data.copy()
    
    # Keep only numeric features and encode categorical ones
    feature_columns = []
    
    # Include basic numeric features
    for col in ml_data.select_dtypes(include=['float64', 'int64']).columns:
        if col not in ['timestamp'] and not pd.isna(ml_data[col]).all():  # Exclude timestamp and completely NA columns
            feature_columns.append(col)
    
    # Encode categorical features
    for col in ml_data.select_dtypes(include=['object', 'category']).columns:
        if col not in ['timestamp'] and ml_data[col].nunique() > 1:  # Exclude timestamp and constant columns
            # Use label encoding for categorical variables
            le = LabelEncoder()
            ml_data[f'{col}_encoded'] = le.fit_transform(ml_data[col].astype(str))
            feature_columns.append(f'{col}_encoded')
    
    # Final dataset for ML
    ml_features = ml_data[feature_columns]
    
    # Check for and remove problematic columns
    # 1. Constant columns
    constant_columns = [col for col in ml_features.columns if ml_features[col].nunique() == 1]
    if constant_columns:
        st.warning(f"Removing constant columns: {', '.join(constant_columns)}")
        ml_features = ml_features.drop(columns=constant_columns)
    
    # 2. Columns with too many NaNs
    high_na_columns = [col for col in ml_features.columns if ml_features[col].isna().mean() > 0.5]  # >50% NaN
    if high_na_columns:
        st.warning(f"Removing columns with too many missing values: {', '.join(high_na_columns)}")
        ml_features = ml_features.drop(columns=high_na_columns)
    
    # 3. Highly correlated features (keep one from each highly correlated pair)
    if len(ml_features.columns) > 10:  # Only if we have many features
        try:
            # Calculate correlation matrix
            corr_matrix = ml_features.corr().abs()
            
            # Extract upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find columns with correlation > 0.95
            to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
            
            if to_drop:
                st.warning(f"Removing highly correlated features: {', '.join(to_drop)}")
                ml_features = ml_features.drop(columns=to_drop)
        except Exception as e:
            st.warning(f"Couldn't perform correlation analysis: {str(e)}")
    
    # Clean data before scaling - handle NaN, infinity and extreme values
    ml_features = ml_features.fillna(0)
    
    # Handle infinity and extreme values
    # Replace inf/-inf with NaN first, then replace with very large/small values
    ml_features = ml_features.replace([np.inf, -np.inf], np.nan)
    
    # Find columns with very large values that might cause scaling problems
    for col in ml_features.columns:
        # Calculate reasonable bounds (99.9% of normal distribution)
        col_mean = ml_features[col].mean()
        col_std = ml_features[col].std()
        if pd.notna(col_mean) and pd.notna(col_std) and col_std > 0:
            upper_bound = col_mean + 5 * col_std
            lower_bound = col_mean - 5 * col_std
            
            # Replace extreme outliers with bounds
            # This preserves outlier signal but prevents scaling issues
            too_large = ml_features[col] > upper_bound
            too_small = ml_features[col] < lower_bound
            if too_large.any():
                ml_features.loc[too_large, col] = upper_bound
            if too_small.any():
                ml_features.loc[too_small, col] = lower_bound
    
    # Final NaN handling
    ml_features = ml_features.fillna(0)
    
    # Try scaling with robust checks
    try:
        # Scale features for ML algorithms
        feature_names = ml_features.columns
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(ml_features)
        
        # Check if scaling produced any NaN or infinite values
        if not np.all(np.isfinite(scaled_features)):
            # If there are problems, use a more robust scaling approach
            st.warning("Standard scaling produced non-finite values. Using robust scaling instead.")
            # Replace problematic values
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        ml_features = pd.DataFrame(scaled_features, columns=feature_names)
    except Exception as e:
        st.warning(f"Scaling error: {str(e)}. Using original features with capped values.")
        # If scaling completely fails, just use the capped values
        # Normalize manually to the 0-1 range for each column
        for col in ml_features.columns:
            col_min = ml_features[col].min()
            col_max = ml_features[col].max()
            if col_max > col_min:
                ml_features[col] = (ml_features[col] - col_min) / (col_max - col_min)
    
    st.success(f"Data preprocessing complete! Created {len(ml_features.columns)} features for analysis.")
    
    return ml_features, processed_data

def detect_contradictions(data):
    """
    Detect contradictory or inconsistent data in energy consumption datasets.
    Adapts to various dataset structures and column naming conventions.
    
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
    
    # Only add the contradiction detection logic on datasets with sufficient size
    min_required_rows = 10
    if len(flagged_data) < min_required_rows:
        st.warning(f"Dataset too small ({len(flagged_data)} rows) for reliable contradiction detection. Skipping this step.")
        return flagged_data
    
    st.info("Analyzing data for potential inconsistencies...")
    
    # 1. Check for physically impossible values in energy consumption
    if 'consumption' in flagged_data.columns:
        # Negative consumption (usually impossible in energy data)
        neg_consumption = flagged_data['consumption'] < 0
        if neg_consumption.any():
            flagged_data.loc[neg_consumption, 'contradiction_flag'] = True
            flagged_data.loc[neg_consumption, 'contradiction_reason'] += 'Negative consumption; '
            st.warning(f"Found {neg_consumption.sum()} negative consumption values")
        
        # Check for statistical outliers (values > 3 standard deviations from mean)
        # Use robust statistics for outlier detection
        mean_consumption = flagged_data['consumption'].median()
        mad = (flagged_data['consumption'] - mean_consumption).abs().median() * 1.4826  # MAD estimator
        
        # Extreme outliers (>5 MADs from median)
        extreme_outliers = (flagged_data['consumption'] - mean_consumption).abs() > (5 * mad)
        if extreme_outliers.any() and mad > 0:
            flagged_data.loc[extreme_outliers, 'contradiction_flag'] = True
            flagged_data.loc[extreme_outliers, 'contradiction_reason'] += 'Extreme consumption outlier; '
            st.warning(f"Found {extreme_outliers.sum()} extreme consumption outliers that may need verification.")
    
    # 2. Detect impossible patterns in time series data
    if 'timestamp' in flagged_data.columns:
        # Sort by timestamp for time-based analysis
        flagged_data = flagged_data.sort_values('timestamp')
        
        # Check for future timestamps (can be valid in some forecast datasets, so just flag them)
        future_dates = flagged_data['timestamp'] > pd.Timestamp.now()
        if future_dates.any():
            flagged_data.loc[future_dates, 'contradiction_flag'] = True
            flagged_data.loc[future_dates, 'contradiction_reason'] += 'Future timestamp; '
            st.info(f"Found {future_dates.sum()} future timestamps (may be forecasts or simulation data)")
        
        # Check for duplicated timestamps
        dup_timestamps = flagged_data.duplicated(subset=['timestamp'])
        if dup_timestamps.any():
            # Not all duplicated timestamps are errors - check for different meter_ids or locations
            if 'meter_id' in flagged_data.columns:
                # Only flag duplicates that have the same timestamp AND meter_id
                real_dups = flagged_data.duplicated(subset=['timestamp', 'meter_id'])
                flagged_data.loc[real_dups, 'contradiction_flag'] = True
                flagged_data.loc[real_dups, 'contradiction_reason'] += 'Duplicated timestamp for same meter; '
                if real_dups.any():
                    st.warning(f"Found {real_dups.sum()} duplicated timestamps for the same meter")
            elif 'location' in flagged_data.columns:
                # Only flag duplicates with same timestamp AND location
                real_dups = flagged_data.duplicated(subset=['timestamp', 'location'])
                flagged_data.loc[real_dups, 'contradiction_flag'] = True
                flagged_data.loc[real_dups, 'contradiction_reason'] += 'Duplicated timestamp for same location; '
                if real_dups.any():
                    st.warning(f"Found {real_dups.sum()} duplicated timestamps for the same location")
            else:
                # No identifiers to check, flag all duplicates
                flagged_data.loc[dup_timestamps, 'contradiction_flag'] = True
                flagged_data.loc[dup_timestamps, 'contradiction_reason'] += 'Duplicated timestamp; '
                st.warning(f"Found {dup_timestamps.sum()} duplicated timestamps")
        
        # Check for time gaps (if dataset should be continuous)
        # First determine the typical time interval
        if len(flagged_data) > 2:
            time_diffs = flagged_data['timestamp'].diff().dropna()
            
            if not time_diffs.empty:
                median_diff = time_diffs.median()
                
                # Only check for gaps if we have regular readings (low std dev in time differences)
                time_diff_std = time_diffs.std()
                if time_diff_std < median_diff * 0.5:  # Indicates fairly regular readings
                    # Look for gaps > 2x the median difference
                    large_gaps = time_diffs > (median_diff * 2)
                    if large_gaps.any():
                        gap_indices = large_gaps[large_gaps].index
                        for idx in gap_indices:
                            flagged_data.loc[idx, 'contradiction_flag'] = True
                            gap_size = time_diffs.loc[idx]
                            flagged_data.loc[idx, 'contradiction_reason'] += f'Time gap of {gap_size}; '
                        
                        st.info(f"Found {large_gaps.sum()} potential missing data points (time gaps)")
    
    # 3. Check for environmental data inconsistencies
    if 'temperature' in flagged_data.columns:
        # Check for physically implausible temperature values (extreme values)
        extreme_low_temp = flagged_data['temperature'] < -50  # -50°C/-58°F is extremely rare
        extreme_high_temp = flagged_data['temperature'] > 55  # 55°C/131°F is extremely rare
        
        if extreme_low_temp.any():
            flagged_data.loc[extreme_low_temp, 'contradiction_flag'] = True
            flagged_data.loc[extreme_low_temp, 'contradiction_reason'] += 'Extremely low temperature; '
            st.warning(f"Found {extreme_low_temp.sum()} extremely low temperature values (< -50°C)")
            
        if extreme_high_temp.any():
            flagged_data.loc[extreme_high_temp, 'contradiction_flag'] = True
            flagged_data.loc[extreme_high_temp, 'contradiction_reason'] += 'Extremely high temperature; '
            st.warning(f"Found {extreme_high_temp.sum()} extremely high temperature values (> 55°C)")
        
        # Check for rapid temperature changes
        if 'timestamp' in flagged_data.columns:
            # Compute time differences in hours
            time_diffs_hours = flagged_data['timestamp'].diff().dt.total_seconds() / 3600
            
            # Compute temperature differences
            temp_diffs = flagged_data['temperature'].diff().abs()
            
            # Calculate rate of change (°C/hour)
            temp_change_rate = temp_diffs / time_diffs_hours.clip(lower=0.01)  # Avoid division by zero
            
            # Flag rapid temperature changes (>10°C per hour is unusual for ambient temps)
            rapid_temp_change = (temp_change_rate > 10) & (~temp_change_rate.isna())
            if rapid_temp_change.any():
                flagged_data.loc[rapid_temp_change, 'contradiction_flag'] = True
                flagged_data.loc[rapid_temp_change, 'contradiction_reason'] += 'Rapid temperature change; '
                st.warning(f"Found {rapid_temp_change.sum()} instances of unusually rapid temperature changes")
    
    # 4. Check for humidity inconsistencies
    if 'humidity' in flagged_data.columns:
        # Humidity should be between 0-100%
        invalid_humidity = (flagged_data['humidity'] < 0) | (flagged_data['humidity'] > 100)
        if invalid_humidity.any():
            flagged_data.loc[invalid_humidity, 'contradiction_flag'] = True
            flagged_data.loc[invalid_humidity, 'contradiction_reason'] += 'Invalid humidity value; '
            st.warning(f"Found {invalid_humidity.sum()} invalid humidity values (outside 0-100% range)")
    
    # 5. Check for logical consumption inconsistencies
    if all(col in flagged_data.columns for col in ['consumption', 'timestamp']):
        # Look for sudden zero consumption values that may indicate meter failures
        # First, check if this is a single-meter dataset or multiple meters
        has_meter_ids = 'meter_id' in flagged_data.columns
        
        if has_meter_ids:
            # Group by meter_id to analyze each separately
            for meter, meter_data in flagged_data.groupby('meter_id'):
                if len(meter_data) > 10:  # Only analyze if enough data for this meter
                    # Sort by timestamp
                    meter_data = meter_data.sort_values('timestamp')
                    
                    # Look for sudden drops to zero surrounded by normal consumption
                    is_zero = meter_data['consumption'] == 0
                    if is_zero.any() and not is_zero.all():  # Some zeros, but not all
                        # Flag isolated zeros (zero values surrounded by non-zeros)
                        for i in range(1, len(meter_data)-1):
                            if (meter_data.iloc[i]['consumption'] == 0 and 
                                meter_data.iloc[i-1]['consumption'] > 0 and 
                                meter_data.iloc[i+1]['consumption'] > 0):
                                idx = meter_data.iloc[i].name
                                flagged_data.loc[idx, 'contradiction_flag'] = True
                                flagged_data.loc[idx, 'contradiction_reason'] += 'Isolated zero consumption; '
        else:
            # Single meter dataset
            if len(flagged_data) > 10:
                # Look for isolated zeros
                flagged_data = flagged_data.sort_values('timestamp')
                consumption_vals = flagged_data['consumption'].values
                
                for i in range(1, len(flagged_data)-1):
                    if (consumption_vals[i] == 0 and 
                        consumption_vals[i-1] > 0 and 
                        consumption_vals[i+1] > 0):
                        idx = flagged_data.index[i]
                        flagged_data.loc[idx, 'contradiction_flag'] = True
                        flagged_data.loc[idx, 'contradiction_reason'] += 'Isolated zero consumption; '
    
    # Trim trailing separators from reason string
    flagged_data['contradiction_reason'] = flagged_data['contradiction_reason'].str.rstrip('; ')
    
    # Summary of contradictions
    contradiction_count = flagged_data['contradiction_flag'].sum()
    if contradiction_count > 0:
        st.warning(f"Detected {contradiction_count} potential data contradictions or inconsistencies.")
    else:
        st.success("No data contradictions or inconsistencies detected.")
    
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
