import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import time
import base64
import io
from datetime import datetime
from pathlib import Path
import json

def get_icon(name):
    """
    Generate SVG icon URL from Feather Icons.
    
    Args:
        name: Icon name from Feather Icons
    
    Returns:
        URL for the SVG icon
    """
    # Map of common icon names to Feather icon names
    icon_map = {
        "home": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-home"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>""",
        "bar-chart": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bar-chart"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>""",
        "bar-chart-2": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bar-chart-2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>""",
        "upload": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>""",
        "download": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-download"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>""",
        "search": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-search"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>""",
        "settings": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-settings"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>""",
        "activity": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-activity"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>""",
        "pie-chart": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-pie-chart"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>""",
        "award": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-award"><circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline></svg>""",
        "bolt": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>"""
    }
    
    # Return the SVG data string or a default if not found
    return icon_map.get(name, icon_map.get("settings"))

def footer():
    """Display footer on every page with original styling."""
    st.markdown("---")
    
    # Footer with original color scheme
    footer_html = """
    <div style="text-align: center; padding: 10px; margin-top: 30px;">
        <p style="color: #4CAF50; font-size: 14px; margin-bottom: 5px;">⚡ Energy Anomaly Detection System</p>
        <p style="color: #90CAF9; font-size: 12px; margin-bottom: 5px;">Offline analysis powered by advanced ML algorithms</p>
        <p style="color: #aaaaaa; font-size: 10px;">© 2025 Opulent Chikwiramakomo. All rights reserved.</p>
    </div>
    """
    
    st.markdown(footer_html, unsafe_allow_html=True)

def create_pdf_report(data, anomalies, model_results):
    """
    Create a PDF report with anomaly detection results.
    
    Args:
        data: DataFrame with processed data
        anomalies: Indices of detected anomalies
        model_results: Dictionary with model results
    
    Returns:
        BytesIO object with the PDF data
    """
    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set styles
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, 'Energy Anomaly Detection Report', 0, 1, 'C')
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(190, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    
    # Summary section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Summary', 0, 1, 'L')
    
    pdf.set_font('Arial', '', 12)
    total_records = len(data)
    anomaly_count = len(anomalies)
    anomaly_percentage = (anomaly_count / total_records) * 100 if total_records > 0 else 0
    
    pdf.cell(190, 8, f'Total Records: {total_records}', 0, 1, 'L')
    pdf.cell(190, 8, f'Anomalies Detected: {anomaly_count} ({anomaly_percentage:.2f}%)', 0, 1, 'L')
    
    # Date range if timestamp exists
    if 'timestamp' in data.columns:
        min_date = data['timestamp'].min().strftime("%Y-%m-%d")
        max_date = data['timestamp'].max().strftime("%Y-%m-%d")
        pdf.cell(190, 8, f'Date Range: {min_date} to {max_date}', 0, 1, 'L')
    
    # Model performance section
    if model_results:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(190, 10, 'Model Performance', 0, 1, 'L')
        
        pdf.set_font('Arial', '', 12)
        for model_name, result in model_results.items():
            if 'metrics' in result and 'accuracy' in result['metrics']:
                pdf.cell(190, 8, f'Model: {model_name}', 0, 1, 'L')
                pdf.cell(190, 8, f'  Accuracy: {result["metrics"]["accuracy"]:.4f}', 0, 1, 'L')
                pdf.cell(190, 8, f'  Precision: {result["metrics"]["precision"]:.4f}', 0, 1, 'L')
                pdf.cell(190, 8, f'  Recall: {result["metrics"]["recall"]:.4f}', 0, 1, 'L')
                pdf.cell(190, 8, f'  Training Time: {result["training_time"]:.2f} seconds', 0, 1, 'L')
                pdf.cell(190, 5, '', 0, 1, 'L')  # Spacing
    
    # Recommendations section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Recommendations', 0, 1, 'L')
    
    pdf.set_font('Arial', '', 12)
    
    # Add recommendations based on anomaly analysis
    recommendations = [
        "Monitor energy consumption during peak anomaly hours",
        "Investigate systems at locations with high anomaly counts",
        "Consider automated alerts for consumption spikes",
        "Implement regular system maintenance during low-usage periods",
        "Review energy efficiency protocols"
    ]
    
    for i, rec in enumerate(recommendations):
        pdf.cell(190, 8, f'{i+1}. {rec}', 0, 1, 'L')
    
    # Footer
    pdf.set_y(-20)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, '© 2025 Opulent Chikwiramakomo. All rights reserved.', 0, 0, 'C')
    
    # Get PDF as bytes
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output())
    pdf_output.seek(0)
    
    return pdf_output

def get_download_link(data, filename, text):
    """
    Generate a download link for data.
    
    Args:
        data: BytesIO object with data
        filename: Filename for download
        text: Link text to display
    
    Returns:
        HTML string with download link
    """
    b64 = base64.b64encode(data.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def save_plot_as_image(fig):
    """
    Save a Plotly figure as an image.
    
    Args:
        fig: Plotly figure object
    
    Returns:
        BytesIO object with the image data
    """
    img_bytes = io.BytesIO()
    fig.write_image(img_bytes, format='png')
    img_bytes.seek(0)
    return img_bytes

def create_user_data_directory(username):
    """
    Create a user data directory if it doesn't exist.
    
    Args:
        username: Username for the directory
    
    Returns:
        Path object for the user directory
    """
    user_dir = Path(f"user_data/{username}")
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def save_user_settings(username, settings):
    """
    Save user settings to a file.
    
    Args:
        username: Username
        settings: Dictionary with settings
    """
    user_dir = create_user_data_directory(username)
    settings_file = user_dir / "settings.json"
    
    with open(settings_file, "w") as f:
        json.dump(settings, f)

def load_user_settings(username):
    """
    Load user settings from a file.
    
    Args:
        username: Username
    
    Returns:
        Dictionary with settings or default settings
    """
    user_dir = create_user_data_directory(username)
    settings_file = user_dir / "settings.json"
    
    default_settings = {
        "anomaly_threshold": 0.5,
        "selected_algorithms": ["isolation_forest", "autoencoder", "kmeans"],
        "theme": "dark"
    }
    
    if settings_file.exists():
        try:
            with open(settings_file, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
    
    return default_settings
