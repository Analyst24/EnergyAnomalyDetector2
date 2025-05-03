#!/usr/bin/env python3
"""
Main entry point for the Energy Anomaly Detection System
This is designed to be run by clicking the Run button in VS Code
"""

import os
import sys
import subprocess
import logging
import time
import threading
import webbrowser
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnergyAnomalySystem')

# Configuration
FLASK_PORT = 5001
STREAMLIT_PORT = 5000

def open_browser():
    """Open the browser to the Streamlit URL after a short delay"""
    time.sleep(3)  # Give Streamlit time to start
    url = f"http://localhost:{STREAMLIT_PORT}"
    webbrowser.open(url)
    logger.info(f"🌟 Opening browser to {url}")

def run_streamlit():
    """Run the Streamlit frontend"""
    logger.info(f"🚀 Starting Streamlit on port {STREAMLIT_PORT}...")
    
    # Check if Streamlit is already running on the port
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(('localhost', STREAMLIT_PORT))
        s.close()
        if result == 0:
            logger.info(f"⚠️ Port {STREAMLIT_PORT} is already in use. Assuming Streamlit is already running.")
            return False
    except:
        pass
    
    streamlit_command = [
        "streamlit", "run", "app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    env = os.environ.copy()
    env['BROWSER_GATHER_USAGE_STATS'] = "0"
    
    try:
        streamlit_process = subprocess.Popen(
            streamlit_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # Monitor Streamlit output in a separate thread
        def monitor_streamlit():
            for line in streamlit_process.stdout:
                if "You can now view your Streamlit app in your browser" in line:
                    logger.info("✅ Streamlit is ready!")
                elif "Error" in line or "Exception" in line:
                    logger.error(f"❌ Streamlit error: {line.strip()}")
                else:
                    logger.debug(f"Streamlit: {line.strip()}")
        
        threading.Thread(target=monitor_streamlit, daemon=True).start()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit: {str(e)}")
        return False

def run_flask():
    """Run the Flask backend"""
    logger.info(f"🚀 Starting Flask on port {FLASK_PORT}...")
    
    flask_command = [
        sys.executable, "app_flask.py"
    ]
    
    env = os.environ.copy()
    env['STREAMLIT_PORT'] = str(STREAMLIT_PORT)
    env['FLASK_PORT'] = str(FLASK_PORT)
    
    try:
        flask_process = subprocess.Popen(
            flask_command,
            env=env
        )
        
        # Give Flask time to start
        time.sleep(2)
        
        # Check if it's still running
        if flask_process.poll() is not None:
            logger.error(f"❌ Flask failed to start! Exit code: {flask_process.returncode}")
            return False
        
        logger.info("✅ Flask is ready!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start Flask: {str(e)}")
        return False

def print_welcome_message():
    """Print a welcome message in the console"""
    welcome = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   Energy Anomaly Detection System                     ║
    ║                                                       ║
    ║   🌟 System is running!                               ║
    ║                                                       ║
    ║   📊 Streamlit Dashboard: http://localhost:5000       ║
    ║   🔌 Flask API Server:    http://localhost:5001       ║
    ║                                                       ║
    ║   Your browser should open automatically...           ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(welcome)
    logger.info("✅ System startup complete!")

if __name__ == "__main__":
    logger.info("🚀 Starting Energy Anomaly Detection System...")
    
    # Start Streamlit first
    streamlit_success = run_streamlit()
    
    # Then start Flask
    if streamlit_success:
        flask_success = run_flask()
    else:
        # If Streamlit failed or is already running, still try to start Flask
        flask_success = run_flask()
    
    # Open browser if both services started successfully
    if streamlit_success or flask_success:
        open_browser()
        print_welcome_message()
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")
            sys.exit(0)
    else:
        logger.error("❌ Failed to start services!")
        sys.exit(1)