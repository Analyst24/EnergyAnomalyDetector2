import subprocess
import os
import time
import signal
import sys
import logging
from threading import Thread
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Get port configurations from environment or use defaults
streamlit_port = int(os.environ.get('STREAMLIT_PORT', 5000))
flask_port = int(os.environ.get('FLASK_PORT', 5001))

# Global variable to hold the Streamlit process
streamlit_process = None

def start_streamlit():
    """
    Start the Streamlit application as a subprocess
    """
    global streamlit_process
    
    if streamlit_process is not None:
        logger.info("Streamlit is already running")
        return

    streamlit_command = [
        "streamlit", "run", "app.py",
        "--server.port", str(streamlit_port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    # Start Streamlit as a subprocess
    logger.info(f"Starting Streamlit with command: {' '.join(streamlit_command)}")
    streamlit_process = subprocess.Popen(
        streamlit_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=dict(os.environ, BROWSER_GATHER_USAGE_STATS="0")
    )
    
    # Monitor Streamlit output
    def monitor_streamlit():
        for line in streamlit_process.stdout:
            logger.info(f"Streamlit: {line.strip()}")
    
    Thread(target=monitor_streamlit, daemon=True).start()
    
    # Wait for Streamlit to start
    logger.info("Waiting for Streamlit to start...")
    time.sleep(5)
    logger.info("Streamlit should be running now")

def stop_streamlit():
    """
    Stop the Streamlit subprocess
    """
    global streamlit_process
    
    if streamlit_process is None:
        logger.info("Streamlit is not running")
        return
    
    logger.info("Stopping Streamlit...")
    
    # Try graceful termination first
    if hasattr(signal, 'SIGTERM'):
        streamlit_process.send_signal(signal.SIGTERM)
        try:
            streamlit_process.wait(timeout=5)
            logger.info("Streamlit terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Streamlit did not terminate gracefully, forcing termination")
            streamlit_process.kill()
    else:
        # Windows doesn't have SIGTERM
        streamlit_process.terminate()
    
    streamlit_process = None
    logger.info("Streamlit stopped")

# Define Flask routes
@app.route('/')
def index():
    """
    Main entry point - redirect to Streamlit application
    """
    return redirect(f"http://localhost:{streamlit_port}")

@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "ok", 
        "streamlit_running": streamlit_process is not None
    })

@app.route('/restart', methods=['POST'])
def restart_streamlit():
    """
    Endpoint to restart the Streamlit application
    """
    stop_streamlit()
    start_streamlit()
    return jsonify({"status": "restarted"})

# API endpoint example - you can add more as needed
@app.route('/api/status', methods=['GET'])
def api_status():
    """
    Example API endpoint
    """
    return jsonify({
        "system": "Energy Anomaly Detection System",
        "status": "operational",
        "streamlit_frontend": f"http://localhost:{streamlit_port}"
    })

# Shutdown handler
def shutdown_handler(signal, frame):
    logger.info("Received shutdown signal")
    stop_streamlit()
    sys.exit(0)

# Register shutdown handlers
if hasattr(signal, 'SIGINT'):
    signal.signal(signal.SIGINT, shutdown_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == '__main__':
    # Start Streamlit first
    start_streamlit()
    
    logger.info(f"Starting Flask on port {flask_port}")
    
    try:
        # Start Flask with debugging disabled for production
        app.run(host='0.0.0.0', port=flask_port, debug=False)
    finally:
        # Ensure Streamlit is stopped when Flask exits
        stop_streamlit()