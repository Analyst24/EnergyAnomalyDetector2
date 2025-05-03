#!/usr/bin/env python3
"""
Production deployment script for Energy Anomaly Detection System
This script launches the system using gunicorn for Flask and
manages the Streamlit process for production deployment
"""

import os
import sys
import subprocess
import argparse
import logging
import signal
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EADSDeployment')

# Global variables
streamlit_process = None
gunicorn_process = None
flask_port = 5001
streamlit_port = 5000

def start_streamlit():
    """Start the Streamlit process"""
    global streamlit_process
    
    streamlit_command = [
        "streamlit", "run", "app.py",
        "--server.port", str(streamlit_port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    env = os.environ.copy()
    env['BROWSER_GATHER_USAGE_STATS'] = "0"
    
    logger.info(f"Starting Streamlit on port {streamlit_port}")
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
            logger.info(f"Streamlit: {line.strip()}")
    
    import threading
    threading.Thread(target=monitor_streamlit, daemon=True).start()
    
    # Wait for Streamlit to start
    logger.info("Waiting for Streamlit to initialize...")
    time.sleep(5)
    
    # Check if process is still running
    if streamlit_process.poll() is not None:
        logger.error(f"Streamlit failed to start! Exit code: {streamlit_process.returncode}")
        sys.exit(1)
    
    logger.info("Streamlit initialized successfully")

def start_gunicorn():
    """Start the Gunicorn process for Flask"""
    global gunicorn_process
    
    gunicorn_command = [
        "gunicorn",
        "-c", "gunicorn_config.py",
        "--bind", f"0.0.0.0:{flask_port}",
        "app_flask:app"
    ]
    
    env = os.environ.copy()
    env['STREAMLIT_PORT'] = str(streamlit_port)
    
    logger.info(f"Starting Gunicorn/Flask on port {flask_port}")
    gunicorn_process = subprocess.Popen(
        gunicorn_command,
        env=env
    )
    
    # Wait for Gunicorn to start
    time.sleep(2)
    
    # Check if process is still running
    if gunicorn_process.poll() is not None:
        logger.error(f"Gunicorn failed to start! Exit code: {gunicorn_process.returncode}")
        sys.exit(1)
    
    logger.info("Gunicorn/Flask initialized successfully")

def stop_processes():
    """Stop all running processes"""
    global streamlit_process, gunicorn_process
    
    if gunicorn_process is not None:
        logger.info("Stopping Gunicorn/Flask...")
        gunicorn_process.terminate()
        try:
            gunicorn_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Gunicorn did not terminate gracefully, forcing shutdown")
            gunicorn_process.kill()
        gunicorn_process = None
    
    if streamlit_process is not None:
        logger.info("Stopping Streamlit...")
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Streamlit did not terminate gracefully, forcing shutdown")
            streamlit_process.kill()
        streamlit_process = None
    
    logger.info("All processes stopped")

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    stop_processes()
    sys.exit(0)

def main():
    """Main deployment function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Deploy Energy Anomaly Detection System')
    parser.add_argument('--flask-port', type=int, default=5001, 
                       help='Port for Flask server (default: 5001)')
    parser.add_argument('--streamlit-port', type=int, default=5000, 
                       help='Port for Streamlit server (default: 5000)')
    args = parser.parse_args()
    
    global flask_port, streamlit_port
    flask_port = args.flask_port
    streamlit_port = args.streamlit_port
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start Streamlit first
        start_streamlit()
        
        # Then start Gunicorn/Flask
        start_gunicorn()
        
        logger.info(f"Energy Anomaly Detection System is running!")
        logger.info(f"Streamlit UI: http://localhost:{streamlit_port}")
        logger.info(f"Flask API: http://localhost:{flask_port}")
        
        # Keep the script running
        while True:
            # Check if processes are still running
            if streamlit_process.poll() is not None:
                logger.error(f"Streamlit stopped unexpectedly! Exit code: {streamlit_process.returncode}")
                break
            
            if gunicorn_process.poll() is not None:
                logger.error(f"Gunicorn stopped unexpectedly! Exit code: {gunicorn_process.returncode}")
                break
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        stop_processes()

if __name__ == "__main__":
    main()