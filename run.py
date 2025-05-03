#!/usr/bin/env python3
"""
Main entry point for the Energy Anomaly Detection System
This script starts both Flask and Streamlit for production deployment
"""

import os
import argparse
import sys
import subprocess
import logging
import time
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnergyAnomalySystem')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Energy Anomaly Detection System')
    parser.add_argument('--flask-port', type=int, default=5001, 
                       help='Port for Flask server (default: 5001)')
    parser.add_argument('--streamlit-port', type=int, default=5000, 
                       help='Port for Streamlit server (default: 5000)')
    parser.add_argument('--streamlit-only', action='store_true',
                       help='Run only the Streamlit application without Flask')
    parser.add_argument('--flask-only', action='store_true',
                       help='Run only the Flask backend (assumes Streamlit is already running)')
    args = parser.parse_args()
    
    # Update environment variables
    os.environ['STREAMLIT_PORT'] = str(args.streamlit_port)
    os.environ['FLASK_PORT'] = str(args.flask_port)
    
    # Run the components based on arguments
    processes = []
    
    try:
        if args.streamlit_only:
            # Run only Streamlit
            logger.info("Starting Streamlit only...")
            run_streamlit(args.streamlit_port)
        elif args.flask_only:
            # Run only Flask
            logger.info("Starting Flask only...")
            run_flask(args.flask_port)
        else:
            # Run both (default)
            logger.info("Starting both Flask and Streamlit...")
            flask_process = Thread(target=run_flask, args=(args.flask_port,))
            flask_process.daemon = True
            flask_process.start()
            
            # Give Flask a moment to start
            time.sleep(2)
            
            # Run Streamlit in the main thread
            run_streamlit(args.streamlit_port)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Shutdown complete")

def run_flask(port):
    """Run the Flask backend"""
    logger.info(f"Starting Flask on port {port}")
    flask_command = [
        sys.executable, "app_flask.py"
    ]
    
    env = os.environ.copy()
    env['FLASK_PORT'] = str(port)
    
    flask_process = subprocess.Popen(
        flask_command,
        env=env
    )
    
    try:
        flask_process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping Flask...")
        flask_process.terminate()
        flask_process.wait()

def run_streamlit(port):
    """Run the Streamlit frontend"""
    logger.info(f"Starting Streamlit on port {port}")
    streamlit_command = [
        "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    env = os.environ.copy()
    env['BROWSER_GATHER_USAGE_STATS'] = "0"
    
    streamlit_process = subprocess.Popen(
        streamlit_command,
        env=env
    )
    
    try:
        streamlit_process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping Streamlit...")
        streamlit_process.terminate()
        streamlit_process.wait()

if __name__ == "__main__":
    main()