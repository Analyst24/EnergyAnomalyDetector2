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
import signal
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnergyAnomalySystem')

# Global variables
streamlit_process = None
flask_process = None

# Configuration
FLASK_PORT = 5001
STREAMLIT_PORT = 5000

def is_port_in_use(port):
    """Check if a port is in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def cleanup():
    """Clean up processes when the script exits"""
    logger.info("🧹 Cleaning up processes...")
    
    global streamlit_process, flask_process
    
    if flask_process is not None:
        logger.info("🛑 Stopping Flask...")
        try:
            if hasattr(signal, 'SIGTERM'):
                flask_process.send_signal(signal.SIGTERM)
            else:
                flask_process.terminate()
            flask_process.wait(timeout=5)
        except:
            if flask_process.poll() is None:
                flask_process.kill()
    
    if streamlit_process is not None:
        logger.info("🛑 Stopping Streamlit...")
        try:
            if hasattr(signal, 'SIGTERM'):
                streamlit_process.send_signal(signal.SIGTERM)
            else:
                streamlit_process.terminate()
            streamlit_process.wait(timeout=5)
        except:
            if streamlit_process.poll() is None:
                streamlit_process.kill()
    
    logger.info("👋 Goodbye!")

def open_browser():
    """Open the browser to the Streamlit URL after a short delay"""
    time.sleep(3)  # Give servers time to start
    url = f"http://localhost:{STREAMLIT_PORT}"
    webbrowser.open(url)
    logger.info(f"🌟 Opening browser to {url}")

def monitor_process_output(process, name):
    """Monitor process output in a separate thread"""
    if process.stdout is None:
        return
    
    for line in process.stdout:
        line = line.strip() if isinstance(line, str) else line.decode('utf-8', errors='replace').strip()
        if not line:
            continue
            
        if name == "Streamlit" and "You can now view your Streamlit app in your browser" in line:
            logger.info("✅ Streamlit is ready!")
        elif "Error" in line or "Exception" in line:
            logger.error(f"❌ {name} error: {line}")
        elif "WARNING" in line:
            logger.warning(f"⚠️ {name}: {line}")
        else:
            logger.debug(f"{name}: {line}")

def start_streamlit():
    """Start the Streamlit frontend"""
    global streamlit_process
    
    logger.info(f"🚀 Starting Streamlit on port {STREAMLIT_PORT}...")
    
    # Check if Streamlit is already running on the port
    if is_port_in_use(STREAMLIT_PORT):
        logger.info(f"⚠️ Port {STREAMLIT_PORT} is already in use. Killing any existing processes...")
        try:
            # Try to kill existing process on the port
            if sys.platform == 'win32':
                os.system(f'taskkill /F /PID $(netstat -ano | findstr {STREAMLIT_PORT} | awk "{{print $5}}") 2>NUL')
            else:
                os.system(f'kill $(lsof -t -i:{STREAMLIT_PORT}) 2>/dev/null')
            time.sleep(1)
        except Exception as e:
            logger.warning(f"⚠️ Could not kill process on port {STREAMLIT_PORT}: {str(e)}")
    
    # Start Streamlit process
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
            env=env,
            bufsize=1  # Line buffered
        )
        
        # Monitor output in a thread
        threading.Thread(
            target=monitor_process_output, 
            args=(streamlit_process, "Streamlit"), 
            daemon=True
        ).start()
        
        # Give it time to start
        time.sleep(2)
        
        # Check if it's still running
        if streamlit_process.poll() is not None:
            logger.error(f"❌ Streamlit failed to start! Exit code: {streamlit_process.returncode}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit: {str(e)}")
        return False

def start_flask():
    """Start the Flask backend"""
    global flask_process
    
    logger.info(f"🚀 Starting Flask on port {FLASK_PORT}...")
    
    # Check if Flask is already running on the port
    if is_port_in_use(FLASK_PORT):
        logger.info(f"⚠️ Port {FLASK_PORT} is already in use. Killing any existing processes...")
        try:
            # Try to kill existing process on the port
            if sys.platform == 'win32':
                os.system(f'taskkill /F /PID $(netstat -ano | findstr {FLASK_PORT} | awk "{{print $5}}") 2>NUL')
            else:
                os.system(f'kill $(lsof -t -i:{FLASK_PORT}) 2>/dev/null')
            time.sleep(1)
        except Exception as e:
            logger.warning(f"⚠️ Could not kill process on port {FLASK_PORT}: {str(e)}")
    
    # Start Flask
    try:
        env = os.environ.copy()
        env['FLASK_APP'] = 'app_flask.py'
        env['STREAMLIT_PORT'] = str(STREAMLIT_PORT)
        env['FLASK_PORT'] = str(FLASK_PORT)
        
        flask_command = [
            sys.executable, "-m", "flask", "run",
            "--host=0.0.0.0",
            f"--port={FLASK_PORT}",
            "--no-debugger"
        ]
        
        flask_process = subprocess.Popen(
            flask_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
            bufsize=1  # Line buffered
        )
        
        # Monitor output in a thread
        threading.Thread(
            target=monitor_process_output, 
            args=(flask_process, "Flask"), 
            daemon=True
        ).start()
        
        # Give it time to start
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
    # Register cleanup handler
    atexit.register(cleanup)
    
    # Register signal handlers
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    
    logger.info("🚀 Starting Energy Anomaly Detection System...")
    
    # Start Streamlit first
    streamlit_started = start_streamlit()
    
    # Then start Flask
    flask_started = start_flask()
    
    # Open browser if both started successfully
    if streamlit_started and flask_started:
        open_browser()
        print_welcome_message()
        
        # Keep the script running
        try:
            while True:
                # Check for health status periodically
                if (streamlit_process and streamlit_process.poll() is not None):
                    logger.error(f"❌ Streamlit exited with code {streamlit_process.returncode}")
                    break
                
                if (flask_process and flask_process.poll() is not None):
                    logger.error(f"❌ Flask exited with code {flask_process.returncode}")
                    break
                
                time.sleep(2)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")
        except Exception as e:
            logger.error(f"❌ Error: {str(e)}")
        finally:
            sys.exit(0)
    else:
        logger.error("❌ Failed to start one or more services!")
        sys.exit(1)