"""
Gunicorn configuration for Flask deployment
"""

import multiprocessing
import os

# Server socket configuration
bind = "0.0.0.0:5001"  # The default Flask port
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"  # Use 'gevent' or 'eventlet' for async support
worker_connections = 1000
timeout = 60
keepalive = 5

# Log configuration
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Post-fork application initialization
def post_fork(server, worker):
    """
    Called after a worker has been forked
    """
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def on_starting(server):
    """
    Called just before the master process is initialized
    """
    server.log.info("Starting Energy Anomaly Detection System")

def on_exit(server):
    """
    Called just before the server exits
    """
    server.log.info("Shutting down Energy Anomaly Detection System")