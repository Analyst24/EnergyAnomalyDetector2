# Energy Anomaly Detection System

An advanced offline energy consumption anomaly detection system that leverages machine learning algorithms to analyze and visualize energy usage patterns without requiring internet connectivity.

## Overview

This system detects unusual patterns in energy consumption data, helping businesses optimize energy efficiency and reduce operational costs. The application provides advanced analytics, visualization capabilities, and actionable insights through an interactive interface.

## Features

- **Anomaly Detection**: Multiple machine learning models including Isolation Forest, Autoencoder, and K-Means
- **Interactive Dashboard**: Dynamic visualizations for data exploration and analysis
- **Offline Operation**: 100% offline compatibility with no internet requirement
- **Multiple Model Comparison**: Compare the performance of different anomaly detection algorithms
- **Actionable Recommendations**: Get specific recommendations based on detected anomalies
- **Visual Analytics**: Advanced visualizations including time-series analysis, distribution plots, and more
- **Flexible Deployment**: Choose between Streamlit-only mode or Flask+Streamlit integration

## Running the System

The recommended way to run the Energy Anomaly Detection System is through the main entry point:

```bash
python main.py
```

This single command will:
- Start the Streamlit dashboard on port 5000
- Start the Flask API server on port 5001
- Open your browser automatically to the dashboard
- Monitor both services and ensure they're running correctly

Simply click the "Run" button in VS Code or execute `python main.py` in your terminal to start the entire system with one command.

### Alternative Deployment Options

While using `main.py` is recommended, these alternative methods are available for specific scenarios:

#### Streamlit Only

If you only need the Streamlit dashboard without the Flask API:

```bash
streamlit run app.py
```

#### Development Flask Server

For API development without the integrated approach:

```bash
python app_flask.py
```

#### Production Deployment

For production environments, use the deployment script with Gunicorn:

```bash
python deploy.py
```

## API Endpoints (Flask Mode Only)

When running in Flask mode, the following API endpoints are available:

- `/health` - System health status
- `/restart` - Restart the Streamlit application (POST)
- `/api/status` - System status information

## Configuration

Both deployment options support configuration through command-line arguments:

- `--flask-port`: Port for the Flask server (default: 5001)
- `--streamlit-port`: Port for the Streamlit server (default: 5000)
- `--streamlit-only`: Run only Streamlit without Flask
- `--flask-only`: Run only Flask (assumes Streamlit is already running)

## Architecture

The system architecture consists of two main components:

1. **Streamlit Frontend**: Handles the user interface, visualizations, and interactive elements
2. **Flask Backend** (optional): Manages deployment, API endpoints, and backend processing

In the integrated mode, Flask serves as the main entry point and manages the Streamlit process, providing a unified deployment model while keeping the user experience unchanged.

## Dependencies

See `dependencies.txt` for a complete list of required packages.

## License

This project is licensed under the MIT License - see the LICENSE file for details.