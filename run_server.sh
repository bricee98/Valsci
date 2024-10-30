#!/bin/bash

# Name of the PM2 process
APP_NAME="ValsciServer"

# Path to the Python script
SCRIPT_PATH="run.py"

# Path to the virtual environment
VENV_PATH="path/to/your/venv"

# Activate the virtual environment
source venv/bin/activate

# Start the application with PM2
pm2 start $SCRIPT_PATH --name $APP_NAME --interpreter python3