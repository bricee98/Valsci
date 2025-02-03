#!/bin/bash

# Name of the PM2 process
APP_NAME="ValsciProcessor"

# Path to the Python script
SCRIPT_PATH="processor.py"

# Activate the virtual environment
source venv/bin/activate

# Start the processor with PM2
pm2 start $SCRIPT_PATH --name $APP_NAME --interpreter python3

# Display status
pm2 status $APP_NAME

echo "Valsci Processor started. View logs with: pm2 logs $APP_NAME" 