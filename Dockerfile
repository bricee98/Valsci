FROM python:3.11-slim

# Set working directory
WORKDIR /valsci

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY semantic_scholar/ semantic_scholar/
COPY run.py processor.py ./

# Create directories for jobs
RUN mkdir -p queued_jobs saved_jobs

# Set up config directory and create placeholder for env_vars.json
RUN mkdir -p app/config
COPY app/config/settings.py app/config/

# Create a placeholder env_vars.json that will be overridden at runtime
RUN echo '{}' > app/config/env_vars.json

# Expose the port the app runs on
EXPOSE 3000

# Command to run the application
CMD ["python", "run.py"]