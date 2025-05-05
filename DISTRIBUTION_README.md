# Valsci Docker Distribution

This package contains a pre-built Docker image for running Valsci without needing to build from source.

## Contents
- `valsci-image.tar.gz` - The pre-built Docker image
- `docker-compose.yml` - Configuration for running the application
- `env_vars.json.example` - Template for configuration
- `DOCKER_README.md` - Detailed documentation about the Docker setup

## Prerequisites
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- At least 8GB of RAM and 20GB of free disk space
- A Semantic Scholar API key (get one at https://www.semanticscholar.org/product/api)

## Quick Setup Guide

0. **Make sure you have extracted the distribution tarball to a new directory**
```bash
   tar -xzvf valsci-docker-dist.tar.gz
```

1. **Load the Docker image**:
   ```bash
   gunzip -c valsci-image.tar.gz | docker load
   ```
   This might take a few minutes depending on your system.

2. **Create your configuration**:
   ```bash
   # Create the config directory
   mkdir -p app/config
   
   # Copy the example config
   cp env_vars.json.example app/config/env_vars.json
   
   # Edit the config file with your API keys and settings
   nano app/config/env_vars.json  # or use any text editor
   ```

3. **Configure required settings**:
   At minimum, you need to set:
   - `FLASK_SECRET_KEY` - Any random string for security
   - `USER_EMAIL` - Your email address
   - `SEMANTIC_SCHOLAR_API_KEY` - Your Semantic Scholar API key
   - `LLM_PROVIDER` - Usually "openai"
   - `LLM_API_KEY` - Your OpenAI API key

4. **Start the containers**:
   ```bash
   docker-compose up -d
   ```

5. **Check that everything is running**:
   ```bash
   docker-compose ps
   ```
   You should see both services running.

6. **Access the application**:
   Open http://localhost:3000 in your browser

7. **Download required datasets**:
   ```bash
   # For minimal test datasets (~2GB)
   docker-compose exec web python -m semantic_scholar.utils.downloader --mini
   
   # Or for full datasets (requires ~1.6TB of disk space)
   # docker-compose exec web python -m semantic_scholar.utils.downloader
   ```

## Management Commands

- **View logs**:
  ```bash
  docker-compose logs -f
  ```

- **Restart services**:
  ```bash
  docker-compose restart
  ```

- **Stop services**:
  ```bash
  docker-compose down
  ```

## Troubleshooting

- **Can't access the web interface**: Check if the container is running with `docker-compose ps`
- **Processor not working**: Check logs with `docker-compose logs processor`
- **Dataset download errors**: Check if you have enough disk space

## For Advanced Users

If you want to modify the code or build from source, please refer to the main repository. This package contains only the pre-built image for ease of use. 