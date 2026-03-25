# Running Valsci with Docker

This guide explains how to run Valsci using Docker containers.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Semantic Scholar API key (get one at https://www.semanticscholar.org/product/api), unless you are using Mock Demo Mode

## Quick Start

1. **Set up configuration**

   Copy the example environment configuration file:

   ```bash
   cp env_vars.json.example app/config/env_vars.json
   ```

   Edit `app/config/env_vars.json` and add your API keys and settings.

2. **Build and start the containers**

   ```bash
   docker-compose up -d
   ```

   This will start two containers:
   - `valsci-web`: The Flask web server (accessible at http://localhost:3000)
   - `valsci-processor`: The background claim processor service

3. **Access the application**

   Open http://localhost:3000 in your browser.

## Container Structure

- **Web Service**: Handles HTTP requests and serves the web interface
- **Processor Service**: Processes claims in the background
- **Shared Volumes**:
  - `semantic_scholar_data`: Stores downloaded Semantic Scholar datasets
  - `queued_jobs`: Stores claims waiting to be processed
  - `saved_jobs`: Stores processed claim results, plus per-claim `traces/*.jsonl` and `issues/*.jsonl`

## Directory Structure in the Container

All application code is mounted in the `/valsci` directory inside the containers:
- `/valsci/app/`: Contains the Flask application code
- `/valsci/semantic_scholar/`: Contains the Semantic Scholar utilities
- `/valsci/queued_jobs/`: Directory for claims waiting to be processed
- `/valsci/saved_jobs/`: Directory for processed claim results and LLM debug traces/issues

## Mock Demo Mode

For fast local UI testing without Semantic Scholar datasets, enable these settings in `app/config/env_vars.json` before starting the stack:

```json
{
  "MOCK_SEMANTIC_SCHOLAR_MODE": true,
  "MOCK_SEMANTIC_SCHOLAR_FIXTURE_PACK": "happy_path",
  "MOCK_SEMANTIC_SCHOLAR_DELAY_SECONDS": 0.4
}
```

Mock mode drives the full claim, arena, progress, and review flow with deterministic synthetic papers and does not require a Semantic Scholar API key.

## Downloading Semantic Scholar Datasets

For the application to function properly, you need to download Semantic Scholar datasets.

1. **Run the downloader utility**:

   ```bash
   docker-compose exec web python -m semantic_scholar.utils.downloader
   ```

   Options:
   - Download minimal datasets: `--mini`
   - Download specific datasets: `--datasets papers abstracts authors`
   - Download without indexing: `--download-only`
   - Create indices for existing datasets: `--index-only`

2. **Verify downloads**:

   ```bash
   docker-compose exec web python -m semantic_scholar.utils.downloader --verify
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

## Configuration Options

The application is configured through `app/config/env_vars.json`. See the comments in that file for details on available options.

### Required Settings

- `FLASK_SECRET_KEY`: Secret key for Flask session security
- `USER_EMAIL`: Your email address
- `SEMANTIC_SCHOLAR_API_KEY`: Your Semantic Scholar API key
- `LLM_PROVIDER`: AI provider to use ("openai", "azure-openai", "azure-inference", or "local")
- `LLM_API_KEY`: API key for the AI provider
- `LLM_EVALUATION_MODEL`: Model to use for evaluation

### Optional Settings

- `REQUIRE_PASSWORD`: Enable password protection
- `ACCESS_PASSWORD`: Password for accessing the application
- `ENABLE_EMAIL_NOTIFICATIONS`: Enable email notifications
- `LOCAL_BACKEND`: `ollama`, `llamacpp`, `vllm`, or `generic_openai_compat` when `LLM_PROVIDER` is `local`
- `LLM_ROUTING`: Task-to-model routing and fallback configuration
- `TRACE_ENABLED` / `TRACE_EMBED_MODE`: Control trace persistence and report embedding behavior
- And more...

### Ollama from Docker

If Ollama runs on your host machine instead of inside Compose:

- macOS/Windows: set `LLM_BASE_URL` to `http://host.docker.internal:11434/v1`
- Linux: use your Docker bridge gateway IP or run Ollama as another Compose service

## Persistent Data

All persistent data is stored in Docker volumes:

- `semantic_scholar_data`: Contains downloaded datasets and indices
- `queued_jobs`: Contains claims waiting to be processed
- `saved_jobs`: Contains processed claim results

To back up this data, you can use Docker's volume backup features.
