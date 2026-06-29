# Running Valsci with Docker

This guide explains how to run Valsci using Docker containers.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Semantic Scholar API key (get one at https://www.semanticscholar.org/product/api)

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
  - `./semantic_scholar/manifests` (read-only): Curated corpus manifests, mounted read-only into both services. The manifest used is selected by the `SEMANTIC_SCHOLAR_MANIFEST` filename setting (default `mendelian_v1.json`).
  - `./app/config` → `/valsci/runtime_config`: Your `env_vars.json` lives here. The compose file mounts the config **directory** (not the single file) and sets `VALSCI_ENV_FILE=/valsci/runtime_config/env_vars.json` so the in-app **Settings** page can rewrite it. A single-file bind mount cannot be rewritten in place (atomic rename fails with "Device or resource busy"), so the directory mount is required for in-app config editing.

### Editing settings in the app

Open **Settings** in the web UI to edit configuration without touching files by hand. Saved changes to most LLM settings (provider, routing, token budgets, timeouts) are picked up by the background processor **automatically within a few seconds** — the Settings page shows a live "Processor is up to date / Applying…" indicator. A few settings (rate limits, email, storage paths) are read once at startup and are marked **Restart**; they only apply after `docker compose restart processor`.

## Directory Structure in the Container

All application code is mounted in the `/valsci` directory inside the containers:
- `/valsci/app/`: Contains the Flask application code
- `/valsci/semantic_scholar/`: Contains the Semantic Scholar utilities
- `/valsci/queued_jobs/`: Directory for claims waiting to be processed
- `/valsci/saved_jobs/`: Directory for processed claim results and LLM debug traces/issues

## Downloading Semantic Scholar Datasets

For the application to function properly, you need to download Semantic Scholar datasets.

1. **Run the downloader utility**:

   ```bash
   docker-compose exec web python -m semantic_scholar.utils.downloader
   ```

   Options:
   - Build the curated manifest-driven mini corpus: `--mini`
   - Create indices for existing datasets: `--index-only`

   `--mini` uses the tracked Mendelian mini manifest at
   `semantic_scholar/manifests/mendelian_v1.json` unless you pass
   `--mini-manifest`. The manifest records fixed dataset-specific IDs. Valsci
   streams the matching Semantic Scholar dataset shards and writes the compact
   runtime release under ignored local data.

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
- `LLM_PROVIDER`: AI provider to use ("openai", "azure-openai", "azure-inference", "openrouter", "ollama", or "llamacpp")
- `LLM_API_KEY`: API key for the AI provider
- `LLM_EVALUATION_MODEL`: Model to use for evaluation

### Optional Settings

- `REQUIRE_PASSWORD`: Enable password protection
- `ACCESS_PASSWORD`: Password for accessing the application
- `ENABLE_EMAIL_NOTIFICATIONS`: Enable email notifications
- `LLM_ROUTING`: Task-to-model routing, fallback, and **per-task output-token budgets**. Each task accepts `max_output_tokens` (e.g. `LLM_ROUTING.tasks.query_generation.max_output_tokens`). This is the output budget — separate from a model's context window. Reasoning ("thinking") models need a high value (e.g. several thousand) or they spend the whole budget on hidden chain-of-thought and return empty content. Applied even when `LLM_ROUTING.enabled` is `false`.
- `TRACE_ENABLED` / `TRACE_EMBED_MODE`: Control trace persistence and report embedding behavior
- And more...

### Ollama (or any local LLM) from Docker

Inside a container, `localhost` refers to the **container**, not your host — so a
host-run Ollama is **not** reachable at `http://localhost:11434`. Use the special
hostname `host.docker.internal` instead. The Compose files already declare
`extra_hosts: ["host.docker.internal:host-gateway"]` for both services, so this
resolves on macOS, Windows, **and** Linux.

For a host-run Ollama, set in `app/config/env_vars.json`:

```json
"LLM_PROVIDER": "ollama",
"LLM_BASE_URL": "http://host.docker.internal:11434/v1",
"OLLAMA_SHOW_URL": "http://host.docker.internal:11434/api/show",
"LLM_EVALUATION_MODEL": "llama3.1:8b"
```

If discovery fails with "Could not reach an Ollama host", Valsci now appends a
hint when the failure looks like this localhost/`host.docker.internal` mixup.

Heads-up: the same `env_vars.json` is shared between bare-metal and Docker runs,
so these two URLs must use `localhost` when running directly on the host and
`host.docker.internal` when running in Docker — flip them depending on how you
launch. (`LLM_API_KEY` can be any non-empty placeholder; Ollama ignores it.)

Verify the container can reach Ollama:

```bash
docker-compose exec web curl -s http://host.docker.internal:11434/api/tags
```

## Persistent Data

All persistent data is stored in Docker volumes:

- `semantic_scholar_data`: Contains downloaded datasets and indices
- `valsci_state`: Canonical claim store, run records, traces, arenas, and data-job history (`/valsci/state`)
- `queued_jobs`: Contains claims waiting to be processed
- `saved_jobs`: Contains processed claim results

> Without the `valsci_state` volume, recreating the container (e.g. `docker compose down` then `up`) would discard your results and job history, since they live under `/valsci/state`.

To back up this data, you can use Docker's volume backup features.
