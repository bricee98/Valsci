# Valsci

**A self-hosted, open-source automated literature review tool.**

## Overview

Valsci is designed to validate scientific claims by leveraging a combination of literature search, paper analysis, and evidence scoring. It automates the process of evaluating claims against existing scientific literature, providing users with detailed reports and insights.

## Features

- **Claim Validation**: Automatically validate scientific claims using a robust pipeline that includes literature search, paper analysis, and evidence scoring.
- **Batch Processing**: Submit multiple claims at once via file upload for batch processing.
- **LLM Evaluation**: Integrate with SOTA LLMs - self-hosted or from providers - to create reasoned, thought-out reports on the literature support for submitted claims.
- **Web Interface**: User-friendly web interface for submitting claims, checking status, and browsing results.
- **API Access**: RESTful API for programmatic access to claim validation and batch processing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PM2 (for production deployment)
- A Semantic Scholar API Key (with S2ORC access)
- Disk space for Semantic Scholar datasets:
  - Papers dataset: ~200GB
  - Abstracts dataset: ~140GB
  - Authors dataset: ~25GB
  - S2ORC dataset: ~1.1TB
  - TLDRs dataset: ~20GB
  - Indices: ~150GB

### Required Python Packages

The required packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

### Configuration

1. **Create configuration file:**

Create a `config/env_vars.json` file with the following structure:

```json
{
    "FLASK_SECRET_KEY": "your_secret_key",
    "USER_EMAIL": "your_email@example.com",
    "SEMANTIC_SCHOLAR_API_KEY": "your_semantic_scholar_api_key",
    "AI_PROVIDER": "openai",  // Can be "openai", "azure", or "local"
    "AI_API_KEY": "your_api_key",  // Required for OpenAI and Azure OpenAI
    "AI_BASE_URL": "http://localhost:8000",  // Required for local AI provider
    "REQUIRE_PASSWORD": "true",  // Optional password protection for internet hosting
    "ACCESS_PASSWORD": "your_access_password",  // Required if REQUIRE_PASSWORD is true
    
    // Optional Azure OpenAI configuration
    "USE_AZURE_OPENAI": "false",
    "AZURE_OPENAI_ENDPOINT": "your_azure_endpoint",
    "AZURE_OPENAI_API_VERSION": "2024-06-01",
    
    // Optional email notification configuration
    "ENABLE_EMAIL_NOTIFICATIONS": "false",
    "EMAIL_SENDER": "your_gmail@gmail.com",
    "EMAIL_APP_PASSWORD": "your_gmail_app_password",
    "SMTP_SERVER": "smtp.gmail.com",
    "SMTP_PORT": "587",
    "BASE_URL": "https://your-domain.com"
}
```

### Downloading and Indexing Semantic Scholar Datasets

Valsci requires local copies of Semantic Scholar datasets for efficient paper lookup and analysis. The datasets are downloaded and indexed using the provided downloader utility.

1. **Basic Usage:**
```bash
python -m semantic_scholar.utils.downloader
```

2. **Download Options:**
```bash
# Download minimal datasets for testing
python -m semantic_scholar.utils.downloader --mini

# Download specific datasets
python -m semantic_scholar.utils.downloader --datasets papers abstracts authors

# Download without indexing
python -m semantic_scholar.utils.downloader --download-only

# Only create indices for downloaded datasets
python -m semantic_scholar.utils.downloader --index-only
```

3. **Verification and Maintenance:**
```bash
# Verify downloads are complete
python -m semantic_scholar.utils.downloader --verify

# Verify index integrity
python -m semantic_scholar.utils.downloader --verify-index

# Show detailed index statistics
python -m semantic_scholar.utils.downloader --count

# Audit datasets and indices
python -m semantic_scholar.utils.downloader --audit
```

The datasets will be downloaded to `semantic_scholar/datasets/` and indexed for fast lookup. The indexing process creates binary indices in `semantic_scholar/datasets/binary_indices/`.

**Note:** S2ORC access requires special API permissions. Visit https://api.semanticscholar.org/s2orc to request access.

### Running the Application

Valsci consists of two main services:

1. **Web Server**: Handles HTTP requests and serves the web interface
2. **Claim Processor**: Processes claims in the background


### AI Integrations

Valsci can be used with any OpenAI-compatible text completion LLM API provider. It's set up for out-of-the-box usage with a locally-hosted llama.cpp server, the OpenAI API, or the Azure-hosted OpenAI API.

To set up a locally-hosted inference server, please see the [llama.cpp repository](https://github.com/ggerganov/llama.cpp) and specifically the [server examples](https://github.com/ggerganov/llama.cpp/tree/master/examples/server). For OpenAI and Azure OpenAI, you must set up appropriate API credentials in your configuration file.

#### Development Mode

Run the web server:
```bash
python run.py
```

Run the processor:
```bash
python processor.py
```

#### Production Mode

Use the provided shell scripts with PM2:

```bash
# Start the web server
./run_server.sh

# Start the claim processor
./run_processor.sh

# Monitor services
pm2 status
pm2 logs
```

The application will be available at `http://localhost:3000`.

## API Routes

### Claims

- `POST /api/v1/claims`: Submit a single claim
- `GET /api/v1/claims/<batch_id>/<claim_id>`: Get claim status
- `GET /api/v1/claims/<batch_id>/<claim_id>/report`: Get claim report
- `GET /api/v1/claims/<claim_id>/download_citations`: Download citations in RIS format

### Batch Processing

- `POST /api/v1/batch`: Submit multiple claims via file upload
- `GET /api/v1/batch/<batch_id>`: Get batch status
- `GET /api/v1/batch/<batch_id>/progress`: Get batch progress
- `GET /api/v1/batch/<batch_id>/download`: Download batch results
- `DELETE /api/v1/batch/<batch_id>`: Delete a batch

### Management

- `GET /api/v1/browse`: Browse saved batches and claims
- `DELETE /api/v1/delete/claim/<claim_id>`: Delete a specific claim

### Web Interface Routes

- `/`: Home page
- `/results`: View claim results
- `/progress`: View processing progress
- `/browser`: Browse saved claims and batches
- `/batch_results`: View batch results

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [bedelman3@gatech.edu](mailto:bedelman3@gatech.edu).

## Version
The current version is 0.1.5. This version will increment when any changes are made to the report generation logic that could affect results.