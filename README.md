# Valsci

**A self-hostable, open-source automated literature review tool.**

## Overview

Valsci is an open-source, self-hostable utility designed to automate large-batch scientific claim verification using any OpenAI-compatible large language model (LLM). It combines retrieval-augmented generation (RAG) with structured bibliometric scoring to efficiently search, evaluate, and summarize evidence from the Semantic Scholar database and other academic sources.

Unlike conventional standalone LLMs, which often suffer from hallucinations and unreliable citations, Valsci grounds its analyses in verifiable published findings through:

1. **Retrieval-Augmented Generation (RAG)**: Seamless integration with Semantic Scholar ensures outputs remain anchored in verifiable sources.
2. **Structured Bibliometric Scoring**: Incorporates author H-index, citation counts, and journal impact for nuanced source credibility assessment.
3. **Guided Chain-of-Thought (CoT)**: Uses specialized prompts to systematically organize retrieved evidence into comprehensive verification reports.
4. **High-Throughput Processing**: Employs asynchronous parallelization to process hundreds of claims per hour.

Valsci has demonstrated significant improvements over base LLM outputs in true/false annotation accuracy and citation hallucination rates across benchmark datasets.

## Features

- **Claim Validation**: Automatically validate scientific claims using a robust pipeline that includes literature search, paper analysis, and evidence scoring.
- **Batch Processing**: Process hundreds of claims per hour through asynchronous parallelization.
- **LLM Integration**: Compatible with any OpenAI-style API, including self-hosted open-source models like LLaMA, Deepseek-R1, and Mistral.
- **Bibliometric Scoring**: Evaluate source credibility using author H-index, citation counts, and estimated journal impact.
- **Structured Reports**: Generate detailed reports with supporting evidence, contradictory findings, and mechanistic evaluations.
- **Verifiable Results**: All citations and excerpts are drawn directly from the Semantic Scholar database, eliminating hallucinations.
- **Web Interface**: User-friendly interface for submitting claims, monitoring progress, and browsing results.

### Rating Scale
Claims are rated on an ordinal scale:
- Contradicted
- Likely False
- Mixed Evidence
- Likely True
- Highly Supported
- No Evidence

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

Valsci's configuration is managed through a JSON file that controls LLM integration, security settings, and optional features. The system supports various LLM backends including OpenAI's API, Azure OpenAI, Azure AI Inference, and local deployments of open-source models.

1. **Create configuration file:**

Create an `app/config/env_vars.json` file with your configuration settings. Below is a template with explanations for each setting:

```json
{
    "FLASK_SECRET_KEY": "your_secret_key",
    "USER_EMAIL": "your_email@example.com",
    "SEMANTIC_SCHOLAR_API_KEY": "your_semantic_scholar_api_key",
    "LLM_PROVIDER": "openai",
    "LLM_API_KEY": "your_api_key",
    "LLM_BASE_URL": "http://localhost:8000",
    "LLM_EVALUATION_MODEL": "gpt-4o",
    "REQUIRE_PASSWORD": "true",
    "ACCESS_PASSWORD": "your_access_password",
    "AZURE_OPENAI_ENDPOINT": "your_azure_endpoint",
    "AZURE_OPENAI_API_VERSION": "2024-06-01",
    "AZURE_AI_INFERENCE_ENDPOINT": "your_azure_ai_inference_endpoint",
    "ENABLE_EMAIL_NOTIFICATIONS": "false",
    "EMAIL_SENDER": "your_gmail@gmail.com",
    "EMAIL_APP_PASSWORD": "your_gmail_app_password",
    "SMTP_SERVER": "smtp.gmail.com",
    "SMTP_PORT": "587",
    "BASE_URL": "https://your-domain.com"
}
```

#### Configuration Options

**Required Settings:**
- `FLASK_SECRET_KEY`: Secret key for Flask session security
- `USER_EMAIL`: Your email address
- `SEMANTIC_SCHOLAR_API_KEY`: Your Semantic Scholar API key
- `LLM_PROVIDER`: AI provider to use ("openai", "azure-openai", "azure-inference", or "local")
- `LLM_API_KEY`: API key for OpenAI, Azure OpenAI, or Azure AI Inference
- `LLM_EVALUATION_MODEL`: Model to use for evaluation (e.g., "gpt-4o" for OpenAI, "Phi-4" for Azure AI Inference)

**Optional Settings:**
- `LLM_BASE_URL`: Base URL for local AI provider (required if using "local" provider)
- `REQUIRE_PASSWORD`: Enable password protection for internet hosting
- `ACCESS_PASSWORD`: Access password (required if REQUIRE_PASSWORD is "true")

**Azure OpenAI Settings (Required for azure-openai provider):**
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version

**Azure AI Inference Settings (Required for azure-inference provider):**
- `AZURE_AI_INFERENCE_ENDPOINT`: Azure AI Inference endpoint URL

**Email Notification Settings (Optional):**
- `ENABLE_EMAIL_NOTIFICATIONS`: Enable email notifications
- `EMAIL_SENDER`: Gmail address for sending notifications
- `EMAIL_APP_PASSWORD`: Gmail app password
- `SMTP_SERVER`: SMTP server (default: smtp.gmail.com)
- `SMTP_PORT`: SMTP port (default: 587)
- `BASE_URL`: Base URL for your deployment

### Downloading and Indexing Semantic Scholar Datasets

Valsci requires local copies of Semantic Scholar datasets for efficient paper lookup and analysis. The datasets are downloaded and indexed using the provided downloader utility.

1. **Basic Usage:**
```bash
python -m semantic_scholar.utils.downloader
```

The full downloader includes `s2orc_v2` as the local full-text dataset. The mini-corpus manifest mirrors that production content path by listing dataset-specific IDs for `s2orc_v2`, `papers`, `abstracts`, `tldrs`, and `authors`; the downloader then streams the matching Semantic Scholar release rows.

2. **Download Options:**
```bash
# Build the curated mini corpus for local production-pipeline testing.
python -m semantic_scholar.utils.downloader --mini

# Download only selected full datasets
python -m semantic_scholar.utils.downloader --datasets papers abstracts authors

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

The web UI also exposes this workflow at `/data`. Use the Data page to inspect local releases, see per-dataset file and index coverage, build the curated mini corpus, download selected full datasets, rebuild indices, verify files, and watch downloader logs while a job runs.

#### Curated Mini Corpus

The `--mini` flag now means "build the curated local mini corpus," not "download the first dataset shard." It is designed for a focused desktop rehearsal of the real production pipeline with live Semantic Scholar search, local dataset lookup, binary indices, local Ollama models, paper analysis, scoring, and final reports.

The default mini manifest path is:

```text
semantic_scholar/manifests/mendelian_v1.json
```

The manifest contains only dataset-specific IDs. Running `--mini` uses the Semantic Scholar dataset API to stream the matching release shards, writes compact extracted rows into `semantic_scholar/datasets/`, and builds the normal binary indices. `semantic_scholar/datasets/` remains ignored local runtime data.

The manifest describes a manually curated Mendelian-disease subset:

```json
{
  "release_id": "2026-05-26",
  "mini_release_id": "2026-05-26-mini-mendelian-v1",
  "datasets": {
    "papers": {"corpus_ids": []},
    "abstracts": {"corpus_ids": []},
    "tldrs": {"corpus_ids": []},
    "s2orc_v2": {"corpus_ids": []},
    "authors": {"author_ids": []}
  }
}
```

Running `--mini` fetches only the listed IDs into a compact release such as `2026-05-26-mini-mendelian-v1`, then builds the normal binary indices. A fresh checkout can therefore build the Mendelian mini corpus without first recreating the curation workflow.

The helper workflow for the Mendelian corpus is:

```bash
# Capture/query provenance and candidate papers.
python -m semantic_scholar.utils.mini_corpus_curator

# Optionally expand selected seed papers through their references.
python -m semantic_scholar.utils.mini_corpus_curator --expand-references --resume

# Scan dataset shards, write ignored curation/cache artifacts, and update the minimal manifest.
python -m semantic_scholar.utils.mini_manifest_builder

# If the ignored curation cache already exists, rebuild the manifest without rescanning remote shards.
python -m semantic_scholar.utils.mini_manifest_builder --reuse-source-extracts
```

The Data page can run the same manifest-backed downloader from the browser; its mini action executes `python -m semantic_scholar.utils.downloader --mini --mini-manifest ...`, streams Semantic Scholar dataset shards for the requested IDs, and shows the job logs.

**Note:** S2ORC v2 access may require Semantic Scholar dataset API permissions. Check the Semantic Scholar Datasets API release metadata for the `s2orc_v2` dataset.

**Test claims:** See [`TEST_CLAIMS.md`](TEST_CLAIMS.md) for 12 ready-to-paste claims that are covered by this mini corpus, plus the `scripts/` helpers for probing coverage and re-expanding the corpus.

### Running the Application

Valsci consists of two main services:

1. **Web Server**: Handles HTTP requests and serves the web interface
2. **Claim Processor**: Processes claims in the background


### AI Integrations

Valsci can be used with any OpenAI-compatible text completion LLM API provider. It supports:

1. **OpenAI API**: Use "openai" as the provider and provide your OpenAI API key
2. **Azure OpenAI Service**: Use "azure-openai" as the provider and provide your Azure OpenAI endpoint and API key
3. **Azure AI Inference SDK**: Use "azure-inference" as the provider for models like Phi-4
4. **Local Deployment**: Use "local" as the provider with a locally-hosted server like llama.cpp

#### Azure AI Inference SDK Integration

To use models like Phi-4 with Azure AI Inference SDK:

1. Set `LLM_PROVIDER` to "azure-inference"
2. Set `LLM_API_KEY` to your Azure AI API key
3. Set `AZURE_AI_INFERENCE_ENDPOINT` to your Azure AI endpoint (e.g., "https://Phi-4-dxdep.eastus.models.ai.azure.com")
4. Set `LLM_EVALUATION_MODEL` to the model name (e.g., "Phi-4")

Example configuration for Phi-4:
```json
{
    "LLM_PROVIDER": "azure-inference",
    "LLM_API_KEY": "your_api_key",
    "AZURE_AI_INFERENCE_ENDPOINT": "https://your-model-endpoint.models.ai.azure.com",
    "LLM_EVALUATION_MODEL": "Phi-4"
}
```

#### Local Model Setup

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

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use Valsci in your research, please cite:

```bibtex
@article{edelman2024valsci,
  title={Valsci: An Open-Source, Self-Hostable Literature Review Utility for Automated Large Batch Scientific Claim Verification Using Large Language Models},
  author={Edelman, Brice and Skolnick, Jeffrey},
  year={2024}
}
```

## Contact

For questions or support, please contact [bedelman3@gatech.edu](mailto:bedelman3@gatech.edu).
