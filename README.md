# Valsci

**Validate scientific claims en masse.**

## Overview

Valsci is a powerful tool designed to validate scientific claims by leveraging a combination of literature search, paper analysis, and evidence scoring. It automates the process of evaluating claims against existing scientific literature, providing users with detailed reports and insights.

## Features

- **Claim Validation**: Automatically validate scientific claims using a robust pipeline that includes literature search, paper analysis, and evidence scoring.
- **Batch Processing**: Submit multiple claims at once via file upload for batch processing.
- **Detailed Reports**: Generate comprehensive reports for each claim, including supporting papers, explanations, and claim ratings.
- **Web Interface**: User-friendly web interface for submitting claims, checking status, and browsing results.
- **API Access**: RESTful API for programmatic access to claim validation and batch processing.

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Required Python Packages

The following packages are required and will be installed via `requirements.txt`:

- Flask
- python-dotenv
- requests
- pandas
- beautifulsoup4
- pymupdf4llm
- PyPDF2
- urllib3
- fake-useragent
- openai
- brotli
- chardet
- pymupdf
- bs4
- python-dateutil
- typing-extensions

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/valsci.git
   cd valsci
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `config/env_vars.json` file in the `config` directory and add the necessary environment variables. Here is an example of what the file should look like:

   ```json
   {
       "FLASK_SECRET_KEY": "your_secret_key",
       "OPENAI_API_KEY": "your_openai_api_key",
       "ANTHROPIC_API_KEY": "your_anthropic_api_key",
       "USER_EMAIL": "your_email@example.com",
       "AZURE_OPENAI_API_KEY": "your_azure_openai_api_key",
       "AZURE_OPENAI_ENDPOINT": "your_azure_openai_endpoint",
       "AZURE_OPENAI_API_VERSION": "2024-06-01",
       "USE_AZURE_OPENAI": "false",
       "REQUIRE_PASSWORD": "true",
       "ACCESS_PASSWORD": "your_access_password"
   }
   ```

   **Note:**
   - `FLASK_SECRET_KEY` and `USER_EMAIL` are required.
   - If `USE_AZURE_OPENAI` is set to `"true"`, you must provide `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`.
   - If `REQUIRE_PASSWORD` is set to `"true"`, you must provide `ACCESS_PASSWORD`.

5. **Run the application:**
   ```bash
   python run.py
   ```

   The application will be available at `http://localhost:3000`.

### Usage

- **Submit a Claim**: Use the web interface to enter a claim and submit it for validation.
- **Upload a File**: Upload a text file containing multiple claims for batch processing.
- **Check Status**: Enter a claim or batch reference ID to check the processing status.
- **Browse Results**: View processed claims and batch results through the browser interface.

## API Documentation

The API provides endpoints for submitting claims, checking status, and retrieving reports. Below are some key endpoints:

- `POST /api/v1/claims`: Submit a single claim for validation.
- `GET /api/v1/claims/<claim_id>`: Retrieve the status of a specific claim.
- `POST /api/v1/batch`: Start a batch job by uploading a file of claims.
- `GET /api/v1/batch/<batch_id>`: Get the status and results of a batch job.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [bricee98@gmail.com](mailto:bricee98@gmail.com).
