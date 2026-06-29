import os
import io
import json
import requests
import urllib3
from http.client import IncompleteRead
from typing import Any, List, Dict, Optional, Tuple, Set
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import math
from datetime import datetime
from tqdm import tqdm
import gzip
import shutil
import hashlib
from pathlib import Path
import sys
import time
from urllib.parse import urlparse, unquote
import re
import multiprocessing
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# Now we can import using the full package path
from semantic_scholar.utils.binary_indexer import BinaryIndexer, IndexEntry, remove_scratch_path
from app.config.settings import Config

BASE_URL = "https://api.semanticscholar.org/datasets/v1"
console = Console()

# Curated corpus manifests are tracked, immutable inputs and live ONLY here.
MANIFESTS_DIR = Path(project_root) / "semantic_scholar" / "manifests"
DEFAULT_MINI_MANIFEST_NAME = "mendelian_v1.json"
# Scratch/build work lives under STATE_DIR so it never pollutes the verified
# dataset/index directories. Cleanup is explicit and job-scoped.
DATA_WORK_DIR = Path(Config.STATE_DIR) / "data_work"
MINI_SUPPORTED_DATASETS = {"papers", "abstracts", "authors", "s2orc_v2", "tldrs"}
S2ORC_DATASETS = {"s2orc_v2"}
DEFAULT_DATASETS = ["papers", "abstracts", "authors", "s2orc_v2", "tldrs"]
SUPPORTED_DATASETS = ["papers", "abstracts", "authors", "s2orc_v2", "tldrs"]


class MiniCorpusManifestError(ValueError):
    """Raised when the curated mini-corpus manifest is missing or invalid."""


def configured_manifest_name() -> str:
    """Return the manifest filename selected via SEMANTIC_SCHOLAR_MANIFEST."""
    name = str(getattr(Config, "SEMANTIC_SCHOLAR_MANIFEST", "") or "").strip()
    return name or DEFAULT_MINI_MANIFEST_NAME


def resolve_manifest_name(name: Optional[str] = None) -> str:
    """Validate that ``name`` (or the configured default) is a bare manifest filename.

    Manifests are resolved only from ``semantic_scholar/manifests/``; anything that
    looks like a path (separators, parent refs, drive/absolute) is rejected loudly.
    """
    candidate = configured_manifest_name() if name is None else str(name).strip()
    if not candidate:
        candidate = DEFAULT_MINI_MANIFEST_NAME
    invalid = (
        candidate in {".", ".."}
        or "/" in candidate
        or "\\" in candidate
        or candidate != os.path.basename(candidate)
        or os.path.isabs(candidate)
        or Path(candidate).name != candidate
    )
    if invalid:
        raise MiniCorpusManifestError(
            "SEMANTIC_SCHOLAR_MANIFEST must be a plain filename with no path separators "
            f"(got {candidate!r}). Place curated manifests in {MANIFESTS_DIR} and "
            "reference them by filename only."
        )
    return candidate


def resolve_manifest_path(name: Optional[str] = None) -> Path:
    """Resolve a manifest filename to an absolute path under ``MANIFESTS_DIR``."""
    return MANIFESTS_DIR / resolve_manifest_name(name)


# Backwards-compatible default path (configured filename resolved within MANIFESTS_DIR).
try:
    DEFAULT_MINI_MANIFEST_PATH = resolve_manifest_path()
except MiniCorpusManifestError:
    DEFAULT_MINI_MANIFEST_PATH = MANIFESTS_DIR / DEFAULT_MINI_MANIFEST_NAME

class RateLimiter:
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.last_request = 0
        self.min_interval = 1.0 / requests_per_second

    def wait(self):
        """Wait if necessary to maintain the rate limit."""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request = time.time()

# (connect timeout, read timeout) applied to every HTTP request. The read
# timeout is the cap on how long a streamed response may go silent between
# bytes; without it a stalled-but-open connection blocks read() forever. With
# it, a stall surfaces as urllib3 ReadTimeoutError, which the resume logic below
# recovers from via an HTTP Range request rather than hanging indefinitely.
DEFAULT_REQUEST_TIMEOUT = (30, 60)

# Presigned dataset shard URLs are signed with short-lived STS credentials
# (the ``ASIA...`` key + ``x-amz-security-token``), which expire ~1h after they
# are issued regardless of the much-later ``Expires`` query value. A full mini
# scan streams tens of millions of rows per shard and can run for hours, so the
# signed URLs are refreshed once they approach this age, and re-fetched on
# demand if a shard is nonetheless rejected for an expired signature.
SIGNED_URL_REFRESH_SECONDS = 40 * 60

# Transient errors that indicate a dropped/interrupted byte stream (as opposed
# to a clean EOF). These are recoverable by resuming or re-requesting.
TRANSIENT_STREAM_ERRORS = (
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.ReadTimeoutError,
    urllib3.exceptions.IncompleteRead,
    IncompleteRead,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionError,
    TimeoutError,
    OSError,
)


class ResumeNotSupported(Exception):
    """Raised when an interrupted stream cannot be resumed via HTTP Range
    (e.g. the server ignored the Range header), so the whole download must
    restart from the beginning instead."""


class _ResumableHTTPReader(io.RawIOBase):
    """A raw, file-like reader over an HTTP source that transparently resumes
    via HTTP Range requests when the underlying connection drops mid-stream.

    It exposes the *raw* (still-compressed) bytes, so a gzip/decompression
    layer wrapped on top is oblivious to reconnects: the byte stream it sees is
    continuous even though it was stitched from multiple HTTP responses.

    Byte-level resume requires the server to honor Range requests (S3/CloudFront
    do). If a resume request is answered with a full 200 instead of 206, we
    raise ResumeNotSupported so the caller can restart the whole download.
    """

    def __init__(
        self,
        downloader: "S2DatasetDownloader",
        url: str,
        *,
        max_resume_attempts: int = 6,
        max_total_resumes: Optional[int] = None,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ):
        super().__init__()
        self._downloader = downloader
        self._url = url
        # Cap on *consecutive* failures without forward progress, and an overall
        # ceiling so a server that dribbles a few bytes before each drop cannot
        # make us reconnect forever.
        self._max_resume_attempts = max_resume_attempts
        self._max_total_resumes = (
            max_total_resumes if max_total_resumes is not None else max(max_resume_attempts * 10, 30)
        )
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._pos = 0  # raw (compressed) bytes successfully consumed
        self._consecutive_failures = 0
        self._total_resumes = 0
        self._resp = None
        self._raw = None
        self._open()

    def _open(self) -> None:
        headers = {}
        if self._pos > 0:
            headers["Range"] = f"bytes={self._pos}-"
        resp = self._downloader.make_request(self._url, stream=True, headers=headers)
        if self._pos > 0 and getattr(resp, "status_code", None) != 206:
            try:
                resp.close()
            except Exception:
                pass
            raise ResumeNotSupported(
                f"Server did not honor Range at byte {self._pos} "
                f"(status {getattr(resp, 'status_code', '?')}); cannot resume."
            )
        self._resp = resp
        self._raw = resp.raw

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while True:
            try:
                chunk = self._raw.read(len(buffer), decode_content=False)
            except TRANSIENT_STREAM_ERRORS as exc:
                self._consecutive_failures += 1
                self._total_resumes += 1
                if (
                    self._consecutive_failures > self._max_resume_attempts
                    or self._total_resumes > self._max_total_resumes
                ):
                    raise
                wait = min(self._backoff_max, self._backoff_base * (2 ** (self._consecutive_failures - 1)))
                console.print(
                    f"[yellow]Stream interrupted at byte {self._pos:,} ({exc}); resuming in "
                    f"{wait:.0f}s (attempt {self._consecutive_failures}/{self._max_resume_attempts})[/yellow]"
                )
                time.sleep(wait)
                self._close_response()
                self._open()  # may raise ResumeNotSupported -> caller restarts
                continue
            count = len(chunk)
            if count:
                buffer[:count] = chunk
                self._pos += count
                self._consecutive_failures = 0  # forward progress resets the streak
            return count

    def _close_response(self) -> None:
        try:
            if self._resp is not None:
                self._resp.close()
        except Exception:
            pass

    def close(self) -> None:
        self._close_response()
        super().close()


class S2DatasetDownloader:
    def __init__(self, version: Optional[str] = None):
        # Use project root for base directory
        self.base_dir = Path(project_root) / "semantic_scholar/datasets"
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(requests_per_second=0.5)  # Reduced to 1 request per 2 seconds
        
        self.api_key = Config.SEMANTIC_SCHOLAR_API_KEY
        if self.api_key:
            self.session.headers.update({
                'x-api-key': self.api_key
            })
        
        # Store the requested version
        self.version = version
        
        self.datasets_to_download = list(DEFAULT_DATASETS)
        self.supported_datasets = list(SUPPORTED_DATASETS)
        
        # Create base and index directories with parents
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = self.base_dir / "binary_indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Scratch/build work area, kept out of the verified dataset/index dirs.
        self.work_dir = DATA_WORK_DIR
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Define which IDs to index for each dataset
        self.dataset_id_fields = {
            'papers': [('corpusid', 'corpus_id')],
            'abstracts': [('corpusid', 'corpus_id')],
            's2orc_v2': [('corpusid', 'corpus_id')],
            'authors': [('authorid', 'author_id')],
            'tldrs': [('corpusid', 'corpus_id')]
        }

        # Initialize binary indexer (its scratch lives under the same work area)
        self.indexer = BinaryIndexer(self.base_dir, work_dir=self.work_dir / "index_tmp")

    @property
    def scratch_root(self) -> Path:
        """Root for build/backup/cache scratch, outside verified dataset dirs."""
        root = Path(getattr(self, "work_dir", None) or DATA_WORK_DIR)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def make_request(self, url: str, method: str = 'get', max_retries: int = 8, **kwargs) -> requests.Response:
        """Make a request with retry logic for rate limits and transient errors.

        A multi-hour mini build depends on these calls, so transient
        connection/TLS/5xx failures (e.g. a load-balanced API node briefly
        serving a bad cert) are retried over a ~2 minute window rather than
        aborting the whole job. Deterministic 4xx client errors are not retried.
        """
        for attempt in range(max_retries):
            try:
                # Wait for rate limit before making request
                self.rate_limiter.wait()

                # Handle headers properly - don't merge if pre-signed URL
                if 'AWSAccessKeyId' in url or 'x-amz-security-token' in url:
                    headers = kwargs.get('headers', {})
                else:
                    headers = {**self.session.headers, **(kwargs.get('headers', {}))}
                kwargs['headers'] = headers
                # Never issue a request without a timeout: a silently stalled
                # stream must raise (and become resumable) rather than hang.
                kwargs.setdefault('timeout', DEFAULT_REQUEST_TIMEOUT)
                
                if method.lower() == 'get':
                    response = requests.get(url, **kwargs)
                elif method.lower() == 'head':
                    response = requests.head(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Handle different error cases
                if response.status_code == 429:  # Rate limit
                    wait_time = min(30, (2 ** attempt) + 1)
                    console.print(f"[yellow]Rate limited. Waiting {wait_time} seconds...[/yellow]")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as e:
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", None)
                body = ""
                if resp is not None:
                    try:
                        body = (resp.text or "").strip()[:500]
                    except Exception:
                        body = ""
                # 4xx client errors (other than the rate-limit 429 handled above)
                # are deterministic: retrying the same request just wastes time.
                # Surface the server's explanation — for S3 this is the XML body
                # that names the real cause (e.g. ExpiredToken) — and raise now so
                # the caller can re-sign the URL rather than hammer a dead one.
                if status is not None and 400 <= status < 500:
                    detail = f" | response: {body}" if body else ""
                    raise requests.exceptions.HTTPError(f"{e}{detail}", response=resp) from None
                if attempt == max_retries - 1:
                    raise
                wait_time = min(30, (2 ** attempt) + 1)
                console.print(f"[yellow]Request failed ({str(e)}). Retrying in {wait_time} seconds...[/yellow]")
                time.sleep(wait_time)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = min(30, (2 ** attempt) + 1)
                console.print(f"[yellow]Request failed ({str(e)}). Retrying in {wait_time} seconds...[/yellow]")
                time.sleep(wait_time)

    def get_latest_release(self) -> str:
        """Get the latest release ID or return specified version."""
        if self.version:
            # Validate version format (YYYY-MM-DD)
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', self.version):
                raise ValueError("Version must be in YYYY-MM-DD format")
            return self.version
        
        response = self.make_request(f"{BASE_URL}/release/latest")
        return response.json()["release_id"]

    def _shard_name_from_url(self, dataset_name: str, url: str) -> str:
        filename = self.get_filename_from_url(url)
        for suffix in [".gz", ".json"]:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
        return filename or dataset_name

    def _normalize_dataset_file_entry(self, dataset_name: str, entry: Any) -> Any:
        if isinstance(entry, dict):
            url = (
                entry.get("url")
                or entry.get("path")
                or entry.get("file")
                or entry.get("source")
                or entry.get("source_file")
            )
            shard = entry.get("shard")
        else:
            url = entry
            shard = None

        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"Dataset file entry for {dataset_name} is missing a URL/path.")

        if dataset_name in S2ORC_DATASETS:
            return {
                "url": url.strip(),
                "shard": str(shard or self._shard_name_from_url(dataset_name, url)).strip(),
            }
        return url.strip()

    def get_dataset_info(self, dataset_name: str, release_id: str) -> Dict:
        """Get information about a specific dataset including download links."""
        if not self.api_key:
            raise ValueError("No Semantic Scholar API key found. Set SEMANTIC_SCHOLAR_API_KEY in env_vars.json")
        if dataset_name in S2ORC_DATASETS:
            # S2ORC datasets use many large shards; normalize entries for consistent naming/progress.
            url = f"{BASE_URL}/release/{release_id}/dataset/{dataset_name}/"
            try:
                response = self.make_request(url)
                data = response.json()
                if data.get('files'):
                    data['files'] = [
                        self._normalize_dataset_file_entry(dataset_name, file_entry)
                        for file_entry in data['files']
                    ]
                    console.print(
                        f"[cyan]Found {len(data['files'])} {dataset_name} shards for release {release_id}[/cyan]"
                    )
                return data
            except requests.exceptions.HTTPError:
                console.print(f"[red]Error accessing {dataset_name} dataset. Make sure your API key has access.[/red]")
                raise
        else:
            # Standard dataset handling
            url = f"{BASE_URL}/release/{release_id}/dataset/{dataset_name}"
            response = self.make_request(url)
            data = response.json()
            if data.get('files'):
                data['files'] = [
                    self._normalize_dataset_file_entry(dataset_name, file_entry)
                    for file_entry in data['files']
                ]
            return data

    def format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        if size_bytes == 0:
            return "Unknown size"
        
        size_names = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def estimate_dataset_size(self, url: str) -> int:
        """Estimate file size using HEAD request."""
        try:
            response = self.make_request(url, method='head')
            return int(response.headers.get('content-length', 0))
        except:
            return 0

    def get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL without query parameters."""
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)
        return os.path.basename(path)

    @staticmethod
    def _safe_slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
        return slug or "curated"

    @staticmethod
    def _parse_json_line(line: bytes) -> Optional[Dict[str, Any]]:
        try:
            decoded = bytes.fromhex(line.strip().decode("ascii")).decode("utf-8")
            data = json.loads(decoded)
        except Exception:
            try:
                data = json.loads(line.strip())
            except Exception:
                return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _first_present(data: Dict[str, Any], keys: List[str]) -> Any:
        for key in keys:
            if key in data:
                return data.get(key)
        return None

    @classmethod
    def _record_corpus_id(cls, data: Dict[str, Any]) -> Optional[str]:
        value = cls._first_present(data, ["corpusid", "corpusId", "corpus_id", "CorpusId"])
        if value is None:
            external_ids = data.get("externalids") or data.get("externalIds") or {}
            if isinstance(external_ids, dict):
                value = cls._first_present(external_ids, ["CorpusId", "corpusId", "corpusid"])
        return str(value) if value is not None and str(value).strip() else None

    @classmethod
    def _record_author_id(cls, data: Dict[str, Any]) -> Optional[str]:
        value = cls._first_present(data, ["authorid", "authorId", "author_id"])
        return str(value) if value is not None and str(value).strip() else None

    @staticmethod
    def _collect_ids(value: Any, keys: Set[str]) -> Set[str]:
        ids: Set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key in keys and item is not None and str(item).strip():
                    ids.add(str(item))
                ids.update(S2DatasetDownloader._collect_ids(item, keys))
        elif isinstance(value, list):
            for item in value:
                ids.update(S2DatasetDownloader._collect_ids(item, keys))
        return ids

    @staticmethod
    def _string_id_set(values: Any) -> Set[str]:
        if values is None:
            return set()
        if not isinstance(values, list):
            raise MiniCorpusManifestError("Manifest ID lists must be arrays.")
        return {str(value).strip() for value in values if str(value).strip()}

    def _mini_manifest_setup_instructions(self, manifest_path: Path) -> str:
        return (
            "Curated mini-corpus manifest not found. "
            f"Restore the tracked Mendelian mini manifest at {manifest_path} or pass --mini-manifest. "
            "The manifest should include release_id and dataset-specific corpus_ids or author_ids; "
            "the downloader fetches the matching rows from Semantic Scholar."
        )

    def _load_mini_manifest(self, manifest_path: Optional[Path] = None) -> Tuple[Path, Dict[str, Any]]:
        if manifest_path is None:
            path = resolve_manifest_path()
        else:
            path = Path(manifest_path).expanduser()
            if not path.is_absolute():
                path = Path(project_root) / path
        path = path.resolve()
        if not path.exists():
            raise MiniCorpusManifestError(self._mini_manifest_setup_instructions(path))

        try:
            with open(path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except json.JSONDecodeError as exc:
            raise MiniCorpusManifestError(f"Mini-corpus manifest is not valid JSON: {path}") from exc

        if not isinstance(manifest, dict):
            raise MiniCorpusManifestError("Mini-corpus manifest must be a JSON object.")
        if not str(manifest.get("release_id", "")).strip():
            raise MiniCorpusManifestError("Mini-corpus manifest must include release_id.")

        dataset_ids = self._manifest_dataset_ids(manifest)
        if not dataset_ids:
            raise MiniCorpusManifestError(
                "Mini-corpus manifest must include dataset-specific corpus_ids or author_ids."
            )
        return path, manifest

    @staticmethod
    def _dataset_id_values(config: Any, dataset: str) -> Set[str]:
        if isinstance(config, list):
            return S2DatasetDownloader._string_id_set(config)
        if not isinstance(config, dict):
            return set()
        if dataset == "authors":
            values = config.get("author_ids")
        else:
            values = config.get("corpus_ids")
        if values is None:
            values = config.get("ids")
        return S2DatasetDownloader._string_id_set(values)

    def _manifest_dataset_ids(self, manifest: Dict[str, Any]) -> Dict[str, Set[str]]:
        dataset_ids: Dict[str, Set[str]] = {}
        dataset_config = manifest.get("datasets")
        if isinstance(dataset_config, dict):
            for dataset, config in dataset_config.items():
                dataset_name = str(dataset).strip()
                if dataset_name not in MINI_SUPPORTED_DATASETS:
                    console.print(f"[yellow]Skipping unsupported mini-corpus dataset: {dataset_name}[/yellow]")
                    continue
                ids = self._dataset_id_values(config, dataset_name)
                if ids:
                    dataset_ids[dataset_name] = ids

        legacy_corpus_ids = self._string_id_set(manifest.get("corpus_ids"))
        legacy_corpus_ids.update(self._collect_ids(manifest.get("papers"), {"corpusid", "corpusId", "corpus_id"}))
        legacy_corpus_ids.update(self._collect_ids(manifest.get("claims"), {"corpusid", "corpusId", "corpus_id"}))
        for dataset in ["papers", "abstracts", "s2orc_v2", "tldrs"]:
            if legacy_corpus_ids and dataset not in dataset_ids:
                dataset_ids[dataset] = set(legacy_corpus_ids)

        legacy_author_ids = self._string_id_set(manifest.get("author_ids"))
        legacy_author_ids.update(self._collect_ids(manifest.get("authors"), {"authorid", "authorId", "author_id"}))
        legacy_author_ids.update(self._collect_ids(manifest.get("claims"), {"authorid", "authorId", "author_id"}))
        if legacy_author_ids and "authors" not in dataset_ids:
            dataset_ids["authors"] = legacy_author_ids

        return dataset_ids

    def _manifest_corpus_ids(self, manifest: Dict[str, Any]) -> Set[str]:
        dataset_ids = self._manifest_dataset_ids(manifest)
        corpus_ids: Set[str] = set()
        for dataset, ids in dataset_ids.items():
            if dataset != "authors":
                corpus_ids.update(ids)
        return corpus_ids

    def _manifest_author_ids(self, manifest: Dict[str, Any]) -> Set[str]:
        return set(self._manifest_dataset_ids(manifest).get("authors", set()))

    @staticmethod
    def _manifest_file_entry_value(entry: Any) -> Optional[str]:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for key in ["path", "url", "file", "source", "source_file"]:
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _manifest_source_files(self, manifest: Dict[str, Any], manifest_dir: Path) -> Dict[str, List[str]]:
        source_config = manifest.get("source_files")
        if source_config is None:
            source_config = manifest.get("datasets")
        if not isinstance(source_config, dict):
            return {}

        source_files: Dict[str, List[str]] = {}
        for dataset, config in source_config.items():
            dataset_name = str(dataset).strip()
            if dataset_name not in MINI_SUPPORTED_DATASETS:
                console.print(f"[yellow]Skipping unsupported mini-corpus dataset: {dataset_name}[/yellow]")
                continue

            entries = config
            if isinstance(config, dict):
                entries = config.get("files") or config.get("source_files") or []
            if not isinstance(entries, list):
                raise MiniCorpusManifestError(f"source_files.{dataset_name} must be an array.")

            resolved_entries: List[str] = []
            for entry in entries:
                value = self._manifest_file_entry_value(entry)
                if not value:
                    raise MiniCorpusManifestError(
                        f"source_files.{dataset_name} entries must be strings or objects with path/url."
                    )
                if re.match(r"^https?://", value):
                    resolved_entries.append(value)
                else:
                    source_path = Path(value).expanduser()
                    if not source_path.is_absolute():
                        source_path = manifest_dir / source_path
                    resolved_entries.append(str(source_path.resolve()))
            if resolved_entries:
                source_files[dataset_name] = resolved_entries
        return source_files

    def _mini_topic_label(self, manifest: Dict[str, Any]) -> str:
        explicit = str(manifest.get("topic_label", "")).strip()
        if explicit:
            return explicit
        mini_release_id = str(manifest.get("mini_release_id", "")).strip()
        if "-mini-" in mini_release_id:
            return mini_release_id.split("-mini-", 1)[1]
        return "curated"

    def _mini_release_id(self, manifest: Dict[str, Any]) -> str:
        explicit = str(manifest.get("mini_release_id", "")).strip()
        if explicit:
            return explicit
        release_id = str(manifest["release_id"]).strip()
        topic_slug = self._safe_slug(self._mini_topic_label(manifest))
        return f"{release_id}-mini-{topic_slug}"

    @staticmethod
    def _is_remote_source(source: str) -> bool:
        return bool(re.match(r"^https?://", source))

    def _iter_source_lines(self, source: str):
        if self._is_remote_source(source):
            yield from self._iter_remote_source_lines(source)
            return

        source_path = Path(source)
        if not source_path.exists():
            raise MiniCorpusManifestError(f"Mini-corpus source file does not exist: {source_path}")
        opener = gzip.open if source_path.suffix == ".gz" else open
        with opener(source_path, "rb") as handle:
            for line in handle:
                yield line

    def _iter_remote_source_lines(self, source: str, whole_restart_attempts: int = 2):
        """Stream lines from a remote shard, resilient to dropped connections.

        Primary recovery is byte-level resume via HTTP Range (see
        _ResumableHTTPReader). If the stream still cannot be completed (e.g. the
        server ignores Range, or the gzip ends up truncated), the whole shard is
        re-downloaded from the start up to ``whole_restart_attempts`` times.
        Re-emitting already-seen lines on a restart is safe: callers de-duplicate
        by record ID, so duplicates are skipped downstream.
        """
        is_gz = self.get_filename_from_url(source).endswith(".gz")
        restartable_errors = TRANSIENT_STREAM_ERRORS + (
            ResumeNotSupported,
            EOFError,
            gzip.BadGzipFile,
        )
        for restart in range(whole_restart_attempts + 1):
            reader = _ResumableHTTPReader(self, source)
            buffered = io.BufferedReader(reader)
            stream = gzip.GzipFile(fileobj=buffered) if is_gz else buffered
            try:
                for line in stream:
                    yield line
                return  # completed cleanly
            except restartable_errors as exc:
                if restart >= whole_restart_attempts:
                    raise
                wait = min(30, 2 ** restart)
                console.print(
                    f"[yellow]Shard stream failed ({exc}); re-downloading from the start "
                    f"in {wait}s (restart {restart + 1}/{whole_restart_attempts})[/yellow]"
                )
                time.sleep(wait)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass
                reader.close()

    def _record_matches_manifest(
        self,
        dataset: str,
        data: Dict[str, Any],
        *,
        target_ids: Set[str],
    ) -> bool:
        if dataset == "authors":
            author_id = self._record_author_id(data)
            return bool(author_id and author_id in target_ids)
        corpus_id = self._record_corpus_id(data)
        return bool(corpus_id and corpus_id in target_ids)

    def _matched_ids_in_sources(self, *, dataset: str, sources: List[str], target_ids: Set[str]) -> Set[str]:
        matched_ids: Set[str] = set()
        for source in sources:
            if self._is_remote_source(source):
                continue
            source_path = Path(source)
            if not source_path.exists():
                continue
            for line in self._iter_source_lines(str(source_path)):
                data = self._parse_json_line(line)
                if not data:
                    continue
                record_id = self._record_author_id(data) if dataset == "authors" else self._record_corpus_id(data)
                if record_id and record_id in target_ids:
                    matched_ids.add(record_id)
            if matched_ids == target_ids:
                break
        return matched_ids

    def _shard_progress_logger(
        self,
        prefix: str,
        status_fn=None,
        *,
        every_lines: int = 100_000,
        every_seconds: float = 15.0,
    ):
        """Return a ``tick()`` to call once per streamed line.

        It prints a throttled progress line (at most once per ``every_seconds``,
        checked every ``every_lines`` rows) so that a shard which is slow but
        alive is visibly distinct from one that has stalled. A genuine hang stops
        producing lines, so ``tick`` goes quiet too — but the request read
        timeout now turns that stall into a logged resume rather than silence.
        """
        state = {"lines": 0, "last": time.time()}

        def tick():
            state["lines"] += 1
            if state["lines"] % every_lines:
                return
            now = time.time()
            if now - state["last"] < every_seconds:
                return
            state["last"] = now
            extra = f" — {status_fn()}" if status_fn else ""
            console.print(
                f"[cyan]  …{prefix}: {state['lines']:,} rows scanned{extra}[/cyan]"
            )

        return tick

    def _shard_descriptor(self, dataset: str, entry: Any) -> Tuple[str, str, str]:
        """Return (source_url, display_label, stable_shard_key) for a dataset
        file entry. The shard key is the filename (query string stripped), which
        is stable across re-signings, so a refreshed URL list can be matched back
        to the same shard."""
        if dataset in S2ORC_DATASETS:
            if isinstance(entry, dict):
                source = entry.get("url") or ""
                label = entry.get("shard") or self.get_filename_from_url(source)
            else:
                source = str(entry or "")
                label = source
        else:
            source = str(entry or "")
            label = self.get_filename_from_url(source)
        key = self.get_filename_from_url(source) if source else str(label or "")
        return source, label, key

    @staticmethod
    def _is_signed_url_rejection(exc: Exception) -> bool:
        """True for the 4xx S3 returns when a presigned URL's STS credentials
        have expired or are otherwise rejected — recoverable by re-signing."""
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status in (400, 403):
            return True
        text = str(exc)
        return "400 " in text or "403 " in text or "Forbidden" in text

    def _scan_shard_for_ids(
        self,
        *,
        dataset: str,
        source: str,
        index: int,
        total: int,
        remaining: Set[str],
        matched_ids: Set[str],
        output,
    ) -> None:
        """Stream one shard, writing matches and removing them from ``remaining``.

        Safe to call again on the same shard after a failure: already-matched
        records are no longer in ``remaining`` and are skipped on the re-scan."""
        tick = self._shard_progress_logger(
            f"{dataset} {index}/{total}",
            lambda: f"{len(matched_ids):,} matched, {len(remaining):,} still missing",
        )
        for line in self._iter_source_lines(source):
            tick()
            data = self._parse_json_line(line)
            if not data:
                continue
            record_id = self._record_author_id(data) if dataset == "authors" else self._record_corpus_id(data)
            if not record_id or record_id not in remaining:
                continue
            output.write(json.dumps(data, ensure_ascii=True) + "\n")
            output.flush()
            matched_ids.add(record_id)
            remaining.discard(record_id)
            if not remaining:
                break

    @staticmethod
    def _scan_fingerprint(target_ids: Set[str]) -> str:
        """Stable digest of the full target-id set a scan is run against.

        Used to invalidate shard checkpoints if the manifest's requested IDs
        change: skipping a previously-scanned shard is only safe when it was
        scanned for (a superset of) the IDs we are now looking for."""
        digest = hashlib.sha256("\n".join(sorted(target_ids)).encode("utf-8"))
        return digest.hexdigest()[:16]

    def _scan_remote_sources_for_ids(
        self,
        *,
        dataset: str,
        release_id: str,
        target_ids: Set[str],
        output_path: Path,
        scan_fingerprint: Optional[str] = None,
    ) -> Tuple[str, Set[str]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        info = self.get_dataset_info(dataset, release_id)
        descriptors = [self._shard_descriptor(dataset, entry) for entry in (info.get("files") or [])]
        shards = [(label, key) for (source, label, key) in descriptors if source]
        shard_sources: Dict[str, str] = {key: source for (source, label, key) in descriptors if source}
        signed_at = time.monotonic()

        # Shard-level resume: a checkpoint next to the cache records which shards
        # have been fully scanned, so an interrupted multi-hour scan resumes at
        # the next unscanned shard instead of re-streaming completed ones. It is
        # only honored when the cache it describes still exists and was built for
        # the same release and target set.
        progress_path = Path(str(output_path) + ".progress.json")
        completed: Set[str] = set()
        if output_path.exists() and progress_path.exists():
            try:
                checkpoint = json.loads(progress_path.read_text(encoding="utf-8"))
                if (
                    checkpoint.get("release_id") == release_id
                    and checkpoint.get("fingerprint") == scan_fingerprint
                ):
                    completed = set(checkpoint.get("completed_shards") or [])
            except Exception:
                completed = set()

        def save_progress() -> None:
            tmp = Path(str(progress_path) + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "release_id": release_id,
                        "fingerprint": scan_fingerprint,
                        "completed_shards": sorted(completed),
                    }
                ),
                encoding="utf-8",
            )
            os.replace(tmp, progress_path)

        def refresh_signed_urls(reason: str) -> None:
            nonlocal signed_at
            console.print(f"[yellow]Refreshing Semantic Scholar signed URLs ({reason}).[/yellow]")
            fresh = self.get_dataset_info(dataset, release_id)
            for fresh_entry in fresh.get("files") or []:
                fresh_source, _label, fresh_key = self._shard_descriptor(dataset, fresh_entry)
                if fresh_source:
                    shard_sources[fresh_key] = fresh_source
            signed_at = time.monotonic()

        remaining = set(target_ids)
        matched_ids: Set[str] = set()
        total = len(shards)
        if completed:
            console.print(
                f"[cyan]Resuming {dataset} scan: {len(completed)}/{total} shards already done.[/cyan]"
            )
        mode = "a" if output_path.exists() else "w"
        with open(output_path, mode, encoding="utf-8") as output:
            for index, (label, shard_key) in enumerate(shards, start=1):
                if shard_key in completed:
                    continue
                # Proactively re-sign before the ~1h STS token behind these URLs
                # expires; a full scan routinely runs longer than that.
                if time.monotonic() - signed_at > SIGNED_URL_REFRESH_SECONDS:
                    refresh_signed_urls("approaching token expiry")
                console.print(f"[cyan]Fetching mini {dataset} source {index}/{total}: {label}[/cyan]")
                for attempt in range(2):
                    source = shard_sources.get(shard_key)
                    if not source:
                        break
                    try:
                        self._scan_shard_for_ids(
                            dataset=dataset,
                            source=source,
                            index=index,
                            total=total,
                            remaining=remaining,
                            matched_ids=matched_ids,
                            output=output,
                        )
                        completed.add(shard_key)
                        save_progress()
                        break
                    except requests.exceptions.HTTPError as exc:
                        # An expired/rejected signature is recoverable: re-sign the
                        # URLs once and retry this shard from the start.
                        if attempt == 0 and self._is_signed_url_rejection(exc):
                            refresh_signed_urls(f"shard {index}/{total} rejected: {exc}")
                            continue
                        raise
                if not remaining:
                    break
        # The scan finished without error; the checkpoint has served its purpose.
        try:
            progress_path.unlink()
        except OSError:
            pass
        return str(output_path.resolve()), matched_ids

    def _mini_sources_for_dataset(
        self,
        *,
        dataset: str,
        release_id: str,
        mini_release_id: str,
        manifest_dir: Path,
        configured_sources: List[str],
        target_ids: Set[str],
    ) -> List[str]:
        sources = list(configured_sources)
        cache_path = self.scratch_root / "mini" / "source_extracts" / self._safe_slug(mini_release_id) / f"{dataset}.jsonl"
        cache_resolved = str(cache_path.resolve())
        if cache_path.exists() and cache_resolved not in sources:
            sources.append(cache_resolved)

        local_matches = self._matched_ids_in_sources(dataset=dataset, sources=sources, target_ids=target_ids)
        missing_ids = target_ids - local_matches
        if missing_ids:
            console.print(
                f"[yellow]{dataset}: {len(missing_ids):,} requested IDs are not in the local mini cache; "
                "streaming Semantic Scholar dataset shards.[/yellow]"
            )
            source, matched_ids = self._scan_remote_sources_for_ids(
                dataset=dataset,
                release_id=release_id,
                target_ids=missing_ids,
                output_path=cache_path,
                # Fingerprint the full requested set (not just the currently
                # missing subset) so the shard checkpoint stays valid across
                # resumes and is invalidated only when the manifest changes.
                scan_fingerprint=self._scan_fingerprint(target_ids),
            )
            if matched_ids and source not in sources:
                sources.append(source)
            unresolved = missing_ids - matched_ids
            if unresolved:
                console.print(f"[yellow]{dataset}: {len(unresolved):,} requested IDs were not found.[/yellow]")
        if not sources:
            raise MiniCorpusManifestError(f"No source rows available for mini-corpus dataset: {dataset}")
        return sources

    def _write_filtered_dataset(
        self,
        *,
        dataset: str,
        sources: List[str],
        output_path: Path,
        target_ids: Set[str],
    ) -> Tuple[int, Set[str]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seen_ids: Set[str] = set()
        written = 0
        with open(output_path, "w", encoding="utf-8") as output:
            for source in sources:
                # A configured source that isn't present here (e.g. a manifest
                # source_files path that is absolute to another machine) is
                # skipped, matching _matched_ids_in_sources — the data is taken
                # from whichever sources do exist (such as the streamed cache).
                if not self._is_remote_source(source) and not Path(source).exists():
                    console.print(f"[yellow]Skipping missing {dataset} source: {source}[/yellow]")
                    continue
                console.print(f"[cyan]Filtering {dataset} source: {source}[/cyan]")
                for line in self._iter_source_lines(source):
                    data = self._parse_json_line(line)
                    if not data or not self._record_matches_manifest(
                        dataset,
                        data,
                        target_ids=target_ids,
                    ):
                        continue

                    record_id = self._record_author_id(data) if dataset == "authors" else self._record_corpus_id(data)
                    if not record_id or record_id in seen_ids:
                        continue
                    seen_ids.add(record_id)
                    output.write(json.dumps(data, ensure_ascii=True) + "\n")
                    written += 1
        return written, seen_ids

    def materialize_mini_corpus(self, manifest_path: Optional[Path] = None) -> str:
        """Build the curated mini corpus described by a local manifest."""
        path, manifest = self._load_mini_manifest(manifest_path)
        mini_release_id = self._mini_release_id(manifest)
        topic_label = self._mini_topic_label(manifest)
        dataset_ids = self._manifest_dataset_ids(manifest)
        source_files = self._manifest_source_files(manifest, path.parent)

        release_dir = self.base_dir / mini_release_id
        if release_dir.exists():
            if "-mini-" not in mini_release_id:
                raise MiniCorpusManifestError(
                    f"Refusing to replace non-mini release directory: {release_dir}"
                )
            console.print(f"[cyan]Existing mini release will be replaced after rebuild succeeds: {release_dir}[/cyan]")

        build_root = self.scratch_root / "mini" / "builds"
        build_root.mkdir(parents=True, exist_ok=True)
        build_dir = build_root / f"{self._safe_slug(mini_release_id)}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        release_metadata = {
            "release_id": mini_release_id,
            "source_release_id": manifest["release_id"],
            "topic_label": topic_label,
            "manifest_path": str(path),
            "dataset_ids_requested": {dataset: len(ids) for dataset, ids in dataset_ids.items()},
        }
        with open(build_dir / "mini_manifest_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(release_metadata, handle, indent=2, ensure_ascii=True)

        written_by_dataset: Dict[str, int] = {}
        matched_ids_by_dataset: Dict[str, Set[str]] = {}
        missing_ids_by_dataset: Dict[str, List[str]] = {}
        dataset_order = DEFAULT_DATASETS
        for dataset in sorted(dataset_ids, key=lambda name: dataset_order.index(name) if name in dataset_order else 999):
            target_ids = dataset_ids[dataset]
            sources = self._mini_sources_for_dataset(
                dataset=dataset,
                release_id=str(manifest["release_id"]),
                mini_release_id=mini_release_id,
                manifest_dir=path.parent,
                configured_sources=source_files.get(dataset, []),
                target_ids=target_ids,
            )
            dataset_dir = build_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "name": dataset,
                        "mini_corpus": True,
                        "source_release_id": manifest["release_id"],
                        "topic_label": topic_label,
                        "requested_id_count": len(target_ids),
                        "source_files": sources,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=True,
                )
            count, matched_ids = self._write_filtered_dataset(
                dataset=dataset,
                sources=sources,
                output_path=dataset_dir / "mini.json",
                target_ids=target_ids,
            )
            written_by_dataset[dataset] = count
            matched_ids_by_dataset[dataset] = matched_ids
            missing_ids_by_dataset[dataset] = sorted(target_ids - matched_ids)
            console.print(f"[green]Wrote {count:,} records for {dataset}[/green]")

        summary = {
            "mini_release_id": mini_release_id,
            "source_release_id": manifest["release_id"],
            "topic_label": topic_label,
            "records_requested": {dataset: len(ids) for dataset, ids in dataset_ids.items()},
            "records_written": written_by_dataset,
            "records_missing": {
                dataset: len(ids)
                for dataset, ids in missing_ids_by_dataset.items()
                if ids
            },
        }
        with open(build_dir / "mini_build_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=True)

        backup_dir: Optional[Path] = None
        try:
            if release_dir.exists():
                backup_root = self.scratch_root / "mini" / "release_backups"
                backup_root.mkdir(parents=True, exist_ok=True)
                backup_dir = backup_root / f"{self._safe_slug(mini_release_id)}-{uuid.uuid4().hex[:8]}"
                shutil.move(str(release_dir), str(backup_dir))
            shutil.move(str(build_dir), str(release_dir))
        except Exception:
            if backup_dir and backup_dir.exists() and not release_dir.exists():
                shutil.move(str(backup_dir), str(release_dir))
            if build_dir.exists():
                shutil.rmtree(build_dir, ignore_errors=True)
            raise

        for dataset, count in written_by_dataset.items():
            if count <= 0:
                console.print(f"[yellow]Skipping index for {dataset}; no matching records were written.[/yellow]")
                continue
            if not self.index_dataset(dataset, mini_release_id):
                raise MiniCorpusManifestError(f"Failed to index mini-corpus dataset: {dataset}")

        # Housekeeping: remove the now-redundant backup (bounded retry). A failure
        # here means scratch is piling up in a verified-data-adjacent area, so the
        # job must fail loudly rather than report success.
        if backup_dir and backup_dir.exists():
            try:
                remove_scratch_path(backup_dir)
            except OSError as exc:
                raise MiniCorpusManifestError(
                    f"Mini corpus built but housekeeping failed: {exc}"
                ) from exc

        console.print(
            f"[green]Curated mini corpus ready: {mini_release_id} "
            f"({release_dir})[/green]"
        )
        return mini_release_id

    @staticmethod
    def _count_jsonl_records(path: Path) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    def verify_mini_corpus(self, manifest_path: Optional[Path] = None) -> bool:
        """Verify that the local mini release matches the manifest and indices."""
        path, manifest = self._load_mini_manifest(manifest_path)
        mini_release_id = self._mini_release_id(manifest)
        release_dir = self.base_dir / mini_release_id
        if not release_dir.exists():
            console.print(f"[red]Mini corpus release not found: {release_dir}[/red]")
            return False

        dataset_ids = self._manifest_dataset_ids(manifest)
        missing_rows: Dict[str, int] = {}
        record_counts: Dict[str, int] = {}
        for dataset, ids in dataset_ids.items():
            dataset_dir = release_dir / dataset
            if not dataset_dir.exists():
                missing_rows[dataset] = len(ids)
                record_counts[dataset] = 0
                continue
            count = sum(
                self._count_jsonl_records(path)
                for path in dataset_dir.glob("*.json")
                if path.name != "metadata.json"
            )
            record_counts[dataset] = count
            if count < len(ids):
                missing_rows[dataset] = len(ids) - count

        if missing_rows:
            console.print(f"[red]Mini corpus release is stale: {mini_release_id}[/red]")
            for dataset, count in missing_rows.items():
                console.print(f"[yellow]{dataset}: {count:,} manifest records missing locally[/yellow]")
            return False

        stats = self.indexer.get_index_stats(mini_release_id)
        dataset_id_fields = getattr(
            self,
            "dataset_id_fields",
            getattr(self.indexer, "dataset_id_fields", {}),
        )
        index_issues: List[str] = []
        for dataset, ids in dataset_ids.items():
            for _, id_type in dataset_id_fields.get(dataset, []):
                key = f"{dataset}_{id_type}"
                info = stats.get(key)
                if not info:
                    index_issues.append(f"{key}: missing")
                    continue
                if not info.get("healthy"):
                    index_issues.append(f"{key}: unhealthy")
                    continue
                if int(info.get("entry_count") or 0) < len(ids):
                    index_issues.append(f"{key}: indexed fewer rows than manifest requests")

        if index_issues:
            console.print(f"[red]Mini corpus indices need attention: {mini_release_id}[/red]")
            for issue in index_issues:
                console.print(f"[yellow]{issue}[/yellow]")
            return False

        console.print(f"[green]Mini corpus release matches manifest: {mini_release_id}[/green]")
        for dataset, count in record_counts.items():
            console.print(f"[green]{dataset}: {count:,} records[/green]")
        return True

    def _parallel_extract_gzip(self, input_path: Path, output_path: Path):
        """Revert to single-thread gzip extraction to avoid misalignment issues."""
        try:
            console.print(f"Extracting {input_path.name} without parallelization...")

            with gzip.open(input_path, 'rb') as gz_in, open(output_path, 'wb') as out:
                shutil.copyfileobj(gz_in, out)

        except Exception as e:
            console.print(f"[red]Error extracting {input_path.name}: {str(e)}[/red]")
            raise

    def verify_file(self, file_path: Path, expected_size: Optional[int] = None) -> bool:
        """Verify if a file is complete based on size."""
        if not file_path.exists():
            return False
        
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                console.print(f"[yellow]File {file_path.name} is incomplete (size: {actual_size} vs expected: {expected_size})[/yellow]")
                return False
        
        return True

    def download_file(self, url: str, output_dir: Path, desc: str = None, resign=None) -> Tuple[bool, Optional[Path]]:
        """Stream a dataset shard to disk via the shared resumable HTTP reader.

        Uses the same transfer path as the mini-corpus build — make_request +
        _ResumableHTTPReader, reached through _iter_source_lines — so it resumes
        byte-level on dropped connections, honors request timeouts, and needs no
        external tools (no wget). Gzipped shards are decompressed on the fly and
        written to a ``.partial`` file that is atomically renamed on success, so
        an interrupted download never leaves a truncated shard behind.

        ``resign`` is an optional zero-arg callable that returns a freshly signed
        URL for this shard. If the download is rejected for an expired/invalid
        signature (the STS token behind the presigned URL), it is called once and
        the download retried — the same recovery the mini build uses.
        """
        filename = self.get_filename_from_url(url)
        is_gzipped = filename.endswith('.gz')
        final_path = output_dir / (filename.replace('.gz', '.json') if is_gzipped else filename)
        if final_path.exists():
            console.print(f"[green]File {final_path.name} already exists[/green]")
            return True, final_path

        output_dir.mkdir(parents=True, exist_ok=True)
        partial_path = final_path.with_name(final_path.name + ".partial")
        console.print(f"[cyan]{desc or f'Downloading {filename}'}[/cyan]")
        for attempt in range(2):
            try:
                with open(partial_path, 'wb') as out:
                    for line in self._iter_source_lines(url):
                        out.write(line)
                os.replace(partial_path, final_path)
                console.print(f"[green]Successfully downloaded: {final_path.name}[/green]")
                return True, final_path
            except Exception as exc:
                if partial_path.exists():
                    try:
                        partial_path.unlink()
                    except OSError:
                        pass
                if (
                    attempt == 0
                    and resign is not None
                    and isinstance(exc, requests.exceptions.HTTPError)
                    and self._is_signed_url_rejection(exc)
                ):
                    fresh = resign()
                    if fresh:
                        console.print(f"[yellow]Re-signing rejected URL for {filename} and retrying.[/yellow]")
                        url = fresh
                        continue
                console.print(f"[red]Failed to download {filename}: {exc}[/red]")
                return False, None
        return False, None

    def download_dataset(self, dataset_name: str, release_id: str = 'latest', index: bool = True) -> bool:
        """Download a specific dataset and optionally build index."""
        try:
            if release_id == 'latest':
                # First try to get local latest release
                local_release = self._get_latest_local_release()
                if local_release:
                    release_id = local_release
                    console.print(f"[cyan]Using latest local release: {release_id}[/cyan]")
                else:
                    # If no local release, get latest from API
                    release_id = self.get_latest_release()
                    console.print(f"[cyan]Using latest API release: {release_id}[/cyan]")
            
            # Get dataset info
            console.print(f"\n[cyan]Getting dataset info for {dataset_name}...[/cyan]")
            dataset_info = self.get_dataset_info(dataset_name, release_id)
            if not dataset_info:
                return False
            
            # Save metadata
            dataset_dir = self.base_dir / release_id / dataset_name
            os.makedirs(dataset_dir, exist_ok=True)
            
            metadata_path = dataset_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)

            # First check which files we need
            missing_files = []
            if dataset_name in S2ORC_DATASETS:
                files = dataset_info['files']
                console.print(f"\n[bold]Checking {len(files)} files for {dataset_name}...[/bold]")
                for file_info in files:
                    output_path = dataset_dir / f"{file_info['shard']}.json"
                    if output_path.exists():
                        size = output_path.stat().st_size / (1024 * 1024)
                        console.print(f"[green]OK Already exists ({size:.1f} MB): {output_path.name}[/green]")
                    else:
                        console.print(f"[yellow]→ Needs download: {output_path.name}[/yellow]")
                        missing_files.append(file_info)
            else:
                files_to_download = dataset_info['files']
                console.print(f"\n[bold]Checking {len(files_to_download)} files for {dataset_name}...[/bold]")
                for file_url in files_to_download:
                    output_path = dataset_dir / self.get_filename_from_url(file_url).replace('.gz', '.json')
                    if output_path.exists():
                        size = output_path.stat().st_size / (1024 * 1024)
                        console.print(f"[green]OK Already exists ({size:.1f} MB): {output_path.name}[/green]")
                    else:
                        console.print(f"[yellow]→ Needs download: {output_path.name}[/yellow]")
                        missing_files.append(file_url)

            # If we have missing files, stream them — re-signing on demand if a
            # shard's presigned URL expires mid-download (a full corpus can take
            # longer than the ~1h STS token lifetime).
            if missing_files:
                console.print(f"\n[cyan]Downloading {len(missing_files)} missing {dataset_name} files...[/cyan]")

                def resign_for(shard_key):
                    def _resign():
                        fresh = self.get_dataset_info(dataset_name, release_id)
                        for fresh_entry in fresh.get('files') or []:
                            fresh_src, _label, fresh_key = self._shard_descriptor(dataset_name, fresh_entry)
                            if fresh_src and fresh_key == shard_key:
                                return fresh_src
                        return None
                    return _resign

                downloaded_files = []
                download_failures = 0
                for file_info in missing_files:
                    src, label, shard_key = self._shard_descriptor(dataset_name, file_info)
                    success, path = self.download_file(
                        src,
                        dataset_dir,
                        desc=f"Downloading {dataset_name} shard {label}",
                        resign=resign_for(shard_key),
                    )
                    if success and path:
                        downloaded_files.append(path)
                    else:
                        download_failures += 1

                if download_failures:
                    console.print(
                        f"[red]{download_failures} of {len(missing_files)} {dataset_name} files failed to download[/red]"
                    )
                    return False

                if index and downloaded_files:
                    if not self.index_dataset(dataset_name, release_id):
                        console.print(f"[red]Indexing failed for {dataset_name}[/red]")
                        return False

            return True
                
        except Exception as e:
            console.print(f"[red]Error downloading dataset {dataset_name}: {str(e)}[/red]")
            return False

    def index_dataset(self, dataset: str, release_id: str, skip_ids: Optional[Set[str]] = None) -> bool:
        """Create binary indices for a dataset"""
        try:
            dataset_dir = self.base_dir / release_id / dataset
            if not dataset_dir.exists():
                console.print(f"[red]Dataset directory not found: {dataset_dir}[/red]")
                return False

            # Get all data files
            files = [f for f in dataset_dir.glob("*.json") if f.name != 'metadata.json']
            if not files:
                console.print(f"[yellow]No files found to index in {dataset_dir}[/yellow]")
                return False

            # Create temporary directory for chunks
            chunk_dir = self.indexer.tmp_dir / f"{release_id}_{dataset}_chunks"
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir)
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Track entries for each ID type
            id_types_seen = set()
            total_entries = 0
            
            console.print(f"[cyan]Processing {len(files)} files for {dataset}...[/cyan]")
            
            # Process each file and write sorted chunks
            for file_num, file_path in enumerate(files):
                console.print(f"[cyan]Processing {file_path.name}...[/cyan]")
                entries_by_id_type = defaultdict(list)
                entries_in_file = 0
                
                with open(file_path, 'rb') as f:
                    offset = 0
                    for line in f:
                        try:
                            # Try hex-encoded JSON first
                            decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                            data = json.loads(decoded)
                        except:
                            # Fall back to regular JSON
                            try:
                                data = json.loads(line.strip())
                            except:
                                console.print(f"[yellow]Warning: Skipping invalid JSON line in {file_path}[/yellow]")
                                offset += len(line)
                                continue

                        # Extract IDs based on dataset type
                        if dataset == 'papers':
                            corpus_id = self._record_corpus_id(data)
                            if corpus_id:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(corpus_id, str(file_path), offset)
                                )
                                entries_in_file += 1
                            paper_id = self._first_present(data, ["paperId", "paperid", "paper_id"])
                            if paper_id:
                                entries_by_id_type['paper_id'].append(
                                    IndexEntry(str(paper_id), str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 'authors':
                            author_id = self._record_author_id(data)
                            if author_id:
                                entries_by_id_type['author_id'].append(
                                    IndexEntry(author_id, str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 'citations':
                            if 'citingcorpusid' in data:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(str(data['citingcorpusid']), str(file_path), offset)
                                )
                                entries_in_file += 1
                            if 'citedcorpusid' in data:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(str(data['citedcorpusid']), str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 'abstracts':
                            corpus_id = self._record_corpus_id(data)
                            if corpus_id:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(corpus_id, str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 's2orc_v2':
                            corpus_id = self._record_corpus_id(data)
                            if corpus_id:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(corpus_id, str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 'tldrs':
                            corpus_id = self._record_corpus_id(data)
                            if corpus_id:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(corpus_id, str(file_path), offset)
                                )
                                entries_in_file += 1
                            
                        offset += len(line)

                # Write sorted chunks for each ID type
                for id_type, entries in entries_by_id_type.items():
                    id_types_seen.add(id_type)
                    entries.sort(key=lambda x: x.id)
                    chunk_path = chunk_dir / f"{id_type}_chunk_{file_num:03d}.idx"

                    # Skip rewriting if chunk file already exists
                    if chunk_path.exists():
                        console.print(f"[yellow]Skipping existing chunk {chunk_path.name}[/yellow]")
                        continue

                    with open(chunk_path, 'wb') as f:
                        for entry in entries:
                            if skip_ids is None or entry.id not in skip_ids:
                                f.write(entry.to_bytes())

                total_entries += entries_in_file
                console.print(f"[green]Created {entries_in_file:,} index entries from {file_path.name}[/green]")

            # Create final indices from chunks
            for id_type in id_types_seen:
                chunks = sorted(chunk_dir.glob(f"{id_type}_chunk_*.idx"))
                if not chunks:
                    continue
                    
                if not self.indexer.create_index_from_chunks(release_id, dataset, id_type, chunks):
                    console.print(f"[red]Failed to create index for {dataset}_{id_type}[/red]")
                    return False

            # Explicit, job-scoped cleanup of the chunk scratch (bounded retry).
            # A cleanup failure fails the index step rather than silently leaving
            # scratch behind.
            try:
                remove_scratch_path(chunk_dir)
            except OSError as exc:
                console.print(f"[red]Failed to clean up index scratch for {dataset}: {exc}[/red]")
                return False

            console.print(f"[green]Successfully created {total_entries:,} total index entries for {dataset}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error indexing dataset {dataset}: {str(e)}[/red]")
            if 'chunk_dir' in locals() and chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)
            return False

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indexer.close()

    def download_all_datasets(self, release_id: str = 'latest'):
        """Download all datasets first, then index them all."""
        if release_id == 'latest' and self.version:
            release_id = self.version
        elif release_id == 'latest':
            release_id = self.get_latest_release()
        
        console.print(f"[bold cyan]Downloading all datasets for release {release_id}...[/bold cyan]")
        
        # First download all datasets without indexing
        for dataset in self.datasets_to_download:
            console.print(f"\n[bold]Downloading {dataset}...[/bold]")
            self.download_dataset(dataset, release_id, index=False)
        
        # Then index all datasets
        console.print(f"\n[bold cyan]Indexing all datasets...[/bold cyan]")
        for dataset in self.datasets_to_download:
            console.print(f"\n[bold]Indexing {dataset}...[/bold]")
            self.index_dataset(dataset, release_id)

    def verify_downloads(self) -> bool:
        """Verify that all datasets were downloaded correctly."""
        try:
            release_id = self.get_latest_release()
            missing_files = {}
            
            for dataset in self.datasets_to_download:
                dataset_info = self.get_dataset_info(dataset, release_id)
                if not dataset_info:
                    continue
                    
                dataset_dir = self.base_dir / release_id / dataset
                if not dataset_dir.exists():
                    console.print(f"[red]Missing dataset directory: {dataset}[/red]")
                    continue
                
                # Get expected files
                if dataset in S2ORC_DATASETS:
                    expected_files = [
                        file_info['shard']
                        for file_info in dataset_info['files']
                    ]
                else:
                    expected_files = [
                        self.get_filename_from_url(url).replace('.gz', '')
                        for url in dataset_info['files']
                    ]
                
                # Get actual files
                actual_files = {
                    f.stem for f in dataset_dir.glob('*.json') 
                    if f.name != 'metadata.json'
                }
                
                # Find missing files
                missing = set(expected_files) - actual_files
                if missing:
                    missing_files[dataset] = missing
                    
            if missing_files:
                console.print("\nVerifying downloads for release {}...".format(release_id))
                for dataset, missing in missing_files.items():
                    console.print(f"Missing files in {dataset}: {missing}")
                return False
                
            return True
            
        except Exception as e:
            console.print(f"[red]Error verifying downloads: {str(e)}[/red]")
            raise

    def update_datasets(self) -> bool:
        """Incrementally update all datasets to the latest release using the official
        diff end-points provided by Semantic Scholar.  If the diff API is not
        available (or something goes wrong mid-way) we gracefully fall back to
        downloading the full dataset for the new release – this guarantees that
        Valsci will continue to work even on the first update attempt.

        The high-level algorithm is:

        1.  Identify the *current* (local) release as well as the *latest*
            release available from the API.
        2.  For each dataset:
            a.  Copy the existing JSONL shard files from the current release to
                a new directory for the latest release.  Hard-links are used on
                filesystems that support them to save disk space; otherwise a
                normal copy is performed.
            b.  Download every «update» diff file and place it in the new
                directory.
            c.  Collect the primary-key values that appear in «delete» diff
                files so we can exclude them when (re)building the binary index
                for the new release.
        3.  Re-index the dataset (calling the enhanced index_dataset method
            which accepts a *skip_ids* set).
        4.  Run a quick verification step to ensure the rebuilt index is
            internally consistent with the source files.

        Returns True on success, False otherwise.
        """

        try:
            current_release = self._get_latest_local_release()
            if not current_release:
                console.print("[yellow]No local datasets found. Please run an initial download first.[/yellow]")
                return False

            latest_release = self.get_latest_release()
            if current_release == latest_release:
                console.print("[green]Datasets already at the latest release – nothing to do.[/green]")
                return True

            console.print(f"[cyan]Updating from {current_release} ➜ {latest_release} using diffs...[/cyan]")

            for dataset in self.datasets_to_download:
                console.print(f"\n[bold]Dataset: {dataset}[/bold]")

                # We will lazily create/populate the destination directory *only*
                # if there are any diff chunks to apply.  This avoids copying a
                # potentially huge dataset when we subsequently fall back to a
                # full download.
                dst_dir: Optional[Path] = None
                deletion_ids: Set[str] = set()

                # ------------------------------------------------------------------
                # Retrieve diff manifest for this dataset
                # ------------------------------------------------------------------
                if dataset in S2ORC_DATASETS:
                    diff_url = f"{BASE_URL}/diffs/{current_release}/to/{latest_release}/{dataset}/"
                else:
                    diff_url = f"{BASE_URL}/diffs/{current_release}/to/{latest_release}/{dataset}"

                try:
                    diffs_resp = self.make_request(diff_url)
                    diffs = diffs_resp.json().get('diffs', [])
                except Exception as e:
                    console.print(f"[yellow]Could not retrieve diff for {dataset} (reason: {e}). Falling back to full download.[/yellow]")
                    # Full download fall-back
                    self.download_dataset(dataset, latest_release, index=True)
                    continue

                total_diff_files = sum(len(d.get('update_files', [])) + len(d.get('delete_files', [])) for d in diffs)
                if total_diff_files == 0:
                    console.print("[green]No changes for this dataset – skipping.[/green]")
                    # Even if there are no changes we still need an index file
                    # for the new release so we copy the existing one.
                    for field, id_type in self.dataset_id_fields[dataset]:
                        old_idx = self.index_dir / f"{current_release}_{dataset}_{id_type}.idx"
                        new_idx = self.index_dir / f"{latest_release}_{dataset}_{id_type}.idx"
                        if old_idx.exists() and not new_idx.exists():
                            shutil.copy2(old_idx, new_idx)
                    continue

                # At this point we know there are some diff files – now we can
                # prepare the destination directory by copying (or hard-linking)
                # the current release.
                dst_dir = self._prepare_dataset_dir_for_update(current_release, latest_release, dataset)

                with Progress() as progress:
                    task_id = progress.add_task(f"[cyan]Applying diffs for {dataset}...", total=total_diff_files)

                    # Loop through every diff segment chronologically.
                    for diff in diffs:
                        # ---------------------------
                        #   UPDATE  files
                        # ---------------------------
                        upd_urls = diff.get('update_files', [])
                        if dataset in S2ORC_DATASETS:
                            upd_urls = [u['url'] for u in upd_urls]

                        for url in upd_urls:
                            success, _ = self.download_file(url, dst_dir)
                            if not success:
                                console.print(f"[red]Failed to download update file: {url}[/red]")
                            progress.advance(task_id)

                        # ---------------------------
                        #   DELETE files
                        # ---------------------------
                        del_urls = diff.get('delete_files', [])
                        if dataset in S2ORC_DATASETS:
                            del_urls = [u['url'] for u in del_urls]

                        for url in del_urls:
                            success, del_path = self.download_file(url, dst_dir)
                            if success and del_path:
                                # Extract primary keys that need to be removed
                                try:
                                    with open(del_path, 'rb') as f_del:
                                        for line in f_del:
                                            try:
                                                decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                                                rec = json.loads(decoded)
                                            except Exception:
                                                try:
                                                    rec = json.loads(line.strip())
                                                except Exception:
                                                    continue

                                            for field, _ in self.dataset_id_fields[dataset]:
                                                if field in rec:
                                                    deletion_ids.add(str(rec[field]))
                                finally:
                                    # Delete the diff delete file – we only
                                    # needed it to collect the IDs.
                                    del_path.unlink(missing_ok=True)
                            progress.advance(task_id)

                # After all diff chunks processed, rebuild the binary indices
                console.print("[cyan]Re-building binary index...[/cyan]")
                if not self.index_dataset(dataset, latest_release, skip_ids=deletion_ids):
                    console.print(f"[red]Failed to rebuild index for {dataset}")
                    return False

                # Quick verification (10% margin).
                self.verify_index_completeness(latest_release, dataset, quick_estimate=True)

            console.print("\n[green]All datasets updated successfully![/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error during incremental update: {e}")
            return False

    def _prepare_dataset_dir_for_update(self, current_release: str, latest_release: str, dataset: str) -> Path:
        """Helper that copies (or hard-links) the current-release dataset files
        into a fresh directory for the *latest_release*.  Returns the new
        directory path.  If the directory already exists it is returned as-is.
        """

        src_dir = self.base_dir / current_release / dataset
        dst_dir = self.base_dir / latest_release / dataset

        if dst_dir.exists():
            return dst_dir

        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            # This should not happen but we guard against it.
            return dst_dir

        for file in src_dir.glob("*.json"):
            # Skip metadata.json here; we'll handle it explicitly below
            if file.name == "metadata.json":
                continue

            dest = dst_dir / file.name
            if dest.exists():
                # File (or hard-link) already in place
                continue
            try:
                os.link(file, dest)  # Hard-link (O(1) if same filesystem)
            except Exception:
                shutil.copy2(file, dest)  # Fall-back to regular copy

        # Copy metadata.json once (if it isn't already present)
        meta_src = src_dir / "metadata.json"
        meta_dst = dst_dir / "metadata.json"
        if meta_src.exists() and not meta_dst.exists():
            try:
                os.link(meta_src, meta_dst)
            except Exception:
                shutil.copy2(meta_src, meta_dst)

        return dst_dir

    def _get_latest_local_release(self) -> Optional[str]:
        """Get the latest release ID from local datasets directory."""
        if not self.base_dir.exists():
            return None
        
        # Get all subdirectories that look like release IDs (YYYY-MM-DD)
        releases = [
            d.name for d in self.base_dir.iterdir() 
            if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', d.name)
        ]
        
        if not releases:
            return None
        
        # Sort by date and return the latest
        return sorted(releases)[-1]

    def audit_datasets(self, release_id: str = 'latest', datasets: List[str] = None):
        """Audit dataset files and indexing status."""
        if release_id == 'latest':
            release_id = self.get_latest_release()
        
        console.print(f"\n[bold cyan]Auditing datasets for release {release_id}...[/bold cyan]")
        
        # Use provided datasets or all datasets
        datasets = datasets or self.datasets_to_download
        
        # Get index stats from binary indexer
        index_stats = self.indexer.get_index_stats(release_id)
        if not index_stats:
            console.print("[red]No index data found for this release[/red]")
            return
        
        table = Table(
            "Dataset", 
            "Expected Files", 
            "Downloaded Files",
            "Index Status",
            title=f"Dataset Audit for Release {release_id}"
        )
        
        for dataset in datasets:
            try:
                # Get expected files from API
                dataset_info = self.get_dataset_info(dataset, release_id)
                if not dataset_info:
                    table.add_row(
                        dataset,
                        "?",
                        "0",
                        "[red]Cannot fetch dataset info[/red]"
                    )
                    continue
                
                expected_count = len(dataset_info['files'])
                
                # Get downloaded files
                dataset_dir = self.base_dir / release_id / dataset
                if dataset in S2ORC_DATASETS:
                    downloaded_files = list(dataset_dir.glob("*.json")) if dataset_dir.exists() else []
                else:
                    downloaded_files = [
                        f for f in dataset_dir.glob("*.json")
                        if f.name != 'metadata.json'
                    ] if dataset_dir.exists() else []
                
                # Get index status for all ID types for this dataset
                index_statuses = []
                for field_name, id_type in self.dataset_id_fields[dataset]:
                    index_key = f"{dataset}_{id_type}"
                    if index_key in index_stats:
                        index_statuses.append(
                            f"{id_type}: {index_stats[index_key]['entry_count']:,}"
                        )
                
                if index_statuses:
                    index_status = "[green]" + "\n".join(index_statuses) + "[/green]"
                else:
                    index_status = "[yellow]Not indexed[/yellow]"
                
                table.add_row(
                    dataset,
                    str(expected_count),
                    str(len(downloaded_files)),
                    index_status
                )
                
            except Exception as e:
                table.add_row(
                    dataset,
                    "?",
                    "?",
                    f"[red]Error: {str(e)}[/red]"
                )
        
        console.print(table)

    def count_indices(self, release_id: str = 'latest'):
        """Print detailed index counts for each file."""
        if release_id == 'latest':
            release_id = self.get_latest_release()
        
        console.print(f"\n[bold cyan]Index counts for release {release_id}...[/bold cyan]")
        
        # Get index stats from binary indexer
        stats = self.indexer.get_index_stats(release_id)
        if not stats:
            console.print("[red]No index data found for this release[/red]")
            return
        
        table = Table(
            "Dataset",
            "ID Type",
            "Entries",
            "Size",
            "Created",
            "Status",
            title=f"Index Statistics for Release {release_id}"
        )
        
        total_entries = 0
        for index_key, info in stats.items():
            parsed = self.indexer._parse_index_key(index_key)
            if not parsed:
                continue
            dataset, id_type = parsed
            total_entries += info['entry_count']
            
            status = "[green]Healthy[/green]" if info['healthy'] else "[red]Unhealthy[/red]"
            
            table.add_row(
                dataset,
                id_type,
                f"{info['entry_count']:,}",
                f"{info['size_mb']:.1f} MB",
                info['created'],
                status
            )
        
        console.print(f"\nTotal index entries: [bold cyan]{total_entries:,}[/bold cyan]\n")
        console.print(table)

    def _verify_db_name(self):
        """This method is no longer needed with binary indexer"""
        pass

    def verify_index_completeness(self, release_id: str, dataset: Optional[str] = None, 
                                sample_size: int = 1000, quick_estimate: bool = False) -> bool:
        """Thin wrapper around BinaryIndexer.verify_index_completeness so callers
        don't need to import the indexer directly.
        """
        try:
            return self.indexer.verify_index_completeness(release_id, dataset, quick_estimate=quick_estimate)
        except Exception as e:
            console.print(f"[red]Error verifying index completeness: {str(e)}[/red]")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download Semantic Scholar datasets')
    parser.add_argument('--release', default='latest', help='Release ID to download')
    parser.add_argument('--version', help='Specific version to download (YYYY-MM-DD format)')
    parser.add_argument(
        '--mini',
        action='store_true',
        help='Build the curated manifest-driven mini corpus. Requires a local mini manifest.',
    )
    parser.add_argument(
        '--mini-manifest',
        default=None,
        help=(
            'Filename (no path) of the curated manifest in semantic_scholar/manifests/ '
            'used by --mini. Defaults to the SEMANTIC_SCHOLAR_MANIFEST setting.'
        ),
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        help='Limit download, verify, or audit operations to specific datasets.',
    )
    parser.add_argument('--verify', action='store_true', help='Verify downloaded datasets')
    parser.add_argument('--verify-index', nargs='*', help='Verify index completeness. Optionally specify datasets to verify')
    parser.add_argument('--audit', nargs='*', help='Audit datasets and indexing status')
    parser.add_argument('--index-only', nargs='*', help='Only run indexing on downloaded files')
    parser.add_argument('--repair', action='store_true', help='Repair/resume incomplete indexes')
    parser.add_argument('--count', action='store_true', help='Show detailed index counts for each file')
    # New flag to perform an incremental update based on diff end-points
    parser.add_argument('--update', action='store_true', help='Incrementally update all local datasets to the latest release using Semantic Scholar diff files')
    args = parser.parse_args()
    
    with S2DatasetDownloader(version=args.version) as downloader:
        def validate_datasets(dataset_list):
            """Helper function to validate dataset names and return filtered list"""
            if not dataset_list:
                return downloader.datasets_to_download
                
            supported_datasets = getattr(downloader, "supported_datasets", downloader.datasets_to_download)
            invalid_datasets = [d for d in dataset_list if d not in supported_datasets]
            if invalid_datasets:
                console.print(f"[red]Invalid dataset names: {', '.join(invalid_datasets)}[/red]")
                console.print(f"[yellow]Valid datasets are: {', '.join(supported_datasets)}[/yellow]")
                return None
            return dataset_list

        def prompt_yes_no(message: str, default: str = "n") -> str:
            """Prompt safely; default in non-interactive runs."""
            console.print(message)
            if not sys.stdin or not sys.stdin.isatty():
                return default.lower()
            try:
                response = input().strip().lower()
            except EOFError:
                return default.lower()
            return response or default.lower()

        if args.mini and not any([
            args.verify,
            args.verify_index is not None,
            args.audit is not None,
            args.index_only is not None,
            args.repair,
            args.count,
            args.update,
        ]):
            try:
                manifest_path = resolve_manifest_path(args.mini_manifest)
                downloader.materialize_mini_corpus(manifest_path)
            except MiniCorpusManifestError as exc:
                console.print(f"[red]{exc}[/red]")
                return False
            except Exception as exc:
                console.print(f"[red]Mini corpus build failed: {exc}[/red]")
                return False
            return True

        if args.verify:
            # Verify downloaded files match expected files from API
            if args.mini:
                try:
                    manifest_path = resolve_manifest_path(args.mini_manifest)
                    return downloader.verify_mini_corpus(manifest_path)
                except MiniCorpusManifestError as exc:
                    console.print(f"[red]{exc}[/red]")
                    return False

            release_id = args.release
            if release_id == 'latest':
                release_id = downloader.get_latest_release()
            
            console.print(f"\n[bold cyan]Verifying downloads for release {release_id}...[/bold cyan]")
            missing_files = {}
            verify_errors = False

            datasets = validate_datasets(args.datasets)
            if datasets is None:
                return False

            for dataset in datasets:
                try:
                    dataset_info = downloader.get_dataset_info(dataset, release_id)
                    if not dataset_info:
                        console.print(f"[yellow]Could not get info for dataset: {dataset}[/yellow]")
                        continue

                    dataset_dir = downloader.base_dir / release_id / dataset
                    if not dataset_dir.exists():
                        missing_files[dataset] = ["entire dataset missing"]
                        continue

                    # Get expected files
                    if dataset in S2ORC_DATASETS:
                        expected_files = [
                            f"{info['shard']}.json"
                            for info in dataset_info['files']
                        ]
                    else:
                        expected_files = [
                            downloader.get_filename_from_url(url).replace('.gz', '.json')
                            for url in dataset_info['files']
                        ]

                    # Check actual files
                    actual_files = {f.name for f in dataset_dir.glob('*.json') if f.name != 'metadata.json'}
                    missing = set(expected_files) - actual_files
                    if missing:
                        missing_files[dataset] = missing

                except Exception as e:
                    console.print(f"[red]Error verifying {dataset}: {str(e)}[/red]")
                    verify_errors = True

            if missing_files or verify_errors:
                if missing_files:
                    console.print("\n[red]Missing files found:[/red]")
                    for dataset, files in missing_files.items():
                        console.print(f"\n[yellow]{dataset}:[/yellow]")
                        for f in files:
                            console.print(f"  • {f}")
                if verify_errors:
                    console.print("\n[red]Verification encountered errors (see above).[/red]")
                return False
            else:
                console.print("\n[green]All expected files are present![/green]")
                return True
                
        elif args.count:
            # Show index statistics
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()
            if not release_id:
                console.print("[red]No local releases found[/red]")
                return False

            stats = downloader.indexer.get_index_stats(release_id)

            # Create a rich table to display stats
            table = Table(title=f"Index Statistics for Release {release_id}")
            table.add_column("Dataset")
            table.add_column("ID Type")
            table.add_column("Entries")
            table.add_column("Size")
            table.add_column("Created")
            table.add_column("Status")
            
            for index_name, info in stats.items():
                parsed = downloader.indexer._parse_index_key(index_name)
                if not parsed:
                    continue
                dataset, id_type = parsed
                status = "[green]Healthy[/green]" if info['healthy'] else "[red]Unhealthy[/red]"
                table.add_row(
                    dataset,
                    id_type,
                    f"{info['entry_count']:,}",
                    f"{info['size_mb']:.1f} MB",
                    info['created'],
                    status
                )
            
            console.print(table)
            
        elif args.audit is not None:
            datasets = validate_datasets(args.audit or args.datasets)
            if datasets is None:
                return False

            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()
            if not release_id:
                console.print("[red]No local releases found[/red]")
                return False

            # Create audit table
            table = Table(title=f"Dataset Audit for Release {release_id}")
            table.add_column("Dataset")
            table.add_column("Files")
            table.add_column("Index Status")
            table.add_column("Health Check")
            
            for dataset in datasets:
                # Check dataset files
                dataset_dir = downloader.base_dir / release_id / dataset
                if not dataset_dir.exists():
                    table.add_row(dataset, "[red]Missing[/red]", "N/A", "N/A")
                    continue
                    
                files = list(dataset_dir.glob("*.json"))
                file_count = len([f for f in files if f.name != 'metadata.json'])
                
                # Get index stats
                stats = downloader.indexer.get_index_stats(release_id)
                index_infos = []
                for index_key, info in stats.items():
                    parsed = downloader.indexer._parse_index_key(index_key)
                    if parsed and parsed[0] == dataset:
                        index_infos.append(info)
                
                if not index_infos:
                    table.add_row(
                        dataset,
                        f"{file_count} files",
                        "[yellow]Not Indexed[/yellow]",
                        "N/A"
                    )
                else:
                    health = "[green]Healthy[/green]" if all(info['healthy'] for info in index_infos) else "[red]Unhealthy[/red]"
                    entry_count = sum(info['entry_count'] for info in index_infos)
                    table.add_row(
                        dataset,
                        f"{file_count} files",
                        f"{entry_count:,} entries",
                        health
                    )
            
            console.print(table)
            
        elif args.verify_index is not None:
            datasets = validate_datasets(args.verify_index)
            if datasets is None:
                return False

            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()
            if not release_id:
                console.print("[red]No local releases found[/red]")
                return False

            console.print(f"[cyan]Verifying indices for {len(datasets)} datasets...[/cyan]")

            # First do a quick estimate check
            console.print(f"\n[bold]Quick estimation check...[/bold]")
            if downloader.verify_index_completeness(release_id, quick_estimate=True):
                console.print(f"[green]OK Quick estimate check passed[/green]")

                # If quick check passes, offer to do detailed verification
                response = prompt_yes_no("\nQuick check passed. Would you like to perform a detailed verification? (y/N)")
                if response == 'y':
                    console.print(f"\n[bold]Performing detailed verification...[/bold]")
                    if downloader.indexer.verify_all_indices(release_id, show_details=True):
                        console.print(f"[green]OK All indices verified successfully[/green]")
                        return True
                    console.print(f"[red]FAIL Indices verification failed[/red]")
                    return False
                return True
            else:
                console.print(f"[red]FAIL Quick estimate check failed[/red]")
                response = prompt_yes_no("\nWould you like to perform a detailed verification to identify issues? (y/N)")
                if response == 'y':
                    console.print(f"\n[bold]Performing detailed verification...[/bold]")
                    downloader.indexer.verify_all_indices(release_id, show_details=True)
                # A failed quick estimate is a verification failure regardless of
                # whether the optional detailed pass was run.
                return False

        elif args.update:
            # Perform an incremental update (diff-based). This downloads only the changes
            # between the current local release and the latest available release, then
            # rebuilds the binary indices.  Falls back to a full download if diffs are
            # unavailable.
            success = downloader.update_datasets()
            # Exit code indicates success (0) or failure (1) for shell scripts
            sys.exit(0 if success else 1)
        elif args.index_only is not None:
            datasets = validate_datasets(args.index_only or args.datasets)
            if datasets is None:
                return False

            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()
            if not release_id:
                console.print("[red]No local releases found[/red]")
                return False

            console.print(f"[cyan]Indexing {len(datasets)} datasets...[/cyan]")

            index_failed = False
            for dataset in datasets:
                console.print(f"\n[bold]Indexing {dataset}...[/bold]")
                if downloader.index_dataset(dataset, release_id):
                    console.print(f"[green]OK Successfully indexed {dataset}[/green]")
                else:
                    console.print(f"[red]FAIL Failed to index {dataset}[/red]")
                    index_failed = True
            if index_failed:
                return False
            return True

        elif args.repair:
            # Repair mode: re-index datasets that are missing or unhealthy for the latest local release
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()

            if not release_id:
                console.print("[red]No local releases found to repair[/red]")
                return False

            console.print(f"[bold cyan]Repairing indices for release {release_id}...[/bold cyan]")

            # Get current index stats (may be empty)
            stats = downloader.indexer.get_index_stats(release_id)

            repair_failed = False
            for dataset in downloader.datasets_to_download:
                # Determine for every id_type if index exists and is healthy
                needs_rebuild = False
                for _, id_type in downloader.dataset_id_fields[dataset]:
                    key = f"{dataset}_{id_type}"
                    if key not in stats:
                        needs_rebuild = True
                        break
                    if not stats[key]['healthy']:
                        needs_rebuild = True
                        break

                if needs_rebuild:
                    console.print(f"\n[bold]Re-indexing {dataset}...[/bold]")
                    if downloader.index_dataset(dataset, release_id):
                        console.print(f"[green]OK Successfully re-indexed {dataset}[/green]")
                    else:
                        console.print(f"[red]FAIL Failed to re-index {dataset}[/red]")
                        repair_failed = True

            console.print("\n[bold cyan]Repair completed[/bold cyan]")
            if repair_failed:
                return False
            return True

        else:
            # Download and index all datasets
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader.get_latest_release()

            console.print(f"[bold cyan]Downloading and indexing datasets for release {release_id}...[/bold cyan]")

            datasets = validate_datasets(args.datasets)
            if datasets is None:
                return False

            download_failed = False
            for dataset in datasets:
                console.print(f"\n[bold]Processing {dataset}...[/bold]")
                if downloader.download_dataset(dataset, release_id, index=True):
                    console.print(f"[green]OK Successfully processed {dataset}[/green]")
                else:
                    console.print(f"[red]FAIL Failed to process {dataset}[/red]")
                    download_failed = True
            if download_failed:
                return False
            return True

if __name__ == "__main__":
    result = main()
    if result is False:
        sys.exit(1)
