import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from semantic_scholar.utils import downloader as downloader_module
from semantic_scholar.utils.binary_indexer import BinaryIndexer
from semantic_scholar.utils.downloader import S2DatasetDownloader

ROOT = Path(__file__).resolve().parents[1]


def _build_downloader(tmp_path: Path) -> S2DatasetDownloader:
    downloader = S2DatasetDownloader.__new__(S2DatasetDownloader)
    downloader.base_dir = tmp_path / "datasets"
    downloader.base_dir.mkdir(parents=True, exist_ok=True)
    downloader.index_dir = downloader.base_dir / "binary_indices"
    downloader.index_dir.mkdir(parents=True, exist_ok=True)
    downloader.work_dir = tmp_path / "data_work"
    downloader.work_dir.mkdir(parents=True, exist_ok=True)
    downloader.indexer = BinaryIndexer(downloader.base_dir, work_dir=downloader.work_dir / "index_tmp")
    downloader.datasets_to_download = list(downloader_module.DEFAULT_DATASETS)
    downloader.supported_datasets = list(downloader_module.SUPPORTED_DATASETS)
    downloader.dataset_id_fields = {
        "papers": [("corpusid", "corpus_id")],
        "abstracts": [("corpusid", "corpus_id")],
        "s2orc_v2": [("corpusid", "corpus_id")],
        "authors": [("authorid", "author_id")],
        "tldrs": [("corpusid", "corpus_id")],
    }
    return downloader


def _write_papers(downloader: S2DatasetDownloader, release_id: str):
    dataset_dir = downloader.base_dir / release_id / "papers"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "papers.json").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"corpusid": 101, "paperid": "p-101", "title": "One"},
                {"corpusid": 202, "paperid": "p-202", "title": "Two"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_index_dataset_returns_false_when_housekeeping_fails(tmp_path, monkeypatch):
    downloader = _build_downloader(tmp_path)
    release_id = "2026-05-26"
    _write_papers(downloader, release_id)

    def boom(path, *args, **kwargs):
        raise OSError("scratch is locked")

    # Simulate a non-transient housekeeping failure after indices are built.
    monkeypatch.setattr(downloader_module, "remove_scratch_path", boom)

    result = downloader.index_dataset("papers", release_id)
    assert result is False
    downloader.indexer.close()


def test_index_dataset_succeeds_and_cleans_scratch(tmp_path):
    downloader = _build_downloader(tmp_path)
    release_id = "2026-05-26"
    _write_papers(downloader, release_id)

    assert downloader.index_dataset("papers", release_id) is True
    # The chunk scratch dir is cleaned up explicitly (job-scoped).
    chunk_dir = downloader.indexer.tmp_dir / f"{release_id}_papers_chunks"
    assert not chunk_dir.exists()
    downloader.indexer.close()


def test_download_dataset_returns_false_on_download_failure(tmp_path, monkeypatch):
    downloader = _build_downloader(tmp_path)
    downloader.api_key = ""
    release_id = "2026-05-26"

    monkeypatch.setattr(
        downloader,
        "get_dataset_info",
        lambda dataset, rid: {"files": ["https://example.org/papers.gz"]},
    )
    monkeypatch.setattr(downloader, "get_filename_from_url", lambda url: "papers.gz")
    # Every file download fails.
    monkeypatch.setattr(downloader, "download_file", lambda *a, **k: (False, None))

    result = downloader.download_dataset("papers", release_id, index=False)
    assert result is False
    downloader.indexer.close()


def test_remove_scratch_path_removes_dir_and_tolerates_missing(tmp_path):
    from semantic_scholar.utils.binary_indexer import remove_scratch_path

    scratch = tmp_path / "scratch" / "sub"
    scratch.mkdir(parents=True)
    (scratch / "f.txt").write_text("x", encoding="utf-8")
    remove_scratch_path(tmp_path / "scratch")
    assert not (tmp_path / "scratch").exists()
    # A missing path is a no-op (idempotent cleanup).
    remove_scratch_path(tmp_path / "scratch")


def test_remove_scratch_path_retries_once_then_raises(tmp_path, monkeypatch):
    import pytest

    import semantic_scholar.utils.binary_indexer as bi

    locked = tmp_path / "locked"
    locked.mkdir()
    calls = {"n": 0}

    def fake_rmtree(path):
        calls["n"] += 1
        raise OSError("locked")

    monkeypatch.setattr(bi.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(bi.time, "sleep", lambda *_a, **_k: None)

    with pytest.raises(OSError):
        bi.remove_scratch_path(locked, attempts=2, delay=0)
    # One retry => two total attempts.
    assert calls["n"] == 2


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "semantic_scholar.utils.downloader", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_cli_path_like_manifest_exits_nonzero():
    proc = _run_cli("--mini", "--mini-manifest", "../escape.json")
    assert proc.returncode != 0


def test_cli_index_only_missing_release_exits_nonzero():
    proc = _run_cli("--index-only", "papers", "--release", "1999-01-01")
    assert proc.returncode != 0


def _has_local_mini_release() -> bool:
    # The CLI verifies against datasets under <project>/semantic_scholar/datasets,
    # which is gitignored. On a machine where a mini corpus has been built, that
    # release exists and `--mini --verify` legitimately succeeds — so this test
    # (which covers the no-local-release case) only applies on a fresh checkout/CI.
    datasets = ROOT / "semantic_scholar" / "datasets"
    if not datasets.is_dir():
        return False
    return any(child.is_dir() and "mini" in child.name for child in datasets.iterdir())


@pytest.mark.skipif(
    _has_local_mini_release(),
    reason="A local mini corpus is present, so '--mini --verify' correctly succeeds; "
    "this test covers the no-local-release case (fresh checkout / CI).",
)
def test_cli_verify_mini_without_local_release_exits_nonzero():
    proc = _run_cli("--mini", "--verify")
    assert proc.returncode != 0
