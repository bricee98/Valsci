import json
import sys
import types
from pathlib import Path


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from semantic_scholar.utils import downloader as downloader_module
from semantic_scholar.utils.binary_indexer import BinaryIndexer
from semantic_scholar.utils.downloader import MiniCorpusManifestError, S2DatasetDownloader
from semantic_scholar.utils.searcher import S2Searcher


def _build_downloader(tmp_path: Path) -> S2DatasetDownloader:
    downloader = S2DatasetDownloader.__new__(S2DatasetDownloader)
    downloader.base_dir = tmp_path / "datasets"
    downloader.base_dir.mkdir(parents=True, exist_ok=True)
    downloader.index_dir = downloader.base_dir / "binary_indices"
    downloader.index_dir.mkdir(parents=True, exist_ok=True)
    # Keep scratch/build work isolated under the test's tmp dir.
    downloader.work_dir = tmp_path / "data_work"
    downloader.work_dir.mkdir(parents=True, exist_ok=True)
    downloader.indexer = BinaryIndexer(downloader.base_dir, work_dir=downloader.work_dir / "index_tmp")
    return downloader


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_mini_manifest_is_required(tmp_path):
    downloader = _build_downloader(tmp_path)

    missing_manifest = tmp_path / "mini" / "manifests" / "mendelian_v1.json"

    try:
        downloader.materialize_mini_corpus(missing_manifest)
    except MiniCorpusManifestError as exc:
        assert "Curated mini-corpus manifest not found" in str(exc)
        assert str(missing_manifest) in str(exc)
    else:
        raise AssertionError("Expected missing mini manifest to fail")


def test_full_downloader_defaults_to_s2orc_v2_without_legacy_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader_module, "project_root", str(tmp_path))

    downloader = S2DatasetDownloader()
    try:
        assert downloader.datasets_to_download == [
            "papers",
            "abstracts",
            "authors",
            "s2orc_v2",
            "tldrs",
        ]
        assert downloader.supported_datasets == downloader.datasets_to_download
        assert downloader.dataset_id_fields["papers"] == [("corpusid", "corpus_id")]
        assert downloader._normalize_dataset_file_entry(
            "s2orc_v2",
            "https://example.org/releases/2026-05-26/s2orc_v2/00042.gz?signature=abc",
        ) == {
            "url": "https://example.org/releases/2026-05-26/s2orc_v2/00042.gz?signature=abc",
            "shard": "00042",
        }
    finally:
        downloader.indexer.close()


def test_manifest_filters_records_and_builds_indices(tmp_path):
    downloader = _build_downloader(tmp_path)
    source_dir = tmp_path / "sources"

    _write_jsonl(
        source_dir / "papers.jsonl",
        [
            {"corpusid": 101, "paperid": "paper-101", "title": "Keep 101", "citationcount": 7},
            {"corpusid": 999, "paperid": "paper-999", "title": "Drop 999", "citationcount": 1},
        ],
    )
    _write_jsonl(
        source_dir / "authors.jsonl",
        [
            {"authorid": "a1", "name": "Kept Author", "hindex": 42},
            {"authorid": "a2", "name": "Dropped Author", "hindex": 2},
        ],
    )
    _write_jsonl(
        source_dir / "s2orc_v2.jsonl",
        [
            {"corpusid": 101, "body": {"text": "S2ORC v2 full text for paper 101."}},
            {"corpusid": 999, "body": {"text": "Dropped full text."}},
        ],
    )
    _write_jsonl(
        source_dir / "abstracts.jsonl",
        [{"corpusid": 303, "abstract": "Abstract text for paper 303."}],
    )
    _write_jsonl(
        source_dir / "tldrs.jsonl",
        [{"corpusid": 404, "text": "TLDR text for paper 404."}],
    )

    manifest = {
        "release_id": "2026-05-26",
        "mini_release_id": "2026-05-26-mini-mendelian-disease-v1",
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "Pathogenic variants in PAH cause phenylketonuria.",
                "corpus_ids": [101],
            }
        ],
        "source_files": {
            "papers": [str(source_dir / "papers.jsonl")],
            "authors": [str(source_dir / "authors.jsonl")],
            "s2orc_v2": [str(source_dir / "s2orc_v2.jsonl")],
            "abstracts": [str(source_dir / "abstracts.jsonl")],
            "tldrs": [str(source_dir / "tldrs.jsonl")],
        },
        "datasets": {
            "papers": {"corpus_ids": [101]},
            "authors": {"author_ids": ["a1"]},
            "s2orc_v2": {"corpus_ids": [101]},
            "abstracts": {"corpus_ids": [303]},
            "tldrs": {"corpus_ids": [404]},
        },
    }
    manifest_path = tmp_path / "manifests" / "mendelian_v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    release_id = downloader.materialize_mini_corpus(manifest_path)

    assert release_id == "2026-05-26-mini-mendelian-disease-v1"
    release_dir = downloader.base_dir / release_id
    papers = [
        json.loads(line)
        for line in (release_dir / "papers" / "mini.json").read_text(encoding="utf-8").splitlines()
    ]
    authors = [
        json.loads(line)
        for line in (release_dir / "authors" / "mini.json").read_text(encoding="utf-8").splitlines()
    ]
    assert [paper["corpusid"] for paper in papers] == [101]
    assert [author["authorid"] for author in authors] == ["a1"]

    indexer = BinaryIndexer(downloader.base_dir)
    try:
        assert indexer.lookup(release_id, "papers", "corpus_id", "101")["title"] == "Keep 101"
        assert indexer.lookup(release_id, "papers", "corpus_id", "999") is None
        assert indexer.lookup(release_id, "authors", "author_id", "a1")["hindex"] == 42
        stats = indexer.get_index_stats(release_id)
        assert "s2orc_v2_corpus_id" in stats
        assert stats["s2orc_v2_corpus_id"]["healthy"]
        assert indexer.verify_index_completeness(release_id, "s2orc_v2", quick_estimate=True)
    finally:
        indexer.close()

    fresh_indexer = BinaryIndexer(downloader.base_dir)
    try:
        assert fresh_indexer.verify_index_completeness(release_id, "s2orc_v2", quick_estimate=True)
    finally:
        fresh_indexer.close()

    summary = json.loads((release_dir / "mini_build_summary.json").read_text(encoding="utf-8"))
    assert summary["records_written"]["s2orc_v2"] == 1
    assert summary["records_requested"]["s2orc_v2"] == 1
    assert "s2orc_v2" not in summary["records_missing"]
    assert downloader.verify_mini_corpus(manifest_path)

    stale_manifest = dict(manifest)
    stale_manifest["datasets"] = dict(manifest["datasets"])
    stale_manifest["datasets"]["s2orc_v2"] = {"corpus_ids": [101, 202]}
    manifest_path.write_text(json.dumps(stale_manifest, indent=2), encoding="utf-8")
    assert not downloader.verify_mini_corpus(manifest_path)


def test_minimal_manifest_fetches_remote_sources_for_dataset_ids(tmp_path, monkeypatch):
    downloader = _build_downloader(tmp_path)

    manifest = {
        "release_id": "2026-05-26",
        "mini_release_id": "2026-05-26-mini-test",
        "datasets": {
            "papers": {"corpus_ids": [101]},
            "authors": {"author_ids": ["a1"]},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        downloader,
        "get_dataset_info",
        lambda dataset, release_id: {"files": [f"https://example.org/{dataset}.gz"]},
    )

    def fake_iter_source_lines(source):
        if source.endswith("papers.gz"):
            rows = [{"corpusid": 101, "title": "Fetched Paper"}]
        elif source.endswith("authors.gz"):
            rows = [{"authorid": "a1", "name": "Fetched Author"}]
        else:
            path = Path(source)
            if path.exists():
                yield from path.read_bytes().splitlines()
            return
        for row in rows:
            yield json.dumps(row).encode("utf-8")

    monkeypatch.setattr(downloader, "_iter_source_lines", fake_iter_source_lines)

    release_id = downloader.materialize_mini_corpus(manifest_path)

    assert release_id == "2026-05-26-mini-test"
    release_dir = downloader.base_dir / release_id
    papers = [
        json.loads(line)
        for line in (release_dir / "papers" / "mini.json").read_text(encoding="utf-8").splitlines()
    ]
    authors = [
        json.loads(line)
        for line in (release_dir / "authors" / "mini.json").read_text(encoding="utf-8").splitlines()
    ]
    assert papers == [{"corpusid": 101, "title": "Fetched Paper"}]
    assert authors == [{"authorid": "a1", "name": "Fetched Author"}]


def test_failed_mini_rebuild_preserves_existing_release(tmp_path, monkeypatch):
    downloader = _build_downloader(tmp_path)
    release_id = "2026-05-26-mini-test"
    existing_dir = downloader.base_dir / release_id / "papers"
    _write_jsonl(existing_dir / "mini.json", [{"corpusid": 1, "title": "Old release"}])

    manifest = {
        "release_id": "2026-05-26",
        "mini_release_id": release_id,
        "datasets": {
            "papers": {"corpus_ids": [2]},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fail_dataset_info(dataset, release):
        raise RuntimeError("remote scan failed")

    monkeypatch.setattr(downloader, "get_dataset_info", fail_dataset_info)

    try:
        downloader.materialize_mini_corpus(manifest_path)
    except RuntimeError as exc:
        assert "remote scan failed" in str(exc)
    else:
        raise AssertionError("Expected rebuild to fail")

    rows = [
        json.loads(line)
        for line in (existing_dir / "mini.json").read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{"corpusid": 1, "title": "Old release"}]


def test_searcher_reads_s2orc_v2_then_abstract_and_tldr_fallbacks(tmp_path):
    downloader = _build_downloader(tmp_path)
    source_dir = tmp_path / "sources"
    for dataset, row in {
        "s2orc_v2": {"corpusid": 101, "body": {"text": "S2ORC v2 text."}},
        "abstracts": {"corpusid": 303, "abstract": "Abstract fallback text."},
        "tldrs": {"corpusid": 404, "text": "TLDR fallback text."},
    }.items():
        _write_jsonl(source_dir / f"{dataset}.jsonl", [row])

    manifest = {
        "release_id": "2026-05-26",
        "source_files": {
            dataset: [str(source_dir / f"{dataset}.jsonl")]
            for dataset in ["s2orc_v2", "abstracts", "tldrs"]
        },
        "datasets": {
            "s2orc_v2": {"corpus_ids": [101]},
            "abstracts": {"corpus_ids": [303]},
            "tldrs": {"corpus_ids": [404]},
        },
    }
    manifest_path = tmp_path / "mendelian_v1.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    release_id = downloader.materialize_mini_corpus(manifest_path)

    searcher = S2Searcher.__new__(S2Searcher)
    searcher.current_release = release_id
    searcher.has_local_data = True
    searcher.indexer = BinaryIndexer(downloader.base_dir)
    try:
        result = searcher.get_paper_content("101")
        assert result["source"] == "s2orc_v2"
        assert result["text"] == "S2ORC v2 text."

        result = searcher.get_paper_content("303")
        assert result["source"] == "abstract"
        assert result["text"] == "Abstract fallback text."

        result = searcher.get_paper_content("404")
        assert result["source"] == "tldr"
        assert result["text"] == "TLDR fallback text."
    finally:
        searcher.indexer.close()
