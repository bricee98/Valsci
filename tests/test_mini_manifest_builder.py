import json
import sys
import types
from pathlib import Path


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from semantic_scholar.utils.downloader import S2DatasetDownloader
from semantic_scholar.utils.mini_manifest_builder import (
    build_manifest_payload,
    scan_dataset_sources,
    scan_source_extract,
    selected_ids,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_downloader(tmp_path: Path) -> S2DatasetDownloader:
    downloader = S2DatasetDownloader.__new__(S2DatasetDownloader)
    downloader.base_dir = tmp_path / "datasets"
    downloader.base_dir.mkdir(parents=True, exist_ok=True)
    return downloader


def test_selected_ids_normalizes_final_selection_ids():
    corpus_ids, author_ids = selected_ids(
        {
            "corpus_ids": [101, "202", None, ""],
            "author_ids": ["a1", 2, ""],
        }
    )

    assert corpus_ids == {"101", "202"}
    assert author_ids == {"a1", "2"}


def test_scan_dataset_sources_finds_only_matching_local_shards(tmp_path):
    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    _write_jsonl(source_a, [{"corpusid": 101}, {"corpusid": 999}])
    _write_jsonl(source_b, [{"corpusid": 202}])
    downloader = _build_downloader(tmp_path)

    result = scan_dataset_sources(
        downloader,
        dataset="papers",
        files=[str(source_a), str(source_b)],
        corpus_ids={"101", "202"},
        author_ids=set(),
        extract_output_path=tmp_path / "extracts" / "papers.jsonl",
    )

    assert result["matched_id_count"] == 2
    assert result["matched_source_count"] == 2
    assert result["records_written"] == 2
    assert result["missing_ids"] == []
    assert [source["matched_ids"] for source in result["sources"]] == [["101"], ["202"]]
    extracted = [
        json.loads(line)
        for line in Path(result["extract_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["corpusid"] for row in extracted] == [101, 202]


def test_build_manifest_payload_uses_matched_sources_only():
    final_selection = {
        "topic_label": "mendelian_disease_v1",
        "selection_policy": {"max_corpus_ids": 2},
        "corpus_ids": ["101", "202"],
        "author_ids": ["a1"],
    }
    claim_bank = {"claims": [{"claim_id": "MENDEL-001"}]}
    shard_map = {
        "datasets": {
            "papers": {
                "requested_id_count": 2,
                "matched_id_count": 1,
                "matched_source_count": 1,
                "records_written": 1,
                "extract_path": "/tmp/extracts/papers.jsonl",
                "missing_ids": ["202"],
                "sources": [
                    {"url": "https://example.org/a.gz", "matched_count": 1, "matched_ids": ["101"]},
                    {"url": "https://example.org/b.gz", "matched_count": 0, "matched_ids": []},
                ],
            }
        }
    }

    manifest = build_manifest_payload(
        final_selection=final_selection,
        claim_bank=claim_bank,
        release_id="2026-05-26",
        mini_release_id="2026-05-26-mini-mendelian-v1",
        shard_map=shard_map,
    )

    assert manifest["release_id"] == "2026-05-26"
    assert manifest["mini_release_id"] == "2026-05-26-mini-mendelian-v1"
    assert manifest == {
        "release_id": "2026-05-26",
        "mini_release_id": "2026-05-26-mini-mendelian-v1",
        "datasets": {
            "papers": {"corpus_ids": ["101"]},
        },
    }


def test_scan_source_extract_reuses_existing_rows(tmp_path):
    extract_path = tmp_path / "extracts" / "papers.jsonl"
    _write_jsonl(extract_path, [{"corpusid": 101}, {"corpusid": 999}])
    downloader = _build_downloader(tmp_path)

    result = scan_source_extract(
        downloader,
        dataset="papers",
        extract_path=extract_path,
        corpus_ids={"101", "202"},
        author_ids=set(),
    )

    assert result["matched_id_count"] == 1
    assert result["records_written"] == 2
    assert result["extract_path"] == str(extract_path)
    assert result["missing_ids"] == ["202"]
    assert result["sources"] == [
        {
            "path": str(extract_path),
            "matched_ids": ["101"],
            "matched_count": 1,
        }
    ]
