"""Build a local manifest for the curated Mendelian mini corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from rich.console import Console


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from semantic_scholar.utils.downloader import S2DatasetDownloader
from semantic_scholar.utils.mini_corpus_curator import utc_now


console = Console()

DEFAULT_FINAL_SELECTION_PATH = (
    project_root / "semantic_scholar/datasets/mini/curation/mendelian_v1_final_selection.json"
)
DEFAULT_CLAIM_BANK_PATH = (
    project_root / "semantic_scholar/datasets/mini/curation/mendelian_v1_claims.json"
)
DEFAULT_OUTPUT_MANIFEST_PATH = (
    project_root / "semantic_scholar/manifests/mendelian_v1.json"
)
DEFAULT_SHARD_MAP_PATH = (
    project_root / "semantic_scholar/datasets/mini/curation/mendelian_v1_shard_map.json"
)
DEFAULT_SOURCE_EXTRACT_DIR = (
    project_root / "semantic_scholar/datasets/mini/source_extracts/mendelian_v1"
)
DEFAULT_DATASETS = ["papers", "authors", "abstracts", "tldrs", "s2orc_v2"]
CORPUS_DATASETS = {"papers", "abstracts", "tldrs", "s2orc_v2"}


def console_safe(value: Any) -> str:
    return str(value).encode("ascii", errors="replace").decode("ascii")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def selected_ids(final_selection: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    corpus_ids = {
        str(value).strip()
        for value in (final_selection.get("corpus_ids") or [])
        if value is not None and str(value).strip()
    }
    author_ids = {
        str(value).strip()
        for value in (final_selection.get("author_ids") or [])
        if value is not None and str(value).strip()
    }
    return corpus_ids, author_ids


def source_url(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("url") or entry.get("path") or "").strip()
    return str(entry or "").strip()


def source_label(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("shard") or source_url(entry)).strip()
    url = source_url(entry)
    parsed = urlparse(url)
    if parsed.path:
        return unquote(Path(parsed.path).name)
    return url


def manifest_source_entry(entry: Any) -> Dict[str, str]:
    if isinstance(entry, dict):
        payload = {"url": source_url(entry)}
        if entry.get("shard"):
            payload["shard"] = str(entry["shard"])
        return payload
    return {"url": source_url(entry)}


def record_id_for_dataset(downloader: S2DatasetDownloader, dataset: str, record: Dict[str, Any]) -> Optional[str]:
    if dataset == "authors":
        return downloader._record_author_id(record)
    return downloader._record_corpus_id(record)


def scan_dataset_sources(
    downloader: S2DatasetDownloader,
    *,
    dataset: str,
    files: List[Any],
    corpus_ids: Set[str],
    author_ids: Set[str],
    max_files: Optional[int] = None,
    target_matches: Optional[int] = None,
    sleep_seconds: float = 0.0,
    extract_output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    remaining = set(author_ids if dataset == "authors" else corpus_ids)
    matched_ids: Set[str] = set()
    matched_sources: List[Dict[str, Any]] = []
    scanned = 0
    files_to_scan = files[:max_files] if max_files is not None else files
    records_written = 0
    extract_handle = None
    written_ids: Set[str] = set()

    if extract_output_path is not None:
        extract_output_path = Path(extract_output_path)
        extract_output_path.parent.mkdir(parents=True, exist_ok=True)
        extract_handle = open(extract_output_path, "w", encoding="utf-8")

    try:
        for index, entry in enumerate(files_to_scan, start=1):
            url = source_url(entry)
            if not url:
                continue
            label = source_label(entry)
            scanned += 1
            source_matches: Set[str] = set()
            console.print(
                f"[cyan]Scanning {dataset} shard {index}/{len(files_to_scan)}: {console_safe(label)}[/cyan]"
            )
            try:
                for line in downloader._iter_source_lines(url):
                    record = downloader._parse_json_line(line)
                    if not record:
                        continue
                    record_id = record_id_for_dataset(downloader, dataset, record)
                    if record_id and record_id in remaining:
                        source_matches.add(record_id)
                        matched_ids.add(record_id)
                        remaining.discard(record_id)
                        if extract_handle is not None and record_id not in written_ids:
                            extract_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                            extract_handle.flush()
                            written_ids.add(record_id)
                            records_written += 1
                        if target_matches is not None and len(matched_ids) >= target_matches:
                            break
            except Exception as exc:
                matched_sources.append(
                    {
                        **manifest_source_entry(entry),
                        "scan_error": str(exc),
                        "matched_ids": [],
                        "matched_count": 0,
                    }
                )
                console.print(f"[red]Shard scan failed for {dataset}: {console_safe(exc)}[/red]")
                continue

            if source_matches:
                matched_sources.append(
                    {
                        **manifest_source_entry(entry),
                        "matched_ids": sorted(source_matches),
                        "matched_count": len(source_matches),
                    }
                )
                console.print(f"[green]Matched {len(source_matches)} IDs in {dataset} shard[/green]")
            if not remaining:
                break
            if target_matches is not None and len(matched_ids) >= target_matches:
                break
            if sleep_seconds > 0 and index < len(files_to_scan):
                time.sleep(sleep_seconds)
    finally:
        if extract_handle is not None:
            extract_handle.close()

    return {
        "dataset": dataset,
        "scanned_file_count": scanned,
        "matched_source_count": len([source for source in matched_sources if source.get("matched_count")]),
        "matched_id_count": len(matched_ids),
        "records_written": records_written,
        "extract_path": str(extract_output_path) if extract_output_path else None,
        "requested_id_count": len(author_ids if dataset == "authors" else corpus_ids),
        "matched_ids": sorted(matched_ids),
        "missing_ids": sorted(remaining),
        "sources": matched_sources,
    }


def scan_source_extract(
    downloader: S2DatasetDownloader,
    *,
    dataset: str,
    extract_path: Path,
    corpus_ids: Set[str],
    author_ids: Set[str],
) -> Dict[str, Any]:
    requested_ids = set(author_ids if dataset == "authors" else corpus_ids)
    remaining = set(requested_ids)
    matched_ids: Set[str] = set()
    records_seen = 0

    extract_path = Path(extract_path)
    if extract_path.exists():
        with open(extract_path, "r", encoding="utf-8") as handle:
            for line in handle:
                record = downloader._parse_json_line(line)
                if not record:
                    continue
                records_seen += 1
                record_id = record_id_for_dataset(downloader, dataset, record)
                if record_id and record_id in requested_ids:
                    matched_ids.add(record_id)
                    remaining.discard(record_id)

    sources = []
    if matched_ids:
        sources.append(
            {
                "path": str(extract_path),
                "matched_ids": sorted(matched_ids),
                "matched_count": len(matched_ids),
            }
        )

    return {
        "dataset": dataset,
        "scanned_file_count": 1 if extract_path.exists() else 0,
        "matched_source_count": 1 if matched_ids else 0,
        "matched_id_count": len(matched_ids),
        "records_written": records_seen,
        "extract_path": str(extract_path) if extract_path.exists() else None,
        "requested_id_count": len(requested_ids),
        "matched_ids": sorted(matched_ids),
        "missing_ids": sorted(remaining),
        "sources": sources,
    }


def build_manifest_payload(
    *,
    final_selection: Dict[str, Any],
    claim_bank: Dict[str, Any],
    release_id: str,
    mini_release_id: str,
    shard_map: Dict[str, Any],
) -> Dict[str, Any]:
    datasets: Dict[str, Dict[str, List[str]]] = {}
    for dataset, result in (shard_map.get("datasets") or {}).items():
        matched_id_set = {
            str(value).strip()
            for value in (result.get("matched_ids") or [])
            if value is not None and str(value).strip()
        }
        for source in result.get("sources") or []:
            if not isinstance(source, dict):
                continue
            matched_id_set.update(
                str(value).strip()
                for value in (source.get("matched_ids") or [])
                if value is not None and str(value).strip()
            )
        matched_ids = list(matched_id_set)
        if not matched_ids:
            continue
        if dataset == "authors":
            datasets[dataset] = {"author_ids": sorted(matched_ids, key=str)}
        else:
            datasets[dataset] = {
                "corpus_ids": sorted(
                    matched_ids,
                    key=lambda value: int(value) if str(value).isdigit() else str(value),
                )
            }

    return {
        "release_id": release_id,
        "mini_release_id": mini_release_id,
        "datasets": datasets,
    }


def build_manifest(
    *,
    final_selection_path: Path = DEFAULT_FINAL_SELECTION_PATH,
    claim_bank_path: Path = DEFAULT_CLAIM_BANK_PATH,
    output_manifest_path: Path = DEFAULT_OUTPUT_MANIFEST_PATH,
    shard_map_path: Path = DEFAULT_SHARD_MAP_PATH,
    source_extract_dir: Path = DEFAULT_SOURCE_EXTRACT_DIR,
    release_id: Optional[str] = None,
    mini_release_id: str = "2026-05-26-mini-mendelian-v1",
    datasets: Optional[List[str]] = None,
    max_s2orc_v2_files: Optional[int] = 60,
    target_s2orc_v2_matches: Optional[int] = 100,
    sleep_seconds: float = 0.0,
    reuse_source_extracts: bool = False,
) -> Dict[str, Any]:
    final_selection_path = Path(final_selection_path).expanduser().resolve()
    claim_bank_path = Path(claim_bank_path).expanduser().resolve()
    output_manifest_path = Path(output_manifest_path).expanduser().resolve()
    shard_map_path = Path(shard_map_path).expanduser().resolve()
    source_extract_dir = Path(source_extract_dir).expanduser().resolve()

    final_selection = load_json(final_selection_path)
    claim_bank = load_json(claim_bank_path)
    corpus_ids, author_ids = selected_ids(final_selection)
    downloader = S2DatasetDownloader()
    release_id = release_id or final_selection.get("release_id") or downloader.get_latest_release()
    datasets = datasets or DEFAULT_DATASETS
    source_extract_dir.mkdir(parents=True, exist_ok=True)

    dataset_results: Dict[str, Any] = {}
    dataset_files: Dict[str, int] = {}
    for dataset in datasets:
        if reuse_source_extracts:
            extract_path = source_extract_dir / f"{dataset}.jsonl"
            dataset_files[dataset] = 1 if extract_path.exists() else 0
            dataset_results[dataset] = scan_source_extract(
                downloader,
                dataset=dataset,
                extract_path=extract_path,
                corpus_ids=corpus_ids,
                author_ids=author_ids,
            )
            continue
        info = downloader.get_dataset_info(dataset, release_id)
        files = info.get("files") or []
        dataset_files[dataset] = len(files)
        max_files = None
        target_matches = None
        if dataset == "s2orc_v2":
            max_files = max_s2orc_v2_files
            target_matches = target_s2orc_v2_matches
        dataset_results[dataset] = scan_dataset_sources(
            downloader,
            dataset=dataset,
            files=files,
            corpus_ids=corpus_ids,
            author_ids=author_ids,
            max_files=max_files,
            target_matches=target_matches,
            sleep_seconds=sleep_seconds,
            extract_output_path=source_extract_dir / f"{dataset}.jsonl",
        )

    shard_map = {
        "created_at": utc_now(),
        "release_id": release_id,
        "final_selection_path": str(final_selection_path),
        "dataset_file_counts": dataset_files,
        "datasets": dataset_results,
    }
    shard_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(shard_map_path, "w", encoding="utf-8") as handle:
        json.dump(shard_map, handle, indent=2, ensure_ascii=True)

    manifest = build_manifest_payload(
        final_selection=final_selection,
        claim_bank=claim_bank,
        release_id=release_id,
        mini_release_id=mini_release_id,
        shard_map=shard_map,
    )
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    return {
        "manifest_path": str(output_manifest_path),
        "shard_map_path": str(shard_map_path),
        "release_id": release_id,
        "mini_release_id": mini_release_id,
        "corpus_id_count": len(corpus_ids),
        "author_id_count": len(author_ids),
        "datasets": {
            dataset: {
                "matched_id_count": result.get("matched_id_count"),
                "matched_source_count": result.get("matched_source_count"),
                "missing_id_count": len(result.get("missing_ids") or []),
            }
            for dataset, result in dataset_results.items()
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Mendelian mini-corpus manifest.")
    parser.add_argument("--final-selection", default=str(DEFAULT_FINAL_SELECTION_PATH))
    parser.add_argument("--claims-bank", default=str(DEFAULT_CLAIM_BANK_PATH))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST_PATH))
    parser.add_argument("--shard-map", default=str(DEFAULT_SHARD_MAP_PATH))
    parser.add_argument("--source-extract-dir", default=str(DEFAULT_SOURCE_EXTRACT_DIR))
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--mini-release-id", default="2026-05-26-mini-mendelian-v1")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-s2orc-v2-files", type=int, default=60)
    parser.add_argument("--target-s2orc-v2-matches", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--reuse-source-extracts",
        action="store_true",
        help="Build the manifest from the ignored local curation cache instead of rescanning remote shards.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_manifest(
            final_selection_path=Path(args.final_selection),
            claim_bank_path=Path(args.claims_bank),
            output_manifest_path=Path(args.output_manifest),
            shard_map_path=Path(args.shard_map),
            source_extract_dir=Path(args.source_extract_dir),
            release_id=args.release_id,
            mini_release_id=args.mini_release_id,
            datasets=args.datasets,
            max_s2orc_v2_files=args.max_s2orc_v2_files,
            target_s2orc_v2_matches=args.target_s2orc_v2_matches,
            sleep_seconds=args.sleep,
            reuse_source_extracts=args.reuse_source_extracts,
        )
    except Exception as exc:
        console.print(f"[red]Manifest builder failed: {console_safe(exc)}[/red]")
        return 1
    console.print(f"[green]Manifest ready:[/green] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
