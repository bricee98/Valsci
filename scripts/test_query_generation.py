#!/usr/bin/env python
"""
Standalone query-generation benchmark script.

Tests which LLM models produce usable Semantic Scholar search queries
for a given set of scientific claims, and how long they take.

Usage:
  python scripts/test_query_generation.py \
    --claim-file scripts/sample_claims.txt \
    --preferred-models gemma3:latest,llama3.2:latest \
    --fallback-models gpt-4o-mini \
    --max-output-tokens 800 \
    --timeout 300 \
    --num-queries 5 \
    --output results.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from app.config.settings import Config  # noqa: E402
from app.services.llm.gateway import LLMGateway, LLMTask  # noqa: E402
from semantic_scholar.utils.searcher import S2Searcher  # noqa: E402


def load_claims(claim_file: str) -> list[str]:
    """Load claims from file, one per line. Lines starting with # are comments."""
    claims = []
    with open(claim_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                claims.append(line)
    return claims


async def run_query_generation(args: argparse.Namespace) -> None:
    claims = load_claims(args.claim_file)
    if not claims:
        print("No claims found in file.")
        return

    # Build model lists
    preferred = [m.strip() for m in args.preferred_models.split(",") if m.strip()] if args.preferred_models else []
    fallback = [m.strip() for m in args.fallback_models.split(",") if m.strip()] if args.fallback_models else []

    # Override routing config for this run
    task_config = {}
    if preferred or fallback:
        task_config["preferred_models"] = preferred
        task_config["fallback_models"] = fallback
    if args.max_output_tokens:
        task_config["max_output_tokens"] = args.max_output_tokens
    if args.timeout:
        task_config["timeout_seconds"] = args.timeout

    if task_config:
        Config.LLM_ROUTING = {
            "enabled": True,
            "locked_models": False,
            "tasks": {
                LLMTask.QUERY_GENERATION: task_config,
            },
        }

    if args.timeout:
        Config.LLM_TIMEOUT_SECONDS = args.timeout

    gateway = LLMGateway()
    searcher = S2Searcher()

    results = []
    batch_id = f"test_qgen_{int(time.time())}"

    print(f"\nBenchmarking query generation")
    print(f"  Models (preferred): {preferred or ['(default: ' + Config.LLM_EVALUATION_MODEL + ')']}")
    print(f"  Models (fallback):  {fallback or ['(none)']}")
    print(f"  Max output tokens:  {args.max_output_tokens}")
    print(f"  Timeout:            {args.timeout or Config.LLM_TIMEOUT_SECONDS}s")
    print(f"  Claims:             {len(claims)}")
    print(f"  Queries per claim:  {args.num_queries}")
    print("=" * 60)

    for i, claim_text in enumerate(claims):
        claim_id = f"claim_{i}"
        print(f"\n--- Claim {i + 1}/{len(claims)} ---")
        print(f"  {claim_text[:120]}{'...' if len(claim_text) > 120 else ''}")

        start = time.perf_counter()
        try:
            queries, usage = await searcher.generate_search_queries(
                claim_text=claim_text,
                num_queries=args.num_queries,
                ai_service=gateway,
                batch_id=batch_id,
                claim_id=claim_id,
            )
            elapsed = time.perf_counter() - start
            result = {
                "claim": claim_text,
                "queries": queries,
                "usage": usage,
                "elapsed_s": round(elapsed, 3),
                "status": "success" if queries else "empty",
                "model": preferred[0] if preferred else Config.LLM_EVALUATION_MODEL,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            result = {
                "claim": claim_text,
                "queries": [],
                "error": str(e),
                "error_type": type(e).__name__,
                "elapsed_s": round(elapsed, 3),
                "status": "error",
            }

        results.append(result)

        status_marker = {"success": "+", "empty": "?", "error": "X"}[result["status"]]
        print(f"  [{status_marker}] {result['status']} | {result['elapsed_s']}s | {len(result.get('queries', []))} queries")
        if result.get("queries"):
            for q in result["queries"]:
                print(f"      - {q}")
        if result.get("error"):
            print(f"      ERROR: {result['error_type']}: {result['error'][:200]}")

    # Summary
    successes = sum(1 for r in results if r["status"] == "success")
    empties = sum(1 for r in results if r["status"] == "empty")
    errors = sum(1 for r in results if r["status"] == "error")
    avg_time = sum(r["elapsed_s"] for r in results) / len(results) if results else 0

    print("\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"  Claims:   {len(results)}")
    print(f"  Success:  {successes}")
    print(f"  Empty:    {empties}")
    print(f"  Errors:   {errors}")
    print(f"  Avg time: {avg_time:.2f}s")

    if results:
        total_input = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
        total_output = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
        total_cost = sum(r.get("usage", {}).get("cost_usd", 0) for r in results)
        if total_input or total_output:
            print(f"  Tokens:   {total_input} in / {total_output} out")
        if total_cost:
            print(f"  Cost:     ${total_cost:.6f}")

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark query generation across different LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--claim-file", required=True, help="File with one claim per line (# for comments)")
    parser.add_argument("--preferred-models", default="", help="Comma-separated preferred model names")
    parser.add_argument("--fallback-models", default="", help="Comma-separated fallback model names")
    parser.add_argument("--max-output-tokens", type=int, default=800, help="Max output tokens (default: 800)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds per LLM call")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of queries to generate per claim (default: 5)")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: print to stdout)")
    args = parser.parse_args()

    if not Path(args.claim_file).exists():
        print(f"Error: claim file not found: {args.claim_file}")
        sys.exit(1)

    asyncio.run(run_query_generation(args))


if __name__ == "__main__":
    main()
