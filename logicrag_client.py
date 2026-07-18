"""
User-friendly LogicRAG client.

This script orchestrates the full LogicRAG pipeline:

1. document_learner.py      -> global_template.json + state_index.json
2. query_processing.py      -> query_subtree.json
3. ifind_data_plugin.py     -> retrieved_materials/*
4. report_generator.py      -> generated_report.md

API keys and iFinD credentials can be passed as CLI hyperparameters or read
from environment variables.  CLI values are used only inside this process.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

from document_learner import (
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    run_from_csv,
)
from ifind_data_plugin import run_data_plugin
from query_processing import run_query_processing
from report_generator import ChatClient, LogicRAGReportGenerator


DEFAULT_CSV = r"dataset\Precious_Metals\data\case_2253.csv"
DEFAULT_OUTPUT_ROOT = "logicrag_outputs"
DEFAULT_THETA = 0.5
DEFAULT_TAU = 0.5


def set_if_present(name: str, value: str | None) -> None:
    if value:
        os.environ[name] = value


def configure_runtime_env(args: argparse.Namespace) -> None:
    """Set process-local environment variables from CLI hyperparameters."""
    set_if_present("DEEPSEEK_API_KEY", args.deepseek_api_key)
    set_if_present("OPENAI_API_KEY", args.openai_api_key)
    set_if_present("DASHSCOPE_API_KEY", args.dashscope_api_key)
    set_if_present("DASHSCOPE_API_BASE", args.dashscope_api_base)
    set_if_present("IFIND_USERNAME", args.ifind_username)
    set_if_present("IFIND_PASSWORD", args.ifind_password)


def validate_inputs(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    dictionary_path = Path(args.dictionary)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Domain dictionary does not exist: {dictionary_path}")


def run_document_learning(args: argparse.Namespace, output_root: Path) -> Dict[str, str]:
    stage_args = SimpleNamespace(
        csv=args.csv,
        row_a=args.row_a,
        row_b=args.row_b,
        theta=args.theta,
        report_pair_id=args.report_pair_id,
        output_dir=str(output_root),
        chat_model=args.chat_model,
        chat_base_url=args.chat_base_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        local_embedding_only=args.local_embedding_only,
    )
    return run_from_csv(stage_args)


def run_query_stage(args: argparse.Namespace, output_root: Path) -> None:
    stage_args = SimpleNamespace(
        query=args.query,
        tau=args.tau,
        template=str(output_root / "global_template.json"),
        index=str(output_root / "state_index.json"),
        output=str(output_root / "query_subtree.json"),
        preview_top_k=args.preview_top_k,
        fallback_top_k=args.fallback_top_k,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        local_embedding_only=args.local_embedding_only,
    )
    run_query_processing(stage_args)


def run_data_stage(args: argparse.Namespace, output_root: Path) -> None:
    stage_args = SimpleNamespace(
        query=args.query,
        query_subtree=str(output_root / "query_subtree.json"),
        dictionary=args.dictionary,
        output_dir=str(output_root / "retrieved_materials"),
        date=args.date,
        asset_name=args.asset_name,
        username=args.ifind_username or os.environ.get("IFIND_USERNAME", ""),
        password=args.ifind_password or os.environ.get("IFIND_PASSWORD", ""),
        dry_run=args.dry_run_data,
    )
    run_data_plugin(stage_args)


def run_generation_stage(args: argparse.Namespace, output_root: Path) -> Dict[str, str]:
    chat_client = ChatClient(
        model=args.chat_model,
        base_url=args.chat_base_url,
        temperature=args.temperature,
    )
    generator = LogicRAGReportGenerator(
        query_subtree_path=output_root / "query_subtree.json",
        materials_dir=output_root / "retrieved_materials",
        chat_client=chat_client,
        dry_run=args.dry_run_report,
        generate_internal=args.generate_internal,
    )
    generator.load_data()
    generator.generate()
    return generator.save_outputs(
        output_md=output_root / "generated_report.md",
        node_outputs_path=output_root / "generated_node_outputs.json",
        trace_path=output_root / "generation_trace.json",
    )


def run_pipeline(args: argparse.Namespace) -> Dict[str, str]:
    configure_runtime_env(args)
    validate_inputs(args)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n=== LogicRAG Client ===")
    print(f"csv: {args.csv}")
    print(f"rows: {args.row_a}, {args.row_b}")
    print(f"query: {args.query}")
    print(f"theta: {args.theta}")
    print(f"tau: {args.tau}")
    print(f"output_root: {output_root}")

    if not args.skip_document_learning:
        print("\n[Stage 1/4] Document learning")
        run_document_learning(args, output_root)
    else:
        print("\n[Stage 1/4] Document learning skipped")

    if not args.skip_query_processing:
        print("\n[Stage 2/4] Query processing")
        run_query_stage(args, output_root)
    else:
        print("\n[Stage 2/4] Query processing skipped")

    if not args.skip_data_retrieval:
        print("\n[Stage 3/4] Data retrieval")
        run_data_stage(args, output_root)
    else:
        print("\n[Stage 3/4] Data retrieval skipped")

    if not args.skip_report_generation:
        print("\n[Stage 4/4] Report generation")
        outputs = run_generation_stage(args, output_root)
    else:
        print("\n[Stage 4/4] Report generation skipped")
        outputs = {
            "report": str(output_root / "generated_report.md"),
            "node_outputs": str(output_root / "generated_node_outputs.json"),
            "trace": str(output_root / "generation_trace.json"),
        }

    print("\nLogicRAG pipeline completed.")
    print(f"- report: {outputs['report']}")
    print(f"- query_subtree: {output_root / 'query_subtree.json'}")
    print(f"- retrieved_materials: {output_root / 'retrieved_materials'}")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full LogicRAG pipeline from two sample reports to a generated report."
    )

    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV containing sample reports.")
    parser.add_argument("--row-a", type=int, default=0, help="First sample report row index.")
    parser.add_argument("--row-b", type=int, default=1, help="Second sample report row index.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Pipeline output directory.")
    parser.add_argument("--dictionary", default="domain_dictionary.csv", help="domain_dictionary.csv path.")

    parser.add_argument("--query", required=True, help="User query.")
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA, help="Document state matching threshold.")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Query-state matching threshold.")
    parser.add_argument("--date", default="", help="Optional date override for iFinD retrieval, YYYY-MM-DD.")
    parser.add_argument("--asset-name", default="", help="Optional lookup term used only against domain_dictionary.csv.")
    parser.add_argument("--report-pair-id", default="", help="Optional report pair id.")

    parser.add_argument("--deepseek-api-key", default="", help="DeepSeek/OpenAI-compatible chat API key.")
    parser.add_argument("--openai-api-key", default="", help="Optional OpenAI API key fallback.")
    parser.add_argument("--dashscope-api-key", default="", help="DashScope Bailian embedding API key.")
    parser.add_argument(
        "--dashscope-api-base",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="DashScope OpenAI-compatible base URL.",
    )
    parser.add_argument("--ifind-username", default="", help="iFinD username.")
    parser.add_argument("--ifind-password", default="", help="iFinD password.")

    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Chat model name.")
    parser.add_argument("--chat-base-url", default=DEFAULT_CHAT_BASE_URL, help="Chat API base URL.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Report generation temperature.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    parser.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL, help="Embedding API base URL.")
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--local-embedding-only", action="store_true", help="Use local hash embeddings.")

    parser.add_argument("--preview-top-k", type=int, default=5, help="Ranked state preview count.")
    parser.add_argument("--fallback-top-k", type=int, default=0, help="Fallback top-k if no state exceeds tau.")

    parser.add_argument("--dry-run-data", action="store_true", help="Do not call iFinD; only build data specs.")
    parser.add_argument("--dry-run-report", action="store_true", help="Do not call LLM in report generation.")
    parser.add_argument("--generate-internal", action="store_true", help="Generate text for internal nodes too.")

    parser.add_argument("--skip-document-learning", action="store_true")
    parser.add_argument("--skip-query-processing", action="store_true")
    parser.add_argument("--skip-data-retrieval", action="store_true")
    parser.add_argument("--skip-report-generation", action="store_true")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
