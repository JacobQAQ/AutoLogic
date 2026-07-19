"""Command-line entry point for the standalone AutoLogic DFA workflow."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from document_learner import (
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    LogicRAGDocumentLearner,
    read_report_from_csv,
)

from autologic.adapters import (
    AutoLogicChatAdapter,
    IFindStateRetriever,
    StateContentGenerator,
    StateEvidence,
)
from autologic.condition_induction import DeepSeekConditionInducer, HeuristicConditionInducer
from autologic.dfa_builder import AutoLogicDFABuilder
from autologic.executor import AutoLogicExecutor, ConditionGrounder, ExecutionResult
from autologic.models import WritingDFA, WritingState


PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_DFA = PROJECT_ROOT / "examples" / "autologic_demo_dfa.json"
DEMO_MATERIALS = PROJECT_ROOT / "examples" / "autologic_demo_materials.json"
CANONICAL_OUTPUTS = (
    "dfa.json",
    "generated_report.md",
    "generated_states.json",
    "execution_trace.json",
    "run_manifest.json",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cannot find required JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return value


def _save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def _prepare_output(directory: Path, filenames: Sequence[str], force: bool) -> None:
    existing = [directory / name for name in filenames if (directory / name).exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {joined}. Pass --force to overwrite them.")
    directory.mkdir(parents=True, exist_ok=True)


class DemoEvidenceRetriever:
    """Read deterministic state evidence from the keyless demo fixture."""

    def __init__(self, materials_path: str | Path, scenario: str) -> None:
        payload = _load_json(materials_path)
        scenarios = payload.get("scenarios", {})
        if scenario not in scenarios:
            raise ValueError(f"Demo materials do not define scenario {scenario!r}.")
        self.query_date = str(payload.get("query_date") or "")
        self.state_evidence = dict(scenarios[scenario].get("state_evidence") or {})

    def retrieve_state(
        self,
        state: WritingState,
        query: str,
        date: str | None = None,
        dry_run: bool = False,
        asset_name: str | None = None,
    ) -> StateEvidence:
        del query, asset_name
        fixture = dict(self.state_evidence.get(state.state_id) or {})
        records = list(fixture.get("records") or [])
        summary = str(
            fixture.get("summary")
            or f"No directional fixture data is required for state {state.state_id}."
        )
        return StateEvidence(
            state_id=state.state_id,
            required_materials=list(state.required_materials),
            resolved_codes=[str(record.get("code")) for record in records if record.get("code")],
            resolved_indicators=["settlement", "changeRatio"] if records else [],
            query_date=date or self.query_date,
            records=records,
            summary=summary,
            status="planned",
            errors=[],
            is_mock=True,
        )

    def close(self) -> None:
        return None


class _RetrievalMode:
    def __init__(self, inner: IFindStateRetriever, dry_run_data: bool) -> None:
        self.inner = inner
        self.dry_run_data = dry_run_data

    def retrieve_state(self, state, query, date=None, dry_run=False, asset_name=None):
        del dry_run
        return self.inner.retrieve_state(
            state,
            query,
            date=date,
            dry_run=self.dry_run_data,
            asset_name=asset_name,
        )

    def close(self) -> None:
        self.inner.close()


class _GenerationMode:
    def __init__(self, inner: StateContentGenerator, dry_run_llm: bool) -> None:
        self.inner = inner
        self.dry_run_llm = dry_run_llm
        self.chat = inner.chat

    def generate_current_state(self, **kwargs):
        kwargs["dry_run"] = self.dry_run_llm
        return self.inner.generate_current_state(**kwargs)


class _GroundingMode:
    def __init__(self, inner: ConditionGrounder, dry_run_llm: bool) -> None:
        self.inner = inner
        self.dry_run_llm = dry_run_llm

    def ground_condition(self, **kwargs):
        kwargs["dry_run"] = self.dry_run_llm
        return self.inner.ground_condition(**kwargs)


def _transition_threshold(args: argparse.Namespace) -> float:
    if getattr(args, "transition_support_threshold", None) is not None:
        return float(args.transition_support_threshold)
    if getattr(args, "theta", None) is not None:
        return float(args.theta)
    return 0.5


def _extract_or_load_sequences(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = getattr(args, "from_state_sequences", None)
    if paths:
        if getattr(args, "csv", None):
            raise ValueError("Use either --csv or --from-state-sequences, not both.")
        return _load_json(paths[0]), _load_json(paths[1])
    if not getattr(args, "csv", None):
        raise ValueError("build/run requires --csv or --from-state-sequences SEQ_A SEQ_B.")

    csv_path = Path(args.csv)
    document_a, text_a = read_report_from_csv(csv_path, args.row_a)
    document_b, text_b = read_report_from_csv(csv_path, args.row_b)
    learner = LogicRAGDocumentLearner(
        theta=args.semantic_merge_threshold,
        chat_model=args.chat_model,
        chat_base_url=args.chat_base_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        allow_api_embedding=not args.local_embedding_only,
    )
    learner.chat.temperature = args.temperature
    sequence_a = learner.extract_state_sequence(document_a, text_a)
    sequence_b = learner.extract_state_sequence(document_b, text_b)
    return sequence_a, sequence_b


def _build(args: argparse.Namespace, output_root: Path) -> Path:
    outputs = ("dfa.json", "build_manifest.json", "document_a_state_sequence.json", "document_b_state_sequence.json")
    _prepare_output(output_root, outputs, args.force)
    sequence_a, sequence_b = _extract_or_load_sequences(args)
    sequence_a_path = output_root / "document_a_state_sequence.json"
    sequence_b_path = output_root / "document_b_state_sequence.json"
    _save_json(sequence_a, sequence_a_path)
    _save_json(sequence_b, sequence_b_path)

    chat = AutoLogicChatAdapter(
        model=args.chat_model,
        base_url=args.chat_base_url,
        temperature=args.temperature,
    )
    if args.condition_mode == "deepseek":
        inducer = DeepSeekConditionInducer(chat_client=chat)
    else:
        inducer = HeuristicConditionInducer()
    embedder = EmbeddingProvider(
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        batch_size=args.embedding_batch_size,
        allow_api=not args.local_embedding_only,
    )
    builder = AutoLogicDFABuilder(
        semantic_merge_threshold=args.semantic_merge_threshold,
        state_support_threshold=args.state_support_threshold,
        transition_support_threshold=_transition_threshold(args),
        condition_inducer=inducer,
        embedder=embedder,
    )
    dfa_path = output_root / "dfa.json"
    dfa = builder.build(
        sequence_a,
        sequence_b,
        dfa_id=args.dfa_id,
        collection_id=args.collection_id,
        output_path=dfa_path,
    )
    validation = dfa.validate()
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "build",
        "csv": str(Path(args.csv).resolve()) if getattr(args, "csv", None) else None,
        "row_a": args.row_a,
        "row_b": args.row_b,
        "from_state_sequences": [str(Path(path).resolve()) for path in args.from_state_sequences]
        if args.from_state_sequences
        else None,
        "homogeneity_contract": "Both reports must be from the same vertical, report type, and institution/author group.",
        "collection_id": args.collection_id,
        "condition_mode": args.condition_mode,
        "semantic_merge_threshold": args.semantic_merge_threshold,
        "state_support_threshold": args.state_support_threshold,
        "transition_support_threshold": _transition_threshold(args),
        "outputs": {
            "dfa": str(dfa_path),
            "document_a_state_sequence": str(sequence_a_path),
            "document_b_state_sequence": str(sequence_b_path),
        },
        "validation_warnings": [issue.__dict__ for issue in validation.warnings],
        "determinism_conflicts": dfa.metadata.get("construction_metadata", {}).get(
            "determinism_conflicts", []
        ),
    }
    _save_json(manifest, output_root / "build_manifest.json")
    return dfa_path


def _execute_dfa(args: argparse.Namespace, dfa_path: Path, output_root: Path, command: str) -> ExecutionResult:
    _prepare_output(output_root, CANONICAL_OUTPUTS, args.force)
    dfa = WritingDFA.load_json(dfa_path)
    validation = dfa.validate()
    validation.raise_if_invalid()
    dry_data = bool(args.dry_run or args.dry_run_data)
    dry_llm = bool(args.dry_run or args.dry_run_llm)
    chat = AutoLogicChatAdapter(
        model=args.chat_model,
        base_url=args.chat_base_url,
        temperature=args.temperature,
    )
    retriever = _RetrievalMode(
        IFindStateRetriever(dictionary_path=args.dictionary),
        dry_run_data=dry_data,
    )
    generator = _GenerationMode(StateContentGenerator(chat), dry_run_llm=dry_llm)
    grounder = _GroundingMode(ConditionGrounder(chat), dry_run_llm=dry_llm)
    executor = AutoLogicExecutor(
        dfa,
        evidence_retriever=retriever,
        content_generator=generator,
        condition_grounder=grounder,
        max_steps=args.max_steps,
        max_visits_per_state=args.max_visits_per_state,
        max_transition_repeats=args.max_transition_repeats,
        dfa_path=str(dfa_path.resolve()),
    )
    result = executor.execute(
        query=args.query,
        date=args.date or None,
        asset_name=args.asset_name or None,
        dry_run=bool(args.dry_run),
    )
    result.run_manifest.update(
        {
            "command": command,
            "dry_run_data": dry_data,
            "dry_run_llm": dry_llm,
            "dictionary": str(Path(args.dictionary).resolve()),
        }
    )
    dfa.save_json(output_root / "dfa.json")
    result.save_outputs(output_root)
    return result


def run_demo(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    _prepare_output(output_root, CANONICAL_OUTPUTS, args.force)
    dfa = WritingDFA.load_json(DEMO_DFA)
    dfa.validate().raise_if_invalid()
    fixture = _load_json(DEMO_MATERIALS)
    query = str(fixture.get("query") or "Write the weekly market report for 2026-01-02.")
    query_date = str(fixture.get("query_date") or "2026-01-02")
    chat = AutoLogicChatAdapter(model="offline-demo")
    executor = AutoLogicExecutor(
        dfa,
        evidence_retriever=DemoEvidenceRetriever(DEMO_MATERIALS, args.scenario),
        content_generator=StateContentGenerator(chat),
        condition_grounder=ConditionGrounder(chat),
        dfa_path=str(DEMO_DFA),
    )
    result = executor.execute(query=query, date=query_date, dry_run=True)
    result.run_manifest.update(
        {
            "command": "demo",
            "scenario": args.scenario,
            "fixture_materials": str(DEMO_MATERIALS),
            "offline": True,
        }
    )
    dfa.save_json(output_root / "dfa.json")
    result.save_outputs(output_root)
    print(f"AutoLogic demo completed: scenario={args.scenario}")
    print("Path: " + " -> ".join(result.run_manifest["executed_state_path"]))
    print(f"Outputs: {output_root}")
    return 0 if result.success else 1


def run_build(args: argparse.Namespace) -> int:
    path = _build(args, Path(args.output_root))
    print(f"AutoLogic DFA built: {path}")
    return 0


def run_generate(args: argparse.Namespace) -> int:
    result = _execute_dfa(args, Path(args.dfa), Path(args.output_root), "generate")
    print("Path: " + " -> ".join(result.run_manifest["executed_state_path"]))
    print(f"Termination: {result.run_manifest['termination_reason']}")
    return 0 if result.success else 1


def run_full(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    dfa_path = _build(args, output_root / "build")
    result = _execute_dfa(args, dfa_path, output_root, "run")
    print(f"AutoLogic run completed: {output_root}")
    print("Path: " + " -> ".join(result.run_manifest["executed_state_path"]))
    return 0 if result.success else 1


def _add_chat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--chat-base-url", default=DEFAULT_CHAT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.2)


def _add_build_arguments(parser: argparse.ArgumentParser, *, output_default: str) -> None:
    parser.add_argument("--csv", default="", help="Two-report historical CSV input.")
    parser.add_argument("--row-a", type=int, default=0)
    parser.add_argument("--row-b", type=int, default=1)
    parser.add_argument("--from-state-sequences", nargs=2, metavar=("SEQ_A", "SEQ_B"))
    parser.add_argument("--theta", type=float, default=None, help="Alias for transition support threshold.")
    parser.add_argument("--semantic-merge-threshold", type=float, default=0.5)
    parser.add_argument("--state-support-threshold", type=float, default=0.5)
    parser.add_argument("--transition-support-threshold", type=float, default=None)
    parser.add_argument("--condition-mode", choices=("heuristic", "deepseek"), default="heuristic")
    parser.add_argument("--collection-id", default="autologic_collection")
    parser.add_argument("--dfa-id", default="autologic_dfa")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--local-embedding-only", action="store_true")
    parser.add_argument("--output-root", default=output_default)
    parser.add_argument("--force", action="store_true")


def _add_generation_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_default: str,
    include_output: bool = True,
) -> None:
    parser.add_argument("--query", required=True)
    parser.add_argument("--dictionary", default="domain_dictionary.csv")
    parser.add_argument("--date", default="")
    parser.add_argument("--asset-name", default="")
    parser.add_argument("--dry-run", action="store_true", help="Skip both iFinD and chat calls.")
    parser.add_argument("--dry-run-data", action="store_true", help="Skip iFinD calls only.")
    parser.add_argument("--dry-run-llm", action="store_true", help="Skip chat calls only.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-visits-per-state", type=int, default=2)
    parser.add_argument("--max-transition-repeats", type=int, default=2)
    if include_output:
        parser.add_argument("--output-root", default=output_default)
        parser.add_argument("--force", action="store_true")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and execute one AutoLogic writing DFA.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the fully offline keyless fixture DFA.")
    demo.add_argument("--scenario", choices=("price-up", "price-down"), default="price-up")
    demo.add_argument("--output-root", default="autologic_outputs/demo")
    demo.add_argument("--force", action="store_true")
    demo.set_defaults(handler=run_demo)

    build = subparsers.add_parser("build", help="Build a DFA from two homogeneous reports.")
    _add_chat_arguments(build)
    _add_build_arguments(build, output_default="autologic_outputs/build")
    build.set_defaults(handler=run_build)

    generate = subparsers.add_parser("generate", help="Execute an existing dfa.json directly.")
    generate.add_argument("--dfa", required=True)
    _add_chat_arguments(generate)
    _add_generation_arguments(generate, output_default="autologic_outputs/generation")
    generate.set_defaults(handler=run_generate)

    run = subparsers.add_parser("run", help="Build a DFA, then execute that exact DFA.")
    _add_chat_arguments(run)
    _add_build_arguments(run, output_default="autologic_outputs/full")
    _add_generation_arguments(run, output_default="autologic_outputs/full", include_output=False)
    run.set_defaults(handler=run_full)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, KeyError) as exc:
        print(f"AutoLogic error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
