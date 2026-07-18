"""Offline two-report builder for AutoLogic's deterministic writing DFA."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from document_learner import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    StateMatch,
    match_common_states,
    merge_materials,
    merge_text_field,
    normalize_state_sequence,
    state_embedding_text,
)

from .condition_induction import ConditionInducer, HeuristicConditionInducer
from .models import WritingDFA, WritingState, WritingTransition
from .validation import DFAValidationError


@dataclass(frozen=True)
class _CandidateState:
    nodes: tuple[tuple[str, Mapping[str, Any], int], ...]
    match_similarity: float | None = None


def _threshold(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


class AutoLogicDFABuilder:
    """Build the Phase-1 matched-plus-unmatched leaf-state union DFA."""

    def __init__(
        self,
        semantic_merge_threshold: float = 0.85,
        state_support_threshold: float = 0.5,
        transition_support_threshold: float = 0.5,
        condition_inducer: ConditionInducer | None = None,
        embedder: EmbeddingProvider | None = None,
        max_support_examples: int = 3,
    ) -> None:
        self.semantic_merge_threshold = _threshold("semantic_merge_threshold", semantic_merge_threshold)
        self.state_support_threshold = _threshold("state_support_threshold", state_support_threshold)
        self.transition_support_threshold = _threshold(
            "transition_support_threshold", transition_support_threshold
        )
        self.condition_inducer = condition_inducer or HeuristicConditionInducer()
        # The builder is offline/keyless by default. Callers may explicitly inject
        # the existing API-capable provider without changing document_learner.py.
        self.embedder = embedder or EmbeddingProvider(allow_api=False)
        if isinstance(max_support_examples, bool) or int(max_support_examples) < 0:
            raise ValueError("max_support_examples must be a non-negative integer.")
        self.max_support_examples = int(max_support_examples)

    @staticmethod
    def _normalize_artifact(artifact: Mapping[str, Any], fallback_id: str) -> dict[str, Any]:
        document_id = str(artifact.get("document_id") or fallback_id)
        normalized = normalize_state_sequence(dict(artifact), document_id)
        raw_nodes = artifact.get("state_sequence")
        if not raw_nodes:
            raw_nodes = artifact.get("node_template", {}).get("nodes")
        if not raw_nodes:
            raw_nodes = artifact.get("nodes")
        raw_by_id = {
            str(node.get("node_id") or node.get("id")): node
            for node in (raw_nodes or [])
            if isinstance(node, Mapping)
        }
        for node in normalized["state_sequence"]:
            raw = raw_by_id.get(str(node["node_id"]), {})
            excerpt = raw.get("source_excerpt") or raw.get("excerpt") or raw.get("source_text")
            if excerpt is not None:
                node["source_excerpt"] = str(excerpt)
        return normalized

    @staticmethod
    def load_artifact(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid state-sequence JSON in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"State-sequence artifact {path} must contain one JSON object.")
        return payload

    @staticmethod
    def _leaf_nodes(sequence: Mapping[str, Any]) -> list[dict[str, Any]]:
        nodes = sequence.get("state_sequence", [])
        leaves = [dict(node) for node in nodes if str(node.get("node_type", "")).lower() == "leaf"]
        if not leaves:
            raise ValueError(f"Document {sequence.get('document_id')!r} has no executable leaf states.")
        return leaves

    @staticmethod
    def _ancestor_chain(node: Mapping[str, Any], all_nodes: Sequence[Mapping[str, Any]]) -> list[str]:
        by_id = {str(item.get("node_id")): item for item in all_nodes}
        chain: list[str] = []
        parent = node.get("parent")
        visited: set[str] = set()
        while parent is not None and str(parent) not in visited:
            parent_id = str(parent)
            visited.add(parent_id)
            chain.append(parent_id)
            parent = by_id.get(parent_id, {}).get("parent")
        chain.reverse()
        return chain

    @staticmethod
    def _provenance(
        document_id: str,
        node: Mapping[str, Any],
        position: int,
        all_nodes: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "source_report_id": document_id,
            "source_node_id": str(node.get("node_id")),
            "ancestor_chain": AutoLogicDFABuilder._ancestor_chain(node, all_nodes),
            "source_position": position,
            "source_excerpt": str(
                node.get("source_excerpt") or node.get("excerpt") or node.get("source_text") or ""
            ),
        }

    def _match_leaves(
        self,
        leaves_a: Sequence[Mapping[str, Any]],
        leaves_b: Sequence[Mapping[str, Any]],
    ) -> list[StateMatch]:
        texts_a = [state_embedding_text(dict(node)) for node in leaves_a]
        texts_b = [state_embedding_text(dict(node)) for node in leaves_b]
        vectors = self.embedder.embed_texts(texts_a + texts_b)
        return match_common_states(
            leaves_a,
            leaves_b,
            vectors[: len(leaves_a)],
            vectors[len(leaves_a) :],
            self.semantic_merge_threshold,
        )

    @staticmethod
    def _candidate_union(
        document_a: str,
        document_b: str,
        leaves_a: Sequence[Mapping[str, Any]],
        leaves_b: Sequence[Mapping[str, Any]],
        matches: Sequence[StateMatch],
    ) -> tuple[list[_CandidateState], dict[tuple[str, int], int]]:
        """Merge matched leaves and retain every unmatched leaf from both reports."""
        candidates: list[_CandidateState] = []
        source_to_candidate: dict[tuple[str, int], int] = {}
        matched_a: set[int] = set()
        matched_b: set[int] = set()

        for match in sorted(matches, key=lambda item: (item.a_index, item.b_index)):
            index = len(candidates)
            candidates.append(
                _CandidateState(
                    nodes=(
                        (document_a, leaves_a[match.a_index], match.a_index),
                        (document_b, leaves_b[match.b_index], match.b_index),
                    ),
                    match_similarity=float(match.similarity),
                )
            )
            source_to_candidate[(document_a, match.a_index)] = index
            source_to_candidate[(document_b, match.b_index)] = index
            matched_a.add(match.a_index)
            matched_b.add(match.b_index)

        for index_a, node in enumerate(leaves_a):
            if index_a not in matched_a:
                candidate_index = len(candidates)
                candidates.append(_CandidateState(nodes=((document_a, node, index_a),)))
                source_to_candidate[(document_a, index_a)] = candidate_index
        for index_b, node in enumerate(leaves_b):
            if index_b not in matched_b:
                candidate_index = len(candidates)
                candidates.append(_CandidateState(nodes=((document_b, node, index_b),)))
                source_to_candidate[(document_b, index_b)] = candidate_index
        return candidates, source_to_candidate

    @staticmethod
    def _state_from_candidate(state_id: str, candidate: _CandidateState) -> WritingState:
        first = candidate.nodes[0][1]
        second = candidate.nodes[1][1] if len(candidate.nodes) == 2 else None
        label = str(first.get("template_description") or "").strip()
        description = str(first.get("content_guideline") or "").strip()
        materials = list(first.get("required_materials") or [])
        if second is not None:
            label = merge_text_field(label, str(second.get("template_description") or ""))
            description = merge_text_field(description, str(second.get("content_guideline") or ""))
            materials = merge_materials(materials, list(second.get("required_materials") or []))
        documents = list(dict.fromkeys(document_id for document_id, _, _ in candidate.nodes))
        return WritingState(
            state_id=state_id,
            state_kind="writing",
            label=label,
            description=description,
            action=description,
            required_materials=materials,
            support_count=len(documents),
            support_documents=documents,
        )

    @staticmethod
    def determinize(
        transitions: Sequence[WritingTransition],
    ) -> tuple[list[WritingTransition], list[dict[str, Any]]]:
        """Resolve duplicate (source, symbol) targets without random sampling."""
        grouped: dict[tuple[str, str], dict[str, list[WritingTransition]]] = defaultdict(lambda: defaultdict(list))
        for transition in transitions:
            grouped[(transition.source, transition.symbol)][transition.target].append(transition)

        kept: list[WritingTransition] = []
        discarded: list[dict[str, Any]] = []
        for (source, symbol), by_target in sorted(grouped.items()):
            ranked: list[WritingTransition] = []
            for target, items in by_target.items():
                support = sum(item.support_count for item in items)
                confidence = max(item.confidence for item in items)
                examples = [example for item in items for example in item.support_examples]
                ranked.append(
                    WritingTransition(
                        source=source,
                        symbol=symbol,
                        condition_description=sorted(
                            (item.condition_description for item in items), key=lambda text: (-len(text), text)
                        )[0],
                        target=target,
                        support_count=support,
                        confidence=confidence,
                        support_examples=examples,
                        metadata=dict(items[0].metadata),
                    )
                )
            ranked.sort(key=lambda item: (-item.support_count, -item.confidence, item.target))
            winner = ranked[0]
            kept.append(winner)
            for loser in ranked[1:]:
                discarded.append(
                    {
                        "source": source,
                        "symbol": symbol,
                        "discarded_target": loser.target,
                        "kept_target": winner.target,
                        "discarded_support_count": loser.support_count,
                        "discarded_confidence": loser.confidence,
                        "reason": "support_count, confidence, then target_id stable tie-break",
                    }
                )
        kept.sort(key=lambda item: (item.source, item.symbol, item.target))
        return kept, discarded

    def build(
        self,
        sequence_a: Mapping[str, Any],
        sequence_b: Mapping[str, Any],
        *,
        dfa_id: str = "autologic_dfa",
        collection_id: str = "autologic_collection",
        matches: Sequence[StateMatch] | None = None,
        output_path: str | Path | None = None,
    ) -> WritingDFA:
        seq_a = self._normalize_artifact(sequence_a, "report_a")
        seq_b = self._normalize_artifact(sequence_b, "report_b")
        document_a = str(seq_a["document_id"])
        document_b = str(seq_b["document_id"])
        if document_a == document_b:
            raise ValueError("The two source reports must have distinct document_id values.")
        leaves_a = self._leaf_nodes(seq_a)
        leaves_b = self._leaf_nodes(seq_b)
        selected_matches = list(matches) if matches is not None else self._match_leaves(leaves_a, leaves_b)
        candidates, source_to_candidate = self._candidate_union(
            document_a, document_b, leaves_a, leaves_b, selected_matches
        )

        retained_candidates = [
            index
            for index, candidate in enumerate(candidates)
            if len({item[0] for item in candidate.nodes}) / 2 >= self.state_support_threshold
        ]
        candidate_to_state = {
            candidate_index: f"S{ordinal:03d}"
            for ordinal, candidate_index in enumerate(retained_candidates, start=1)
        }
        if not candidate_to_state:
            raise ValueError("state_support_threshold removed every candidate state.")

        states: dict[str, WritingState] = {}
        provenance: dict[str, list[dict[str, Any]]] = {}
        all_nodes = {
            document_a: seq_a["state_sequence"],
            document_b: seq_b["state_sequence"],
        }
        for candidate_index in retained_candidates:
            state_id = candidate_to_state[candidate_index]
            candidate = candidates[candidate_index]
            states[state_id] = self._state_from_candidate(state_id, candidate)
            provenance[state_id] = [
                self._provenance(
                    document_id,
                    node,
                    next(
                        index
                        for index, full_node in enumerate(all_nodes[document_id])
                        if str(full_node.get("node_id")) == str(node.get("node_id"))
                    ),
                    all_nodes[document_id],
                )
                for document_id, node, _position in candidate.nodes
            ]

        projected: dict[str, list[str]] = {}
        for document_id, leaves in ((document_a, leaves_a), (document_b, leaves_b)):
            sequence: list[str] = []
            for leaf_index in range(len(leaves)):
                candidate_index = source_to_candidate[(document_id, leaf_index)]
                state_id = candidate_to_state.get(candidate_index)
                if state_id is not None:
                    sequence.append(state_id)
            projected[document_id] = sequence
        if any(not sequence for sequence in projected.values()):
            raise ValueError("Support filtering removed every projected state from a source report.")

        first_counts = Counter(sequence[0] for sequence in projected.values())
        last_counts = Counter(sequence[-1] for sequence in projected.values())
        initial_state = sorted(first_counts, key=lambda state_id: (-first_counts[state_id], state_id))[0]

        edge_documents: dict[tuple[str, str], set[str]] = defaultdict(set)
        edge_examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        source_occurrence_documents: dict[str, set[str]] = defaultdict(set)
        for document_id, sequence in projected.items():
            for source in sequence[:-1]:
                source_occurrence_documents[source].add(document_id)
            for position, (source, target) in enumerate(zip(sequence, sequence[1:])):
                edge_documents[(source, target)].add(document_id)
                source_entry = next(
                    (entry for entry in provenance[source] if entry["source_report_id"] == document_id),
                    provenance[source][0],
                )
                target_entry = next(
                    (entry for entry in provenance[target] if entry["source_report_id"] == document_id),
                    provenance[target][0],
                )
                excerpt = source_entry.get("source_excerpt", "")
                if excerpt:
                    edge_examples[(source, target)].append(
                        {
                            "document_id": document_id,
                            "source_node_id": source_entry["source_node_id"],
                            "target_node_id": target_entry["source_node_id"],
                            "evidence_excerpt": excerpt,
                        }
                    )

        # If a writing state ends one historical report but continues in another,
        # model termination explicitly instead of marking that branching state final.
        outgoing_observed = {source for source, _ in edge_documents}
        mixed_termination_sources = sorted(set(last_counts) & outgoing_observed)
        terminal_state_id: str | None = None
        if mixed_termination_sources:
            terminal_state_id = f"S{len(states) + 1:03d}"
            states[terminal_state_id] = WritingState(
                state_id=terminal_state_id,
                state_kind="terminal",
                label="Report complete",
                description="Dedicated terminal control state reached through END.",
                action="",
                required_materials=[],
                support_count=0,
                support_documents=[],
                is_final=True,
            )
            for document_id, sequence in projected.items():
                source = sequence[-1]
                if source not in mixed_termination_sources:
                    continue
                edge_documents[(source, terminal_state_id)].add(document_id)
                source_occurrence_documents[source].add(document_id)
                source_entry = next(
                    (entry for entry in provenance[source] if entry["source_report_id"] == document_id),
                    provenance[source][0],
                )
                excerpt = source_entry.get("source_excerpt", "")
                if excerpt:
                    edge_examples[(source, terminal_state_id)].append(
                        {
                            "document_id": document_id,
                            "source_node_id": source_entry["source_node_id"],
                            "target_node_id": terminal_state_id,
                            "evidence_excerpt": excerpt,
                        }
                    )

        retained_edges: dict[str, list[str]] = defaultdict(list)
        dropped_transitions: list[dict[str, Any]] = []
        for (source, target), documents in sorted(edge_documents.items()):
            denominator = max(1, len(source_occurrence_documents[source]))
            ratio = len(documents) / denominator
            if ratio >= self.transition_support_threshold:
                retained_edges[source].append(target)
            else:
                dropped_transitions.append(
                    {"source": source, "target": target, "support_ratio": ratio, "support_documents": sorted(documents)}
                )

        transitions: list[WritingTransition] = []
        assumptions: list[dict[str, Any]] = []
        for source, target_ids in sorted(retained_edges.items()):
            unique_targets = sorted(set(target_ids))
            support_texts = {
                target: [item["evidence_excerpt"] for item in edge_examples[(source, target)]]
                for target in unique_targets
            }
            conditions = self.condition_inducer.induce(
                states[source], [states[target] for target in unique_targets], support_texts
            )
            by_target = {condition.target: condition for condition in conditions}
            if set(by_target) != set(unique_targets):
                raise ValueError(f"Condition inducer did not cover every target for source {source}.")
            for target in unique_targets:
                condition = by_target[target]
                documents = sorted(edge_documents[(source, target)])
                is_terminal_target = target == terminal_state_id
                transitions.append(
                    WritingTransition(
                        source=source,
                        symbol="END" if is_terminal_target else condition.symbol,
                        condition_description=(
                            "The current evidence and writing requirements indicate that the report should end."
                            if is_terminal_target
                            else condition.condition_description
                        ),
                        target=target,
                        support_count=len(documents),
                        confidence=condition.confidence,
                        support_examples=edge_examples[(source, target)][: self.max_support_examples],
                        metadata={"unconditional": True} if is_terminal_target else dict(condition.metadata),
                    )
                )
                if len(unique_targets) == 1 and condition.symbol == "COMPLETE" and not is_terminal_target:
                    assumptions.append(
                        {
                            "source": source,
                            "target": target,
                            "support_count": len(documents),
                            "reason": "MVP single-successor unconditional engineering assumption",
                        }
                    )

        transitions, discarded_conflicts = self.determinize(transitions)
        outgoing_sources = {transition.source for transition in transitions}
        reachable_from_initial: set[str] = {initial_state}
        pending = [initial_state]
        while pending:
            source = pending.pop()
            for transition in transitions:
                if transition.source == source and transition.target not in reachable_from_initial:
                    reachable_from_initial.add(transition.target)
                    pending.append(transition.target)
        final_states = {state_id for state_id in reachable_from_initial if state_id not in outgoing_sources}
        for state in states.values():
            state.is_initial = state.state_id == initial_state
            state.is_final = state.state_id in final_states

        dropped_states = [
            {
                "candidate_index": index,
                "support_documents": sorted({item[0] for item in candidates[index].nodes}),
                "reason": "below state_support_threshold",
            }
            for index in range(len(candidates))
            if index not in candidate_to_state
        ]
        metadata = {
            "schema_version": "1.0",
            "dfa_id": dfa_id,
            "collection_id": collection_id,
            "description": "AutoLogic two-report leaf-state union DFA",
            "build_config": {
                "semantic_merge_threshold": self.semantic_merge_threshold,
                "state_support_threshold": self.state_support_threshold,
                "transition_support_threshold": self.transition_support_threshold,
                "chat_model": DEFAULT_CHAT_MODEL,
                "embedding_model": self.embedder.model if self.embedder.backend == "api" else "local-char-ngram-hash",
                "max_support_examples": self.max_support_examples,
            },
            "construction_metadata": {
                "source_documents": [document_a, document_b],
                "source_document_count": 2,
                "state_provenance": provenance,
                "symbol_catalog": {},
                "determinism_conflicts": discarded_conflicts,
                "discarded_conflicts": discarded_conflicts,
                "dropped_states": dropped_states,
                "dropped_transitions": dropped_transitions,
                "single_successor_unconditional_assumptions": assumptions,
                "initial_state_selection": {
                    "rule": "maximum historical first-state count, then state_id ascending",
                    "counts": dict(sorted(first_counts.items())),
                },
                "historical_last_state_counts": dict(sorted(last_counts.items())),
                "final_state_selection": {
                    "rule": "reachable states with no outgoing valid transitions",
                    "states": sorted(final_states),
                },
                "projected_sequences": projected,
                "candidate_state_rule": "merged matches union unmatched A union unmatched B",
            },
        }
        dfa = WritingDFA(states, transitions, initial_state, final_states, metadata)
        validation = dfa.validate()
        validation.raise_if_invalid()
        if output_path is not None:
            dfa.save_json(output_path)
        return dfa

    def build_from_files(
        self,
        sequence_a_path: str | Path,
        sequence_b_path: str | Path,
        **kwargs: Any,
    ) -> WritingDFA:
        return self.build(
            self.load_artifact(sequence_a_path),
            self.load_artifact(sequence_b_path),
            **kwargs,
        )
