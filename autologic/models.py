"""Typed data models for AutoLogic's persisted writing DFA."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


STATE_ID_PATTERN = re.compile(r"^S[0-9]{3,}$")
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
RESERVED_NO_MATCH = "NO_MATCH"
IMPLICIT_REJECTION_STATE = "s_bottom"


class AutoLogicError(Exception):
    """Base class for clear AutoLogic failures."""


class ModelValidationError(AutoLogicError, ValueError):
    """Raised when one model contains invalid field values."""


class UndefinedTransitionError(AutoLogicError, LookupError):
    """Raised when the sparse transition table has no requested edge."""


class NonDeterministicTransitionError(AutoLogicError, ValueError):
    """Raised when one (source, symbol) pair has multiple targets."""


def normalize_symbol(value: str) -> str:
    """Normalize arbitrary condition text to a finite UPPER_SNAKE_CASE symbol."""
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError("Transition symbol must be a non-empty string.")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        raise ModelValidationError("Transition symbol is empty after normalization.")
    if normalized[0].isdigit():
        normalized = f"CONDITION_{normalized}"
    if len(normalized) > 64:
        normalized = normalized[:64].rstrip("_")
    if not SYMBOL_PATTERN.fullmatch(normalized):
        raise ModelValidationError(f"Invalid normalized transition symbol: {normalized!r}.")
    return normalized


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ModelValidationError(f"{field_name} must be a list of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ModelValidationError(f"{field_name} must contain only non-empty strings.")
        clean = item.strip()
        if clean not in result:
            result.append(clean)
    return result


@dataclass
class WritingState:
    state_id: str
    label: str
    description: str
    action: str
    required_materials: list[str]
    support_count: int
    support_documents: list[str]
    is_initial: bool = False
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    state_kind: str = "writing"

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not STATE_ID_PATTERN.fullmatch(self.state_id):
            raise ModelValidationError("state_id must match ^S[0-9]{3,}$.")
        if self.state_kind not in {"writing", "terminal"}:
            raise ModelValidationError("state_kind must be 'writing' or 'terminal'.")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ModelValidationError("State label must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ModelValidationError("State description must be a non-empty string.")
        if not isinstance(self.action, str):
            raise ModelValidationError("State action must be a string.")
        self.required_materials = _string_list(self.required_materials, "required_materials")
        self.support_documents = _string_list(self.support_documents, "support_documents")
        if isinstance(self.support_count, bool) or not isinstance(self.support_count, int):
            raise ModelValidationError("support_count must be an integer.")
        if not isinstance(self.is_initial, bool) or not isinstance(self.is_final, bool):
            raise ModelValidationError("is_initial and is_final must be booleans.")
        if not isinstance(self.metadata, dict):
            raise ModelValidationError("State metadata must be a dictionary.")
        if self.support_count != len(self.support_documents):
            raise ModelValidationError("support_count must equal len(support_documents).")
        if self.state_kind == "writing":
            if self.support_count < 1 or not self.action.strip():
                raise ModelValidationError(
                    "Writing states require positive support and a non-empty action."
                )
        else:
            if self.support_count != 0 or self.support_documents:
                raise ModelValidationError("Terminal states require zero historical support.")
            if self.required_materials or self.action != "" or not self.is_final:
                raise ModelValidationError(
                    "Terminal states require empty materials/action and is_final=true."
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WritingState":
        if not isinstance(data, Mapping):
            raise ModelValidationError("WritingState input must be an object.")
        state_id = data.get("state_id") or data.get("id") or data.get("node_id")
        label = (
            data.get("label")
            or data.get("template_name")
            or data.get("template_description")
            or ""
        )
        description = data.get("description") or data.get("content_guideline") or data.get("desc") or label
        action = data.get("action")
        if action is None:
            action = data.get("content_guideline") or description
        support_documents = data.get("support_documents")
        if support_documents is None:
            support = data.get("support", [])
            if isinstance(support, list):
                support_documents = [
                    str(item.get("document_id"))
                    for item in support
                    if isinstance(item, Mapping) and item.get("document_id")
                ]
            else:
                support_documents = []
        support_documents = list(dict.fromkeys(support_documents or []))
        support_count = data.get("support_count", len(support_documents))
        state_kind = str(data.get("state_kind", "writing"))
        return cls(
            state_id=str(state_id or ""),
            state_kind=state_kind,
            label=str(label or "").strip(),
            description=str(description or "").strip(),
            action=str(action or "") if state_kind == "writing" else str(action or ""),
            required_materials=list(data.get("required_materials") or []),
            support_count=support_count,
            support_documents=support_documents,
            is_initial=data.get("is_initial", False),
            is_final=data.get("is_final", False),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "state_id": self.state_id,
            "state_kind": self.state_kind,
            "label": self.label,
            "description": self.description,
            "action": self.action,
            "required_materials": list(self.required_materials),
            "support_count": self.support_count,
            "support_documents": list(self.support_documents),
            "is_initial": self.is_initial,
            "is_final": self.is_final,
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass
class WritingTransition:
    source: str
    symbol: str
    condition_description: str
    target: str
    support_count: int
    confidence: float
    support_examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ModelValidationError("Transition source must be a non-empty string.")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ModelValidationError("Transition target must be a non-empty string.")
        self.symbol = normalize_symbol(self.symbol)
        if self.symbol == RESERVED_NO_MATCH:
            raise ModelValidationError("NO_MATCH is reserved and cannot be persisted as a valid edge.")
        if not isinstance(self.condition_description, str) or not self.condition_description.strip():
            raise ModelValidationError("condition_description must be a non-empty string.")
        if isinstance(self.support_count, bool) or not isinstance(self.support_count, int) or self.support_count < 1:
            raise ModelValidationError("Transition support_count must be an integer >= 1.")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ModelValidationError("Transition confidence must be numeric.")
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ModelValidationError("Transition confidence must be between 0 and 1.")
        if not isinstance(self.support_examples, list) or not all(
            isinstance(item, dict) for item in self.support_examples
        ):
            raise ModelValidationError("support_examples must be a list of objects.")
        if not isinstance(self.metadata, dict):
            raise ModelValidationError("Transition metadata must be a dictionary.")
        if set(self.metadata) - {"unconditional"}:
            raise ModelValidationError("Transition metadata only permits 'unconditional'.")
        if "unconditional" in self.metadata and not isinstance(self.metadata["unconditional"], bool):
            raise ModelValidationError("metadata.unconditional must be boolean.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WritingTransition":
        if not isinstance(data, Mapping):
            raise ModelValidationError("WritingTransition input must be an object.")
        return cls(
            source=str(data.get("source") or ""),
            symbol=str(data.get("symbol") or ""),
            condition_description=str(data.get("condition_description") or data.get("condition") or ""),
            target=str(data.get("target") or ""),
            support_count=data.get("support_count", data.get("frequency", 1)),
            confidence=data.get("confidence", 1.0),
            support_examples=list(data.get("support_examples") or []),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "source": self.source,
            "symbol": self.symbol,
            "condition_description": self.condition_description,
            "target": self.target,
            "support_count": self.support_count,
            "confidence": self.confidence,
            "support_examples": [dict(item) for item in self.support_examples],
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass
class WritingDFA:
    states: dict[str, WritingState]
    transitions: list[WritingTransition]
    initial_state: str
    final_states: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.states, dict):
            raise ModelValidationError("states must be a dictionary keyed by state_id.")
        normalized: dict[str, WritingState] = {}
        for key, value in self.states.items():
            state = value if isinstance(value, WritingState) else WritingState.from_dict(value)
            if str(key) != state.state_id:
                raise ModelValidationError(f"State dictionary key {key!r} does not match {state.state_id!r}.")
            normalized[state.state_id] = state
        self.states = normalized
        self.transitions = [
            item if isinstance(item, WritingTransition) else WritingTransition.from_dict(item)
            for item in self.transitions
        ]
        self.final_states = set(self.final_states)
        if not isinstance(self.initial_state, str) or not self.initial_state:
            raise ModelValidationError("initial_state must be a non-empty string.")
        if not isinstance(self.metadata, dict):
            raise ModelValidationError("DFA metadata must be a dictionary.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WritingDFA":
        if not isinstance(data, Mapping):
            raise ModelValidationError("WritingDFA input must be an object.")
        raw_states = data.get("states", [])
        if isinstance(raw_states, Mapping):
            states = {str(key): WritingState.from_dict(value) for key, value in raw_states.items()}
        elif isinstance(raw_states, list):
            parsed = [WritingState.from_dict(item) for item in raw_states]
            states = {state.state_id: state for state in parsed}
            if len(states) != len(parsed):
                raise ModelValidationError("Duplicate state_id values are not allowed.")
        else:
            raise ModelValidationError("states must be an array or object.")
        schema_keys = {
            "schema_version", "dfa_id", "collection_id", "description", "alphabet",
            "implicit_rejection_state", "build_config", "construction_metadata",
        }
        metadata = {key: data[key] for key in schema_keys if key in data}
        if "metadata" in data:  # compatibility with early/internal representations
            metadata.update(dict(data.get("metadata") or {}))
        return cls(
            states=states,
            transitions=[WritingTransition.from_dict(item) for item in data.get("transitions", [])],
            initial_state=str(data.get("initial_state") or ""),
            final_states=set(data.get("final_states") or []),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        symbols = sorted({transition.symbol for transition in self.transitions} | {RESERVED_NO_MATCH})
        construction = dict(self.metadata.get("construction_metadata") or {})
        construction.setdefault("source_documents", [])
        construction.setdefault("source_document_count", len(construction["source_documents"]))
        construction.setdefault("state_provenance", {})
        construction.setdefault("symbol_catalog", {})
        construction.setdefault("determinism_conflicts", [])
        construction.setdefault("dropped_states", [])
        construction.setdefault("dropped_transitions", [])
        construction.setdefault("single_successor_unconditional_assumptions", [])
        build_config = dict(self.metadata.get("build_config") or {})
        build_config.setdefault("semantic_merge_threshold", 0.85)
        build_config.setdefault("state_support_threshold", 0.5)
        build_config.setdefault("transition_support_threshold", 0.5)
        build_config.setdefault("chat_model", "deepseek-chat")
        build_config.setdefault("embedding_model", "local-char-ngram-hash")
        build_config.setdefault("max_support_examples", 3)
        return {
            "schema_version": self.metadata.get("schema_version", "1.0"),
            "dfa_id": self.metadata.get("dfa_id", "autologic_dfa"),
            "collection_id": self.metadata.get("collection_id", "autologic_collection"),
            "description": self.metadata.get("description", ""),
            "alphabet": symbols,
            "implicit_rejection_state": IMPLICIT_REJECTION_STATE,
            "initial_state": self.initial_state,
            "final_states": sorted(self.final_states),
            "states": [self.states[key].to_dict() for key in sorted(self.states)],
            "transitions": [item.to_dict() for item in self.transitions],
            "build_config": build_config,
            "construction_metadata": construction,
        }

    @classmethod
    def load_json(cls, path: str | Path) -> "WritingDFA":
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as file:
                return cls.from_dict(json.load(file))
        except json.JSONDecodeError as exc:
            raise ModelValidationError(f"Invalid DFA JSON in {path}: {exc}") from exc

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    def outgoing(self, state_id: str) -> list[WritingTransition]:
        if state_id not in self.states:
            raise KeyError(f"Unknown DFA state: {state_id!r}.")
        return [transition for transition in self.transitions if transition.source == state_id]

    def delta(self, state_id: str, symbol: str) -> str:
        if state_id not in self.states:
            raise KeyError(f"Unknown DFA state: {state_id!r}.")
        normalized = normalize_symbol(symbol)
        matches = [
            transition.target
            for transition in self.transitions
            if transition.source == state_id and transition.symbol == normalized
        ]
        unique = sorted(set(matches))
        if not unique:
            raise UndefinedTransitionError(
                f"Undefined transition ({state_id}, {normalized}); formally it maps to {IMPLICIT_REJECTION_STATE}."
            )
        if len(unique) != 1:
            raise NonDeterministicTransitionError(
                f"Non-deterministic transition ({state_id}, {normalized}) has targets {unique}."
            )
        return unique[0]

    def is_final(self, state_id: str) -> bool:
        if state_id not in self.states:
            raise KeyError(f"Unknown DFA state: {state_id!r}.")
        return state_id in self.final_states

    def reachable_states(self) -> set[str]:
        if self.initial_state not in self.states:
            return set()
        reached = {self.initial_state}
        pending = [self.initial_state]
        while pending:
            source = pending.pop()
            for transition in self.transitions:
                if transition.source == source and transition.target not in reached:
                    reached.add(transition.target)
                    pending.append(transition.target)
        return reached

    def validate(self):
        from .validation import validate_dfa

        return validate_dfa(self)
