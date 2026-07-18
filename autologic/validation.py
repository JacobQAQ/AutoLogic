"""Semantic validation for AutoLogic DFA artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .models import RESERVED_NO_MATCH, STATE_ID_PATTERN, SYMBOL_PATTERN

if TYPE_CHECKING:
    from .models import WritingDFA


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, code: str, message: str, **context: Any) -> None:
        self.errors.append(ValidationIssue("error", code, message, context))

    def add_warning(self, code: str, message: str, **context: Any) -> None:
        self.warnings.append(ValidationIssue("warning", code, message, context))

    def raise_if_invalid(self) -> None:
        if self.errors:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
            raise DFAValidationError(details)


class DFAValidationError(ValueError):
    """Raised when a DFA has fatal semantic validation errors."""


def _states_reaching_final(dfa: "WritingDFA") -> set[str]:
    reverse: dict[str, set[str]] = {state_id: set() for state_id in dfa.states}
    for transition in dfa.transitions:
        if transition.source in reverse and transition.target in reverse:
            reverse[transition.target].add(transition.source)
    reached = {state_id for state_id in dfa.final_states if state_id in dfa.states}
    pending = list(reached)
    while pending:
        target = pending.pop()
        for source in reverse[target]:
            if source not in reached:
                reached.add(source)
                pending.append(source)
    return reached


def _strongly_connected_components(dfa: "WritingDFA", reachable: set[str]) -> list[set[str]]:
    adjacency: dict[str, list[str]] = {state_id: [] for state_id in reachable}
    for transition in dfa.transitions:
        if transition.source in reachable and transition.target in reachable:
            adjacency[transition.source].append(transition.target)

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in adjacency[node]:
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] == indices[node]:
            component: set[str] = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            components.append(component)

    for state_id in sorted(reachable):
        if state_id not in indices:
            visit(state_id)
    return components


def validate_dfa(dfa: "WritingDFA") -> ValidationResult:
    """Validate structural, deterministic, provenance, and termination invariants."""
    result = ValidationResult()
    state_ids = set(dfa.states)

    if dfa.initial_state not in state_ids:
        result.add_error("INITIAL_STATE_MISSING", "initial_state does not exist.", state=dfa.initial_state)
    missing_finals = sorted(dfa.final_states - state_ids)
    if missing_finals:
        result.add_error("FINAL_STATE_MISSING", "final_states contains unknown states.", states=missing_finals)

    pair_targets: dict[tuple[str, str], set[str]] = {}
    pair_counts: Counter[tuple[str, str]] = Counter()
    outgoing_counts = {state_id: 0 for state_id in state_ids}
    for index, transition in enumerate(dfa.transitions):
        if transition.source not in state_ids:
            result.add_error(
                "DANGLING_TRANSITION_SOURCE", "Transition source does not exist.", index=index, source=transition.source
            )
        else:
            outgoing_counts[transition.source] += 1
        if transition.target not in state_ids:
            result.add_error(
                "DANGLING_TRANSITION_TARGET", "Transition target does not exist.", index=index, target=transition.target
            )
        if not SYMBOL_PATTERN.fullmatch(transition.symbol) or transition.symbol == RESERVED_NO_MATCH:
            result.add_error("INVALID_SYMBOL", "Transition symbol is invalid or reserved.", symbol=transition.symbol)
        if not STATE_ID_PATTERN.fullmatch(transition.source) or not STATE_ID_PATTERN.fullmatch(transition.target):
            result.add_error(
                "INVALID_TRANSITION_ENDPOINT",
                "Transition endpoints must match the persisted state ID format.",
                source=transition.source,
                target=transition.target,
            )
        pair_targets.setdefault((transition.source, transition.symbol), set()).add(transition.target)
        pair_counts[(transition.source, transition.symbol)] += 1

    for (source, symbol), targets in sorted(pair_targets.items()):
        if len(targets) > 1:
            result.add_error(
                "NONDETERMINISTIC_TRANSITION",
                "One (source, symbol) pair points to multiple targets.",
                source=source,
                symbol=symbol,
                targets=sorted(targets),
            )
        elif pair_counts[(source, symbol)] > 1:
            result.add_error(
                "DUPLICATE_TRANSITION",
                "Persisted (source, symbol) pairs must be unique even when targets agree.",
                source=source,
                symbol=symbol,
            )

    initial_flags = sorted(state.state_id for state in dfa.states.values() if state.is_initial)
    if initial_flags != ([dfa.initial_state] if dfa.initial_state in state_ids else []):
        result.add_error(
            "INITIAL_FLAG_MISMATCH",
            "Exactly the initial_state must have is_initial=true.",
            flagged=initial_flags,
        )
    final_flags = {state.state_id for state in dfa.states.values() if state.is_final}
    if final_flags != dfa.final_states:
        result.add_error(
            "FINAL_FLAG_MISMATCH",
            "State is_final flags must exactly match final_states.",
            flagged=sorted(final_flags),
            declared=sorted(dfa.final_states),
        )

    construction = dfa.metadata.get("construction_metadata", {})
    provenance = construction.get("state_provenance", {}) if isinstance(construction, dict) else {}
    for state_id, state in sorted(dfa.states.items()):
        if not isinstance(state.required_materials, list) or not all(
            isinstance(item, str) and item for item in state.required_materials
        ):
            result.add_error("INVALID_REQUIRED_MATERIALS", "required_materials must be a string list.", state=state_id)
        entries = provenance.get(state_id, []) if isinstance(provenance, dict) else []
        if state.support_count != len(state.support_documents):
            result.add_error(
                "STATE_SUPPORT_MISMATCH",
                "support_count must equal len(support_documents).",
                state=state_id,
            )
        if state.state_kind == "writing":
            if state.support_count < 1 or not state.support_documents or not state.action.strip():
                result.add_error(
                    "INVALID_WRITING_STATE",
                    "Writing state requires positive support, support documents, and non-empty action.",
                    state=state_id,
                )
            if not isinstance(entries, list) or not entries:
                result.add_error("WRITING_PROVENANCE_MISSING", "Writing state requires provenance.", state=state_id)
        else:
            if (
                state.support_count != 0
                or state.support_documents
                or state.required_materials
                or state.action != ""
                or not state.is_final
            ):
                result.add_error(
                    "INVALID_TERMINAL_STATE",
                    "Terminal state requires zero support, empty documents/materials/action, and final status.",
                    state=state_id,
                )
            if outgoing_counts[state_id]:
                result.add_error("TERMINAL_HAS_OUTGOING", "Terminal control state cannot have outgoing edges.", state=state_id)
            incoming = [item for item in dfa.transitions if item.target == state_id]
            if any(item.symbol != "END" for item in incoming):
                result.add_error("TERMINAL_NON_END_INCOMING", "Terminal control state may only be targeted by END.", state=state_id)
            if entries:
                result.add_warning("TERMINAL_PROVENANCE_PRESENT", "Terminal provenance should be absent or empty.", state=state_id)

    expected_alphabet = sorted({item.symbol for item in dfa.transitions} | {RESERVED_NO_MATCH})
    persisted_alphabet = dfa.metadata.get("alphabet")
    if persisted_alphabet is not None and sorted(persisted_alphabet) != expected_alphabet:
        result.add_error(
            "ALPHABET_MISMATCH",
            "alphabet must equal valid transition symbols union NO_MATCH.",
            expected=expected_alphabet,
        )
    rejection = dfa.metadata.get("implicit_rejection_state")
    if rejection is not None and rejection != "s_bottom":
        result.add_error("INVALID_REJECTION_STATE", "implicit_rejection_state must be s_bottom.")

    for state_id in sorted(dfa.final_states & state_ids):
        if outgoing_counts[state_id]:
            result.add_error("FINAL_HAS_OUTGOING", "Final state must be a valid-edge sink.", state=state_id)

    reachable = dfa.reachable_states()
    unreachable = sorted(state_ids - reachable)
    if unreachable:
        result.add_warning("UNREACHABLE_STATES", "DFA contains unreachable states.", states=unreachable)
    for state_id in sorted(reachable - dfa.final_states):
        if outgoing_counts.get(state_id, 0) == 0:
            result.add_error(
                "REACHABLE_NONFINAL_SINK", "Reachable non-final state must have an outgoing edge.", state=state_id
            )
    if not (reachable & dfa.final_states):
        result.add_error("NO_REACHABLE_FINAL", "At least one final state must be reachable from initial_state.")

    can_reach_final = _states_reaching_final(dfa)
    for component in _strongly_connected_components(dfa, reachable):
        has_self_loop = any(
            transition.source == transition.target and transition.source in component
            for transition in dfa.transitions
        )
        if len(component) > 1 or has_self_loop:
            if not (component & can_reach_final):
                result.add_warning(
                    "POTENTIALLY_UNBOUNDED_CYCLE",
                    "Reachable cycle has no path to a final state; runtime guards are required.",
                    states=sorted(component),
                )
            else:
                result.add_warning(
                    "LEGAL_CYCLE_REQUIRES_GUARD",
                    "Reachable cycle can terminate but still requires max-step/visit guards.",
                    states=sorted(component),
                )

    try:
        encoded = json.dumps(dfa.to_dict(), ensure_ascii=False)
        type(dfa).from_dict(json.loads(encoded))
    except (TypeError, ValueError) as exc:
        result.add_error("JSON_ROUNDTRIP_FAILED", "DFA cannot round-trip through JSON.", error=str(exc))
    return result
