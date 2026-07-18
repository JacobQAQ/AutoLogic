"""Dynamic online execution of one complete AutoLogic writing DFA."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from .adapters import (
    AutoLogicChatAdapter,
    GeneratedContent,
    IFindStateRetriever,
    StateContentGenerator,
    StateEvidence,
)
from .models import WritingDFA, WritingState, WritingTransition


class ExecutionError(RuntimeError):
    """Base error for incomplete AutoLogic execution."""


class ExecutionGuardError(ExecutionError):
    """Raised when a deterministic cycle/step guard is exceeded."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass(frozen=True)
class ConditionDecision:
    symbol: str | None
    reason: str
    confidence: float
    used_fallback: bool = False
    fallback_reason: str = ""
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "reason": self.reason,
            "confidence": self.confidence,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "raw_response": self.raw_response,
        }


class EvidenceRetriever(Protocol):
    def retrieve_state(
        self,
        state: WritingState,
        query: str,
        date: str | None = None,
        dry_run: bool = False,
        asset_name: str | None = None,
    ) -> StateEvidence: ...


class ContentGenerator(Protocol):
    def generate_current_state(
        self,
        *,
        query: str,
        state: WritingState,
        evidence: StateEvidence,
        memory: str,
        dry_run: bool = False,
    ) -> GeneratedContent: ...


@dataclass
class CompactMemory:
    max_entries: int = 3
    max_chars: int = 1200
    entries: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        text = json.dumps(self.entries[-self.max_entries :], ensure_ascii=False, default=str)
        return text[-self.max_chars :]

    def update(
        self,
        state: WritingState,
        content: str,
        evidence: StateEvidence,
        decision: ConditionDecision,
    ) -> None:
        self.entries.append(
            {
                "state_id": state.state_id,
                "summary": re.sub(r"\s+", " ", content).strip()[:280],
                "evidence": re.sub(r"\s+", " ", evidence.summary).strip()[:280],
                "missing_materials": list(evidence.errors)[:3],
                "selected_symbol": decision.symbol,
                "condition_reason": decision.reason[:240],
            }
        )
        self.entries = self.entries[-self.max_entries :]


def _numeric_direction(records: Sequence[dict[str, Any]]) -> int:
    preferred = {"changeratio", "change_ratio", "change", "chg_settlement", "change_settlement"}
    values: list[float] = []
    for record in records:
        for key, value in record.items():
            if str(key).replace("%", "").casefold() not in preferred:
                continue
            if isinstance(value, bool):
                continue
            try:
                number = float(str(value).strip().rstrip("%"))
            except (TypeError, ValueError):
                continue
            if number != 0:
                values.append(number)
    if not values:
        return 0
    if all(value > 0 for value in values):
        return 1
    if all(value < 0 for value in values):
        return -1
    return 0


def _text_direction(text: str) -> int:
    lowered = text.casefold()
    up_patterns = (
        r"上涨", r"上升", r"走高", r"\bincrease(?:d|s|ing)?\b", r"\b(?:rise|rises|rose|rising)\b",
        r"\bpositive\b", r"\bbullish\b", r"\bprice_up\b",
    )
    down_patterns = (
        r"下跌", r"下降", r"走低", r"\bdecrease(?:d|s|ing)?\b", r"\b(?:fall|falls|fell|falling)\b",
        r"\bnegative\b", r"\bbearish\b", r"\bprice_down\b",
    )
    if any(re.search(pattern, lowered) for pattern in up_patterns):
        return 1
    if any(re.search(pattern, lowered) for pattern in down_patterns):
        return -1
    return 0


def _rank_transition(transition: WritingTransition) -> tuple[float, float, str, str]:
    return (-transition.support_count, -transition.confidence, transition.symbol, transition.target)


class ConditionGrounder:
    """Ground one finite outgoing symbol without allowing target generation."""

    def __init__(self, chat: AutoLogicChatAdapter | None = None) -> None:
        self.chat = chat or AutoLogicChatAdapter()

    @staticmethod
    def _direct_transition(outgoing: Sequence[WritingTransition]) -> WritingTransition | None:
        if len(outgoing) != 1:
            return None
        transition = outgoing[0]
        if transition.symbol in {"COMPLETE", "END"} or transition.metadata.get("unconditional") is True:
            return transition
        return None

    @staticmethod
    def _deterministic_decision(
        evidence: StateEvidence,
        generated_content: str,
        memory: str,
        outgoing: Sequence[WritingTransition],
    ) -> ConditionDecision | None:
        by_symbol = {transition.symbol: transition for transition in outgoing}
        direction = _numeric_direction(evidence.records)
        reason_source = "structured current-state records"
        if direction == 0:
            direction = _text_direction(f"{evidence.summary}\n{generated_content}")
            reason_source = "current-state evidence summary and generated content"
        if direction > 0 and "PRICE_UP" in by_symbol:
            return ConditionDecision("PRICE_UP", f"Positive price direction found in {reason_source}.", 1.0)
        if direction < 0 and "PRICE_DOWN" in by_symbol:
            return ConditionDecision("PRICE_DOWN", f"Negative price direction found in {reason_source}.", 1.0)

        # For non-price conditions, look for meaningful condition words only in
        # current evidence/content. Memory is deliberately a secondary aid.
        current_text = f"{evidence.summary} {generated_content}".casefold()
        memory_text = memory.casefold()
        scored: list[tuple[int, WritingTransition]] = []
        ignored = {
            "current", "evidence", "shows", "indicates", "that", "this", "state",
            "condition", "the", "and", "with", "from", "data", "report",
        }
        for transition in outgoing:
            words = {
                word.casefold()
                for word in re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", transition.condition_description)
                if word.casefold() not in ignored
            }
            score = sum(2 for word in words if word in current_text)
            score += sum(1 for word in words if word in memory_text)
            if score:
                scored.append((score, transition))
        if scored:
            scored.sort(key=lambda item: (-item[0], *_rank_transition(item[1])))
            best = scored[0]
            if len(scored) == 1 or best[0] > scored[1][0]:
                return ConditionDecision(
                    best[1].symbol,
                    "Condition-description keywords are grounded in current evidence/content.",
                    min(0.95, 0.55 + 0.1 * best[0]),
                )
        return None

    @staticmethod
    def _fallback(outgoing: Sequence[WritingTransition], reason: str) -> ConditionDecision:
        selected = sorted(outgoing, key=_rank_transition)[0]
        return ConditionDecision(
            symbol=selected.symbol,
            reason="Classifier failed; selected the deterministic highest-support legal transition.",
            confidence=selected.confidence,
            used_fallback=True,
            fallback_reason=reason,
        )

    def ground_condition(
        self,
        *,
        state: WritingState,
        evidence: StateEvidence,
        generated_content: str,
        memory: str,
        outgoing_transitions: Sequence[WritingTransition],
        dry_run: bool = False,
    ) -> ConditionDecision:
        del state
        outgoing = list(outgoing_transitions)
        if not outgoing:
            return ConditionDecision(None, "Current state has no outgoing transitions.", 1.0)
        direct = self._direct_transition(outgoing)
        if direct is not None:
            return ConditionDecision(
                direct.symbol,
                "The only outgoing transition is explicitly unconditional.",
                1.0,
            )

        deterministic = self._deterministic_decision(evidence, generated_content, memory, outgoing)
        if deterministic is not None:
            return deterministic
        if dry_run:
            return self._fallback(outgoing, "DRY_RUN_NO_EXPLICIT_CONDITION_MATCH")

        candidates = [
            {"symbol": transition.symbol, "condition_description": transition.condition_description}
            for transition in outgoing
        ]
        allowed = {transition.symbol for transition in outgoing}
        prompt = f"""
Select exactly one satisfied symbol from the finite candidates, or NO_MATCH if none is factually satisfied.
Do not generate or mention a next state or target identity.
Return JSON only: {{"symbol":"PRICE_UP","reason":"...","confidence":0.93}}

Candidates:
{json.dumps(candidates, ensure_ascii=False)}

Current evidence:
{json.dumps(evidence.to_dict(), ensure_ascii=False, default=str)}

Current generated content:
{generated_content}

Bounded prior memory:
{memory or "None."}
""".strip()
        errors: list[str] = []
        for attempt in range(2):
            try:
                payload = self.chat.complete_json(
                    system="You are a strict evidence-grounded finite-condition classifier.",
                    user=prompt,
                    max_tokens=500,
                    retries=0,
                )
                symbol = str(payload.get("symbol") or "").strip().upper()
                reason = str(payload.get("reason") or "").strip()
                confidence = float(payload.get("confidence"))
                if symbol == "NO_MATCH":
                    if not 0.0 <= confidence <= 1.0 or not reason:
                        raise ValueError("NO_MATCH response has invalid reason/confidence.")
                    return ConditionDecision(symbol, reason, confidence, raw_response=json.dumps(payload, ensure_ascii=False))
                if symbol not in allowed:
                    raise ValueError(f"Classifier symbol {symbol!r} is not in the outgoing candidate set.")
                if not reason or not 0.0 <= confidence <= 1.0:
                    raise ValueError("Classifier reason/confidence is invalid.")
                return ConditionDecision(symbol, reason, confidence, raw_response=json.dumps(payload, ensure_ascii=False))
            except Exception as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
        return self._fallback(outgoing, "INVALID_CLASSIFIER_OUTPUT_AFTER_RETRY: " + " | ".join(errors))


def ground_condition(
    state: WritingState,
    evidence: StateEvidence,
    generated_content: str,
    memory: str,
    outgoing_transitions: Sequence[WritingTransition],
    *,
    chat: AutoLogicChatAdapter | None = None,
    dry_run: bool = False,
) -> ConditionDecision:
    return ConditionGrounder(chat).ground_condition(
        state=state,
        evidence=evidence,
        generated_content=generated_content,
        memory=memory,
        outgoing_transitions=outgoing_transitions,
        dry_run=dry_run,
    )


@dataclass
class ExecutionResult:
    generated_report: str
    generated_states: list[dict[str, Any]]
    execution_trace: list[dict[str, Any]]
    run_manifest: dict[str, Any]
    success: bool

    def save_outputs(self, output_dir: str | Path) -> dict[str, str]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "generated_report": directory / "generated_report.md",
            "generated_states": directory / "generated_states.json",
            "execution_trace": directory / "execution_trace.json",
            "run_manifest": directory / "run_manifest.json",
        }
        paths["generated_report"].write_text(self.generated_report, encoding="utf-8")
        for key, payload in (
            ("generated_states", self.generated_states),
            ("execution_trace", self.execution_trace),
            ("run_manifest", self.run_manifest),
        ):
            with paths[key].open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        return {key: str(path) for key, path in paths.items()}


class AutoLogicExecutor:
    """Execute only the states selected dynamically through WritingDFA.delta()."""

    def __init__(
        self,
        dfa: WritingDFA,
        *,
        evidence_retriever: EvidenceRetriever | None = None,
        content_generator: ContentGenerator | None = None,
        condition_grounder: ConditionGrounder | None = None,
        max_steps: int | None = None,
        max_visits_per_state: int = 2,
        max_transition_repeats: int = 2,
        dfa_path: str = "",
    ) -> None:
        self.dfa = dfa
        self.evidence_retriever = evidence_retriever or IFindStateRetriever()
        self.content_generator = content_generator or StateContentGenerator()
        self.condition_grounder = condition_grounder or ConditionGrounder()
        self.max_steps = max_steps if max_steps is not None else max(2 * len(dfa.states), 20)
        self.max_visits_per_state = max_visits_per_state
        self.max_transition_repeats = max_transition_repeats
        self.dfa_path = dfa_path
        for name, value in (
            ("max_steps", self.max_steps),
            ("max_visits_per_state", max_visits_per_state),
            ("max_transition_repeats", max_transition_repeats),
        ):
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    def _model_name(self) -> str:
        chat = getattr(self.content_generator, "chat", None)
        return str(getattr(chat, "model", "custom-or-injected"))

    @staticmethod
    def _candidate_payload(outgoing: Sequence[WritingTransition]) -> list[dict[str, Any]]:
        return [
            {
                "symbol": item.symbol,
                "condition_description": item.condition_description,
                "support_count": item.support_count,
                "confidence": item.confidence,
            }
            for item in outgoing
        ]

    def execute(
        self,
        *,
        query: str,
        date: str | None = None,
        asset_name: str | None = None,
        dry_run: bool = False,
        output_dir: str | Path | None = None,
    ) -> ExecutionResult:
        validation = self.dfa.validate()
        validation.raise_if_invalid()
        current = self.dfa.initial_state
        memory = CompactMemory()
        generated_states: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        report_segments: list[str] = []
        visit_counts: Counter[str] = Counter()
        transition_counts: Counter[tuple[str, str, str]] = Counter()
        termination_reason = ""
        success = False
        step = 0

        try:
            while True:
                if step >= self.max_steps:
                    raise ExecutionGuardError(
                        "MAX_STEPS", f"Execution exceeded max_steps={self.max_steps}."
                    )
                step += 1
                visit_counts[current] += 1
                if visit_counts[current] > self.max_visits_per_state:
                    raise ExecutionGuardError(
                        "MAX_VISITS_PER_STATE",
                        f"State {current} exceeded max_visits_per_state={self.max_visits_per_state}.",
                    )

                state = self.dfa.states[current]
                if state.state_kind == "terminal":
                    generated_states.append(
                        {
                            "step": step,
                            "state_id": state.state_id,
                            "state_kind": state.state_kind,
                            "label": state.label,
                            "action": state.action,
                            "generated_content": "",
                            "generation_metadata": {"terminal_control": True},
                            "evidence": None,
                        }
                    )
                    trace.append(
                        {
                            "step": step,
                            "current_state": state.state_id,
                            "state_label": state.label,
                            "state_action": state.action,
                            "evidence_status": "not_applicable",
                            "evidence_summary": "Terminal control state does not retrieve data.",
                            "generated_content": "",
                            "candidate_conditions": [],
                            "selected_condition": None,
                            "condition_reason": "Terminal control state reached.",
                            "condition_confidence": 1.0,
                            "next_state": None,
                            "used_fallback": False,
                            "fallback_reason": "",
                        }
                    )
                    termination_reason = "FINAL_TERMINAL_STATE"
                    success = True
                    break

                evidence = self.evidence_retriever.retrieve_state(
                    state, query, date=date, dry_run=dry_run, asset_name=asset_name
                )
                generated = self.content_generator.generate_current_state(
                    query=query,
                    state=state,
                    evidence=evidence,
                    memory=memory.render(),
                    dry_run=dry_run,
                )
                content = generated.generated_content
                report_segments.append(content)
                generated_states.append(
                    {
                        "step": step,
                        "state_id": state.state_id,
                        "state_kind": state.state_kind,
                        "label": state.label,
                        "action": state.action,
                        "generated_content": content,
                        "generation_metadata": dict(generated.metadata),
                        "evidence": evidence.to_dict(),
                    }
                )

                outgoing = self.dfa.outgoing(current)
                trace_record = {
                    "step": step,
                    "current_state": state.state_id,
                    "state_label": state.label,
                    "state_action": state.action,
                    "evidence_status": evidence.status,
                    "evidence_summary": evidence.summary,
                    "generated_content": content,
                    "candidate_conditions": self._candidate_payload(outgoing),
                    "selected_condition": None,
                    "condition_reason": "",
                    "condition_confidence": 0.0,
                    "next_state": None,
                    "used_fallback": False,
                    "fallback_reason": "",
                }

                if self.dfa.is_final(current):
                    trace_record["condition_reason"] = "Final writing state generated successfully."
                    trace_record["condition_confidence"] = 1.0
                    trace.append(trace_record)
                    termination_reason = "FINAL_WRITING_STATE"
                    success = True
                    break
                if not outgoing:
                    trace_record["condition_reason"] = "No outgoing transition is available."
                    trace.append(trace_record)
                    termination_reason = "NO_OUTGOING_TRANSITION"
                    success = True
                    break

                decision = self.condition_grounder.ground_condition(
                    state=state,
                    evidence=evidence,
                    generated_content=content,
                    memory=memory.render(),
                    outgoing_transitions=outgoing,
                    dry_run=dry_run,
                )
                trace_record.update(
                    {
                        "selected_condition": decision.symbol,
                        "condition_reason": decision.reason,
                        "condition_confidence": decision.confidence,
                        "used_fallback": decision.used_fallback,
                        "fallback_reason": decision.fallback_reason,
                    }
                )
                if decision.symbol == "NO_MATCH":
                    trace_record["next_state"] = "s_bottom"
                    trace_record["status"] = "CONDITION_NOT_SATISFIED"
                    trace.append(trace_record)
                    termination_reason = "CONDITION_NOT_SATISFIED"
                    success = False
                    break
                if decision.symbol is None:
                    raise ExecutionError(f"Condition grounding returned no symbol for non-final state {current}.")

                next_state = self.dfa.delta(current, decision.symbol)
                edge = (current, decision.symbol, next_state)
                transition_counts[edge] += 1
                if transition_counts[edge] > self.max_transition_repeats:
                    raise ExecutionGuardError(
                        "MAX_TRANSITION_REPEATS",
                        f"Transition {edge} exceeded max_transition_repeats={self.max_transition_repeats}.",
                    )
                trace_record["next_state"] = next_state
                trace.append(trace_record)
                memory.update(state, content, evidence, decision)
                current = next_state
        finally:
            close = getattr(self.evidence_retriever, "close", None)
            if callable(close):
                close()

        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "dfa_path": self.dfa_path,
            "dfa_id": self.dfa.metadata.get("dfa_id", ""),
            "executed_state_path": [item["state_id"] for item in generated_states],
            "dry_run": dry_run,
            "model": self._model_name(),
            "max_steps": self.max_steps,
            "max_visits_per_state": self.max_visits_per_state,
            "max_transition_repeats": self.max_transition_repeats,
            "total_executed_states": len(generated_states),
            "final_state": current,
            "termination_reason": termination_reason,
            "success": success,
        }
        result = ExecutionResult(
            generated_report="\n\n".join(report_segments).strip(),
            generated_states=generated_states,
            execution_trace=trace,
            run_manifest=manifest,
            success=success,
        )
        if output_dir is not None:
            result.save_outputs(output_dir)
        return result
