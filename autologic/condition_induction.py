"""Offline induction of finite, evidence-groundable transition conditions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from document_learner import extract_json_object
from report_generator import ChatClient

from .models import ModelValidationError, WritingState, normalize_symbol


@dataclass(frozen=True)
class InducedCondition:
    target: str
    symbol: str
    condition_description: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target:
            raise ModelValidationError("Induced condition target cannot be empty.")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not self.condition_description.strip():
            raise ModelValidationError("Induced condition description cannot be empty.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ModelValidationError("Induced condition confidence must be between 0 and 1.")


class ConditionInducer(Protocol):
    def induce(
        self,
        source: WritingState,
        targets: Sequence[WritingState],
        support_texts: Mapping[str, Sequence[str]] | None = None,
    ) -> list[InducedCondition]:
        """Return exactly one normalized condition for each candidate target."""


_CONCEPT_PATTERNS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("price", "up"), "PRICE_UP", "Current evidence shows that price is increasing."),
    (("price", "rise"), "PRICE_UP", "Current evidence shows that price is increasing."),
    (("价格", "上涨"), "PRICE_UP", "当前证据显示价格上涨。"),
    (("price", "down"), "PRICE_DOWN", "Current evidence shows that price is decreasing."),
    (("price", "fall"), "PRICE_DOWN", "Current evidence shows that price is decreasing."),
    (("价格", "下跌"), "PRICE_DOWN", "当前证据显示价格下跌。"),
    (("supply", "tight"), "SUPPLY_TIGHT", "Current evidence shows that supply is tight."),
    (("供给", "紧"), "SUPPLY_TIGHT", "当前证据显示供给偏紧。"),
    (("supply", "loose"), "SUPPLY_LOOSE", "Current evidence shows that supply is loose."),
    (("供给", "宽"), "SUPPLY_LOOSE", "当前证据显示供给宽松。"),
    (("demand", "strong"), "DEMAND_STRONG", "Current evidence shows strong demand."),
    (("需求", "强"), "DEMAND_STRONG", "当前证据显示需求较强。"),
    (("demand", "weak"), "DEMAND_WEAK", "Current evidence shows weak demand."),
    (("需求", "弱"), "DEMAND_WEAK", "当前证据显示需求较弱。"),
    (("risk", "increase"), "RISK_INCREASED", "Current evidence shows increased risk."),
    (("风险", "上升"), "RISK_INCREASED", "当前证据显示风险上升。"),
)


def _semantic_slug(text: str) -> str:
    latin = re.findall(r"[A-Za-z0-9]+", text.upper())
    ignored = {"THE", "A", "AN", "AND", "OR", "STATE", "WRITING", "ANALYSIS"}
    words = [word for word in latin if word not in ignored][:4]
    if words:
        return normalize_symbol("_".join(words))
    # Do not derive a symbol from target ID. A digest of semantic text is stable
    # and only disambiguates otherwise unclassifiable offline fixture branches.
    codepoints = sum((index + 1) * ord(char) for index, char in enumerate(text)) % 10000
    return f"EVIDENCE_PATTERN_{codepoints:04d}"


def _contains_concept(text: str, token: str) -> bool:
    if token.isascii() and re.fullmatch(r"[A-Za-z0-9]+", token):
        return re.search(rf"\b{re.escape(token.casefold())}\b", text) is not None
    return token.casefold() in text


class HeuristicConditionInducer:
    """Deterministic, keyless condition induction for tests and dry-runs."""

    def induce(
        self,
        source: WritingState,
        targets: Sequence[WritingState],
        support_texts: Mapping[str, Sequence[str]] | None = None,
    ) -> list[InducedCondition]:
        del source  # target contrasts are allowed offline; runtime conditions remain target-free.
        if not targets:
            return []
        ordered = sorted(targets, key=lambda item: (item.label, item.description, item.state_id))
        if len(ordered) == 1:
            target = ordered[0]
            return [
                InducedCondition(
                    target=target.state_id,
                    symbol="COMPLETE",
                    condition_description="The current writing action is complete.",
                    confidence=1.0,
                    metadata={"unconditional": True},
                )
            ]

        used: set[str] = set()
        results: list[InducedCondition] = []
        for ordinal, target in enumerate(ordered, start=1):
            evidence = " ".join((support_texts or {}).get(target.state_id, []))
            semantic_text = f"{target.label} {target.description} {evidence}".strip()
            lowered = semantic_text.casefold()
            symbol = ""
            description = ""
            for tokens, candidate, candidate_description in _CONCEPT_PATTERNS:
                if all(_contains_concept(lowered, token) for token in tokens):
                    symbol = candidate
                    description = candidate_description
                    break
            if not symbol:
                slug = _semantic_slug(semantic_text)
                symbol = f"EVIDENCE_{slug}" if not slug.startswith("EVIDENCE_") else slug
                description = f"Current source evidence satisfies the observable {symbol.lower()} condition."
            base = symbol
            suffix = 2
            while symbol in used:
                symbol = f"{base}_{suffix}"
                suffix += 1
            used.add(symbol)
            results.append(
                InducedCondition(
                    target=target.state_id,
                    symbol=symbol,
                    condition_description=description,
                    confidence=0.6,
                )
            )
        return results


class DeepSeekConditionInducer:
    """Structured condition induction using the existing OpenAI-compatible client."""

    def __init__(
        self,
        chat_client: ChatClient | None = None,
        fallback: ConditionInducer | None = None,
    ) -> None:
        self.chat_client = chat_client or ChatClient()
        self.fallback = fallback or HeuristicConditionInducer()

    @staticmethod
    def _prompt(
        source: WritingState,
        targets: Sequence[WritingState],
        support_texts: Mapping[str, Sequence[str]],
    ) -> str:
        branches = [
            {
                "target_ref": target.state_id,
                "target_semantics": f"{target.label}: {target.description}",
                "source_evidence": list(support_texts.get(target.state_id, []))[:3],
            }
            for target in targets
        ]
        return (
            "Induce one mutually distinguishable observable condition per branch. "
            "Target semantics are offline contrast only: condition_description must be testable "
            "using source evidence/current data/current generated content, and must not name a target. "
            "Use UPPER_SNAKE_CASE symbols. Return JSON only as "
            "{\"conditions\":[{\"target_ref\":str,\"symbol\":str,"
            "\"condition_description\":str,\"confidence\":number}]}.\n"
            f"Source: {source.label}: {source.description}\n"
            f"Branches: {json.dumps(branches, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse(raw: str, targets: Sequence[WritingState]) -> list[InducedCondition]:
        payload = extract_json_object(raw)
        conditions = payload.get("conditions")
        if not isinstance(conditions, list):
            raise ModelValidationError("DeepSeek output must contain a conditions array.")
        target_ids = {target.state_id for target in targets}
        results: list[InducedCondition] = []
        seen_targets: set[str] = set()
        seen_symbols: set[str] = set()
        for item in conditions:
            if not isinstance(item, dict):
                raise ModelValidationError("Every induced condition must be an object.")
            target = str(item.get("target_ref") or "")
            description = str(item.get("condition_description") or "").strip()
            if target not in target_ids or target in seen_targets:
                raise ModelValidationError("DeepSeek output has an unknown or duplicate target_ref.")
            if any(target_id.casefold() in description.casefold() for target_id in target_ids):
                raise ModelValidationError("Runtime condition_description must not depend on target identity.")
            condition = InducedCondition(
                target=target,
                symbol=str(item.get("symbol") or ""),
                condition_description=description,
                confidence=float(item.get("confidence")),
            )
            if condition.symbol in seen_symbols:
                raise ModelValidationError("DeepSeek conditions must use distinct symbols for one source.")
            seen_targets.add(target)
            seen_symbols.add(condition.symbol)
            results.append(condition)
        if seen_targets != target_ids:
            raise ModelValidationError("DeepSeek output must cover every candidate target exactly once.")
        return sorted(results, key=lambda item: item.target)

    def induce(
        self,
        source: WritingState,
        targets: Sequence[WritingState],
        support_texts: Mapping[str, Sequence[str]] | None = None,
    ) -> list[InducedCondition]:
        if len(targets) <= 1:
            return self.fallback.induce(source, targets, support_texts)
        evidence = support_texts or {}
        prompt = self._prompt(source, targets, evidence)
        system = "You induce finite, evidence-grounded DFA conditions and return strict JSON only."
        for _ in range(2):
            try:
                raw = self.chat_client.complete(system=system, user=prompt, max_tokens=1200)
                return self._parse(raw, targets)
            except Exception:
                continue
        return self.fallback.induce(source, targets, evidence)
