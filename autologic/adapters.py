"""Compatibility adapters for AutoLogic chat generation and state-level iFinD retrieval."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from document_learner import extract_json_object
from ifind_data_plugin import (
    DEFAULT_DICTIONARY,
    IFINDDataClient,
    RequiredMaterialResolver,
    fetch_data_for_specs,
    parse_query_date,
    records_to_raw_text,
    unique_keep_order,
)
from report_generator import (
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    ChatClient,
    clean_markdown,
)

from .models import WritingState


class AutoLogicChatError(RuntimeError):
    """Raised when an AutoLogic chat request exhausts its retries."""


def _redact_error(value: BaseException | str) -> str:
    text = str(value)
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "LOGICRAG_CHAT_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "***REDACTED***")
    text = re.sub(r"\b(?:sk|key)-[A-Za-z0-9_-]{8,}\b", "***REDACTED***", text)
    return text


class AutoLogicChatAdapter:
    """Retrying/timeout wrapper around the unchanged LogicRAG ChatClient."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_CHAT_MODEL,
        base_url: str = DEFAULT_CHAT_BASE_URL,
        temperature: float = 0.2,
        timeout: float = 60.0,
        retries: int = 1,
        chat_client: ChatClient | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("Chat timeout must be positive.")
        if isinstance(retries, bool) or retries < 0:
            raise ValueError("Chat retries must be a non-negative integer.")
        self.model = model
        self.base_url = base_url
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.chat_client = chat_client or ChatClient(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    def _call_with_timeout(self, function: Callable[[], str]) -> str:
        responses: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                responses.put((True, function()))
            except BaseException as exc:  # transfer the provider error to the caller thread
                responses.put((False, exc))

        thread = threading.Thread(target=invoke, name="autologic-chat", daemon=True)
        thread.start()
        try:
            succeeded, value = responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"Chat completion timed out after {self.timeout:g} seconds.") from exc
        if not succeeded:
            raise value
        return str(value)

    def _attempt(self, system: str, user: str, max_tokens: int) -> str:
        return self._call_with_timeout(
            lambda: self.chat_client.complete(system=system, user=user, max_tokens=max_tokens)
        )

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1200,
        retries: int | None = None,
    ) -> str:
        attempts = self.retries if retries is None else retries
        if isinstance(attempts, bool) or attempts < 0:
            raise ValueError("retries must be a non-negative integer.")
        errors: list[str] = []
        for _ in range(attempts + 1):
            try:
                response = self._attempt(system, user, max_tokens)
                if not response.strip():
                    raise ValueError("Chat provider returned an empty response.")
                return response
            except Exception as exc:
                errors.append(_redact_error(exc))
        raise AutoLogicChatError(
            f"Chat text completion failed after {attempts + 1} attempt(s): {errors[-1]}"
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1200,
        retries: int | None = None,
    ) -> str:
        """Compatibility alias for ordinary text completion."""
        return self.complete_text(
            system=system,
            user=user,
            max_tokens=max_tokens,
            retries=retries,
        )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1200,
        retries: int | None = None,
    ) -> dict[str, Any]:
        attempts = self.retries if retries is None else retries
        if isinstance(attempts, bool) or attempts < 0:
            raise ValueError("retries must be a non-negative integer.")
        errors: list[str] = []
        for _ in range(attempts + 1):
            try:
                raw = self._attempt(system, user, max_tokens)
                parsed = extract_json_object(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Strict JSON completion must return one JSON object.")
                return parsed
            except Exception as exc:
                errors.append(_redact_error(exc))
        raise AutoLogicChatError(
            f"Strict JSON completion failed after {attempts + 1} attempt(s): {errors[-1]}"
        )


@dataclass
class StateEvidence:
    state_id: str
    required_materials: list[str]
    resolved_codes: list[str]
    resolved_indicators: list[str]
    query_date: str
    records: list[dict[str, Any]]
    summary: str
    status: str
    errors: list[str] = field(default_factory=list)
    is_mock: bool = False
    bindings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "required_materials": list(self.required_materials),
            "resolved_codes": list(self.resolved_codes),
            "resolved_indicators": list(self.resolved_indicators),
            "query_date": self.query_date,
            "records": [dict(record) for record in self.records],
            "summary": self.summary,
            "status": self.status,
            "errors": list(self.errors),
            "is_mock": self.is_mock,
            "bindings": [dict(binding) for binding in self.bindings],
        }


def _dry_run_scenario(text: str) -> str:
    lowered = text.casefold()
    up_words = ("increase", "rise", "positive", "bullish")
    down_words = ("decrease", "fall", "negative", "bearish")
    if (
        any(token in lowered for token in ("上涨", "上升", "price_up"))
        or any(re.search(rf"\b{word}\b", lowered) for word in up_words)
        or re.search(r"\bprice\s+up\b", lowered)
    ):
        return "PRICE_UP"
    if (
        any(token in lowered for token in ("下跌", "下降", "price_down"))
        or any(re.search(rf"\b{word}\b", lowered) for word in down_words)
        or re.search(r"\bprice\s+down\b", lowered)
    ):
        return "PRICE_DOWN"
    return "UNSPECIFIED"


class IFindStateRetriever:
    """Resolve and fetch materials for exactly one currently visited state."""

    def __init__(
        self,
        *,
        resolver: RequiredMaterialResolver | None = None,
        client: IFINDDataClient | None = None,
        dictionary_path: str | Path = DEFAULT_DICTIONARY,
    ) -> None:
        self.resolver = resolver or RequiredMaterialResolver(dictionary_path)
        self.client = client or IFINDDataClient()

    @staticmethod
    def _one_state_template(state: WritingState) -> dict[str, Any]:
        return {
            "template_id": "autologic_runtime_current_state",
            "node_template": {
                "nodes": [
                    {
                        "node_id": state.state_id,
                        "template_description": state.label,
                        "content_guideline": state.description,
                        "required_materials": list(state.required_materials),
                        "children": [],
                    }
                ]
            },
        }

    def retrieve_state(
        self,
        state: WritingState,
        query: str,
        date: str | None = None,
        dry_run: bool = False,
        asset_name: str | None = None,
    ) -> StateEvidence:
        if state.state_kind != "writing":
            raise ValueError("iFinD retrieval is only valid for writing states.")
        try:
            query_date = date.strip() if date and date.strip() else parse_query_date(query)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "A reproducible query date is required. Pass date=YYYY-MM-DD or include a supported date in query."
            ) from exc

        template = self._one_state_template(state)
        specs = self.resolver.build_specs(template, query, query_date, asset_name=asset_name)
        if not specs:
            return StateEvidence(
                state_id=state.state_id,
                required_materials=list(state.required_materials),
                resolved_codes=[],
                resolved_indicators=[],
                query_date=query_date,
                records=[],
                summary="No required materials are declared for this state.",
                status="not_required",
                is_mock=dry_run,
            )

        records, bindings = fetch_data_for_specs(specs, self.client, dry_run=dry_run)
        codes = unique_keep_order(code for spec in specs for code in spec.codes)
        indicators = unique_keep_order(indicator for spec in specs for indicator in spec.indicators)
        errors = [str(binding["error"]) for binding in bindings if binding.get("error")]
        statuses = [str(binding.get("status", "")) for binding in bindings]

        if dry_run:
            scenario = _dry_run_scenario(f"{query}\n{state.label}\n{state.description}\n{state.action}")
            records = [
                {
                    "mock": True,
                    "scenario": scenario,
                    "query_date": query_date,
                    "codes": codes,
                    "indicators": indicators,
                }
            ]
            plan = (
                f"Dry-run retrieval plan for {state.state_id}: scenario={scenario}; "
                f"CODES={','.join(codes) or 'UNRESOLVED'}; "
                f"INDICATORS={','.join(indicators) or 'NONE'}; DATE={query_date}."
            )
            status = "planned" if codes else "unresolved"
            summary = plan
        else:
            raw_text = records_to_raw_text(records)
            if records:
                status = "found"
            elif "error" in statuses:
                status = "error"
            elif "unresolved" in statuses:
                status = "unresolved"
            else:
                status = "empty"
            summary = raw_text or (
                f"No records retrieved; status={status}; CODES={','.join(codes) or 'UNRESOLVED'}; "
                f"INDICATORS={','.join(indicators) or 'NONE'}; DATE={query_date}."
            )

        return StateEvidence(
            state_id=state.state_id,
            required_materials=list(state.required_materials),
            resolved_codes=codes,
            resolved_indicators=indicators,
            query_date=query_date,
            records=records,
            summary=summary,
            status=status,
            errors=errors,
            is_mock=dry_run,
            bindings=bindings,
        )

    def close(self) -> None:
        self.client.logout()


IFINDStateRetriever = IFindStateRetriever
AutoLogicIFindAdapter = IFindStateRetriever


def retrieve_state(
    state: WritingState,
    query: str,
    date: str | None = None,
    dry_run: bool = False,
    *,
    asset_name: str | None = None,
    retriever: IFindStateRetriever | None = None,
) -> StateEvidence:
    """Convenience entry point that still retrieves exactly one state."""
    owned = retriever is None
    active = retriever or IFindStateRetriever()
    try:
        return active.retrieve_state(
            state,
            query,
            date=date,
            dry_run=dry_run,
            asset_name=asset_name,
        )
    finally:
        if owned:
            active.close()


@dataclass(frozen=True)
class GeneratedContent:
    generated_content: str
    metadata: dict[str, Any]


class StateContentGenerator:
    """Generate only the current writing state's report body."""

    def __init__(self, chat: AutoLogicChatAdapter | None = None) -> None:
        self.chat = chat or AutoLogicChatAdapter()

    @staticmethod
    def _prompt(
        query: str,
        state: WritingState,
        evidence: StateEvidence,
        memory: str,
    ) -> str:
        records = json.dumps(evidence.records, ensure_ascii=False, default=str)
        return f"""
Write only the body content required by the current AutoLogic writing state.

User query and global scope:
{query}

Current state:
- state_id: {state.state_id}
- label: {state.label}
- action: {state.action}
- description: {state.description}
- required_materials: {json.dumps(state.required_materials, ensure_ascii=False)}

Bounded prior memory (coherence only; it must not determine the path):
{memory or "None."}

Current-state evidence:
- status: {evidence.status}
- summary: {evidence.summary}
- records: {records}
- errors: {json.dumps(evidence.errors, ensure_ascii=False)}

Requirements:
1. Write only this state's action; do not anticipate or write any later state.
2. Use numerical facts only when present in the current-state evidence records.
3. Never invent missing values; state missing evidence conservatively when necessary.
4. Maintain coherence with prior memory without copying earlier sections.
5. Return plain report body text only, not JSON, metadata, headings, or explanations.
""".strip()

    def generate_current_state(
        self,
        *,
        query: str,
        state: WritingState,
        evidence: StateEvidence,
        memory: str,
        dry_run: bool = False,
    ) -> GeneratedContent:
        if state.state_kind != "writing":
            raise ValueError("Terminal control states do not generate report body content.")
        if dry_run:
            scenario = _dry_run_scenario(
                f"{query}\n{evidence.summary}\n{json.dumps(evidence.records, ensure_ascii=False)}"
            )
            direction = {
                "PRICE_UP": "PRICE_UP：当前 mock evidence 表示价格上涨。",
                "PRICE_DOWN": "PRICE_DOWN：当前 mock evidence 表示价格下跌。",
                "UNSPECIFIED": "当前 mock evidence 未给出明确方向。",
            }[scenario]
            content = f"[DRY RUN] {state.label} — {state.action} {direction}"
            return GeneratedContent(
                generated_content=content,
                metadata={"dry_run": True, "model": self.chat.model, "scenario": scenario},
            )

        prompt = self._prompt(query, state, evidence, memory)
        raw = self.chat.complete_text(
            system=(
                "You write concise, factual financial research report sections. "
                "Return only the current section's plain body text."
            ),
            user=prompt,
            max_tokens=1400,
        )
        content = clean_markdown(raw)
        if not content:
            raise AutoLogicChatError(f"State generation returned empty content for {state.state_id}.")
        return GeneratedContent(
            generated_content=content,
            metadata={"dry_run": False, "model": self.chat.model},
        )
