from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autologic.adapters import (
    AutoLogicChatAdapter,
    GeneratedContent,
    IFindStateRetriever,
    StateContentGenerator,
    StateEvidence,
)
from autologic.executor import (
    AutoLogicExecutor,
    ConditionDecision,
    ConditionGrounder,
    ExecutionGuardError,
)
from autologic.models import UndefinedTransitionError, WritingDFA, WritingState, WritingTransition


def state(state_id: str, label: str, *, initial: bool = False, final: bool = False) -> WritingState:
    return WritingState(
        state_id=state_id,
        label=label,
        description=f"Analyze {label}",
        action=f"Write {label}",
        required_materials=["COMEX gold price and change ratio"],
        support_count=1,
        support_documents=["fixture-report"],
        is_initial=initial,
        is_final=final,
    )


def metadata(*state_ids: str) -> dict:
    return {
        "dfa_id": "executor-fixture",
        "collection_id": "executor-tests",
        "construction_metadata": {
            "source_documents": ["fixture-report"],
            "source_document_count": 1,
            "state_provenance": {
                state_id: [{"source_report_id": "fixture-report"}] for state_id in state_ids
            },
            "single_successor_unconditional_assumptions": [],
        },
    }


def branching_dfa() -> WritingDFA:
    review = state("S001", "market_review", initial=True)
    bullish = state("S002", "bullish_analysis", final=True)
    bearish = state("S003", "bearish_analysis", final=True)
    return WritingDFA(
        states={item.state_id: item for item in (review, bullish, bearish)},
        transitions=[
            WritingTransition(
                "S001", "PRICE_UP", "Current evidence shows that price is increasing.", "S002", 2, 0.9
            ),
            WritingTransition(
                "S001", "PRICE_DOWN", "Current evidence shows that price is decreasing.", "S003", 1, 0.8
            ),
        ],
        initial_state="S001",
        final_states={"S002", "S003"},
        metadata=metadata("S001", "S002", "S003"),
    )


def linear_dfa() -> WritingDFA:
    first = state("S001", "market_review", initial=True)
    final = state("S002", "strategy", final=True)
    return WritingDFA(
        states={first.state_id: first, final.state_id: final},
        transitions=[
            WritingTransition(
                "S001",
                "COMPLETE",
                "The current writing action is complete.",
                "S002",
                1,
                1.0,
                metadata={"unconditional": True},
            )
        ],
        initial_state="S001",
        final_states={"S002"},
        metadata=metadata("S001", "S002"),
    )


def cycle_dfa() -> WritingDFA:
    first = state("S001", "cycle_a", initial=True)
    second = state("S002", "cycle_b")
    final = state("S003", "cycle_exit", final=True)
    return WritingDFA(
        states={item.state_id: item for item in (first, second, final)},
        transitions=[
            WritingTransition("S001", "LOOP", "Loop evidence remains active.", "S002", 2, 0.9),
            WritingTransition("S001", "EXIT", "Exit evidence is active.", "S003", 1, 0.8),
            WritingTransition(
                "S002", "COMPLETE", "The loop section is complete.", "S001", 1, 1.0,
                metadata={"unconditional": True},
            ),
        ],
        initial_state="S001",
        final_states={"S003"},
        metadata=metadata("S001", "S002", "S003"),
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    def retrieve_state(self, state, query, date=None, dry_run=False, asset_name=None):
        del asset_name
        self.calls.append(state.state_id)
        lowered = query.casefold()
        if "up" in lowered or "上涨" in lowered:
            ratio = 1.5
            summary = "Price increased and is 上涨."
        elif "down" in lowered or "下跌" in lowered:
            ratio = -1.5
            summary = "Price decreased and is 下跌."
        else:
            ratio = 0.0
            summary = "Direction is neutral and unspecified."
        return StateEvidence(
            state_id=state.state_id,
            required_materials=list(state.required_materials),
            resolved_codes=["@GC0Y.CMX"],
            resolved_indicators=["changeRatio"],
            query_date=date or "2026-01-02",
            records=[{"changeRatio": ratio}],
            summary=summary,
            status="planned" if dry_run else "found",
            is_mock=dry_run,
        )

    def close(self):
        self.closed = True


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_current_state(self, *, query, state, evidence, memory, dry_run=False):
        del query, memory
        self.calls.append(state.state_id)
        return GeneratedContent(
            f"CONTENT:{state.label}; {evidence.summary}",
            {"dry_run": dry_run, "model": "fake"},
        )


class NetworkBombChat:
    model = "network-bomb"

    def complete_text(self, **kwargs):
        raise AssertionError(f"DeepSeek network path called: {kwargs}")

    def complete_json(self, **kwargs):
        raise AssertionError(f"DeepSeek network path called: {kwargs}")


class SequenceJSONChat:
    model = "fake-json"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete_json(self, **kwargs):
        del kwargs
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RawSequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, **kwargs):
        del kwargs
        self.calls += 1
        return self.responses.pop(0)


class LoopGrounder:
    def ground_condition(self, **kwargs):
        outgoing = kwargs["outgoing_transitions"]
        symbol = "LOOP" if any(item.symbol == "LOOP" for item in outgoing) else outgoing[0].symbol
        return ConditionDecision(symbol, "Fixture keeps the legal loop active.", 1.0)


class BadGrounder:
    def ground_condition(self, **kwargs):
        del kwargs
        return ConditionDecision("UNKNOWN_SYMBOL", "Invalid injected decision.", 1.0)


class BombIFindClient:
    def __init__(self) -> None:
        self.queries = 0
        self.logins = 0
        self.logouts = 0

    def login(self):
        self.logins += 1
        raise AssertionError("iFinD login called during dry-run")

    def cmd_history_quotation(self, *args, **kwargs):
        del args, kwargs
        self.queries += 1
        raise AssertionError("iFinD query called during dry-run")

    def logout(self):
        self.logouts += 1


class ChatAndRetrievalAdapterTests(unittest.TestCase):
    def test_strict_json_retries_without_network(self) -> None:
        raw = RawSequenceClient(["not-json", '{"symbol":"PRICE_UP"}'])
        adapter = AutoLogicChatAdapter(chat_client=raw, retries=1, timeout=1)
        self.assertEqual(adapter.complete_json(system="x", user="y"), {"symbol": "PRICE_UP"})
        self.assertEqual(raw.calls, 2)

    def test_ifind_dry_run_resolves_plan_without_login_or_query(self) -> None:
        client = BombIFindClient()
        retriever = IFindStateRetriever(client=client)
        evidence = retriever.retrieve_state(
            state("S001", "market_review", initial=True),
            "COMEX gold price up on 2026-01-02",
            dry_run=True,
        )
        self.assertEqual(client.logins, 0)
        self.assertEqual(client.queries, 0)
        self.assertIn("@GC0Y.CMX", evidence.resolved_codes)
        self.assertIn("changeRatio", evidence.resolved_indicators)
        self.assertTrue(evidence.is_mock)
        self.assertIn("PRICE_UP", evidence.summary)


class GroundConditionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dfa = branching_dfa()
        self.state = self.dfa.states["S001"]
        self.outgoing = self.dfa.outgoing("S001")
        self.neutral = StateEvidence(
            "S001", [], [], [], "2026-01-02", [], "No directional observation.", "empty"
        )

    def test_invalid_condition_first_attempt_retries(self) -> None:
        chat = SequenceJSONChat(
            [
                {"symbol": "NOT_ALLOWED", "reason": "bad", "confidence": 0.5},
                {"symbol": "PRICE_DOWN", "reason": "Validated on retry", "confidence": 0.8},
            ]
        )
        decision = ConditionGrounder(chat).ground_condition(
            state=self.state,
            evidence=self.neutral,
            generated_content="No directional observation.",
            memory="",
            outgoing_transitions=self.outgoing,
        )
        self.assertEqual(decision.symbol, "PRICE_DOWN")
        self.assertEqual(chat.calls, 2)
        self.assertFalse(decision.used_fallback)

    def test_invalid_condition_falls_back_deterministically(self) -> None:
        chat = SequenceJSONChat([ValueError("bad-json"), {"symbol": "BAD", "reason": "bad", "confidence": 2}])
        decision = ConditionGrounder(chat).ground_condition(
            state=self.state,
            evidence=self.neutral,
            generated_content="No directional observation.",
            memory="",
            outgoing_transitions=self.outgoing,
        )
        self.assertEqual(decision.symbol, "PRICE_UP")
        self.assertTrue(decision.used_fallback)
        self.assertIn("INVALID_CLASSIFIER_OUTPUT_AFTER_RETRY", decision.fallback_reason)


class ExecutorTests(unittest.TestCase):
    def execute(self, dfa, query, **kwargs):
        retriever = kwargs.pop("retriever", FakeRetriever())
        generator = kwargs.pop("generator", FakeGenerator())
        executor = AutoLogicExecutor(
            dfa,
            evidence_retriever=retriever,
            content_generator=generator,
            condition_grounder=kwargs.pop("grounder", ConditionGrounder(NetworkBombChat())),
            **kwargs,
        )
        return executor.execute(query=query, date="2026-01-02", dry_run=True), retriever, generator

    def test_single_unconditional_edge_is_selected_without_deepseek(self) -> None:
        result, retriever, _ = self.execute(linear_dfa(), "neutral")
        self.assertTrue(result.success)
        self.assertEqual([item["state_id"] for item in result.generated_states], ["S001", "S002"])
        self.assertEqual(result.execution_trace[0]["selected_condition"], "COMPLETE")
        self.assertTrue(retriever.closed)

    def test_price_up_and_price_down_take_different_paths(self) -> None:
        up, _, _ = self.execute(branching_dfa(), "price up")
        down, _, _ = self.execute(branching_dfa(), "price down")
        self.assertEqual([item["state_id"] for item in up.generated_states], ["S001", "S002"])
        self.assertEqual([item["state_id"] for item in down.generated_states], ["S001", "S003"])
        self.assertEqual(up.execution_trace[0]["selected_condition"], "PRICE_UP")
        self.assertEqual(down.execution_trace[0]["selected_condition"], "PRICE_DOWN")

    def test_final_state_content_is_generated_and_report_has_only_actual_path(self) -> None:
        result, _, _ = self.execute(branching_dfa(), "price up")
        self.assertIn("CONTENT:bullish_analysis", result.generated_report)
        self.assertNotIn("bearish_analysis", result.generated_report)
        self.assertEqual(result.run_manifest["termination_reason"], "FINAL_WRITING_STATE")

    def test_no_outgoing_final_state_terminates_normally(self) -> None:
        only = state("S001", "only_state", initial=True, final=True)
        dfa = WritingDFA({"S001": only}, [], "S001", {"S001"}, metadata("S001"))
        result, _, _ = self.execute(dfa, "neutral")
        self.assertTrue(result.success)
        self.assertEqual(len(result.generated_states), 1)

    def test_undefined_delta_raises_clear_error(self) -> None:
        with self.assertRaises(UndefinedTransitionError):
            self.execute(branching_dfa(), "neutral", grounder=BadGrounder())

    def test_max_steps_guard(self) -> None:
        with self.assertRaisesRegex(ExecutionGuardError, "MAX_STEPS"):
            self.execute(
                cycle_dfa(),
                "neutral",
                grounder=LoopGrounder(),
                max_steps=2,
                max_visits_per_state=10,
                max_transition_repeats=10,
            )

    def test_max_visits_per_state_guard(self) -> None:
        with self.assertRaisesRegex(ExecutionGuardError, "MAX_VISITS_PER_STATE"):
            self.execute(
                cycle_dfa(),
                "neutral",
                grounder=LoopGrounder(),
                max_steps=10,
                max_visits_per_state=1,
                max_transition_repeats=10,
            )

    def test_execution_trace_fields_and_saved_outputs(self) -> None:
        result, _, _ = self.execute(branching_dfa(), "price down")
        required = {
            "step", "current_state", "state_label", "state_action", "evidence_status",
            "evidence_summary", "generated_content", "candidate_conditions", "selected_condition",
            "condition_reason", "condition_confidence", "next_state", "used_fallback", "fallback_reason",
        }
        self.assertTrue(all(required <= set(item) for item in result.execution_trace))
        with tempfile.TemporaryDirectory() as directory:
            paths = result.save_outputs(directory)
            self.assertEqual(set(paths), {"generated_report", "generated_states", "execution_trace", "run_manifest"})
            states_payload = json.loads(Path(paths["generated_states"]).read_text(encoding="utf-8"))
            self.assertEqual([item["state_id"] for item in states_payload], ["S001", "S003"])

    def test_dry_run_generator_never_calls_deepseek(self) -> None:
        retriever = FakeRetriever()
        generator = StateContentGenerator(chat=NetworkBombChat())
        executor = AutoLogicExecutor(
            branching_dfa(),
            evidence_retriever=retriever,
            content_generator=generator,
            condition_grounder=ConditionGrounder(NetworkBombChat()),
        )
        result = executor.execute(query="price up", date="2026-01-02", dry_run=True)
        self.assertTrue(result.success)
        self.assertIn("PRICE_UP", result.generated_report)

    def test_no_match_stops_at_implicit_rejection_without_forcing_branch(self) -> None:
        retriever = FakeRetriever()
        generator = FakeGenerator()
        chat = SequenceJSONChat(
            [{"symbol": "NO_MATCH", "reason": "Neither price condition is satisfied.", "confidence": 0.9}]
        )
        executor = AutoLogicExecutor(
            branching_dfa(),
            evidence_retriever=retriever,
            content_generator=generator,
            condition_grounder=ConditionGrounder(chat),
        )
        result = executor.execute(query="neutral", date="2026-01-02", dry_run=False)
        self.assertFalse(result.success)
        self.assertEqual(result.run_manifest["termination_reason"], "CONDITION_NOT_SATISFIED")
        self.assertEqual(result.execution_trace[-1]["next_state"], "s_bottom")
        self.assertEqual([item["state_id"] for item in result.generated_states], ["S001"])


if __name__ == "__main__":
    unittest.main()
