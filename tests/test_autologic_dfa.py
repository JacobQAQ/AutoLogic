from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_learner import StateMatch

from autologic.condition_induction import DeepSeekConditionInducer, HeuristicConditionInducer
from autologic.dfa_builder import AutoLogicDFABuilder
from autologic.models import (
    ModelValidationError,
    UndefinedTransitionError,
    WritingDFA,
    WritingState,
    WritingTransition,
)


def writing_state(
    state_id: str,
    label: str,
    *,
    initial: bool = False,
    final: bool = False,
) -> WritingState:
    return WritingState(
        state_id=state_id,
        label=label,
        description=f"Describe {label}",
        action=f"Write {label}",
        required_materials=["market data"],
        support_count=1,
        support_documents=["report-a"],
        is_initial=initial,
        is_final=final,
    )


def valid_dfa() -> WritingDFA:
    first = writing_state("S001", "Opening", initial=True)
    second = writing_state("S002", "Conclusion", final=True)
    return WritingDFA(
        states={first.state_id: first, second.state_id: second},
        transitions=[
            WritingTransition(
                source="S001",
                symbol="COMPLETE",
                condition_description="The opening action is complete.",
                target="S002",
                support_count=1,
                confidence=1.0,
                metadata={"unconditional": True},
            )
        ],
        initial_state="S001",
        final_states={"S002"},
        metadata={
            "dfa_id": "fixture",
            "collection_id": "fixture-collection",
            "construction_metadata": {
                "source_documents": ["report-a"],
                "source_document_count": 1,
                "state_provenance": {
                    "S001": [{"source_report_id": "report-a"}],
                    "S002": [{"source_report_id": "report-a"}],
                },
                "single_successor_unconditional_assumptions": [
                    {"source": "S001", "target": "S002"}
                ],
            },
        },
    )


def report(document_id: str, labels: list[str]) -> dict:
    nodes = []
    for index, label in enumerate(labels, start=1):
        nodes.append(
            {
                "node_id": str(index),
                "node_type": "leaf",
                "template_description": label,
                "level": 0,
                "parent": None,
                "children": [],
                "content_guideline": f"Write {label}",
                "required_materials": ["market data"],
                "length": 100,
                "source_excerpt": f"Evidence for {label}",
            }
        )
    return {"document_id": document_id, "state_sequence": nodes}


class WritingModelTests(unittest.TestCase):
    def test_writing_state_serialization_and_legacy_mapping(self) -> None:
        state = WritingState.from_dict(
            {
                "node_id": "S001",
                "template_name": "Market review",
                "content_guideline": "Review prices",
                "required_materials": ["price"],
                "support": [{"document_id": "report-a"}],
                "metadata": {"legacy": True},
            }
        )
        restored = WritingState.from_dict(state.to_dict())
        self.assertEqual(restored, state)
        self.assertEqual(state.action, "Review prices")

    def test_terminal_state_needs_no_fabricated_support(self) -> None:
        terminal = WritingState(
            state_id="S999",
            state_kind="terminal",
            label="Report complete",
            description="Terminal control state",
            action="",
            required_materials=[],
            support_count=0,
            support_documents=[],
            is_final=True,
        )
        self.assertEqual(terminal.support_documents, [])
        with self.assertRaises(ModelValidationError):
            WritingState(
                state_id="S998",
                state_kind="terminal",
                label="Bad terminal",
                description="Bad terminal",
                action="write",
                required_materials=[],
                support_count=0,
                support_documents=[],
                is_final=True,
            )

    def test_transition_symbol_normalization(self) -> None:
        transition = WritingTransition(
            source="S001",
            symbol=" price up ",
            condition_description="Price is up.",
            target="S002",
            support_count=1,
            confidence=0.8,
        )
        self.assertEqual(transition.symbol, "PRICE_UP")

    def test_dfa_serialization_roundtrip_and_file_io(self) -> None:
        dfa = valid_dfa()
        restored = WritingDFA.from_dict(json.loads(json.dumps(dfa.to_dict())))
        self.assertEqual(restored.to_dict(), dfa.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dfa.json"
            dfa.save_json(path)
            self.assertEqual(WritingDFA.load_json(path).to_dict(), dfa.to_dict())


class ValidationTests(unittest.TestCase):
    def test_valid_dfa(self) -> None:
        result = valid_dfa().validate()
        self.assertTrue(result.is_valid, result.errors)

    def test_dangling_transition_detection(self) -> None:
        dfa = valid_dfa()
        dfa.transitions[0].target = "S777"
        codes = {issue.code for issue in dfa.validate().errors}
        self.assertIn("DANGLING_TRANSITION_TARGET", codes)

    def test_initial_state_missing(self) -> None:
        dfa = valid_dfa()
        dfa.initial_state = "S777"
        codes = {issue.code for issue in dfa.validate().errors}
        self.assertIn("INITIAL_STATE_MISSING", codes)

    def test_final_state_missing(self) -> None:
        dfa = valid_dfa()
        dfa.final_states = {"S777"}
        codes = {issue.code for issue in dfa.validate().errors}
        self.assertIn("FINAL_STATE_MISSING", codes)

    def test_duplicate_pair_with_different_targets(self) -> None:
        dfa = valid_dfa()
        third = writing_state("S003", "Alternate")
        dfa.states[third.state_id] = third
        dfa.metadata["construction_metadata"]["state_provenance"]["S003"] = [
            {"source_report_id": "report-a"}
        ]
        dfa.transitions.append(
            WritingTransition(
                source="S001",
                symbol="COMPLETE",
                condition_description="Alternate completion.",
                target="S003",
                support_count=1,
                confidence=0.7,
            )
        )
        codes = {issue.code for issue in dfa.validate().errors}
        self.assertIn("NONDETERMINISTIC_TRANSITION", codes)

    def test_unreachable_state_warning(self) -> None:
        dfa = valid_dfa()
        third = writing_state("S003", "Unreachable", final=True)
        dfa.states[third.state_id] = third
        dfa.final_states.add("S003")
        dfa.metadata["construction_metadata"]["state_provenance"]["S003"] = [
            {"source_report_id": "report-a"}
        ]
        codes = {issue.code for issue in dfa.validate().warnings}
        self.assertIn("UNREACHABLE_STATES", codes)

    def test_cycle_is_reported_but_not_blanket_forbidden(self) -> None:
        dfa = valid_dfa()
        dfa.transitions.append(
            WritingTransition(
                source="S001",
                symbol="RETRY",
                condition_description="More opening evidence is required.",
                target="S001",
                support_count=1,
                confidence=0.5,
            )
        )
        result = dfa.validate()
        self.assertTrue(result.is_valid, result.errors)
        self.assertIn("LEGAL_CYCLE_REQUIRES_GUARD", {issue.code for issue in result.warnings})


class DeltaAndConditionTests(unittest.TestCase):
    def test_delta_returns_target_and_undefined_raises(self) -> None:
        dfa = valid_dfa()
        self.assertEqual(dfa.delta("S001", "complete"), "S002")
        with self.assertRaisesRegex(UndefinedTransitionError, "s_bottom"):
            dfa.delta("S001", "PRICE_UP")

    def test_single_target_generates_complete_without_network(self) -> None:
        source = writing_state("S001", "Source")
        target = writing_state("S002", "Target")
        with patch("report_generator.ChatClient.complete", side_effect=AssertionError("network called")):
            result = HeuristicConditionInducer().induce(source, [target])
        self.assertEqual(result[0].symbol, "COMPLETE")
        self.assertTrue(result[0].metadata["unconditional"])

    def test_multiple_targets_generate_distinct_semantic_symbols(self) -> None:
        source = writing_state("S001", "Market analysis")
        up = writing_state("S002", "Price up")
        down = writing_state("S003", "Price down")
        result = HeuristicConditionInducer().induce(source, [up, down])
        self.assertEqual({item.symbol for item in result}, {"PRICE_UP", "PRICE_DOWN"})
        self.assertNotIn("S002", {item.symbol for item in result})

    def test_deepseek_malformed_json_retries_then_falls_back(self) -> None:
        class FakeChat:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, **_: object) -> str:
                self.calls += 1
                return "not-json"

        chat = FakeChat()
        source = writing_state("S001", "Market analysis")
        targets = [writing_state("S002", "Price up"), writing_state("S003", "Price down")]
        result = DeepSeekConditionInducer(chat_client=chat).induce(source, targets)
        self.assertEqual(chat.calls, 2)
        self.assertEqual({item.symbol for item in result}, {"PRICE_UP", "PRICE_DOWN"})


class BuilderTests(unittest.TestCase):
    def test_two_report_union_s0_f_and_branch_conditions(self) -> None:
        builder = AutoLogicDFABuilder(
            semantic_merge_threshold=0.5,
            state_support_threshold=0.5,
            transition_support_threshold=0.5,
        )
        with patch("report_generator.ChatClient.complete", side_effect=AssertionError("network called")):
            dfa = builder.build(
                report("report-a", ["Market setup", "Price up"]),
                report("report-b", ["Market setup", "Price down"]),
                matches=[StateMatch(0, 0, 1.0)],
            )
        self.assertEqual(set(dfa.states), {"S001", "S002", "S003"})
        self.assertEqual(dfa.initial_state, "S001")
        self.assertEqual(dfa.final_states, {"S002", "S003"})
        self.assertEqual(dfa.delta("S001", "PRICE_UP"), "S002")
        self.assertEqual(dfa.delta("S001", "PRICE_DOWN"), "S003")
        self.assertTrue(dfa.validate().is_valid)

    def test_mixed_termination_uses_dedicated_terminal(self) -> None:
        builder = AutoLogicDFABuilder()
        dfa = builder.build(
            report("report-a", ["Market setup", "Price up"]),
            report("report-b", ["Market setup"]),
            matches=[StateMatch(0, 0, 1.0)],
        )
        terminal = next(state for state in dfa.states.values() if state.state_kind == "terminal")
        self.assertEqual(terminal.support_count, 0)
        self.assertEqual(terminal.support_documents, [])
        self.assertTrue(terminal.is_final)
        self.assertEqual(dfa.delta("S001", "END"), terminal.state_id)
        self.assertFalse(dfa.states["S001"].is_final)

    def test_initial_tie_break_is_stable(self) -> None:
        builder = AutoLogicDFABuilder()
        dfa = builder.build(
            report("report-a", ["Alpha"]),
            report("report-b", ["Beta"]),
            matches=[],
        )
        self.assertEqual(dfa.initial_state, "S001")

    def test_conflict_resolution_is_deterministic(self) -> None:
        transitions = [
            WritingTransition("S001", "PRICE_UP", "Up.", "S003", 1, 0.9),
            WritingTransition("S001", "PRICE_UP", "Up.", "S002", 1, 0.9),
            WritingTransition("S001", "PRICE_UP", "Up again.", "S002", 1, 0.8),
        ]
        kept, discarded = AutoLogicDFABuilder.determinize(transitions)
        self.assertEqual(kept[0].target, "S002")
        self.assertEqual(kept[0].support_count, 2)
        self.assertEqual(discarded[0]["discarded_target"], "S003")


if __name__ == "__main__":
    unittest.main()
