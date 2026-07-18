"""AutoLogic offline deterministic writing-DFA toolkit."""

from .models import (
    AutoLogicError,
    ModelValidationError,
    NonDeterministicTransitionError,
    UndefinedTransitionError,
    WritingDFA,
    WritingState,
    WritingTransition,
    normalize_symbol,
)
from .condition_induction import (
    ConditionInducer,
    DeepSeekConditionInducer,
    HeuristicConditionInducer,
    InducedCondition,
)
from .dfa_builder import AutoLogicDFABuilder
from .validation import DFAValidationError, ValidationIssue, ValidationResult, validate_dfa
from .adapters import (
    AutoLogicChatAdapter,
    AutoLogicChatError,
    GeneratedContent,
    AutoLogicIFindAdapter,
    IFINDStateRetriever,
    IFindStateRetriever,
    StateContentGenerator,
    StateEvidence,
    retrieve_state,
)
from .executor import (
    AutoLogicExecutor,
    CompactMemory,
    ConditionDecision,
    ConditionGrounder,
    ExecutionError,
    ExecutionGuardError,
    ExecutionResult,
    ground_condition,
)

__all__ = [
    "AutoLogicError",
    "ModelValidationError",
    "NonDeterministicTransitionError",
    "UndefinedTransitionError",
    "WritingDFA",
    "WritingState",
    "WritingTransition",
    "normalize_symbol",
    "AutoLogicDFABuilder",
    "ConditionInducer",
    "DeepSeekConditionInducer",
    "HeuristicConditionInducer",
    "InducedCondition",
    "DFAValidationError",
    "ValidationIssue",
    "ValidationResult",
    "validate_dfa",
    "AutoLogicChatAdapter",
    "AutoLogicChatError",
    "GeneratedContent",
    "AutoLogicIFindAdapter",
    "IFINDStateRetriever",
    "IFindStateRetriever",
    "StateContentGenerator",
    "StateEvidence",
    "retrieve_state",
    "AutoLogicExecutor",
    "CompactMemory",
    "ConditionDecision",
    "ConditionGrounder",
    "ExecutionError",
    "ExecutionGuardError",
    "ExecutionResult",
    "ground_condition",
]
