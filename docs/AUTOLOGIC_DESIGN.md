# AutoLogic Design Freeze

## 1. Status and Scope

This document freezes the implementation design for a minimal runnable AutoLogic project built beside the existing LogicRAG prototype. It is a design artifact only; no AutoLogic production code is introduced in this phase.

AutoLogic learns and executes exactly one deterministic finite automaton (DFA) from one homogeneous historical report collection:

\[
H = \{h_1, \ldots, h_n\}, \qquad L = (S, \Sigma, \delta, s_0, F)
\]

All reports in `H` must belong to the same vertical, report type, and institution or stable author group, while covering different report dates. A different vertical is a separate experiment and produces a separate DFA.

Let `S_json` be the serialized executable writing states plus any explicit `END` terminal control state. The formal DFA state set is `S = S_json union {s_bottom}`. The persisted transition table contains only valid non-rejection transitions and is therefore a partial function `delta_valid`. The standard total DFA function is defined by adding the implicit rejection state `s_bottom`:

\[
\delta(s, a) =
\begin{cases}
\delta_{valid}(s, a), & (s,a) \in \operatorname{dom}(\delta_{valid}) \\
s_{bottom}, & \text{otherwise}
\end{cases}
\]

and `delta(s_bottom, a) = s_bottom` for every `a` in `Sigma`. `NO_MATCH` is a reserved member of `Sigma` with no valid persisted edge; it represents the observation that no outgoing condition holds and therefore maps to `s_bottom`. `s_bottom` is not serialized in `states` or `transitions`. Normal successful execution remains on `delta_valid`; a deliberate `NO_MATCH` produces a controlled non-success termination, while any other unexpected entry to `s_bottom` indicates an executor or artifact defect.

The core method explicitly excludes:

- multiple-DFA selection;
- cross-domain state fusion;
- query-specific SubDFA or query subtree construction;
- query-state FAISS or embedding matching;
- Markov-chain or probabilistic-path execution;
- random transition choice;
- precomputing a complete writing path from the query.

The existing LogicRAG modules and `logicrag_client.py` remain behaviorally unchanged. AutoLogic is added through a separate package and CLI.

## 2. Findings That Constrain the Design

The design is based on the current repository implementation rather than module names:

- `document_learner.py` reads one CSV row as one report, concatenating every non-empty column except `report_id`. Its LLM prompt extracts a hierarchical `state_sequence`. Each normalized state currently contains `node_id`, `node_type`, `template_description`, `level`, `parent`, `children`, `content_guideline`, `required_materials`, and `length`.
- State semantics are compared with embeddings of `template_description + content_guideline`. `match_common_states()` performs a greedy one-to-one match between exactly two reports at cosine similarity `>= theta`.
- `build_global_template()` merges matched labels, descriptions, materials, and lengths. It retains support entries for two source documents only.
- `build_global_transitions()` counts ordered adjacent retained states from the two documents. Existing edges contain only `source`, `target`, `frequency`, and `support_documents`; they do not contain conditions or symbols.
- `query_processing.py` embeds the query, selects states above `tau`, finds their lowest common ancestor, and returns that ancestor's complete descendant subtree. AutoLogic must not call or import this module.
- `RequiredMaterialResolver.build_specs()` expects `node_template.nodes`, and maps each node's `required_materials` to iFinD `codes` and `indicators`. The date is supplied separately, normally from `parse_query_date(query)` or `--date`.
- `LogicRAGReportGenerator.generate()` does not traverse transitions. It sorts nodes by hierarchical `node_id`, normally generates leaf nodes only, and passes an LLM summary of the immediately previous generated node to the next node.
- `ChatClient` is the reusable OpenAI-compatible text client. It resolves `DEEPSEEK_API_KEY`, then `OPENAI_API_KEY`, then `LOGICRAG_CHAT_API_KEY`, and uses `LOGICRAG_CHAT_MODEL` / `LOGICRAG_CHAT_BASE_URL` defaults.
- The repository currently has no `requirements.txt` or `pyproject.toml`. The observed runtime dependencies include `pandas`, `openai`, and, for live iFinD mode, `iFinDPy`.

These constraints require an AutoLogic-owned DFA schema and executor while preserving adapters to the old node/material formats.

## 3. Target Architecture

The planned minimal additive structure is:

```text
autologic/
    __init__.py
    models.py
    validation.py
    condition_induction.py
    dfa_builder.py
    adapters.py
    executor.py

autologic_client.py

examples/
    autologic_demo_dfa.json
    autologic_demo_materials.json

tests/
    test_autologic_dfa.py
    test_autologic_executor.py
    test_autologic_cli.py
```

Only the two design documents are created in the current phase. The structure above is frozen for the implementation phase unless a concrete repository constraint requires a small adjustment.

Responsibilities:

| Component | Frozen responsibility |
|---|---|
| `models.py` | Typed in-memory representations for state, transition, DFA, execution step, and validation issue. |
| `validation.py` | Schema checks, referential integrity, determinism, reachability, terminal-state, symbol, and cycle-safety validation. |
| `condition_induction.py` | Evidence collection, candidate condition induction, symbol normalization, grounding checks, and conflict re-induction. |
| `dfa_builder.py` | MVP two-report leaf-state mapping, support counting/filtering, adjacency counting, `s_0`/`F` selection, condition attachment, and final DFA assembly; arbitrary-`n` clustering is a later extension. |
| `adapters.py` | Compatibility boundary for LogicRAG state sequences, `RequiredMaterialResolver`, materials files, and `ChatClient`. |
| `executor.py` | Online full-DFA traversal, state-level retrieval/generation, symbol recognition, memory compression, guards, trace, and output assembly. |
| `autologic_client.py` | Separate `demo`, `build`, `generate`, `run`, and optional `validate` CLI. It must not alter or wrap the old LogicRAG pipeline in a way that changes old behavior. |

The execution flow is:

```text
Homogeneous two-report MVP input or prebuilt fixture
  -> existing-compatible report loading and state extraction
  -> leaf-only executable-state adapter
  -> existing pairwise semantic matching and support filtering
  -> retained adjacent-pair counting
  -> condition induction and normalization
  -> determinism resolution and validation
  -> dfa.json

query + dfa.json
  -> s_0
  -> retrieve current-state materials
  -> generate current-state content
  -> stop if final, otherwise recognize one outgoing symbol
  -> delta(current_state, symbol)
  -> update compressed memory
  -> repeat
```

## 4. Offline DFA Construction

### 4.1 Inputs and Homogeneity Contract

The MVP builder accepts the existing LogicRAG two-report CSV workflow (`row-a`, `row-b`) or two explicit report texts. CSV compatibility follows `build_report_text_from_row()`:

- `report_id` is optional but recommended;
- every other non-empty string column is concatenated as `column_name:\nvalue`;
- exactly the selected two rows are used by the MVP builder;
- the caller is responsible for supplying a homogeneous collection.

The run manifest records `collection_id`, source path, selected row indices or report IDs, report count, and an optional human-entered homogeneity description. The builder does not automatically select among domains. Arbitrary `n`-report collection ingestion is a Phase 2 extension, not a prerequisite for the minimal demo or dynamic executor.

### 4.2 Per-Report State Extraction

For each of the two MVP reports:

1. Reuse the existing report-loading convention and the state extraction semantics in `build_state_sequence_prompt()` / `LogicRAGDocumentLearner.extract_state_sequence()`.
2. Normalize the output with `normalize_state_sequence()`.
3. Select executable atomic writing units. By default, only normalized nodes with `node_type == "leaf"` are eligible. Root and child nodes are retained only as structural/provenance metadata and are never flattened into the execution sequence.
4. Convert each eligible leaf into an AutoLogic candidate state:
   - `state_id` is provisional until clustering;
   - `label` comes from `template_description`;
   - `description` comes from `content_guideline`;
   - `required_materials` is preserved;
   - `action` maps directly from `content_guideline` in the MVP; no extra per-state LLM call is made;
   - source report ID, source node ID, ordinal position, and initial/final occurrence are recorded.
5. Obtain `source_excerpt` from an AutoLogic-specific extraction field when available, or from the corresponding original CSV section text. A second evidence-grounding LLM pass is optional enhancement only, not an MVP prerequisite.
6. Order only the eligible executable leaves by their actual report order to form the candidate-state sequence. Preserve each leaf's ancestor/root/child chain as provenance metadata without executing those ancestors.

All per-source provenance is stored outside the runtime state objects at `construction_metadata.state_provenance[state_id]`. Each provenance entry contains `source_report_id`, `source_node_id`, `ancestor_chain`, `source_position`, and `source_excerpt`. A merged state has one entry per contributing source leaf. This keeps the `additionalProperties: false` runtime State Schema compact and makes provenance location unambiguous.

Every content-bearing DFA state is one executable atomic writing unit. A title becomes a state only when extraction explicitly identifies it as a leaf-level executable writing action; a root/child label that merely names a section is not executable. An explicit terminal control sentinel may be introduced only for an `END` transition as described below; it emits no report content and is not a mapped LogicRAG node.

### 4.3 Semantic State Merging

AutoLogic reuses:

- `state_embedding_text()` as the canonical label-plus-description text;
- `EmbeddingProvider` and its API/local deterministic fallback;
- `cosine_similarity()`;
- `semantic_merge_threshold` (also denoted `rho`) as the minimum semantic-equivalence similarity.

The MVP reuses the current two-document greedy matcher over eligible leaf states, with its threshold argument renamed at the AutoLogic boundary to `semantic_merge_threshold` / `rho`. The existing function may still receive that value through its legacy `theta` parameter; the AutoLogic artifact and CLI must not call it `theta`.

`match_common_states()` identifies only which A/B leaf pairs should merge. It must never be used to delete unmatched leaves. For leaf sets `L_A` and `L_B`, the candidate state space is the union:

```text
S_candidate =
    merged_matched_states
    union unmatched_states_from_A
    union unmatched_states_from_B
```

Two-report construction pseudocode:

```text
leaves_A = executable_leaf_nodes(sequence_A)
leaves_B = executable_leaf_nodes(sequence_B)
matches = match_common_states(leaves_A, leaves_B, rho)

S_candidate = []
map_A = {}
map_B = {}

for (a_index, b_index) in matches:
    state = merge(leaves_A[a_index], leaves_B[b_index])
    S_candidate.append(state)
    map_A[a_index] = state.state_id
    map_B[b_index] = state.state_id

for each unmatched a_index in leaves_A:
    state = retain_as_independent_state(leaves_A[a_index])
    S_candidate.append(state)
    map_A[a_index] = state.state_id

for each unmatched b_index in leaves_B:
    state = retain_as_independent_state(leaves_B[b_index])
    S_candidate.append(state)
    map_B[b_index] = state.state_id

S_retained = apply_state_support_threshold(S_candidate)
projected_A = project_in_original_order(leaves_A, map_A, S_retained)
projected_B = project_in_original_order(leaves_B, map_B, S_retained)
adjacent_counts = count_adjacent_pairs(projected_A, projected_B)
```

Threshold comparison is inclusive. With `|H| = 2` and `state_support_threshold = 0.5`, a state supported by one report has support ratio `1/2 = 0.5` and must be retained. This is necessary for report-specific branch targets.

Example:

```text
Report A: A -> B
Report B: A -> C

S_candidate = {A, B, C}

delta_valid(A, PRICE_UP)   = B
delta_valid(A, PRICE_DOWN) = C
```

`B` and `C` remain independent unmatched states. Deleting them merely because they were not semantically matched would erase the branch and is invalid.

Phase 2 may add deterministic arbitrary-`n` incremental clustering:

1. Process reports in stable `(report_id, source_position)` order.
2. The first candidate creates the first cluster.
3. For each later candidate, compare it with every existing cluster representative.
4. Candidates with similarity `< semantic_merge_threshold` are ineligible.
5. Choose the eligible cluster with greatest similarity; ties are resolved by larger support count and then lexical cluster ID.
6. If no cluster is eligible, create a new cluster.
7. Update the representative using merged label, description, action, materials, and a centroid embedding; keep all source support records.
8. A single report may contribute at most one support document count to a state cluster even if the action repeats within that report. Occurrence count and document support count are stored separately in construction metadata.

Incremental clustering, centroid updates, and complex arbitrary-`n` support accounting are explicitly Phase 2 work. They are not required to run `demo`, execute a fixture DFA, or prove that `PRICE_UP` and `PRICE_DOWN` produce different paths. The public state schema still exposes `support_count` and `support_documents` so the MVP and Phase 2 share one runtime artifact contract.

### 4.4 State and Transition Stability Filtering

Threshold meanings are strictly separated:

- `semantic_merge_threshold` or `rho`: embedding similarity threshold for semantic state merging; it is an implementation parameter;
- `state_support_threshold`: minimum `support_count / |H|` for a state;
- `transition_support_threshold` or `theta`: minimum `support_count / eligible_source_occurrences` for an adjacent transition.

No single parameter name represents both semantic similarity and frequency support. A paper description may foreground the frequency threshold `theta`; `rho` remains an implementation parameter for semantic merging. `state_support_threshold` remains separately named even when an experiment assigns it the same numeric value as another threshold.

Filtering order:

1. map only executable leaf candidates and merge equivalent states using `semantic_merge_threshold` / `rho`;
2. retain clusters whose normalized document support meets `state_support_threshold`;
3. project each historical sequence separately onto the retained union state space while preserving its own order;
4. count adjacent retained state pairs per document and total occurrences;
5. retain adjacent pairs meeting `transition_support_threshold`;
6. remove unreachable retained states after `s_0` selection, but record them in the build report instead of silently discarding them.

No transition is inferred from non-adjacent target labels. Projection may connect retained states that become adjacent after unstable states are filtered, matching the current `build_global_transitions()` behavior.

### 4.5 Adjacent Pair Statistics

For each retained pair `(source, target)`, collect:

- `support_count`: distinct reports containing the adjacent pair;
- occurrence count in construction metadata;
- source-state support count;
- bounded `support_examples` containing report ID, source node ID, target node ID, and source supporting excerpt;
- empirical `confidence = support_count / eligible_source_documents`.

`confidence` is a construction diagnostic and tie-break signal. It is never sampled and does not make execution probabilistic or Markovian.

### 4.6 Condition Induction

Use `COMPLETE` only when historical evidence supports that finishing the source action unconditionally leads to the target. A single observed successor is not proof of logical unconditionality.

For the MVP, a single-successor edge may use `COMPLETE` as an engineering simplification when no contradictory evidence is available. Every such edge must be recorded in `construction_metadata.single_successor_unconditional_assumptions` with source, target, support, and reason. Validation emits a warning rather than presenting the assumption as a theorem of the learned DFA.

For a source with multiple retained successors:

1. Build one evidence bundle per `(source, target)` from:
   - original source-state excerpts and bounded surrounding text;
   - structured facts associated with the source state's required materials, when available in historical artifacts;
   - source-state action and semantics;
   - target action and semantics, used only during offline induction to distinguish candidate branches.
2. Ask the OpenAI-compatible chat client for an observable condition, a short candidate symbol, the evidence fields needed to decide it, and supporting example IDs.
3. Explicitly prohibit using the target label as the condition and prohibit full natural-language generations as symbols.
4. Compare all conditions leaving the same source jointly. Rewrite them until they are mutually distinguishable using current retrieved data, current generated content, or compressed prior memory.
5. Make branching conditions as mutually exclusive as practicable and cover the common observed evidence space. Deliberately uncovered observations remain valid and produce runtime `NO_MATCH`; they must not be forced onto a branch.
6. Before persistence, remove target identity and target-derived language from the condition. The final condition must be verifiable solely from source-state evidence, current retrieved data, current generated content, or bounded prior memory.
7. Reject conditions that depend on unavailable future information, target identity, or the query's predicted full path.
8. Ground each accepted condition against at least one support example. Unsupported candidates enter re-induction; persistent failures are dropped and recorded.

### 4.7 Symbol Normalization

Symbols must match `^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$` and must be finite identifiers.

Normalization steps:

1. trim and Unicode-normalize the candidate phrase;
2. map known synonyms to a controlled vocabulary, for example `RISE`, `INCREASE`, and `BULLISH_PRICE` to `PRICE_UP` when their grounded meaning is price direction;
3. convert the remaining concise concept to English `UPPER_SNAKE_CASE`;
4. reject values longer than 64 characters or containing prose/punctuation;
5. within one source, merge semantically equivalent symbols and union their support;
6. reserve `COMPLETE` for evidence-supported unconditional completion, or a recorded MVP single-successor assumption;
7. reserve `END` for an explicit transition to a dedicated terminal control state when a source may either terminate or continue;
8. reserve `DEFAULT` only for an explicitly validated catch-all fallback edge, at most one per source. The normal builder should prefer observable domain symbols over `DEFAULT`.

The canonical vocabulary is stored in `construction_metadata.symbol_catalog`, including descriptions and observed aliases. It is finite for each DFA, even if the implementation later supports extending the catalog during a new offline rebuild.

### 4.8 Determinism Conflict Resolution

The persisted valid-transition invariant is:

```text
(source, symbol) in dom(delta_valid) -> exactly one non-rejection target
(source, symbol) not in dom(delta_valid) -> implicit s_bottom
```

When induction yields the same `(source, symbol)` for multiple targets:

1. attempt joint re-induction using the conflicting support bundles;
2. preserve a relabeled edge only if its new condition is observable and grounded;
3. if the conflict remains, keep the target with greatest `support_count`;
4. tie-break by greater `confidence`, then greater target-state `support_count`, then lexical `target` ID;
5. drop other edges and record them in `construction_metadata.determinism_conflicts` with the resolution reason;
6. re-run the complete validator.

No random or query-dependent tie-breaking is permitted. A source left with one target after conflict resolution is not automatically normalized to `COMPLETE`; evidence must support unconditional completion, otherwise the existing observable symbol remains. The MVP assumption is allowed only when explicitly recorded.

### 4.9 Initial State `s_0`

Count the first retained state in every projected historical sequence. Select:

1. greatest initial `support_count`;
2. greatest overall state `support_count`;
3. smallest mean historical position;
4. lexical `state_id`.

Exactly one state receives `is_initial: true`, and `initial_state` must equal it. Other states receive `is_initial: false`.

### 4.10 Final States `F`

In the MVP, `F` contains only reachable states with no outgoing valid non-rejection transitions. Historical terminal frequency is diagnostic metadata and never makes a state final while that state still has an outgoing valid edge.

If a source sometimes terminates and sometimes continues, add an explicit `END` transition to a dedicated, deterministically numbered terminal control state such as `S999`. The terminal state has `support_count: 0`, `support_documents: []`, `required_materials: []`, empty `action`, optional/empty provenance, `is_final: true`, no outgoing valid transitions, and emits no report content. The branching source itself is not final. This sentinel is the sole control-state exception to the rule that mapped content states are executable atomic writing units; its support must never be fabricated from a historical node.

All final states receive `is_final: true`; all others receive `false`. `final_states` must exactly match those flags, and every final state must have zero outgoing valid transitions.

## 5. JSON Schemas

The frozen schemas use JSON Schema Draft 2020-12. Implementations may add non-semantic build metadata only under the declared metadata objects; they must not silently add execution fields to states or transitions.

### 5.1 State Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://autologic.local/schema/state.schema.json",
  "title": "AutoLogicState",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "state_id",
    "state_kind",
    "label",
    "description",
    "action",
    "required_materials",
    "support_count",
    "support_documents",
    "is_initial",
    "is_final"
  ],
  "properties": {
    "state_id": {
      "type": "string",
      "pattern": "^S[0-9]{3,}$"
    },
    "state_kind": {
      "type": "string",
      "enum": ["writing", "terminal"]
    },
    "label": {
      "type": "string",
      "minLength": 1
    },
    "description": {
      "type": "string",
      "minLength": 1
    },
    "action": {
      "type": "string"
    },
    "required_materials": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true
    },
    "support_count": {
      "type": "integer",
      "minimum": 0
    },
    "support_documents": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true
    },
    "is_initial": { "type": "boolean" },
    "is_final": { "type": "boolean" }
  },
  "allOf": [
    {
      "if": {
        "properties": { "state_kind": { "const": "writing" } }
      },
      "then": {
        "properties": {
          "action": { "type": "string", "minLength": 1 },
          "support_count": { "type": "integer", "minimum": 1 },
          "support_documents": { "type": "array", "minItems": 1 }
        }
      }
    },
    {
      "if": {
        "properties": { "state_kind": { "const": "terminal" } }
      },
      "then": {
        "properties": {
          "action": { "const": "" },
          "required_materials": { "type": "array", "maxItems": 0 },
          "support_count": { "const": 0 },
          "support_documents": { "type": "array", "maxItems": 0 },
          "is_final": { "const": true }
        }
      }
    }
  ]
}
```

Cross-field validation additionally requires:

- for `state_kind == "writing"`, `support_count >= 1`, at least one `support_document`, at least one `construction_metadata.state_provenance[state_id]` entry, non-empty `action`, and body generation is allowed;
- for `state_kind == "terminal"`, `support_count == 0`, `support_documents == []`, `required_materials == []`, provenance may be absent or empty, `is_final == true`, no outgoing valid transition exists, and no body is generated;
- `support_count == len(support_documents)` for both kinds.

A terminal control state must never use a fabricated source report or provenance entry merely to satisfy validation.

### 5.2 Transition Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://autologic.local/schema/transition.schema.json",
  "title": "AutoLogicTransition",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "source",
    "symbol",
    "condition_description",
    "target",
    "support_count",
    "confidence"
  ],
  "properties": {
    "source": {
      "type": "string",
      "pattern": "^S[0-9]{3,}$"
    },
    "symbol": {
      "type": "string",
      "pattern": "^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
      "maxLength": 64
    },
    "condition_description": {
      "type": "string",
      "minLength": 1
    },
    "target": {
      "type": "string",
      "pattern": "^S[0-9]{3,}$"
    },
    "support_count": {
      "type": "integer",
      "minimum": 1
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "support_examples": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["document_id", "source_node_id", "target_node_id", "evidence_excerpt"],
        "properties": {
          "document_id": { "type": "string", "minLength": 1 },
          "source_node_id": { "type": "string", "minLength": 1 },
          "target_node_id": { "type": "string", "minLength": 1 },
          "evidence_excerpt": { "type": "string", "minLength": 1 }
        }
      }
    },
    "metadata": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "unconditional": { "type": "boolean", "default": false }
      }
    }
  }
}
```

### 5.3 Complete `dfa.json` Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://autologic.local/schema/dfa.schema.json",
  "title": "AutoLogicDFA",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "dfa_id",
    "collection_id",
    "description",
    "alphabet",
    "implicit_rejection_state",
    "initial_state",
    "final_states",
    "states",
    "transitions",
    "build_config",
    "construction_metadata"
  ],
  "properties": {
    "schema_version": { "const": "1.0" },
    "dfa_id": { "type": "string", "minLength": 1 },
    "collection_id": { "type": "string", "minLength": 1 },
    "description": { "type": "string" },
    "implicit_rejection_state": { "const": "s_bottom" },
    "alphabet": {
      "type": "array",
      "items": {
        "type": "string",
        "pattern": "^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
        "maxLength": 64
      },
      "uniqueItems": true
    },
    "initial_state": {
      "type": "string",
      "pattern": "^S[0-9]{3,}$"
    },
    "final_states": {
      "type": "array",
      "items": {
        "type": "string",
        "pattern": "^S[0-9]{3,}$"
      },
      "uniqueItems": true,
      "minItems": 1
    },
    "states": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/state" }
    },
    "transitions": {
      "type": "array",
      "items": { "$ref": "#/$defs/transition" }
    },
    "build_config": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "semantic_merge_threshold",
        "state_support_threshold",
        "transition_support_threshold",
        "chat_model",
        "embedding_model",
        "max_support_examples"
      ],
      "properties": {
        "semantic_merge_threshold": { "type": "number", "minimum": 0, "maximum": 1 },
        "state_support_threshold": { "type": "number", "minimum": 0, "maximum": 1 },
        "transition_support_threshold": { "type": "number", "minimum": 0, "maximum": 1 },
        "chat_model": { "type": "string", "minLength": 1 },
        "embedding_model": { "type": "string", "minLength": 1 },
        "max_support_examples": { "type": "integer", "minimum": 0 }
      }
    },
    "construction_metadata": {
      "type": "object",
      "additionalProperties": true,
      "required": [
        "source_documents",
        "source_document_count",
        "state_provenance",
        "symbol_catalog",
        "determinism_conflicts",
        "dropped_states",
        "dropped_transitions",
        "single_successor_unconditional_assumptions"
      ],
      "properties": {
        "source_documents": {
          "type": "array",
          "items": { "type": "string", "minLength": 1 },
          "uniqueItems": true
        },
        "source_document_count": { "type": "integer", "minimum": 1 },
        "state_provenance": {
          "type": "object",
          "additionalProperties": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": [
                "source_report_id",
                "source_node_id",
                "ancestor_chain",
                "source_position",
                "source_excerpt"
              ],
              "properties": {
                "source_report_id": { "type": "string", "minLength": 1 },
                "source_node_id": { "type": "string", "minLength": 1 },
                "ancestor_chain": {
                  "type": "array",
                  "items": { "type": "string", "minLength": 1 }
                },
                "source_position": { "type": "integer", "minimum": 0 },
                "source_excerpt": { "type": "string" }
              }
            }
          }
        },
        "symbol_catalog": { "type": "object" },
        "determinism_conflicts": { "type": "array" },
        "dropped_states": { "type": "array" },
        "dropped_transitions": { "type": "array" },
        "single_successor_unconditional_assumptions": { "type": "array" }
      }
    }
  },
  "$defs": {
    "state": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "state_id",
        "state_kind",
        "label",
        "description",
        "action",
        "required_materials",
        "support_count",
        "support_documents",
        "is_initial",
        "is_final"
      ],
      "properties": {
        "state_id": { "type": "string", "pattern": "^S[0-9]{3,}$" },
        "state_kind": { "type": "string", "enum": ["writing", "terminal"] },
        "label": { "type": "string", "minLength": 1 },
        "description": { "type": "string", "minLength": 1 },
        "action": { "type": "string" },
        "required_materials": {
          "type": "array",
          "items": { "type": "string", "minLength": 1 },
          "uniqueItems": true
        },
        "support_count": { "type": "integer", "minimum": 0 },
        "support_documents": {
          "type": "array",
          "items": { "type": "string", "minLength": 1 },
          "uniqueItems": true
        },
        "is_initial": { "type": "boolean" },
        "is_final": { "type": "boolean" }
      },
      "allOf": [
        {
          "if": { "properties": { "state_kind": { "const": "writing" } } },
          "then": {
            "properties": {
              "action": { "type": "string", "minLength": 1 },
              "support_count": { "type": "integer", "minimum": 1 },
              "support_documents": { "type": "array", "minItems": 1 }
            }
          }
        },
        {
          "if": { "properties": { "state_kind": { "const": "terminal" } } },
          "then": {
            "properties": {
              "action": { "const": "" },
              "required_materials": { "type": "array", "maxItems": 0 },
              "support_count": { "const": 0 },
              "support_documents": { "type": "array", "maxItems": 0 },
              "is_final": { "const": true }
            }
          }
        }
      ]
    },
    "transition": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "source",
        "symbol",
        "condition_description",
        "target",
        "support_count",
        "confidence"
      ],
      "properties": {
        "source": { "type": "string", "pattern": "^S[0-9]{3,}$" },
        "symbol": {
          "type": "string",
          "pattern": "^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
          "maxLength": 64
        },
        "condition_description": { "type": "string", "minLength": 1 },
        "target": { "type": "string", "pattern": "^S[0-9]{3,}$" },
        "support_count": { "type": "integer", "minimum": 1 },
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
        "support_examples": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["document_id", "source_node_id", "target_node_id", "evidence_excerpt"],
            "properties": {
              "document_id": { "type": "string", "minLength": 1 },
              "source_node_id": { "type": "string", "minLength": 1 },
              "target_node_id": { "type": "string", "minLength": 1 },
              "evidence_excerpt": { "type": "string", "minLength": 1 }
            }
          }
        },
        "metadata": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "unconditional": { "type": "boolean", "default": false }
          }
        }
      }
    }
  }
}
```

The JSON Schema is supplemented by semantic validation:

- state IDs are unique;
- every writing-state ID has a non-empty `construction_metadata.state_provenance[state_id]` array; merged writing states list both reports and unmatched writing states list their single source; terminal-state provenance may be absent or empty;
- transition endpoints exist;
- `alphabet` equals the unique valid transition symbols union the reserved observation `NO_MATCH`; no persisted valid transition may use `NO_MATCH` as its symbol;
- `implicit_rejection_state` is `s_bottom`, is absent from serialized `states`, and every unrecorded `(state, symbol)` is formally interpreted as a transition to it;
- exactly one initial state exists and matches `initial_state`;
- state final flags exactly match `final_states`;
- `(source, symbol)` pairs are unique;
- every mapped LogicRAG state has `state_kind: writing`, comes from an executable leaf by default, and has a non-empty atomic `action`;
- `state_kind: terminal` is allowed only for a dedicated `END` target, has zero support/provenance requirements as defined above, emits no content, and has no outgoing valid transition;
- every final state has no outgoing valid transition;
- `COMPLETE` edges are evidence-supported or listed in `single_successor_unconditional_assumptions`;
- condition grounding may be skipped only for `COMPLETE`, `END`, or a transition whose `metadata.unconditional` is `true`;
- all retained states are reachable from `initial_state` unless explicitly allowed for diagnostics;
- at least one final state is reachable;
- confidence and support counts are reproducible from construction metadata.

## 6. Online Execution

### 6.1 Query Role

The query may specify:

- report topic;
- report date or time range;
- asset or business scope;
- global writing constraints;
- retrieval hints.

It may not select states, build a subtree, rank states, or decide a complete path before execution.

### 6.2 State-Level Retrieval Adapter

For the current AutoLogic state only, `adapters.py` constructs an in-memory LogicRAG-compatible payload:

```json
{
  "template_id": "autologic_runtime_current_state",
  "node_template": {
    "nodes": [
      {
        "node_id": "S003",
        "template_description": "Review market price",
        "content_guideline": "Describe the observed price movement and key figures.",
        "required_materials": ["COMEX gold price"],
        "children": []
      }
    ]
  }
}
```

It then calls `RequiredMaterialResolver.build_specs(payload, query, date, asset_name)`, followed by `fetch_data_for_specs()` with `IFINDDataClient`. This reuses the existing `CODES`, `INDICATORS`, and `DATE` behavior without making AutoLogic depend on `query_processing.py` or duplicating iFinD code.

Date resolution order is deterministic:

1. explicit CLI `--date`;
2. `parse_query_date(query)`;
3. fail with an actionable input error. No current-date fallback is allowed because it would make runs irreproducible.

### 6.3 State Generation

The state-generation prompt reuses the useful constraints from `LogicRAGReportGenerator._build_state_prompt()` but is AutoLogic-owned because it must include `action` and full-DFA runtime context. Inputs are limited to:

- query scope and global writing requirements;
- current state's label, description, action, and target length if available;
- current-state retrieved materials;
- compressed memory from already executed states;
- explicit missing-material status.

The prompt keeps the existing safeguards: use retrieved facts, do not invent missing data, generate only the current state, and return plain report text. `ChatClient.complete()` is reused unchanged.

### 6.4 Condition Grounding

For every outgoing transition, the runtime builds a condition-evaluation record:

```text
candidate symbol
condition description
current retrieval records and statuses
current generated state content
compressed prior memory, only when required
```

Target-state semantics are not part of this runtime record. They may be used only during offline induction to contrast candidate branches. A persisted `condition_description`, deterministic rule, or classifier prompt must not mention or require the target identity. It must be independently verifiable from source-state evidence, current retrieved data, current generated content, or bounded prior memory.

Grounding priority is:

1. deterministic structured-data rules when the condition can be directly computed, such as `changeRatio > 0 -> PRICE_UP`;
2. exact factual statements in current generated content that are traceable to current materials;
3. a constrained LLM classifier over the finite outgoing symbol set;
4. compressed memory only for conditions explicitly declared as needing prior context.

The condition evaluator returns either one provided valid outgoing symbol or the reserved observation `NO_MATCH`, plus a short evidence citation. It cannot invent a target. `NO_MATCH` means no outgoing condition is factually satisfied; under the total DFA definition `delta(current, NO_MATCH) = s_bottom`. The executor records `CONDITION_NOT_SATISFIED`, preserves already generated content, and ends in a controlled non-success status. A malformed unrecognized symbol remains an evaluator-format error, not `NO_MATCH`.

For a single outgoing transition, grounding may be skipped and the edge selected directly only when its symbol is `COMPLETE`, its symbol is `END`, or `transition.metadata.unconditional == true`. A sole conditional edge such as `PRICE_UP` must still be evaluated against current evidence and generated content. Cardinality alone never proves that its condition holds.

### 6.5 Online Pseudocode

```text
function execute(dfa, query, config):
    validate(dfa)
    current = dfa.initial_state
    memory = EMPTY
    report_segments = []
    trace = []
    visit_count = Counter()

    for step_number in 1..config.max_steps:
        visit_count[current] += 1
        if visit_count[current] > config.max_visits_per_state:
            stop_with_guard("MAX_VISITS_PER_STATE", keep=report_segments)

        state = dfa.states[current]
        if state.state_kind == "terminal":
            assert current in dfa.final_states
            trace.append(terminal_control_record(...))
            return SUCCESS(report_segments, trace)

        specs = resolve_required_materials_for_current_state(
            state=state,
            query=query,
            date=config.date,
            asset_name=config.asset_name
        )
        materials = retrieve_or_plan(specs, dry_run=config.dry_run)
        content = generate_or_placeholder(
            state=state,
            query=query,
            materials=materials,
            memory=memory,
            dry_run=config.dry_run
        )
        report_segments.append(content)

        if current in dfa.final_states:  # reachable writing sink
            trace.append(final_step_record(...))
            return SUCCESS(report_segments, trace)

        candidates = outgoing_transitions(current)
        if candidates is empty:
            fail_validation_or_stop("NON_FINAL_DEAD_END", keep=report_segments)
        else if count(candidates) == 1 and is_unconditional(candidates[0]):
            selected_symbol = candidates[0].symbol
            selection_mode = "UNCONDITIONAL"
        else:
            decision, selection_mode, evidence = ground_conditions(
                candidates=candidates,
                materials=materials,
                content=content,
                memory=memory
            )
            if decision == NO_MATCH:
                next_state = s_bottom
                trace.append(status="CONDITION_NOT_SATISFIED", ...)
                return CONTROLLED_FAILURE(report_segments, trace)
            if decision is classifier_format_error:
                decision, selection_mode = retry_or_evidence_backed_fallback(
                    candidates, materials, content, memory
                )
            selected_symbol = decision

        assert selected_symbol in valid_outgoing_symbols(current)
        next_state = delta_valid[(current, selected_symbol)]
        assert next_state != s_bottom  # NO_MATCH was handled above
        trace.append(step_record(...))
        memory = compress(memory, state, content, materials, selected_symbol)
        current = next_state

    stop_with_guard("MAX_STEPS", keep=report_segments)
```

The content of a final writing state is always appended before stopping. A dedicated terminal control state reached by `END` emits no content and stops immediately.

`is_unconditional(transition)` is true only when `symbol` is `COMPLETE`, `symbol` is `END`, or `metadata.unconditional == true`. It is never inferred from `count(outgoing) == 1`.

### 6.6 Fallback Mechanism

Fallbacks are conservative and deterministic:

- **Unresolved materials:** preserve `unresolved`, `empty`, or `error` status; generate a conservative missing-data statement or dry-run placeholder. Do not fabricate data.
- **All conditions factually false:** return `NO_MATCH`, transition formally to `s_bottom`, record `CONDITION_NOT_SATISFIED`, preserve generated content, and stop non-successfully. Support-based fallback is forbidden.
- **Condition classifier invalid output format:** retry once with the exact finite symbol list plus `NO_MATCH` and JSON-only output.
- **Evidence supports at least one condition but classification/formatting still fails:** fallback may choose only among conditions independently marked evidence-satisfied, using greatest `support_count`, then `confidence`, then lexical `symbol`/`target`. Record `EVIDENCE_BACKED_CLASSIFIER_FALLBACK`.
- **No independently evidence-satisfied candidate after classifier failure:** stop with a controlled classifier error; do not use global support to force a branch.
- **No outgoing edge on a non-final state:** stop with `NON_FINAL_DEAD_END`; retain all already generated content and return a non-zero CLI status.
- **Chat API failure:** no implicit provider switch beyond the existing key precedence. In normal mode, stop and preserve trace; in `--continue-on-generation-error`, emit an explicit error placeholder and continue using transition grounding from available data only.
- **iFinD failure:** keep the existing per-spec error capture. Execution may continue if the state prompt can truthfully express missing data.

Fallback never performs random choice, never derives a full path from the query, and never overrides a factual `NO_MATCH`. For multiple outgoing conditions, induction should make conditions mutually exclusive and cover common observations, but any uncovered observation still terminates through `s_bottom` rather than forcing the highest-support edge.

### 6.7 Memory Compression

After transition selection, store a bounded memory object containing:

- one short factual summary of generated content;
- material facts actually used;
- missing-material notes;
- selected symbol and its grounding evidence;
- current state ID.

The next state receives this compressed object, not unbounded full history. The initial implementation may reuse `ChatClient` for summarization following `_generate_summary()`; dry-run uses deterministic truncation. Memory size is capped by characters or tokens in configuration.

### 6.8 Cycle and Maximum-Step Protection

A DFA may legally contain cycles, so the executor implements:

- `--max-steps`, default `max(2 * |S|, 20)`;
- `--max-visits-per-state`, default `2`;
- a repeated `(source, symbol, target)` counter, default maximum `2`;
- validation that at least one final state is reachable from `s_0`;
- a trace status for every guard-triggered stop.

Reaching a guard is a controlled incomplete run, not success. Generated segments are retained, and the trace explains the exact guard.

## 7. MVP Scope and Phase Boundary

### 7.1 Phase 1 MVP

Phase 1 prioritizes the dynamic executor and the smallest path-sensitive proof:

1. support the existing LogicRAG two-report workflow (`row-a`, `row-b`);
2. map executable leaf nodes only;
3. accept a prebuilt fixture `dfa.json` without rebuilding;
4. provide a keyless demo DFA with one branching price state and valid `PRICE_UP` and `PRICE_DOWN` paths;
5. prove that different grounded conditions produce different executed state sequences;
6. implement sparse valid transitions plus implicit `s_bottom` totalization;
7. write the frozen five-file output contract;
8. preserve all old LogicRAG behavior.

The executor and fixture demo are not blocked on production-quality condition induction from arbitrary corpora.

### 7.2 Phase 2 Extensions

The following are explicitly deferred:

- arbitrary `n`-report ingestion;
- deterministic incremental clustering across many reports;
- centroid embedding updates;
- complex support accounting and filtering across repeated occurrences;
- optional second-pass evidence-grounding LLM calls;
- richer domain condition vocabularies and rule induction.

Phase 2 must retain the same `dfa.json` and executor contracts.

## 8. Dry-Run and Keyless Demo Design

- `demo` requires no API keys and no iFinD installation. It loads the fixture DFA and fixture materials, runs once with positive price evidence to take `PRICE_UP`, and once with negative price evidence to take `PRICE_DOWN`.
- `generate --dry-run` and `run --dry-run` validate the DFA, plan materials, skip iFinD/chat calls, generate deterministic placeholders from `action`, and choose only valid outgoing symbols.
- `build --from-state-sequences` can use two existing normalized state-sequence JSON files without calling extraction APIs. Branches that lack grounded conditions must fail validation rather than invent conditions.
- Dry-run writes the same five canonical output files as normal execution, with `dry_run: true` in `run_manifest.json` and `execution_trace.json`.

Dry-run and demo validate wiring, schemas, dynamic branching, guards, and output contracts; they are not quality evaluations.

## 9. CLI Design

The separate entry point is `autologic_client.py`. It must expose `demo`, `build`, `generate`, and `run`; `validate` is an allowed additional command. It must not modify `logicrag_client.py`.

### 9.1 Demo

```powershell
python autologic_client.py demo --output-root "autologic_outputs/demo"
```

The command performs two fully keyless fixture executions. Their artifacts must be isolated and must never overwrite one another:

```text
autologic_outputs/demo/price_up/
    dfa.json
    generated_report.md
    generated_states.json
    execution_trace.json
    run_manifest.json

autologic_outputs/demo/price_down/
    dfa.json
    generated_report.md
    generated_states.json
    execution_trace.json
    run_manifest.json
```

An optional `--scenario price_up|price_down` may execute only one scenario, writing to the corresponding subdirectory. Without `--scenario`, both subdirectories are produced. Each manifest and trace must demonstrate its own `PRICE_UP` or `PRICE_DOWN` path.

The minimal fixture flow is:

```text
S001 (write an atomic price review)
  -- PRICE_UP   --> S002 (write the upward-case implication; final writing sink)
  -- PRICE_DOWN --> S003 (write the downside-case implication; final writing sink)

all other (state, symbol) pairs --> implicit s_bottom
```

Fixture materials provide a positive `changeRatio` for one run and a negative `changeRatio` for the other. The executor grounds the symbol from that source-state data, never from `S002`/`S003` identity.

### 9.2 Build

```powershell
python autologic_client.py build `
  --csv "data/case_11.csv" `
  --row-a 0 `
  --row-b 1 `
  --collection-id "agriculture_weekly_same_institution" `
  --output-root "autologic_outputs/build" `
  --semantic-merge-threshold 0.5 `
  --state-support-threshold 0.5 `
  --transition-support-threshold 0.5
```

| Argument | Meaning |
|---|---|
| `--csv`, `--row-a`, `--row-b` | Existing two-report input contract for Phase 1. |
| `--collection-id` | Stable identifier for the homogeneous pair. |
| `--output-root` | Build output directory. |
| `--semantic-merge-threshold` | `rho`, semantic equivalence only. |
| `--state-support-threshold` | State document-support threshold. |
| `--transition-support-threshold` | Frequency threshold `theta` for adjacent transitions. |
| `--chat-model`, `--chat-base-url` | Existing OpenAI-compatible defaults. |
| `--embedding-model`, `--embedding-base-url`, `--embedding-batch-size` | Existing embedding configuration. |
| `--local-embedding-only` | Existing deterministic local hash embedding. |
| `--from-state-sequences` | Two pre-extracted normalized sequences. |

### 9.3 Generate

`generate` directly executes an existing `dfa.json`; it never rebuilds the DFA:

```powershell
python autologic_client.py generate `
  --dfa "autologic_outputs/build/dfa.json" `
  --query "write the weekly report for February 28, 2025" `
  --date "2025-02-28" `
  --output-root "autologic_outputs/generation"
```

It accepts the runtime options `--asset-name`, `--dictionary`, chat/iFinD configuration, `--dry-run`, `--max-steps`, `--max-visits-per-state`, and `--continue-on-generation-error`.

### 9.4 Run

`run` is the convenience pipeline: build from the two reports, then generate from the newly written `dfa.json`.

```powershell
python autologic_client.py run `
  --csv "data/case_11.csv" --row-a 0 --row-b 1 `
  --collection-id "agriculture_weekly_same_institution" `
  --query "write the weekly report for February 28, 2025" `
  --date "2025-02-28" `
  --output-root "autologic_outputs/run_2025-02-28"
```

### 9.5 Validate

```powershell
python autologic_client.py validate --dfa "autologic_outputs/build/dfa.json"
```

It performs schema, partial-table totalization, determinism, reachability, terminal, grounding, and guard checks. No AutoLogic CLI command accepts `tau`, a query-state index, or a query-subtree option.

## 10. Output Files

Every scenario directory produced by `demo`, and every `generate` or completed `run` directory, exposes these five canonical artifacts. The demo parent directory is only a container for `price_up/` and `price_down/` and contains no shared mutable runtime output:

```text
autologic_outputs/<run_id>/
    dfa.json
    generated_report.md
    generated_states.json
    execution_trace.json
    run_manifest.json
```

- `dfa.json`: exact automaton executed by the run. For `generate`, copy or reference-identically materialize the supplied DFA so the run is self-describing.
- `generated_report.md`: writing-state content concatenated in actual execution order. Terminal control states emit no content.
- `generated_states.json`: a step-sorted array, not an object keyed by state ID. Repeated visits caused by cycles therefore remain representable. Each step includes state ID, action, materials reference, content, bounded summary, selected symbol, next state, and status.
- `execution_trace.json`: executed path, valid outgoing symbols considered, condition evidence, selected symbol or `NO_MATCH`, fallback, guards, and errors. It records whether `s_bottom` was entered and distinguishes controlled `CONDITION_NOT_SATISFIED` from executor defects.
- `run_manifest.json`: command/mode, inputs, hashes, model/backend names, thresholds, dry-run flag, output paths, status, and timestamps; never credentials.

Build-only diagnostic files such as validation and condition-induction reports may be placed in a subordinate diagnostics directory, but they do not replace or rename the five canonical runtime artifacts. Retrieval bindings may likewise live under `retrieved_materials/` as auxiliary data.

## 11. Validation and Failure Severity

Fatal validation errors include:

- malformed schema;
- a writing state with zero support, empty support documents, empty action, or missing/empty provenance;
- a terminal state with nonzero support, non-empty support documents/materials/action, `is_final: false`, or an outgoing valid transition;
- duplicate state IDs;
- missing transition endpoints;
- zero or multiple initial states;
- missing final states;
- duplicate `(source, symbol)` with different targets;
- invalid symbols;
- unreachable final set;
- a final state with an outgoing valid transition;
- an unrecorded runtime symbol or any attempted normal transition to implicit `s_bottom`;
- direct selection of a sole conditional edge that is neither `COMPLETE`, `END`, nor explicitly `metadata.unconditional: true`;
- branching conditions without observable grounding;
- a persisted condition that depends on target identity or target semantics;
- mismatch between `alphabet` and transitions.
- use of support fallback after a factual `NO_MATCH`.

Warnings include:

- reachable non-final dead ends;
- an MVP `single_successor_unconditional_assumption` used for `COMPLETE`;
- cycles that rely on runtime guards;
- weak support near the configured threshold;
- unresolved required materials in dry-run previews;
- source encoding replacement characters or suspicious mojibake.

Only a DFA with no fatal errors may run in normal mode. `--dry-run` does not bypass fatal structural or determinism errors.

## 12. Minimal-Modification Principles

1. Do not delete, rename, or change the behavior of existing modules.
2. Do not modify `logicrag_client.py` or make it dispatch AutoLogic.
3. Do not import or call `query_processing.py` from AutoLogic.
4. Reuse existing classes and functions through adapters instead of copying iFinD or chat implementations.
5. Keep AutoLogic artifacts under `autologic_outputs`, never overwrite `logicrag_outputs`.
6. Preserve existing environment variables and OpenAI-compatible configuration.
7. Use an AutoLogic-owned schema instead of adding condition fields to old global-template edges.
8. Keep source compatibility conversions in `autologic/adapters.py`; core DFA models must not know the LogicRAG tree schema.
9. Add dependencies only through a future repository manifest; avoid requiring a graph library for the minimal implementation.
10. Treat observed character-encoding corruption in the dictionary/source artifacts as a validation concern, not as permission to silently rewrite user data.
11. In Phase 1, map LogicRAG leaves only; root/child hierarchy remains provenance and is never executed.
12. Keep arbitrary-`n` clustering and centroid updates out of the MVP critical path.

## 13. Frozen Implementation Decisions

- One homogeneous collection produces one DFA.
- The online executor always begins at `s_0` and traverses the complete DFA dynamically.
- The query never builds a SubDFA and never ranks states.
- Every mapped content state is an executable atomic leaf writing action; root/child hierarchy is provenance only.
- A title is a state only when explicitly extracted as an executable leaf action.
- The two-report candidate space is the union of merged matches, unmatched A leaves, and unmatched B leaves; matching never deletes unmatched branch states.
- State provenance is centralized at `construction_metadata.state_provenance[state_id]`, not added ad hoc to runtime State objects.
- The persisted transition table is partial; implicit `s_bottom` totalizes the standard DFA and is not a normal runtime path.
- Final states are reachable sinks with no outgoing valid transitions. Optional termination uses explicit `END` to a dedicated terminal control state.
- Terminal control states have zero support, empty documents/materials/action, optional empty provenance, and never generate body content.
- `semantic_merge_threshold` / `rho`, `state_support_threshold`, and `transition_support_threshold` / `theta` have separate meanings.
- `COMPLETE` requires unconditional historical evidence; an MVP single-successor simplification is explicitly recorded in metadata.
- A sole outgoing edge skips grounding only for `COMPLETE`, `END`, or explicit `metadata.unconditional: true`; a sole conditional edge is still evaluated.
- `NO_MATCH` maps to implicit `s_bottom`, records `CONDITION_NOT_SATISFIED`, preserves generated content, and ends non-successfully without support fallback.
- Branch conditions must be observable from source evidence, current data, generated content, or bounded prior memory and must not depend on target identity.
- The Phase 1 contract includes `demo`, `build`, `generate`, and `run`, plus optional `validate`.
- The keyless demo proves distinct `PRICE_UP` and `PRICE_DOWN` execution paths.
- Demo scenario artifacts are isolated under `demo/price_up/` and `demo/price_down/`.
- Canonical outputs are `dfa.json`, `generated_report.md`, `generated_states.json`, `execution_trace.json`, and `run_manifest.json`.
- Phase 1 supports the existing two-report workflow and fixture DFA; arbitrary-`n` clustering is Phase 2.
- iFinD and chat access are reused, not copied.
- Classifier-format fallbacks are deterministic, evidence-backed, and traceable; factual `NO_MATCH` is never overridden.
- Old LogicRAG behavior and artifacts remain intact.
