# AutoLogic CLI and Keyless Demo

## 1. Minimal method

AutoLogic learns one deterministic writing automaton from two homogeneous historical reports and executes it from `initial_state`. Each executable state is an atomic writing action. After retrieving and generating the current state, the executor grounds one finite outgoing symbol and calls `dfa.delta(current_state, symbol)` to obtain the next state.

The query supplies topic, date, asset scope, and writing requirements. It does not select states or precompute the report path.

## 2. Why there is no SubDFA

AutoLogic executes the complete learned DFA dynamically. It does not import `query_processing.py`, perform query-state similarity matching, build a query subtree, use FAISS, or create a query-specific SubDFA. Only states actually reached through `delta` appear in the report.

## 3. Added project structure

```text
autologic/
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
```

## 4. Fully offline demo

The default scenario is `price-up`. It requires no API key, iFinD installation, or network connection:

```powershell
python autologic_client.py demo --output-root "autologic_outputs/demo"
```

The fixture path is:

```text
start --COMPLETE--> market_review
market_review --PRICE_UP--> bullish_analysis
market_review --PRICE_DOWN--> bearish_analysis
bullish_analysis --COMPLETE--> risk_warning
bearish_analysis --COMPLETE--> risk_warning
risk_warning [final]
```

## 5. Price-up and price-down scenarios

Run the scenarios into separate directories so their five artifacts cannot overwrite one another:

```powershell
python autologic_client.py demo --scenario price-up --output-root "autologic_outputs/demo_up"
python autologic_client.py demo --scenario price-down --output-root "autologic_outputs/demo_down"
```

Both runs use the same neutral report query. `--scenario` selects only the fixture evidence: positive `changeRatio` grounds `PRICE_UP`, while negative `changeRatio` grounds `PRICE_DOWN`. This option exists only on `demo`; production `generate` has no `--scenario` argument.

## 6. Build from two homogeneous reports

The two rows must be from the same vertical, report type, and institution or stable author group:

```powershell
python autologic_client.py build --csv "data/case_11.csv" --row-a 0 --row-b 1 --theta 0.5 --semantic-merge-threshold 0.5 --state-support-threshold 0.5 --condition-mode heuristic --collection-id "case_11_same_domain" --output-root "autologic_outputs/build"
```

`--theta` is the compatibility alias for `--transition-support-threshold`; it is not the semantic merge threshold. To reuse two existing normalized state sequences without running report extraction:

```powershell
python autologic_client.py build --from-state-sequences "logicrag_outputs/document_a_state_sequence.json" "logicrag_outputs/document_b_state_sequence.json" --condition-mode heuristic --local-embedding-only --output-root "autologic_outputs/build_from_sequences"
```

Use `--condition-mode deepseek` when live LLM condition induction is desired. Build writes `dfa.json`, `build_manifest.json`, and the two normalized state-sequence files.

## 7. Dry-run generate

`generate` executes an existing DFA directly and never rebuilds or extracts a SubDFA:

```powershell
python autologic_client.py generate --dfa "autologic_outputs/build/dfa.json" --query "撰写2025年2月28日有色金属周报，使用当前检索证据判断价格方向" --date "2025-02-28" --dictionary "domain_dictionary.csv" --output-root "autologic_outputs/generation_dry" --dry-run
```

Dry-run skips both iFinD and chat calls. `--dry-run-data` skips only iFinD; `--dry-run-llm` skips only chat generation and classification.

## 8. Real DeepSeek and iFinD execution

After setting credentials, execute a built DFA with live state-level retrieval and generation:

```powershell
python autologic_client.py generate --dfa "autologic_outputs/build/dfa.json" --query "撰写2025年2月28日有色金属周报" --date "2025-02-28" --asset-name "沪铜" --dictionary "domain_dictionary.csv" --chat-model "deepseek-chat" --chat-base-url "https://api.deepseek.com" --temperature 0.2 --output-root "autologic_outputs/generation_live"
```

Build and immediately execute the exact resulting DFA:

```powershell
python autologic_client.py run --csv "data/case_11.csv" --row-a 0 --row-b 1 --theta 0.5 --query "撰写2025年2月28日有色金属周报" --date "2025-02-28" --dictionary "domain_dictionary.csv" --condition-mode deepseek --output-root "autologic_outputs/full"
```

`run` places build intermediates under `autologic_outputs/full/build/` and the five runtime artifacts under `autologic_outputs/full/`.

## 9. PowerShell environment variables

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
$env:DASHSCOPE_API_KEY = "your-key"
$env:IFIND_USERNAME = "your-username"
$env:IFIND_PASSWORD = "your-password"
```

`DASHSCOPE_API_KEY` is used only when the existing document learner requests API embeddings. Use `--local-embedding-only` to force deterministic local embeddings. The CLI never prints credential values.

## 10. Linux and macOS environment variables

```bash
export DEEPSEEK_API_KEY="your-key"
export DASHSCOPE_API_KEY="your-key"
export IFIND_USERNAME="your-username"
export IFIND_PASSWORD="your-password"
```

## 11. Output files

Every successful `demo`, `generate`, or `run` directory contains:

- `dfa.json`: the exact DFA executed by this run.
- `generated_report.md`: generated writing-state content in actual visit order.
- `generated_states.json`: step-sorted visited states; cycles may repeat a state.
- `execution_trace.json`: evidence status, finite candidate symbols, selected condition, fallback, guard, and next state for each executed step.
- `run_manifest.json`: command, inputs, model/dry-run configuration, actual state path, termination reason, and success status.

Build additionally writes `build_manifest.json` and two normalized state-sequence files.

## 12. Reading execution_trace.json

For each step, inspect:

1. `current_state`, `state_label`, and `state_action`;
2. `evidence_status` and `evidence_summary`;
3. `candidate_conditions`, which is the complete finite outgoing set;
4. `selected_condition`, which must be one candidate or `NO_MATCH`;
5. `next_state`, which is returned by `dfa.delta`; `s_bottom` means controlled condition failure;
6. `used_fallback`, `fallback_reason`, and any guard `status`.

## 13. Current MVP limitations

- Offline construction is optimized for two reports.
- Report extraction and DeepSeek condition induction require configured chat credentials.
- The domain dictionary contains a small futures vocabulary and unresolved assets remain explicit.
- iFinD live mode requires the local `iFinDPy` environment and credentials.
- Heuristic condition induction is deterministic but intentionally limited.
- There is no arbitrary-N incremental clustering, centroid update, multi-DFA selection, or query-specific path planning.

## 14. Confirming that delta determines the path

Run both demo scenarios and compare `run_manifest.json`:

```text
price-up:   S001 -> S002 -> S003 -> S005
price-down: S001 -> S002 -> S004 -> S005
```

Then inspect step `S002` in each `execution_trace.json`. The candidate set is identical (`PRICE_UP`, `PRICE_DOWN`), but the selected symbol differs because fixture evidence differs. The recorded `next_state` matches the corresponding transition in `dfa.json`. No query-state ranking or precomputed path artifact exists.
