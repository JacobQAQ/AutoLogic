# AutoLogic Experiment Guide

This guide describes the repository's reproducible sample experiment. It learns one deterministic writing automaton from two homogeneous reports in `data/case_6115.csv`, then generates a Chinese non-ferrous-metals weekly report dated 2025-02-28.

All commands use repository-relative paths. No command contains a local user directory, account name, password, token, or API key value.

## 1. Method boundary

AutoLogic learns and executes one DFA for one homogeneous historical collection:

```text
historical reports -> leaf writing states -> deterministic DFA -> state-level retrieval -> state generation -> grounded symbol -> delta -> next state
```

The runtime does not import `query_processing.py`, build a query subtree or SubDFA, perform FAISS query-state matching, or precompute the complete report path. The query provides the report topic, date, asset scope, and global writing requirements. Actual state transitions are chosen during execution from current evidence and generated content.

## 2. Anonymous repository requirements

Before publishing or archiving an experiment:

- use only relative paths in commands and manifests intended for release;
- never place credential values in Markdown, shell history, JSON fixtures, or source code;
- do not publish generated runtime manifests without reviewing path fields;
- do not commit `autologic_outputs/` or local SDK files;
- keep API credentials in process environment variables;
- redact account-specific iFinD errors if they contain subscription or account identifiers.

AutoLogic reads the following environment variables:

```text
DEEPSEEK_API_KEY
IFIND_USERNAME
IFIND_PASSWORD
```

`IFIND_REFRESH_TOKEN` is not used by the current SDK-based `IFINDDataClient`. `DASHSCOPE_API_KEY` is optional and unnecessary when `--local-embedding-only` is enabled.

Check only whether variables exist; do not print their values:

```powershell
@("DEEPSEEK_API_KEY","IFIND_USERNAME","IFIND_PASSWORD") | ForEach-Object { "$_=" + (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_))) }
```

```bash
python -c 'import os; print({name: bool(os.environ.get(name)) for name in ("DEEPSEEK_API_KEY","IFIND_USERNAME","IFIND_PASSWORD")})'
```

## 3. Prerequisites

Run all commands from the repository root.

Install repository-managed dependencies:

```text
python -m pip install -r requirements.txt
```

Core Python imports:

```text
python -c "import openai, pandas, numpy; print('core dependencies: OK')"
```

Live iFinD retrieval additionally requires the vendor SDK:

```text
python -c "import iFinDPy; print('iFinDPy: OK')"
```

If `iFinDPy` is unavailable, install/configure the official package for the same Python interpreter used to run AutoLogic. A common installation command is:

```text
python -m pip install iFinDAPI
```

The offline demo and `--dry-run` do not require iFinD, network access, or API credentials.

## 4. Sample data

The standard experiment uses:

```text
data/case_6115.csv
```

The CSV has one header row and three report rows. AutoLogic uses zero-based data-row indices:

| Physical CSV row | CLI index | Role |
|---:|---:|---|
| 1 | — | Header |
| 2 | 0 | Historical report A |
| 3 | 1 | Historical report B |
| 4 | 2 | Held-out/optional report |

Rows 0 and 1 are from the same vertical, report type, and stable institution/author group. The build command must therefore use `--row-a 0 --row-b 1`.

Input fields are:

```text
report_id, 投资建议, 行业观点, 行情回顾, 行业跟踪, 风险提示
```

## 5. Offline executor preflight

Run a keyless fixture before the live experiment:

```text
python autologic_client.py demo --scenario price-up --output-root "autologic_outputs/preflight_demo" --force
```

Expected fixture path:

```text
S001 -> S002 -> S003 -> S005
```

## 6. Stage 1: build the DFA

Copy and run this as one line:

```text
python autologic_client.py build --csv "data/case_6115.csv" --row-a 0 --row-b 1 --semantic-merge-threshold 0.5 --state-support-threshold 0.5 --transition-support-threshold 0.5 --condition-mode deepseek --local-embedding-only --chat-model "deepseek-chat" --temperature 0.1 --collection-id "nonferrous_weekly_case_6115" --dfa-id "autologic_case_6115" --output-root "autologic_outputs/experiment_6115/build" --force
```

This command:

1. extracts a leaf-state sequence from each historical report;
2. merges semantically matched states while retaining unmatched states from both reports;
3. projects both sequences into the union state space;
4. counts state and adjacent-transition support;
5. induces normalized transition symbols;
6. resolves deterministic `(source, symbol)` conflicts;
7. validates and saves the DFA.

Build outputs:

```text
autologic_outputs/experiment_6115/build/dfa.json
autologic_outputs/experiment_6115/build/build_manifest.json
autologic_outputs/experiment_6115/build/document_a_state_sequence.json
autologic_outputs/experiment_6115/build/document_b_state_sequence.json
```

Validate the persisted DFA:

```text
python -c "from autologic.models import WritingDFA; d=WritingDFA.load_json('autologic_outputs/experiment_6115/build/dfa.json'); r=d.validate(); print('states=',len(d.states),'transitions=',len(d.transitions),'initial=',d.initial_state,'finals=',sorted(d.final_states),'valid=',r.is_valid,'warnings=',[w.code for w in r.warnings]); r.raise_if_invalid()"
```

## 7. Stage 2: offline execution of the learned DFA

This step validates DFA loading and dynamic execution without calling DeepSeek or iFinD:

```text
python autologic_client.py generate --dfa "autologic_outputs/experiment_6115/build/dfa.json" --query "撰写截至2025年2月28日的中文有色金属行业跟踪周报；保持历史样本的章节组织和专业表达，只使用当前状态证据，不编造缺失数据。" --date "2025-02-28" --dictionary "domain_dictionary.csv" --output-root "autologic_outputs/experiment_6115/dry_run" --dry-run --max-steps 50 --max-visits-per-state 2 --max-transition-repeats 2 --force
```

Dry-run evidence is synthetic and deterministic. It verifies control flow, not market-data correctness or report quality.

## 8. Stage 3: live report generation

After confirming that `iFinDPy` imports successfully and credentials are present, run the standard sample command as one line:

```text
python autologic_client.py generate --dfa "autologic_outputs/experiment_6115/build/dfa.json" --query "请撰写一篇截至2025年2月28日的中文有色金属行业跟踪周报。尽量保持历史样本的章节组织、研究深度和信息密度；分品种跟踪沪铜、伦铜、沪铝、伦铝、沪金和COMEX黄金。结合当前状态可用证据分析价格变化、供给、需求、库存、宏观环境和主要驱动因素，并给出后市判断、配置建议与风险提示。只能使用检索证据和此前正文的压缩记忆；证据未提供的数值必须明确说明数据不足，不得估算或虚构。严格按照DFA逐状态生成当前章节，不提前撰写后续章节，也不预先决定完整写作路径。" --date "2025-02-28" --dictionary "domain_dictionary.csv" --chat-model "deepseek-chat" --temperature 0.1 --max-steps 50 --max-visits-per-state 2 --max-transition-repeats 2 --output-root "autologic_outputs/experiment_6115/live_report" --force
```

The query deliberately states the desired coverage but does not select states or targets. It lists supported metal names so material-local code resolution can identify instruments when they are explicitly present in the current `required_material`.

Final report:

```text
autologic_outputs/experiment_6115/live_report/generated_report.md
```

## 9. iFinD request isolation

Live retrieval follows these rules:

1. resolve instruments from the current `required_material` first;
2. then inspect only the current state's label and description;
3. use an explicit `asset_name` only as a controlled fallback;
4. use query fallback only when the query resolves to exactly one code;
5. never propagate every asset from a multi-asset query to every material;
6. call `THS_HQ` separately for each code;
7. retain successful records when another code fails due to permission or availability.

For a material such as `伦铜/沪铜价格`, an account without LME permission may produce:

```text
00CAD.LME -> error
CU00.SHF  -> found
binding status -> partial
```

The successful domestic record remains available to the state generator. Per-code results are stored in `evidence.bindings[].code_results`.

Generic macro, supply, demand, or risk materials are not automatically treated as futures quotations. If no supported code can be grounded from the current material/state, the material remains `unresolved` rather than receiving unrelated codes from the global query.

## 10. Optional one-command experiment

For debugging and reproducibility, the staged workflow is recommended. After it works, the equivalent build-and-generate command is:

```text
python autologic_client.py run --csv "data/case_6115.csv" --row-a 0 --row-b 1 --semantic-merge-threshold 0.5 --state-support-threshold 0.5 --transition-support-threshold 0.5 --condition-mode deepseek --local-embedding-only --query "请撰写一篇截至2025年2月28日的中文有色金属行业跟踪周报。保持历史样本的章节组织、研究深度和信息密度，只使用当前状态的检索证据，不得编造缺失数据。" --date "2025-02-28" --dictionary "domain_dictionary.csv" --chat-model "deepseek-chat" --temperature 0.1 --max-steps 50 --max-visits-per-state 2 --max-transition-repeats 2 --collection-id "nonferrous_weekly_case_6115" --dfa-id "autologic_case_6115" --output-root "autologic_outputs/experiment_6115_full" --force
```

`run` stores build artifacts under `autologic_outputs/experiment_6115_full/build/` and runtime artifacts directly under `autologic_outputs/experiment_6115_full/`.

## 11. Runtime outputs

Every `generate` or `run` writes:

| File | Purpose |
|---|---|
| `dfa.json` | Exact DFA executed in this run. |
| `generated_report.md` | Report containing only actually visited writing states. |
| `generated_states.json` | Step-ordered visited states, generated content, and StateEvidence. |
| `execution_trace.json` | Candidate conditions, grounded symbol, confidence, fallback, and next state. |
| `run_manifest.json` | Model, guards, actual state path, termination reason, and success flag. |

Build additionally writes `build_manifest.json` and the two normalized state-sequence files.

## 12. Acceptance checks

A successful DFA execution should satisfy:

- `run_manifest.json.success` is `true`;
- termination is `FINAL_WRITING_STATE` or `FINAL_TERMINAL_STATE`;
- the executed path starts at `initial_state` and reaches a final state;
- the report contains only states listed in `generated_states.json`;
- each selected condition belongs to the current finite candidate set;
- `next_state` equals the target persisted for `(current_state, selected_condition)`;
- final-state writing content is present in `generated_report.md`.

Data retrieval must be assessed separately from DFA completion. Inspect `generated_states.json`:

```text
python -c "import json; s=json.load(open('autologic_outputs/experiment_6115/live_report/generated_states.json',encoding='utf-8')); [print(x['step'],x['state_id'],(x.get('evidence') or {}).get('status'),(x.get('evidence') or {}).get('resolved_codes'),(x.get('evidence') or {}).get('errors')) for x in s]"
```

Interpretation:

- `found`: at least one record is available;
- `not_required`: the state declares no data requirement;
- `unresolved`: no supported code was grounded for that material;
- `empty`: the request succeeded but returned no records;
- `error`: all attempted code requests failed;
- binding-level `partial`: at least one code succeeded and at least one failed.

Do not treat `success=true` as proof that every material was retrieved. It means the DFA reached a final state.

## 13. Known MVP limitations

- The builder is optimized for two homogeneous reports.
- `domain_dictionary.csv` covers only a limited set of instruments.
- The live adapter currently uses `THS_HQ` for historical quotations; macro, inventory, industry-index, and specialized supply-demand materials require additional data backends.
- The current quotation request uses one date, not a full weekly date range.
- Some overseas instruments require account-specific iFinD permissions.
- Heuristic condition induction and local embeddings are deterministic but intentionally lightweight.
- AutoLogic does not implement arbitrary-N incremental clustering, multi-DFA selection, or query-specific path planning.

## 14. Tests

Run the offline test suite:

```text
python -m compileall autologic
python -m py_compile autologic_client.py ifind_data_plugin.py
python -m unittest discover -s tests -p "test_autologic*.py" -v
python -m unittest tests.test_ifind_data_plugin -v
```

The tests use mocks/fixtures and do not require API keys, iFinD login, or network access.
