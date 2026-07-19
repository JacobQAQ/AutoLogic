# AutoLogic

AutoLogic is a minimal research prototype for condition-driven financial report generation. It learns one deterministic finite automaton (DFA) from two homogeneous historical reports and executes that DFA state by state using current evidence.

This anonymous repository does not contain API keys, account credentials, personal filesystem paths, or user-specific configuration.

## Method

Each executable DFA state is an atomic writing action learned from a leaf node in a historical report. Internal root/child nodes remain construction metadata and are not flattened into the runtime path.

At runtime AutoLogic repeatedly:

1. loads the current state;
2. retrieves only that state's required materials;
3. generates only that state's content;
4. grounds one finite outgoing condition symbol;
5. calls `delta(current_state, symbol)`;
6. continues until a final state or controlled termination.

The query specifies topic, date, scope, and global writing requirements. It does not precompute the report path. AutoLogic does not create a query subtree or SubDFA and does not perform FAISS query-state matching.

Undefined `(state, symbol)` pairs map implicitly to the rejection state `s_bottom`. Runtime guards bound steps, state visits, and repeated transitions while preserving partial outputs.

## Repository layout

```text
autologic/
    models.py
    validation.py
    condition_induction.py
    dfa_builder.py
    adapters.py
    executor.py

autologic_client.py
document_learner.py
ifind_data_plugin.py
report_generator.py
domain_dictionary.csv

examples/
    autologic_demo_dfa.json
    autologic_demo_materials.json

docs/
    AUTOLOGIC_DESIGN.md
    AUTOLOGIC_REUSE_MAP.md
    AUTOLOGIC_DEMO.md
    EXPERIMENT_GUIDE.md
```

The minimal repository retains only the support modules directly reused by AutoLogic. It does not depend on the legacy query-processing or LogicRAG client pipeline.

## Requirements

The observed core Python dependencies are:

```text
openai
pandas
numpy
```

Install the repository-managed dependencies with:

```text
python -m pip install -r requirements.txt
```

`numpy` is installed transitively by `pandas`.

Live iFinD retrieval additionally requires the vendor's `iFinDPy` environment. The offline demo and dry-run mode require neither credentials nor network access.

Check imports:

```text
python -c "import openai, pandas, numpy; print('core dependencies: OK')"
python -c "import iFinDPy; print('iFinDPy: OK')"
```

## Credentials

Set credentials outside the repository using environment variables:

```text
DEEPSEEK_API_KEY
IFIND_USERNAME
IFIND_PASSWORD
```

Do not place credential values in CLI arguments, source files, Markdown, fixtures, manifests, or committed shell scripts. `IFIND_REFRESH_TOKEN` is not used by the current SDK adapter. `DASHSCOPE_API_KEY` is optional when API embeddings are enabled; the documented sample uses `--local-embedding-only` instead.

## Offline quick start

Run the keyless deterministic demo:

```text
python autologic_client.py demo --scenario price-up --output-root "autologic_outputs/demo_up" --force
python autologic_client.py demo --scenario price-down --output-root "autologic_outputs/demo_down" --force
```

Expected paths:

```text
price-up:   S001 -> S002 -> S003 -> S005
price-down: S001 -> S002 -> S004 -> S005
```

The query is identical in both scenarios; fixture evidence selects different symbols, demonstrating that `delta` rather than query keywords determines the path.

## Sample experiment

The standard repository experiment learns from physical CSV rows 2 and 3 of `data/case_6115.csv`, corresponding to zero-based data indices 0 and 1.

Build the DFA:

```text
python autologic_client.py build --csv "data/case_6115.csv" --row-a 0 --row-b 1 --semantic-merge-threshold 0.5 --state-support-threshold 0.5 --transition-support-threshold 0.5 --condition-mode deepseek --local-embedding-only --chat-model "deepseek-chat" --temperature 0.1 --collection-id "nonferrous_weekly_case_6115" --dfa-id "autologic_case_6115" --output-root "autologic_outputs/experiment_6115/build" --force
```

Generate the dated report:

```text
python autologic_client.py generate --dfa "autologic_outputs/experiment_6115/build/dfa.json" --query "请撰写一篇截至2025年2月28日的中文有色金属行业跟踪周报。尽量保持历史样本的章节组织、研究深度和信息密度；分品种跟踪沪铜、伦铜、沪铝、伦铝、沪金和COMEX黄金。结合当前状态可用证据分析价格变化、供给、需求、库存、宏观环境和主要驱动因素，并给出后市判断、配置建议与风险提示。只能使用检索证据和此前正文的压缩记忆；证据未提供的数值必须明确说明数据不足，不得估算或虚构。严格按照DFA逐状态生成当前章节，不提前撰写后续章节，也不预先决定完整写作路径。" --date "2025-02-28" --dictionary "domain_dictionary.csv" --chat-model "deepseek-chat" --temperature 0.1 --max-steps 50 --max-visits-per-state 2 --max-transition-repeats 2 --output-root "autologic_outputs/experiment_6115/live_report" --force
```

Final report:

```text
autologic_outputs/experiment_6115/live_report/generated_report.md
```

See [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md) for prerequisites, dry-run execution, acceptance checks, and troubleshooting.

## State-level iFinD isolation

The resolver prioritizes the current `required_material`, then the current state's semantics, then an explicit single-asset fallback. Assets from a multi-asset query are not propagated to every material.

Quotation requests are split by code. If an overseas code fails because of subscription permissions while a domestic code succeeds, successful records are retained and the binding is marked `partial`. Per-code results are persisted in `evidence.bindings[].code_results`.

Generic macro or supply-demand materials are not silently converted into unrelated futures quotation requests. Unsupported materials remain explicit as `unresolved`.

## Runtime outputs

Successful `demo`, `generate`, and `run` executions write:

```text
dfa.json
generated_report.md
generated_states.json
execution_trace.json
run_manifest.json
```

`generated_states.json` contains only actually visited states, ordered by step. `execution_trace.json` records candidates, selected symbols, evidence summaries, next states, fallbacks, and guards. `run_manifest.json` separates DFA completion from the detailed retrieval evidence stored per state.

## Tests

```text
python -m compileall autologic
python -m py_compile autologic_client.py ifind_data_plugin.py
python -m unittest discover -s tests -p "test_autologic*.py" -v
python -m unittest tests.test_ifind_data_plugin -v
```

The test suite is offline and does not require API keys or iFinD access.

## MVP limitations

- DFA construction is optimized for two homogeneous reports.
- The bundled domain dictionary covers a limited set of instruments.
- The current live backend retrieves historical quotations through `THS_HQ`; macro, inventory, industry-index, and specialized supply-demand evidence require additional backends.
- The current live request uses a single date rather than a complete weekly interval.
- Overseas instruments may require account-specific market-data permissions.
- Arbitrary-N incremental clustering, multi-DFA selection, and query-specific path planning are outside the MVP.

## Citation

```bibtex
@article{autologic2026,
  title={AutoLogic: Condition-Driven Automata for Financial Report Generation},
  author={Anonymous},
  year={2026}
}
```
