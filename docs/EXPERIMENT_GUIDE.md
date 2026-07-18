# LogicRAG Experiment Guide

This guide explains how to configure the runtime environment and run the full LogicRAG prototype.

## 1. Environment Variables

You can provide API keys and account credentials either as environment variables or as CLI hyperparameters. For DeepSeek API, you can apply for an API key at https://platform.deepseek.com/. For `DASHSCOPE_API_KEY`, you can create or view an API key in the Alibaba Cloud Bailian console at https://bailian.console.aliyun.com/. For `IFIND_USERNAME` and `IFIND_PASSWORD`, you can apply for a free trial account at https://quantapi.10jqka.com.cn/.

### PowerShell Temporary Configuration

These variables are valid only in the current PowerShell session:

```powershell
cd "**The file path where you store LogicRAG**"

$env:DEEPSEEK_API_KEY="your_deepseek_api_key"

$env:DASHSCOPE_API_KEY="your_dashscope_api_key"
$env:DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

$env:IFIND_USERNAME="your_ifind_username"
$env:IFIND_PASSWORD="your_ifind_password"
```

### Windows User-Level Configuration

These variables persist after reopening PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "your_deepseek_api_key", "User")
[Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "your_dashscope_api_key", "User")
[Environment]::SetEnvironmentVariable("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1", "User")
[Environment]::SetEnvironmentVariable("IFIND_USERNAME", "your_ifind_username", "User")
[Environment]::SetEnvironmentVariable("IFIND_PASSWORD", "your_ifind_password", "User")
```

Restart PowerShell after setting user-level variables.

## 2. Input Dataset Format

`document_learner.py` expects a CSV file where each row is one sample report.

Recommended format:

```csv
report_id,section_1,section_2,section_3,risk_warning
report_a,text...,text...,text...,text...
report_b,text...,text...,text...,text...
```

Requirements:

- The CSV must contain at least two report rows.
- `report_id` is optional but recommended.
- All non-empty columns except `report_id` are concatenated as the full report text.
- The two selected rows should come from the same report type or vertical.
- The file should be saved as UTF-8 or UTF-8-SIG.

Example dataset path used during development:

```text
dataset\Precious_Metals\data\case_2253.csv
```

## 3. One-Command Client

The easiest way to run the full pipeline is:

```powershell
python logicrag_client.py `
  --csv "dataset\Precious_Metals\data\case_2253.csv" `
  --row-a 0 `
  --row-b 1 `
  --query "write a precious metals research report for February 28, 2025" `
  --theta 0.5 `
  --tau 0.5 `
  --date "2025-02-28"
```


## 4. Hyperparameters

Main hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `--theta` | `0.5` | Threshold for matching states across the two sample reports in `document_learner.py`. |
| `--tau` | `0.5` | Threshold for matching query embedding to state embeddings in `query_processing.py`. |
| `--query` | required | User query for the generation task. This value must be provided explicitly from the CLI. |
| `--row-a` | `0` | First sample report row index. |
| `--row-b` | `1` | Second sample report row index. |
| `--output-root` | `logicrag_outputs` | Output directory for all intermediate and final artifacts. |
| `--date` | empty | Optional manual override for the iFinD query date. |
| `--local-embedding-only` | off | Use deterministic local hash embeddings instead of DashScope embeddings. |

### Domain Dictionary for iFinD CODES

`ifind_data_plugin.py` resolves iFinD `CODES` from `domain_dictionary.csv`. The plugin does not use built-in futures aliases, direct-code fallback, or a default precious-metals portfolio.

Because the iFinD API platform does not provide a complete public `CODES` mapping table, the provided `domain_dictionary.csv` contains only the partial mappings needed for the current experiments. If you want to retrieve data for another domain, exchange, contract, or instrument, manually add the corresponding mapping rows to `domain_dictionary.csv` before running data retrieval. You can find CODES for your domain in the iFinD official website https://quantapi.10jqka.com.cn/.

The required format is `CODE,Name`; an optional `Aliases` column can be added for controlled fuzzy matching when the material name and dictionary name differ.

```csv
CODE,Name,Aliases
@GC0Y.CMX,纽约金连一,COMEX黄金|COMEX金|纽约金|COMEX gold
@GC0Y.LME,伦金连续,LME金|LME金|LME黄金|LME金|LME gold
```

API and account parameters can also be passed directly:

```powershell
python logicrag_client.py `
  --query "write a precious metals research report for February 28, 2025" `
  --deepseek-api-key "your_deepseek_api_key" `
  --dashscope-api-key "your_dashscope_api_key" `
  --ifind-username "your_ifind_username" `
  --ifind-password "your_ifind_password"
```

## 5. Step-by-Step Execution

You can also run each stage separately.

### 5.1 Check DashScope Embedding

```powershell
python check_dashscope_embedding.py
```

Expected result:

```text
Embedding API check succeeded.
dimension=1024
```

### 5.2 Document Logic Extraction

```powershell
python document_learner.py `
  --csv "dataset\Precious_Metals\data\case_2253.csv" `
  --row-a 0 `
  --row-b 1 `
  --theta 0.5 `
  --output-dir "logicrag_outputs"
```

Outputs:

```text
logicrag_outputs/document_a_state_sequence.json
logicrag_outputs/document_b_state_sequence.json
logicrag_outputs/global_template.json
logicrag_outputs/state_index.json
logicrag_outputs/run_manifest.json
```

### 5.3 Query Processing

```powershell
python query_processing.py `
  --query "write a precious metals research report for February 28, 2025" `
  --tau 0.5 `
  --template "logicrag_outputs/global_template.json" `
  --index "logicrag_outputs/state_index.json" `
  --output "logicrag_outputs/query_subtree.json"
```

Output:

```text
logicrag_outputs/query_subtree.json
```

### 5.4 Data Retrieval

Dry run first:

```powershell
python ifind_data_plugin.py `
  --query "write a precious metals research report for February 28, 2025" `
  --query-subtree "logicrag_outputs/query_subtree.json" `
  --dictionary "domain_dictionary.csv" `
  --output-dir "logicrag_outputs/retrieved_materials" `
  --dry-run
```

Actual retrieval:

```powershell
python ifind_data_plugin.py `
  --query "write a precious metals research report for February 28, 2025" `
  --date "2025-02-28" `
  --query-subtree "logicrag_outputs/query_subtree.json" `
  --dictionary "domain_dictionary.csv" `
  --output-dir "logicrag_outputs/retrieved_materials"
```

Outputs:

```text
logicrag_outputs/retrieved_materials/retrieved_data.csv
logicrag_outputs/retrieved_materials/data_bindings.json
logicrag_outputs/retrieved_materials/materials_index.json
logicrag_outputs/retrieved_materials/materials_*.json
```

### 5.5 Report Generation

Dry run:

```powershell
python report_generator.py --dry-run
```

Actual generation:

```powershell
python report_generator.py `
  --query-subtree "logicrag_outputs/query_subtree.json" `
  --materials-dir "logicrag_outputs/retrieved_materials" `
  --output-md "logicrag_outputs/generated_report.md"
```

Outputs:

```text
logicrag_outputs/generated_report.md
logicrag_outputs/generated_node_outputs.json
logicrag_outputs/generation_trace.json
```

## 6. Debugging Tips

- If `global_template.json` loses obvious shared states, reduce `--theta` or inspect `match_reason` in the global template.
- If `query_subtree.json` is too small, reduce `--tau`.
- If market data is empty, check whether the query date is a trading day.
- If iFinD login fails, verify `IFIND_USERNAME`, `IFIND_PASSWORD`, network access, and market-data permissions.
- If DashScope embeddings fail, run with `--local-embedding-only` to test the rest of the pipeline.
- If final generation is slow or expensive, use `--dry-run-report` first.

## 7. Recommended Debug Sequence

```powershell
python check_dashscope_embedding.py
python logicrag_client.py --query "write a precious metals research report for February 28, 2025" --dry-run-data --dry-run-report --local-embedding-only
python logicrag_client.py --query "write a precious metals research report for February 28, 2025" --dry-run-data --dry-run-report
python logicrag_client.py --query "write a precious metals research report for February 28, 2025" --dry-run-report --date "2025-02-28"
python logicrag_client.py --query "write a precious metals research report for February 28, 2025" --date "2025-02-28"
```
