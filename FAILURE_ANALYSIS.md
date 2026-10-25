# Autonomous Data Analyst Agent — Failure Analysis Report

**Document Date:** 2026-10-25  
**Scope:** Phase 1 & Phase 2 System Architecture  
**Execution Context:** Windows 11, Python 3.13.14, Scikit-Learn 1.8.0, NVIDIA NIM (`meta/muse-glimmer-30b`), Docker CLI 29.9.2  

---

## 1. Overview

This document records **actual observed failures**, **controlled failure test results**, **root causes**, and **remedial actions** implemented across the Autonomous Data Analyst Agent project.

Failures are categorized into three distinct classes:
1. **Empirical Live Workflow Failures:** Failures observed during real multi-agent LLM invocations with NVIDIA Muse Glimmer.
2. **Controlled Failure Tests:** Deterministic tests engineered to verify error capture, self-correction, timeout enforcement, and fallbacks.
3. **Environment & Sandbox Failures:** Docker daemon and host environment boundary constraints.

---

## 2. Test Suite Status (86 Tests Executed)

| Test Module / Category | Scope | Tests Executed | Passed | Failed |
|---|---|:---:|:---:|:---:|
| `TestCSVValidation` | CSV file format, existence, empty checks | 7 | 7 | 0 |
| `TestDatasetProfiling` | Data types, missing values, duplicates, summary stats | 6 | 6 | 0 |
| `TestPlanner` | Structured plan creation, schema validation | 4 | 4 | 0 |
| `TestReviewer` | Acceptance decisions, error extraction, missing steps | 4 | 4 | 0 |
| `TestCodeExecutor` | Subprocess execution, timeouts, blocked keywords, hashing | 8 | 8 | 0 |
| `TestReportGenerator` | Markdown report generation, empirical stats, labels | 3 | 3 | 0 |
| `TestWorkflow` | LangGraph full-state pipeline & correction loops | 4 | 4 | 0 |
| `TestConfiguration` | `.env` parsing, model factory, secret masking, live NIM call | 12 | 12 | 0 |
| `TestStateModels` | Pydantic state model serialization & defaults | 4 | 4 | 0 |
| `TestCoder` | Deterministic demo code generation & ML workflows | 3 | 3 | 0 |
| `TestCoderHardeningAndEdgeCases` | Code injection, non-finite values, rare classes, fallbacks | 13 | 13 | 0 |
| `TestFailureHandlingScenarios` | 7 controlled failure scenarios (API crash, malformed JSON, etc.) | 9 | 9 | 0 |
| `TestDockerExecutor` | Container command construction, non-root user, daemon availability | 11 | 11 | 0 |
| **Total Test Suite** | **Comprehensive Regression & Security Suite** | **86** | **86** | **0** |

*All 86 automated tests pass with 100% success rate.*

---

## 3. Empirical Live Workflow Failures (NVIDIA Muse Glimmer Run)

During the live end-to-end evaluation on the Iris dataset with NVIDIA Muse Glimmer (`meta/muse-glimmer-30b`), four genuine execution failures were captured across 4 iterations (732.22s total run time).

### Failure 1: Safety Validator Keyword Substring False Positive
* **Observed In:** Iteration 0 (Initial execution)
* **Error Message:** `Code rejected: Blocked import/function detected: exec`
* **Exit Code:** `-1` (execution halted before subprocess start)
* **Root Cause:**
  In `app/tools/code_executor.py`, `validate_code_safety()` performed a naive substring check:
  ```python
  for blocked in BLOCKED_IMPORTS:
      if blocked in code:  # 'exec' in code!
          warnings.append(f"Blocked import/function detected: {blocked}")
  ```
  The generated code contained ordinary English words like `execution_time` or `def execute_analysis()`. Because `"exec"` is a substring of `"execution"`, legitimate code was erroneously rejected as a blocked call to Python's built-in `exec()`.
* **Fix Applied:**
  Replaced naive substring matching with strict regex word-boundary matching:
  ```python
  pattern = r"\b" + re.escape(blocked).replace(r"\.", r"\s*\.\s*") + r"\b"
  if re.search(pattern, code):
      warnings.append(f"Blocked import/function detected: {blocked}")
  ```
* **Verification:** Confirmed by `test_clean_code_passes_validation` and `test_blocked_import`.

---

### Failure 2: Scikit-Learn 1.8.0 API Incompatibility
* **Observed In:** Iteration 1 & Iteration 3
* **Error Message:** `TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'`
* **Exit Code:** `1`
* **Root Cause:**
  The installed environment uses `scikit-learn 1.8.0`. In earlier scikit-learn releases ($< 1.5$), `LogisticRegression(multi_class='multinomial')` was the standard syntax for multiclass problems. Scikit-learn deprecated and subsequently removed `multi_class`, making its inclusion raise an immediate `TypeError`. The LLM's pre-training data generated code with `multi_class='multinomial'`.
* **Agent Behavior & Self-Correction:**
  - The Reviewer agent captured the exact traceback from stderr:
    `"TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'"`
  - The Reviewer generated explicit instructions:
    `"Fix the LogisticRegression initialization to be compatible with the installed scikit-learn version, e.g. use LogisticRegression(solver='lbfgs', max_iter=200) without multi_class..."`
  - The Coder received the error and instructions on subsequent iterations.
* **Fix Applied:**
  Updated `CODER_SYSTEM_PROMPT` in `app/prompts/coder_prompt.py` to include explicit guidance:
  `- If doing ML: split data properly, avoid leakage, report appropriate metrics. For scikit-learn models, use modern API compatibility (e.g. for LogisticRegression do not pass 'multi_class'; or use RandomForestClassifier / RandomForestRegressor).`

---

### Failure 3: LLM Code Generation Syntax Error
* **Observed In:** Iteration 2
* **Error Message:** `SyntaxError: '(' was never closed` (line 160)
* **Exit Code:** `1`
* **Root Cause:**
  During an aggressive code edit in iteration 2, the model truncated a multi-line `plt.savefig()` call, leaving an open parenthesis unclosed at line 160.
* **Agent Behavior & Self-Correction:**
  - The SubprocessExecutor captured the `SyntaxError` and line number from stderr.
  - The Reviewer identified the unbalanced parenthesis:
    `"Fix the syntax error in analysis_script.py line 160 by closing the plt.savefig call correctly..."`
  - In Iteration 3, the Coder corrected the syntax error, producing a valid 5,674-character script that executed successfully through all EDA and plotting steps.

---

### Failure 4: Maximum Correction Limit Reached
* **Observed In:** Iteration 3 (Terminal state)
* **Outcome:** Clean termination and routing to `failure_handler`
* **Root Cause:**
  After 3 correction attempts (`max_correction_attempts = 3`), the multiclass classification step still failed due to `multi_class`.
* **Agent Behavior & Self-Correction:**
  - The Reviewer evaluated iteration 3, noted `current_iteration >= max_iterations`, and set status to `terminated`:
    `"Per policy after 3 total correction attempts, the analysis is terminated."`
  - The LangGraph conditional router routed to `failure_handler`.
  - The report generator produced `artifacts/analysis_report.md` (3,788 chars), explicitly titled `# Data Analysis Report — iris.csv [TERMINATED / FAILED]`, displaying all 6 generated charts and the full 3-iteration history.
* **Verification:** Confirmed that the system **never falsifies success** when an error remains unresolved.

---

## 4. Controlled Failure Tests & Verification

Dedicated automated test suites verify that every component handles failures gracefully:

### 4.1 Malformed Model Output
* **Test Case:** `test_planner_malformed_llm_output`, `test_reviewer_malformed_llm_output`
* **Mechanism:** Mocked LLM returns non-JSON conversational text without curly braces.
* **Observed Behavior:** Catches `ValueError` / `json.JSONDecodeError`; sets `is_fallback: True`, `generation_mode: "fallback"`, records `fallback_reason`, and gracefully substitutes deterministic fallbacks.
* **Status:** **PASS**

### 4.2 Model API Failures (HTTP 503 / Rate Limits / Timeouts)
* **Test Case:** `test_api_failure_triggers_honest_fallback`
* **Mechanism:** Mocked LLM raises `RuntimeError("503 Service Unavailable")`.
* **Observed Behavior:** Enters exception handler; logs warning; sets `is_fallback: True`; sets `fallback_reason`; renders disclaimer banner in Markdown report:
  `> ⚠️ **EXECUTION MODE: DETERMINISTIC FALLBACK (LLM FAILED/UNAVAILABLE)**`
* **Status:** **PASS**

### 4.3 Python Code Syntax Errors
* **Test Case:** `test_syntax_error_execution_and_review`
* **Mechanism:** Injects unclosed string literal into Python script.
* **Observed Behavior:** Subprocess fails with exit code 1; `SyntaxError` extracted from stderr into `error_summary`; Reviewer assigns `needs_correction`.
* **Status:** **PASS**

### 4.4 Execution Timeouts
* **Test Case:** `test_execution_timeout`
* **Mechanism:** Injects `time.sleep(10)` with a 1-second timeout.
* **Observed Behavior:** Subprocess times out; `subprocess.TimeoutExpired` caught; returns `exit_code: -1` and `error_summary: "Execution timed out after 1 seconds"`.
* **Status:** **PASS**

### 4.5 Missing Artifacts Detection
* **Test Case:** `test_missing_chart_detected_by_reviewer`
* **Mechanism:** Script completes with exit code 0 but generates zero charts despite plan requiring plots.
* **Observed Behavior:** Reviewer compares planned outputs against generated files; flags missing plots in `missing_steps`; sets status to `needs_correction`.
* **Status:** **PASS**

### 4.6 Empty & Malformed Datasets
* **Test Case:** `test_empty_dataset_error`, `test_malformed_dataset_error`
* **Mechanism:** Empty file (0 bytes) and corrupted binary files uploaded.
* **Observed Behavior:** `load_csv()` raises `DatasetError`; blocks dataset from reaching the model; graph halts cleanly at `inspect_dataset`.
* **Status:** **PASS**

### 4.7 Docker Sandbox Unavailability
* **Test Case:** `test_docker_unavailable_fails_explicitly_without_host_execution`
* **Mechanism:** Invokes `DockerExecutor.execute()` when Docker daemon is not running.
* **Observed Behavior:** Returns `ExecutionResult(success=False, exit_code=-1)` with explicit error:
  `"Docker daemon is unavailable or not running. Execution was NOT run on host to preserve isolation boundaries."`
* **Status:** **PASS** (Zero silent fallbacks to host execution).

### 4.8 Docker Container Timeout & Resource Cleanup
* **Test Case:** `test_mocked_docker_execution_timeout`
* **Mechanism:** Simulates `subprocess.TimeoutExpired` during container execution.
* **Observed Behavior:** Executes `docker kill <container_name>` and `docker rm -f <container_name>` in `finally` block; cleans temporary host script directory.
* **Status:** **PASS**

---

## 5. Security Limitations & Phase 2 Sandbox Status

| Security Requirement | Implementation | Current Verification Status |
|---|---|:---:|
| **Subprocess Execution Isolation** | Generated code runs in separate child process via `sys.executable` | **Verified (Phase 1)** |
| **Streamlit Process Protection** | Zero `exec()` or `eval()` calls in UI process | **Verified (Phase 1)** |
| **Environment Variable Stripping** | Child processes run in sanitized environment excluding `NVIDIA_API_KEY` | **Verified (Phase 1)** |
| **Execution Timeouts** | 120s hard timeout via `subprocess.run(timeout=120)` | **Verified (Phase 1)** |
| **Code Size & Keyword Filters** | 100 KB limit; blocked imports (`subprocess`, `socket`, `eval`, `exec`) | **Verified (Phase 1)** |
| **Docker Sandbox Command Construction** | `--network none`, `--user 1000:1000`, `--cpus`, `--memory`, `--read-only` | **Verified (Phase 2 Unit Tests)** |
| **Docker Unavailability Transparency** | Explicit error; no silent fallback to host execution | **Verified (Phase 2 Unit Tests)** |
| **Live Docker Container Execution** | Active Docker daemon running on evaluation machine | **Verified (Live Run: 1.75s, Iris)** |

> **Note on Live Docker Container Execution:**
> Docker Desktop daemon (`desktop-linux`, Server Version 29.9.2) was initialized on the host machine. The sandbox container image `data-analyst-sandbox:latest` was built from `docker/Dockerfile`, and a live execution run with `DockerExecutor` executed successfully on the Iris dataset in 1.75 seconds, verifying network isolation (`--network none`), read-only dataset mounting, non-root user execution, and artifact retrieval.
