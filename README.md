# Autonomous Data Analyst Agent

An AI-powered CSV analysis system built with **LangGraph**. Upload a dataset, let specialized agents plan an analysis, generate and execute Python code, review the results, and produce a concise report with charts.

This project is designed around the MLSA ML Team challenge: a **Planner**, **Coder/Executor**, and **Reviewer** collaborate through a bounded self-correction loop. It includes a five-dataset evaluation workflow and an execution-backend abstraction.

## What it does

- Accepts CSV datasets through a Streamlit interface.
- Profiles columns, data types, missing values, duplicates, and basic statistics.
- Creates a structured analysis plan from dataset metadata.
- Generates Pandas / scikit-learn analysis code and executes it through a configurable execution backend.
- Reviews execution results against the plan and routes failures back for correction.
- Stops after a bounded number of correction attempts and records why analysis could not complete.
- Produces a Markdown report and charts.
- Supports a deterministic demonstration mode when an LLM API key is not configured.
- Provides a benchmark script for five public datasets.

## Agent workflow

```text
START
  |
  v
inspect_dataset  --> Load CSV, inspect schema, profile data quality
  |
  v
planner          --> Build a structured plan and decide whether ML is useful
  |
  v
coder            --> Generate Python analysis code
  |
  v
execute_analysis --> Run code via the configured execution backend
  |
  v
reviewer         --> Check results against the plan and execution errors
  |
  v
conditional_router
  |-- Needs correction --> coder (bounded retry loop)
  |-- Accepted ---------> report_generator --> END
  |-- Retry limit ------> failure_handler --> END
```

### Agent responsibilities

| Role | Responsibility | Output |
|---|---|---|
| **Planner** | Uses dataset profile and task context to define analysis steps, charts, and optional ML work. | Structured analysis plan |
| **Coder / Executor** | Converts the plan into Python, runs it, and uses errors or feedback to correct code. | Code, execution status, outputs |
| **Reviewer** | Checks whether the output covers the plan, whether execution succeeded, and whether required charts/results exist. | Accept, request correction, or terminate |

The workflow uses a bounded correction loop to reduce repeated failures and avoid unending agent execution.

## Technology stack

- Python 3.11+
- LangGraph and LangChain
- Streamlit
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Pydantic
- python-dotenv
- pytest
- NVIDIA NIM through an OpenAI-compatible API by default

## Repository layout

```text
app/
  main.py                 Streamlit user interface
  graph.py                LangGraph workflow and routing
  state.py                Shared state and structured models
  config.py               Environment and LLM configuration
  agents/
    planner.py            Planner agent
    coder.py              Code-generation agent
    reviewer.py           Review agent
  tools/
    dataset_tools.py      CSV ingestion and profiling
    code_executor.py      Execution backend abstraction
    report_generator.py   Markdown report and chart packaging
  prompts/                Agent prompt templates
docker/
  Dockerfile              Container image for Docker execution
scripts/
  evaluate.py             Five-dataset evaluation runner
tests/                    Unit, workflow, safety, and failure tests
datasets/                  Benchmark datasets and evaluation results
reports/                   Example reports and charts
```

## Requirements

- Python 3.11 or newer
- pip
- Optional: NVIDIA NIM API key (or another supported OpenAI-compatible API)
- Optional: Docker, for containerized execution

## Setup

### 1. Clone and enter the repository

```bash
git clone https://github.com/aarush006-cyber/autonomous-data-analyst-agent.git
cd autonomous-data-analyst-agent
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the model (optional)

Copy `.env.example` to `.env` and edit the values. For NVIDIA NIM, set:

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_actual_api_key
LLM_MODEL=meta/muse-glimmer-30b
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
```

Never commit your real `.env` file or API keys. The repository's `.env.example` should contain placeholder values only.

Without a configured API key, the application can run in **Demonstration Mode**, which provides deterministic analysis without live model calls. The UI/report should identify demo-mode output clearly.

## Run the application

```bash
streamlit run app/main.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`, then upload a CSV and start the analysis workflow.

## Run the tests

```bash
python -m pytest tests/ -v
```

Tests cover CSV validation and profiling, agent output validation, executor behavior, report generation, workflow routing, correction-loop limits, configuration, demonstration mode, and failure handling.

## Evaluate on five datasets

Run:

```bash
python scripts/evaluate.py
```

The benchmark suite covers a mix of classification and regression tasks:

| Dataset | Analysis type | What it tests |
|---|---|---|
| Iris | Multiclass classification | Clean numerical features and target discovery |
| Titanic | Classification | Mixed column types, categorical fields, and missing data |
| Wine Quality | Classification / quality prediction | Numerical feature analysis and target selection |
| Breast Cancer Wisconsin | Binary classification | Higher-dimensional numerical data |
| California Housing | Regression | Continuous target and larger dataset handling |

The script writes aggregate results to `datasets/evaluation_results.json`. See [DATASET_EVALUATION.md](DATASET_EVALUATION.md) for the recorded evaluation details and [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md) for known failure modes and their causes.

## Execution and safety

The project supports a modular executor design:

- **Subprocess executor:** Runs analysis code in a separate process with a timeout and a restricted environment. This helps isolate crashes but is **not a secure sandbox**; generated code still executes on the host.
- **Docker executor:** Runs code in a container with tighter restrictions, including no network access, resource limits, reduced privileges, and read-only mounts for inputs where configured.

Configure `EXECUTION_BACKEND` in `.env` or your environment:

| Value | Behavior |
|---|---|
| `auto` | Prefer Docker when available; otherwise use the configured fallback behavior |
| `docker` | Require Docker and fail clearly if it is unavailable |
| `subprocess` | Force subprocess execution |

Build the image with:

```bash
docker build -t data-analyst-sandbox:latest -f docker/Dockerfile .
```

**Security note:** Do not run this project against untrusted code in production using the subprocess backend. Import/keyword filtering is not equivalent to operating-system isolation. Review the implementation and Docker configuration before treating any execution mode as a security boundary.

## Known limitations

- LLM-generated code quality depends on the selected model and may need correction.
- Demonstration mode is deterministic rather than fully task-adaptive.
- Very large datasets can increase execution time and memory use.
- Session persistence and report retention may be limited by the current Streamlit implementation.
- The five-dataset benchmark is a useful smoke/evaluation suite, not proof of general reliability across all possible CSVs.
- Docker execution requires a working Docker engine and compatible host configuration.

## Project documentation

- [Dataset evaluation](DATASET_EVALUATION.md)
- [End-to-end test report](END_TO_END_TEST_REPORT.md)
- [Failure analysis](FAILURE_ANALYSIS.md)
- [Final verification](FINAL_VERIFICATION.md)
- [GitHub preparation report](GITHUB_PREPARATION_REPORT.md)

## Challenge alignment

This repository implements the requested core shape: CSV-driven analysis, at least three agent roles, a bounded self-correction loop, report/chart generation, and a five-dataset evaluation path. The actual completeness of the challenge requirements should be judged using the code, tests, and evaluation artifacts—not README claims alone.

## License

Add a repository-level license file before redistributing this project if you intend it to be open source. Dataset licenses and source attributions are documented in the evaluation materials.
