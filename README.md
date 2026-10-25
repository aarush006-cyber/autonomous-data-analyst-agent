# Autonomous Data Analyst Agent

An AI-powered data analysis application that accepts CSV files and autonomously inspects, plans, codes, executes, reviews, and reports on data analysis — all orchestrated by a LangGraph workflow with three distinct agents.

## Architecture

```
START
  │
  ▼
inspect_dataset  →  Load CSV, profile data, detect quality issues
  │
  ▼
planner          →  Create structured analysis plan
  │
  ▼
coder            →  Generate Python analysis code
  │
  ▼
execute_analysis →  Run code in isolated execution service
  │
  ▼
reviewer         →  Evaluate results against plan
  │
  ▼
conditional_router
  ├── Needs correction → coder (max 3 retries)
  ├── Accepted        → report_generator → END
  └── Limit reached   → failure_handler  → END
```

### Three Agents

| Agent | Role | Key Outputs |
|-------|------|-------------|
| **Planner** | Inspects dataset metadata, creates structured analysis plan, decides if ML is appropriate | `AnalysisPlan` (steps, visualizations, ML decision) |
| **Coder/Executor** | Generates Python code from the plan, handles corrections with error context | Analysis code + execution results |
| **Reviewer** | Validates execution output against plan, checks for missing steps and errors | `ReviewResult` (accept / correct / terminate) |

## Technology Stack

- **Python 3.11+** — Runtime
- **LangGraph** — Workflow orchestration (state graph with conditional routing)
- **LangChain** — LLM integration (OpenAI-compatible API)
- **Streamlit** — Web interface
- **Pandas / NumPy** — Data analysis
- **Matplotlib / Seaborn** — Visualization
- **scikit-learn** — Machine learning and statistical analysis
- **Pydantic** — Structured output validation
- **python-dotenv** — Environment configuration
- **pytest** — Automated testing

## Prerequisites

- Python 3.11 or later
- pip

## Environment Configuration

1. Copy the example environment file:
   ```
   cp .env.example .env
   ```

2. Edit `.env` with your LLM credentials:
   ```
   LLM_PROVIDER=nvidia
   NVIDIA_API_KEY=your_actual_api_key
   LLM_MODEL=meta/muse-glimmer-30b
   LLM_BASE_URL=https://integrate.api.nvidia.com/v1
   ```

3. Without an API key, the application runs in **Demonstration Mode** — deterministic analysis without LLM calls. Demo mode is clearly labeled in the UI and reports.

### LLM Provider Setup

The application uses NVIDIA NIM via LangChain's OpenAI-compatible integration (`langchain-openai`). It connects to NVIDIA NIM with Muse Glimmer by default:

- **Model**: `meta/muse-glimmer-30b`
- **Base URL**: `https://integrate.api.nvidia.com/v1`
- **API Key**: `NVIDIA_API_KEY` (also supports `LLM_API_KEY`)

- **NVIDIA NIM**: Set `LLM_BASE_URL=https://integrate.api.nvidia.com/v1`
- **OpenAI**: Set `LLM_BASE_URL=https://api.openai.com/v1` and adjust model name
- **Local (Ollama, vLLM)**: Set `LLM_BASE_URL=http://localhost:11434/v1`

## Installation

```bash
cd autonomous-data-analyst
pip install -r requirements.txt
```

## How to Launch

```bash
streamlit run app/main.py
```

The application opens at `http://localhost:8501`.

## How to Run Automated Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- CSV validation (empty, malformed, valid files)
- Dataset profiling (missing values, duplicates, types)
- Planner output validation
- Reviewer output validation
- Code executor (success, timeout, blocked imports)
- Report generation
- LangGraph workflow routing
- Correction loop termination
- Configuration validation
- Demonstration mode without API key

All unit tests run without API credentials (LLM calls are bypassed in demo mode).

## Five-Dataset Evaluation

```bash
python scripts/evaluate.py
```

Evaluates on:
1. **Iris** — Classification (sklearn built-in)
2. **Titanic** — Classification (seaborn built-in)
3. **Wine Quality** — Classification (sklearn built-in)
4. **Breast Cancer Wisconsin** — Classification (sklearn built-in)
5. **California Housing** — Regression (sklearn built-in)

Results are saved to `datasets/evaluation_results.json`.

### Data Sources & Licensing

| Dataset | Source | License |
|---------|--------|---------|
| Iris | `sklearn.datasets.load_iris` | Public domain (Fisher, 1936) |
| Titanic | `seaborn.load_dataset("titanic")` | Public domain |
| Wine Quality | `sklearn.datasets.load_wine` | BSD (sklearn) / CC BY 4.0 (UCI) |
| Breast Cancer | `sklearn.datasets.load_breast_cancer` | Public domain (UCI) |
| California Housing | `sklearn.datasets.fetch_california_housing` | Public domain (StatLib) |

## How the Correction Loop Works

1. **Coder** generates analysis code from the plan.
2. **Executor** runs the code in a subprocess with timeout.
3. **Reviewer** checks results against the plan.
4. If the reviewer finds issues:
   - The error, stderr, and reviewer feedback are sent back to the Coder.
   - The Coder generates corrected code using the full error context.
   - This repeats up to **3 correction attempts**.
5. After 3 failed corrections, the workflow terminates via `failure_handler`.
6. Repeated identical code submissions are detected via code hashing.

## Execution Backends: Subprocess & Phase 2 Docker Sandboxing

The application features a modular execution service abstraction via `ExecutorInterface` with two primary backends:

1. **Phase 1: `SubprocessExecutor` (Host Subprocess)**
   - Executes generated code out-of-process via `sys.executable`.
   - Protects the Streamlit UI from crashes and memory corruption.
   - Enforces a 120-second timeout, input path validation, clean environment (no API keys passed), and keyword blocking (`subprocess`, `os.system`, `socket`, `eval`, `exec`).
   - *Limitation:* Does not provide OS kernel-level hardware isolation.

2. **Phase 2: `DockerExecutor` (Containerized Sandbox)**
   - Executes generated Python code inside disposable Docker containers.
   - **Non-root Execution:** Runs as non-root user (`--user 1000:1000`).
   - **Filesystem Isolation:** Mounts input dataset as read-only (`:ro`), script as read-only (`:ro`), and output directory as read-write (`:rw`).
   - **Network Isolation:** Completely disables network access for untrusted code (`--network none`).
   - **Resource Limits:** Enforces CPU limit (`--cpus 2.0`), memory limit (`--memory 512m`), and process count ceiling (`--pids-limit 64`).
   - **Privilege Hardening:** Drops all Linux capabilities (`--cap-drop ALL`) and enforces `--security-opt no-new-privileges:true`. No Docker socket or privileged flags.
   - **Secret Hygiene:** Host secrets, including `NVIDIA_API_KEY`, are completely withheld from the container environment.
   - **Explicit Availability:** Never silently runs on the host while claiming sandboxed execution. If Docker daemon is unavailable, it fails explicitly with an actionable message.

### Configuring Execution Backend

Set `EXECUTION_BACKEND` in `.env` or system environment:
- `EXECUTION_BACKEND=auto` (default) — Uses `DockerExecutor` if Docker daemon is running; otherwise falls back to `SubprocessExecutor` with transparent logging.
- `EXECUTION_BACKEND=docker` — Strictly requires Docker; fails explicitly if Docker daemon is stopped.
- `EXECUTION_BACKEND=subprocess` — Forces host subprocess execution.

### Building the Sandbox Container Image

```bash
docker build -t data-analyst-sandbox:latest -f docker/Dockerfile .
```

## Security Limitations

- Phase 1 subprocess executor is **NOT** a secure sandbox
- LLM-generated code runs on the host (development only)
- Import filtering is basic and can be circumvented
- No network isolation for executed code
- API credentials are never passed to the execution environment
- CSV contents are treated as untrusted (truncated before LLM prompts)

## Known Issues

- Demonstration mode produces deterministic analysis that may not be optimally tailored
- Large datasets may slow down the subprocess executor
- The Streamlit session does not persist analysis results across page refreshes
- LLM-generated code quality depends on the model used

## Project Structure

```
autonomous-data-analyst/
├── app/
│   ├── main.py                 # Streamlit UI
│   ├── graph.py                # LangGraph workflow
│   ├── state.py                # State & Pydantic models
│   ├── config.py               # Configuration loader
│   ├── agents/
│   │   ├── planner.py          # Planner agent
│   │   ├── coder.py            # Coder/Executor agent
│   │   └── reviewer.py         # Reviewer agent
│   ├── tools/
│   │   ├── dataset_tools.py    # CSV ingestion & profiling
│   │   ├── code_executor.py    # Subprocess & Docker execution abstraction
│   │   └── report_generator.py # Markdown report generator
│   └── prompts/
│       ├── planner_prompt.py   # Planner prompt templates
│       ├── coder_prompt.py     # Coder prompt templates
│       └── reviewer_prompt.py  # Reviewer prompt templates
├── docker/
│   └── Dockerfile              # Phase 2 isolated sandbox container definition
├── tests/
│   ├── test_analyst.py         # Core & safety test suite
│   ├── test_docker_executor.py # Docker sandbox isolation tests
│   └── test_failure_handling.py# Controlled failure scenario tests
├── scripts/
│   └── evaluate.py             # 5-dataset evaluation suite
├── datasets/                   # Benchmark CSV datasets
├── reports/                    # Generated Markdown reports and PNG charts
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── DATASET_EVALUATION.md
├── FAILURE_ANALYSIS.md
└── FINAL_VERIFICATION.md
```

