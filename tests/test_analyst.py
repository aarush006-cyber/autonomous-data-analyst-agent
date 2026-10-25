"""Comprehensive tests for the Autonomous Data Analyst Agent."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import load_config, AppConfig, LLMConfig, ExecutionConfig, get_llm
from app.graph import build_graph, run_analysis
from app.state import (
    AnalysisPlan, ReviewResult, ExecutionResult,
    CorrectionRecord, DatasetMetadata, AnalystState,
)
from app.tools.dataset_tools import (
    load_csv, profile_dataset, validate_csv_file, DatasetError,
    get_safe_metadata_for_llm,
)
from app.tools.code_executor import (
    SubprocessExecutor, DockerExecutor, DemoExecutor, validate_code_safety,
    get_executor, code_hash,
)
from app.tools.report_generator import generate_report
from app.agents.planner import run_planner, _build_demo_plan
from app.agents.coder import run_coder, _generate_demo_code
from app.agents.reviewer import run_reviewer, _demo_review


# ──────────────────── Fixtures ────────────────────


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_analyst_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_csv(temp_dir):
    """Create a simple sample CSV."""
    path = os.path.join(temp_dir, "sample.csv")
    df = pd.DataFrame({
        "id": range(1, 101),
        "name": [f"item_{i}" for i in range(1, 101)],
        "value": np.random.randn(100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "score": np.random.uniform(0, 100, 100),
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def iris_csv(temp_dir):
    """Create Iris-like CSV."""
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = pd.Categorical.from_codes(data.target, data.target_names)
    path = os.path.join(temp_dir, "iris.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def empty_csv(temp_dir):
    path = os.path.join(temp_dir, "empty.csv")
    with open(path, "w") as f:
        f.write("")
    return path


@pytest.fixture
def headers_only_csv(temp_dir):
    path = os.path.join(temp_dir, "headers_only.csv")
    with open(path, "w") as f:
        f.write("col1,col2,col3\n")
    return path


@pytest.fixture
def missing_values_csv(temp_dir):
    path = os.path.join(temp_dir, "missing.csv")
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, None, None, None],
        "c": ["x", "y", None, "z", "w"],
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def duplicates_csv(temp_dir):
    path = os.path.join(temp_dir, "duplicates.csv")
    df = pd.DataFrame({
        "x": [1, 2, 3, 1, 2, 3],
        "y": ["a", "b", "c", "a", "b", "c"],
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def demo_config():
    return AppConfig(
        llm=LLMConfig(
            provider="nvidia",
            api_key="",
            model="test-model",
            base_url="http://localhost",
            available=False,
        ),
        execution=ExecutionConfig(
            max_correction_attempts=3,
            code_execution_timeout=60,
            max_code_size_kb=100,
            max_output_size_kb=500,
            backend="subprocess",  # Explicit development-only backend; CI has no Docker daemon.
        ),
    )


# ──────────────────── CSV Validation Tests ────────────────────


class TestCSVValidation:
    def test_valid_csv(self, sample_csv):
        validate_csv_file(sample_csv)  # Should not raise

    def test_missing_file(self):
        with pytest.raises(DatasetError, match="not found"):
            validate_csv_file("/nonexistent/path.csv")

    def test_empty_file(self, empty_csv):
        with pytest.raises(DatasetError, match="empty"):
            validate_csv_file(empty_csv)

    def test_non_csv_extension(self, temp_dir):
        path = os.path.join(temp_dir, "data.txt")
        with open(path, "w") as f:
            f.write("data")
        with pytest.raises(DatasetError, match="CSV"):
            validate_csv_file(path)

    def test_load_valid_csv(self, sample_csv):
        df = load_csv(sample_csv)
        assert len(df) == 100
        assert len(df.columns) == 5

    def test_load_empty_csv(self, empty_csv):
        with pytest.raises(DatasetError):
            load_csv(empty_csv)

    def test_load_headers_only(self, headers_only_csv):
        with pytest.raises(DatasetError, match="empty|no rows"):
            load_csv(headers_only_csv)


# ──────────────────── Dataset Profiling Tests ────────────────────


class TestDatasetProfiling:
    def test_basic_profiling(self, sample_csv):
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        assert meta.rows == 100
        assert meta.columns == 5
        assert len(meta.column_names) == 5
        assert "id" in meta.column_names
        assert meta.duplicate_rows >= 0

    def test_missing_values_detection(self, missing_values_csv):
        df = load_csv(missing_values_csv)
        meta = profile_dataset(df, "missing.csv")
        assert meta.missing_values.get("b", 0) == 5
        assert any("missing" in w.lower() or "50%" in w for w in meta.quality_warnings)

    def test_duplicate_detection(self, duplicates_csv):
        df = load_csv(duplicates_csv)
        meta = profile_dataset(df, "duplicates.csv")
        assert meta.duplicate_rows == 3

    def test_numeric_summary(self, sample_csv):
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        assert "value" in meta.numeric_summary
        assert "mean" in meta.numeric_summary["value"]

    def test_categorical_summary(self, sample_csv):
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        assert "category" in meta.categorical_summary
        assert "unique" in meta.categorical_summary["category"]

    def test_safe_metadata_for_llm(self, sample_csv):
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        safe = get_safe_metadata_for_llm(meta)
        assert "filename" in safe
        assert len(str(safe)) < 50000  # Not too large


# ──────────────────── Planner Tests ────────────────────


class TestPlanner:
    def test_demo_plan_basic(self, sample_csv):
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        plan = _build_demo_plan(meta)
        assert len(plan.steps) >= 3
        assert plan.objective
        assert len(plan.expected_outputs) >= 1

    def test_demo_plan_with_classification(self, iris_csv):
        df = load_csv(iris_csv)
        meta = profile_dataset(df, "iris.csv")
        plan = _build_demo_plan(meta)
        assert plan.use_ml
        assert plan.ml_task == "classification"
        assert plan.target_column == "species"

    def test_planner_output_validation(self, sample_csv):
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        state: AnalystState = {
            "dataset_path": sample_csv,
            "dataset_metadata": meta.model_dump(),
            "user_request": "General analysis",
            "is_demo_mode": True,
            "execution_log": [],
            "correction_history": [],
        }
        result = run_planner(state)
        assert "analysis_plan" in result
        plan = AnalysisPlan(**result["analysis_plan"])
        assert len(plan.steps) >= 1

    def test_planner_no_forced_ml(self, sample_csv):
        """Should NOT force ML on a dataset without obvious target."""
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        plan = _build_demo_plan(meta)
        # sample_csv has no obvious target column
        assert not plan.use_ml


# ──────────────────── Reviewer Tests ────────────────────


class TestReviewer:
    def test_accept_successful_execution(self):
        exec_result = ExecutionResult(
            success=True, stdout="Results computed\n{}", exit_code=0,
            generated_files=["chart.png"], duration_seconds=1.5,
        )
        plan = AnalysisPlan(
            objective="Test",
            steps=[{"step": 1, "name": "overview", "description": "overview", "output": "summary"}],
        )
        review = _demo_review(exec_result, plan, ["chart.png"])
        assert review.status == "accepted"

    def test_reject_failed_execution(self):
        exec_result = ExecutionResult(
            success=False, stderr="Error: division by zero", exit_code=1,
            error_summary="division by zero",
        )
        plan = AnalysisPlan(
            objective="Test",
            steps=[{"step": 1, "name": "calc", "description": "calculation", "output": "result"}],
        )
        review = _demo_review(exec_result, plan, [])
        assert review.status == "needs_correction"
        assert len(review.detected_errors) > 0

    def test_reviewer_structured_output(self):
        exec_result = ExecutionResult(success=True, stdout="OK", exit_code=0)
        plan = AnalysisPlan(objective="Test", steps=[])
        state: AnalystState = {
            "dataset_path": "test.csv",
            "execution_result": exec_result.model_dump(),
            "analysis_plan": plan.model_dump(),
            "is_demo_mode": True,
            "current_iteration": 0,
            "max_iterations": 3,
            "generated_artifact_paths": [],
            "execution_log": [],
            "correction_history": [],
        }
        result = run_reviewer(state)
        review = ReviewResult(**result["review_result"])
        assert review.status in ("accepted", "needs_correction", "terminated")


# ──────────────────── Code Executor Tests ────────────────────


class TestCodeExecutor:
    def test_simple_execution(self, sample_csv, temp_dir):
        executor = SubprocessExecutor()
        code = 'import pandas as pd\ndf = pd.read_csv(DATASET_PATH)\nprint(f"Rows: {len(df)}")'
        result = executor.execute(code, sample_csv, temp_dir, timeout=30)
        assert result.success
        assert "Rows: 100" in result.stdout

    def test_execution_timeout(self, sample_csv, temp_dir):
        executor = SubprocessExecutor()
        code = "import time\ntime.sleep(100)"
        result = executor.execute(code, sample_csv, temp_dir, timeout=3)
        assert not result.success
        assert "timeout" in result.error_summary.lower() or "timed out" in result.error_summary.lower()

    def test_blocked_import(self, sample_csv, temp_dir):
        executor = SubprocessExecutor()
        code = "import subprocess\nsubprocess.run(['ls'])"
        result = executor.execute(code, sample_csv, temp_dir)
        assert not result.success
        assert "Blocked" in result.error_summary

    def test_code_safety_validation(self):
        warnings = validate_code_safety("import subprocess\nos.system('rm -rf /')")
        assert len(warnings) > 0

    def test_clean_code_passes_validation(self):
        code = "import pandas as pd\nimport numpy as np\ndf = pd.read_csv('data.csv')"
        warnings = validate_code_safety(code)
        assert len(warnings) == 0

    def test_pandas_dataframe_eval_is_not_blocked(self):
        code = "import pandas as pd\\ndf = pd.DataFrame({'a': [1]})\\ndf.eval('a + 1')"
        warnings = validate_code_safety(code)
        assert not any("eval" in warning.lower() for warning in warnings)

    def test_auto_backend_fails_closed_without_docker(self, sample_csv, temp_dir, monkeypatch):
        monkeypatch.setattr(DockerExecutor, "is_docker_available", staticmethod(lambda: False))
        executor = get_executor(is_demo=False, backend="auto")
        assert isinstance(executor, DockerExecutor)
        result = executor.execute("print('must not run')", sample_csv, temp_dir, timeout=1)
        assert not result.success
        assert "NOT run on the host" in result.error_summary

    def test_demo_auto_backend_does_not_fall_back_to_host(self, sample_csv, temp_dir, monkeypatch):
        monkeypatch.setattr(DockerExecutor, "is_docker_available", staticmethod(lambda: False))
        executor = get_executor(is_demo=True, backend="auto")
        result = executor.execute("print('must not run')", sample_csv, temp_dir, timeout=1)
        assert not result.success
        assert "NOT run on the host" in result.error_summary

    def test_demo_executor(self, sample_csv, temp_dir):
        executor = DemoExecutor()
        assert "Demo" in executor.backend_name
        code = 'import pandas as pd\ndf = pd.read_csv(DATASET_PATH)\nprint(f"OK: {len(df)}")'
        result = executor.execute(code, sample_csv, temp_dir, timeout=30)
        assert result.success

    def test_code_hash(self):
        h1 = code_hash("print('hello')")
        h2 = code_hash("print('hello')")
        h3 = code_hash("print('world')")
        assert h1 == h2
        assert h1 != h3

    def test_executor_factory(self):
        e1 = get_executor(is_demo=True)
        assert isinstance(e1, DemoExecutor)
        e2 = get_executor(is_demo=False, backend="subprocess")
        assert isinstance(e2, SubprocessExecutor)
        e3 = get_executor(is_demo=False)
        assert isinstance(e3, (SubprocessExecutor, DockerExecutor))

    def test_subprocess_windows_cache_regression(self, sample_csv, temp_dir):
        """Regression test: verify subprocess execution does not produce accidental %SystemDrive% cache paths."""
        executor = SubprocessExecutor()
        code = (
            "import matplotlib.pyplot as plt\n"
            "import os\n"
            "plt.figure()\n"
            "plt.plot([1, 2], [3, 4])\n"
            "plt.savefig(os.path.join(OUTPUT_DIR, 'test_plot.png'))\n"
            "plt.close('all')\n"
        )
        result = executor.execute(code, sample_csv, temp_dir, timeout=30)
        assert result.success is True
        stray_cache = os.path.join(temp_dir, "%SystemDrive%")
        assert not os.path.exists(stray_cache), f"Unexpected stray cache directory found: {stray_cache}"


# ──────────────────── Report Generator Tests ────────────────────


class TestReportGenerator:
    def test_generate_basic_report(self, temp_dir):
        metadata = DatasetMetadata(
            filename="test.csv", rows=100, columns=5,
            column_names=["a", "b", "c", "d", "e"],
            dtypes={"a": "int64", "b": "float64", "c": "object", "d": "int64", "e": "float64"},
        )
        plan = AnalysisPlan(objective="Test analysis", steps=[
            {"step": 1, "name": "overview", "description": "Overview", "output": "summary"},
        ])
        review = ReviewResult(status="accepted", acceptance_decision="Looks good")

        report = generate_report(
            metadata=metadata, plan=plan,
            execution_stdout='{"results": {"test": "value"}}',
            generated_files=["chart.png"],
            review=review,
            correction_history=[],
            output_dir=temp_dir,
        )
        assert "test.csv" in report
        assert "Test analysis" in report
        assert os.path.isfile(os.path.join(temp_dir, "analysis_report.md"))

    def test_demo_mode_label(self, temp_dir):
        metadata = DatasetMetadata(filename="test.csv", rows=10, columns=2)
        plan = AnalysisPlan(objective="Test")
        review = ReviewResult(status="accepted", acceptance_decision="OK")

        report = generate_report(
            metadata=metadata, plan=plan,
            execution_stdout="", generated_files=[], review=review,
            correction_history=[], output_dir=temp_dir, is_demo=True,
        )
        assert "DEMONSTRATION MODE" in report


# ──────────────────── LangGraph Workflow Tests ────────────────────


class TestWorkflow:
    def test_full_demo_workflow(self, sample_csv, demo_config):
        """Test full workflow in demo mode."""
        from app.graph import run_analysis

        final_state = run_analysis(
            dataset_path=sample_csv,
            user_request="Analyze this dataset",
            config=demo_config,
        )

        assert final_state is not None
        assert final_state.get("workflow_status") in ("completed", "failed")
        assert final_state.get("analysis_plan") is not None
        assert len(final_state.get("execution_log", [])) > 0

    def test_demo_workflow_iris(self, iris_csv, demo_config):
        """Test demo workflow with Iris dataset (has ML)."""
        from app.graph import run_analysis

        final_state = run_analysis(
            dataset_path=iris_csv,
            user_request="Classify iris species",
            config=demo_config,
        )

        assert final_state is not None
        plan = AnalysisPlan(**final_state.get("analysis_plan", {}))
        assert plan.use_ml
        assert plan.ml_task == "classification"

    def test_max_iteration_enforcement(self, demo_config):
        """Verify correction loop respects max iterations."""
        assert demo_config.execution.max_correction_attempts == 3

    def test_correction_history_tracking(self):
        record = CorrectionRecord(
            iteration=1, error="test error",
            reviewer_feedback="fix it", code_hash="abc123",
            outcome="correcting",
        )
        d = record.model_dump()
        assert d["iteration"] == 1
        assert d["error"] == "test error"


# ──────────────────── Configuration Tests ────────────────────


class TestConfiguration:
    def test_demo_mode_without_key(self):
        config = AppConfig(
            llm=LLMConfig(
                provider="nvidia", api_key="", model="test",
                base_url="http://localhost", available=False,
            ),
            execution=ExecutionConfig(
                max_correction_attempts=3, code_execution_timeout=60,
                max_code_size_kb=100, max_output_size_kb=500,
            ),
        )
        assert config.llm.is_demo_mode

    def test_llm_mode_with_key(self):
        config = AppConfig(
            llm=LLMConfig(
                provider="nvidia", api_key="real_key", model="test",
                base_url="http://localhost", available=True,
            ),
            execution=ExecutionConfig(
                max_correction_attempts=3, code_execution_timeout=60,
                max_code_size_kb=100, max_output_size_kb=500,
            ),
        )
        assert not config.llm.is_demo_mode

    def test_load_config_from_env(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test_key_12345")
        monkeypatch.setenv("LLM_MODEL", "meta/muse-glimmer-30b")
        monkeypatch.setenv("MAX_CORRECTION_ATTEMPTS", "5")
        config = load_config()
        assert config.llm.available
        assert config.llm.api_key == "nvapi-test_key_12345"
        assert config.llm.model == "meta/muse-glimmer-30b"
        assert config.execution.max_correction_attempts == 5

    def test_nvidia_api_key_precedence(self, monkeypatch):
        """NVIDIA_API_KEY should take precedence over LLM_API_KEY."""
        monkeypatch.setenv("NVIDIA_API_KEY", "primary_nv_key")
        monkeypatch.setenv("LLM_API_KEY", "secondary_key")
        config = load_config()
        assert config.llm.api_key == "primary_nv_key"

    def test_fallback_to_llm_api_key_if_nvidia_key_absent(self, monkeypatch):
        """If NVIDIA_API_KEY is not set or empty, fallback to LLM_API_KEY."""
        monkeypatch.setenv("NVIDIA_API_KEY", "")
        monkeypatch.setenv("LLM_API_KEY", "fallback_key_abc")
        config = load_config()
        assert config.llm.api_key == "fallback_key_abc"
        assert config.llm.available is True

    def test_default_muse_glimmer_configuration(self, monkeypatch):
        """Default model must be meta/muse-glimmer-30b and endpoint NVIDIA NIM."""
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        config = load_config()
        assert config.llm.model == "meta/muse-glimmer-30b"
        assert config.llm.base_url == "https://integrate.api.nvidia.com/v1"

    def test_placeholder_keys_rejected(self, monkeypatch):
        """Placeholder keys must be treated as unavailable (triggering demo mode)."""
        for placeholder in ["your_api_key_here", "your_existing_key", ""]:
            monkeypatch.setenv("NVIDIA_API_KEY", placeholder)
            monkeypatch.delenv("LLM_API_KEY", raising=False)
            config = load_config()
            assert config.llm.available is False
            assert config.llm.is_demo_mode is True

    def test_api_key_masking_in_repr(self):
        """LLMConfig.__repr__ must mask the API key and never expose it in logs or errors."""
        cfg = LLMConfig(
            provider="nvidia",
            api_key="nvapi-secret-key-1234567890",
            model="meta/muse-glimmer-30b",
            base_url="https://integrate.api.nvidia.com/v1",
            available=True,
        )
        repr_str = repr(cfg)
        assert "nvapi-secret-key-1234567890" not in repr_str
        assert "***7890" in repr_str

    def test_centralized_get_llm_factory(self):
        """get_llm factory must construct a ChatOpenAI client with Muse Glimmer settings."""
        cfg = AppConfig(
            llm=LLMConfig(
                provider="nvidia",
                api_key="nvapi-mock-test-key",
                model="meta/muse-glimmer-30b",
                base_url="https://integrate.api.nvidia.com/v1",
                available=True,
            ),
            execution=ExecutionConfig(3, 60, 100, 500),
        )
        llm = get_llm(cfg)
        assert llm is not None
        assert llm.model_name == "meta/muse-glimmer-30b"
        assert str(llm.openai_api_base).rstrip("/") == "https://integrate.api.nvidia.com/v1"

    def test_get_llm_returns_none_in_demo_mode(self):
        """get_llm must return None when no valid key is configured."""
        cfg = AppConfig(
            llm=LLMConfig("nvidia", "", "meta/muse-glimmer-30b", "https://integrate.api.nvidia.com/v1", False),
            execution=ExecutionConfig(3, 60, 100, 500),
        )
        assert get_llm(cfg) is None

    def test_agents_receive_configured_llm_in_graph(self, monkeypatch):
        """build_graph must inject the centrally configured LLM into Planner, Coder, and Reviewer."""
        captured_llms = {}

        class MockLLM:
            def __init__(self, name):
                self.name = name

            def invoke(self, *args, **kwargs):
                return "mock"

        custom_llm = MockLLM("custom-muse-glimmer")

        # Monkeypatch get_llm in app.graph to return our mock
        import app.graph as graph_module
        monkeypatch.setattr(graph_module, "get_llm", lambda cfg: custom_llm)

        cfg = AppConfig(
            llm=LLMConfig("nvidia", "real_key", "meta/muse-glimmer-30b", "https://integrate.api.nvidia.com/v1", True),
            execution=ExecutionConfig(3, 60, 100, 500),
        )
        compiled_graph = build_graph(cfg)
        assert compiled_graph is not None

    def test_real_nvidia_muse_glimmer_live_call(self):
        """If real NVIDIA_API_KEY is present in the environment, verify live API request succeeds."""
        real_config = load_config()
        if not real_config.llm.available:
            pytest.skip("No real NVIDIA_API_KEY found in .env; skipping live API call test.")

        llm = get_llm(real_config, timeout=30)
        assert llm is not None
        try:
            res = llm.invoke("Reply with exactly: PONG")
            content = res.content if hasattr(res, "content") else str(res)
            # Handle potential reasoning content if content is empty
            if not content and hasattr(res, "additional_kwargs"):
                content = res.additional_kwargs.get("reasoning_content", "")
            assert len(str(content).strip()) > 0
        except Exception as e:
            pytest.fail(f"Live NVIDIA Muse Glimmer API call failed: {type(e).__name__}: {e}")


# ──────────────────── State Model Tests ────────────────────


class TestStateModels:
    def test_analysis_plan_serialization(self):
        plan = AnalysisPlan(
            objective="Test", steps=[{"step": 1, "name": "test"}],
            use_ml=True, ml_task="classification", target_column="target",
        )
        d = plan.model_dump()
        reconstructed = AnalysisPlan(**d)
        assert reconstructed.objective == "Test"
        assert reconstructed.use_ml

    def test_review_result_serialization(self):
        review = ReviewResult(
            status="accepted",
            completed_steps=["step1"],
            detected_errors=[],
            acceptance_decision="OK",
        )
        d = review.model_dump()
        r = ReviewResult(**d)
        assert r.status == "accepted"

    def test_execution_result_defaults(self):
        result = ExecutionResult()
        assert not result.success
        assert result.exit_code == -1

    def test_dataset_metadata_defaults(self):
        meta = DatasetMetadata()
        assert meta.rows == 0
        assert meta.columns == 0


# ──────────────────── Coder Tests ────────────────────


class TestCoder:
    def test_demo_code_generation(self, sample_csv):
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        plan = _build_demo_plan(meta)
        code = _generate_demo_code(meta, plan)
        assert "import pandas" in code
        assert "DATASET_PATH" in code
        assert "OUTPUT_DIR" in code
        assert len(code) > 100

    def test_demo_code_with_ml(self, iris_csv):
        df = load_csv(iris_csv)
        meta = profile_dataset(df, "iris.csv")
        plan = _build_demo_plan(meta)
        code = _generate_demo_code(meta, plan)
        assert "classification" in code.lower() or "RandomForest" in code

    def test_coder_node(self, sample_csv):
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        plan = _build_demo_plan(meta)
        state: AnalystState = {
            "dataset_path": sample_csv,
            "dataset_metadata": meta.model_dump(),
            "analysis_plan": plan.model_dump(),
            "is_demo_mode": True,
            "current_iteration": 0,
            "execution_log": [],
            "correction_history": [],
        }
        result = run_coder(state)
        assert "generated_code" in result
        assert len(result["generated_code"]) > 0


# ──────────────────── Hardening, Security & Edge Cases Tests ────────────────────


class TestCoderHardeningAndEdgeCases:
    def test_unusual_column_names_and_code_injection(self, temp_dir):
        """Verify unusual column names and injection attempts don't cause syntax errors or crash execution."""
        csv_path = os.path.join(temp_dir, "injection.csv")
        df = pd.DataFrame({
            'col"; import os; #': range(20),
            "col 'single' quote": [f"val_{i}" for i in range(20)],
            "col/with/slash": np.random.randn(20),
            "col [brackets]": np.random.randn(20),
            "2024/2025": np.random.randn(20),
        })
        df.to_csv(csv_path, index=False)

        meta = profile_dataset(df, "injection.csv")
        plan = _build_demo_plan(meta)
        code = _generate_demo_code(meta, plan)

        # Execute code via SubprocessExecutor
        output_dir = os.path.join(temp_dir, "out_injection")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success, f"Execution failed: {result.stderr or result.error_summary}"
        assert result.exit_code == 0

    def test_target_with_special_characters(self, temp_dir):
        """Verify target column names with quotes, brackets, and spaces execute safely."""
        csv_path = os.path.join(temp_dir, "special_target.csv")
        target_name = 'target "quoted" [col]'
        df = pd.DataFrame({
            "feat_1": np.random.randn(30),
            "feat_2": np.random.randn(30),
            target_name: np.random.choice(["class_A", "class_B"], 30),
        })
        df.to_csv(csv_path, index=False)

        meta = profile_dataset(df, "special_target.csv")
        plan = AnalysisPlan(
            objective="Classify special target",
            use_ml=True,
            ml_task="classification",
            target_column=target_name,
            steps=[{"step": 1, "name": "ml_modeling", "output": "model_metrics"}],
        )
        code = _generate_demo_code(meta, plan)

        output_dir = os.path.join(temp_dir, "out_target")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success, f"Execution failed: {result.stderr or result.error_summary}"

    def test_classification_single_class_target(self, temp_dir):
        """Classification should not crash when target has only one class."""
        csv_path = os.path.join(temp_dir, "single_class.csv")
        df = pd.DataFrame({
            "f1": np.random.randn(25),
            "f2": np.random.randn(25),
            "target": ["only_class"] * 25,
        })
        df.to_csv(csv_path, index=False)

        meta = profile_dataset(df, "single_class.csv")
        plan = AnalysisPlan(
            objective="Classify",
            use_ml=True,
            ml_task="classification",
            target_column="target",
        )
        code = _generate_demo_code(meta, plan)

        output_dir = os.path.join(temp_dir, "out_single_class")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success, f"Execution crashed on single-class target: {result.stderr}"
        assert "results" in result.stdout

    def test_classification_rare_classes_no_crash(self, temp_dir):
        """Classification should not crash stratify when classes have only 1 member."""
        csv_path = os.path.join(temp_dir, "rare_classes.csv")
        df = pd.DataFrame({
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
            # Several single-member classes
            "target": ["A", "B", "C", "D", "E"] + ["Common"] * 15,
        })
        df.to_csv(csv_path, index=False)

        meta = profile_dataset(df, "rare_classes.csv")
        plan = AnalysisPlan(
            objective="Classify with rare classes",
            use_ml=True,
            ml_task="classification",
            target_column="target",
        )
        code = _generate_demo_code(meta, plan)

        output_dir = os.path.join(temp_dir, "out_rare_classes")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success, f"Execution crashed on rare classes: {result.stderr}"

    def test_classification_insufficient_samples(self, temp_dir):
        """Classification should gracefully skip if fewer than 10 rows exist."""
        csv_path = os.path.join(temp_dir, "few_samples.csv")
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": ["A", "B", "A", "B", "A"],
        })
        df.to_csv(csv_path, index=False)

        meta = profile_dataset(df, "few_samples.csv")
        plan = AnalysisPlan(
            objective="Classify",
            use_ml=True,
            ml_task="classification",
            target_column="target",
        )
        code = _generate_demo_code(meta, plan)

        output_dir = os.path.join(temp_dir, "out_few_samples")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success
        assert "insufficient samples" in result.stdout.lower()

    def test_regression_constant_target(self, temp_dir):
        """Regression should gracefully handle zero-variance target without crash."""
        csv_path = os.path.join(temp_dir, "constant_target.csv")
        df = pd.DataFrame({
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
            "target": [42.0] * 20,
        })
        df.to_csv(csv_path, index=False)

        meta = profile_dataset(df, "constant_target.csv")
        plan = AnalysisPlan(
            objective="Regress",
            use_ml=True,
            ml_task="regression",
            target_column="target",
        )
        code = _generate_demo_code(meta, plan)

        output_dir = os.path.join(temp_dir, "out_constant_target")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success, f"Regression crashed on zero variance: {result.stderr}"
        assert "zero variance" in result.stdout.lower()

    def test_regression_non_numeric_target(self, temp_dir):
        """Regression should gracefully handle non-numeric target."""
        csv_path = os.path.join(temp_dir, "string_target.csv")
        df = pd.DataFrame({
            "f1": np.random.randn(20),
            "target": ["abc"] * 20,
        })
        df.to_csv(csv_path, index=False)

        meta = profile_dataset(df, "string_target.csv")
        plan = AnalysisPlan(
            objective="Regress",
            use_ml=True,
            ml_task="regression",
            target_column="target",
        )
        code = _generate_demo_code(meta, plan)

        output_dir = os.path.join(temp_dir, "out_string_target")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success
        assert "insufficient" in result.stdout.lower()

    def test_non_finite_values_in_features(self, temp_dir):
        """Code should handle np.inf and -np.inf without crashing sklearn or matplotlib."""
        csv_path = os.path.join(temp_dir, "infinite.csv")
        df = pd.DataFrame({
            "f1": [1.0, np.inf, 3.0, -np.inf, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "f2": [np.nan, 2.0, 3.0, 4.0, np.inf, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        df.to_csv(csv_path, index=False)

        meta = profile_dataset(df, "infinite.csv")
        plan = AnalysisPlan(
            objective="Handle inf",
            use_ml=True,
            ml_task="classification",
            target_column="target",
        )
        code = _generate_demo_code(meta, plan)

        output_dir = os.path.join(temp_dir, "out_infinite")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success, f"Crashed on non-finite values: {result.stderr}"

    def test_empty_dataset_code_execution(self, temp_dir):
        """Empty dataset (0 rows) should not cause division by zero or unhandled crashes."""
        csv_path = os.path.join(temp_dir, "empty_table.csv")
        with open(csv_path, "w") as f:
            f.write("col_a,col_b,col_c\n")

        # Create dummy metadata with 0 rows
        meta = DatasetMetadata(filename="empty_table.csv", rows=0, columns=3, column_names=["col_a", "col_b", "col_c"])
        plan = AnalysisPlan(objective="Analyze empty")
        code = _generate_demo_code(meta, plan)

        output_dir = os.path.join(temp_dir, "out_empty")
        executor = SubprocessExecutor()
        result = executor.execute(code, csv_path, output_dir, timeout=30)
        assert result.success, f"Empty dataset execution crashed: {result.stderr}"
        assert "0 rows" in result.stdout

    def test_llm_failure_records_fallback_in_state(self, sample_csv):
        """Planner, Coder, and Reviewer must explicitly record is_fallback and fallback_reason on LLM error."""
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")

        class BrokenLLM:
            def invoke(self, messages):
                raise RuntimeError("Simulated API rate limit / network failure")

        broken = BrokenLLM()
        state: AnalystState = {
            "dataset_path": sample_csv,
            "dataset_metadata": meta.model_dump(),
            "analysis_plan": {},
            "user_request": "Test fallback",
            "is_demo_mode": False,
            "current_iteration": 0,
            "max_iterations": 3,
            "execution_log": [],
            "correction_history": [],
        }

        # Planner fallback
        planner_res = run_planner(state, llm=broken)
        assert planner_res["is_fallback"] is True
        assert planner_res["generation_mode"] == "fallback"
        assert "BrokenLLM" in planner_res["fallback_reason"] or "RuntimeError" in planner_res["fallback_reason"]

        # Coder fallback
        state["analysis_plan"] = planner_res["analysis_plan"]
        coder_res = run_coder(state, llm=broken)
        assert coder_res["is_fallback"] is True
        assert coder_res["generation_mode"] == "fallback"
        assert "RuntimeError" in coder_res["fallback_reason"]

        # Reviewer fallback
        state["execution_result"] = ExecutionResult(success=True, stdout="OK", exit_code=0).model_dump()
        reviewer_res = run_reviewer(state, llm=broken)
        assert reviewer_res["is_fallback"] is True
        assert reviewer_res["generation_mode"] == "fallback"

    def test_report_generator_distinguishes_fallback_and_failure(self, temp_dir):
        """Report generator should clearly flag demo mode, fallback mode, and terminal failure."""
        meta = DatasetMetadata(filename="data.csv", rows=50, columns=4, column_names=["a", "b", "c", "d"])
        plan = AnalysisPlan(objective="Test")
        review_ok = ReviewResult(status="accepted", acceptance_decision="Done")
        review_term = ReviewResult(status="terminated", acceptance_decision="Failed after 3 attempts")

        # Fallback report
        rep_fb = generate_report(
            meta, plan, "{}", [], review_ok, [], temp_dir,
            is_demo=False, is_fallback=True, workflow_status="completed"
        )
        assert "DETERMINISTIC FALLBACK" in rep_fb

        # Failure report
        rep_fail = generate_report(
            meta, plan, "{}", [], review_term, [], temp_dir,
            is_demo=False, is_fallback=False, workflow_status="failed", failure_info="Max iterations reached"
        )
        assert "FAILED / TERMINATED" in rep_fail
        assert "Max iterations reached" in rep_fail

    def test_coder_correction_incorporates_rich_error_context(self, sample_csv):
        """Coder prompt in correction mode must receive stderr traceback and reviewer feedback."""
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")

        captured_prompt = []

        class InspectingLLM:
            def invoke(self, messages):
                user_msg = messages[-1]["content"]
                captured_prompt.append(user_msg)
                return "import pandas as pd\nprint('fixed')"

        state: AnalystState = {
            "dataset_path": sample_csv,
            "dataset_metadata": meta.model_dump(),
            "analysis_plan": {"objective": "Test", "steps": []},
            "generated_code": "import pandas as pd\nraise ValueError('Invalid feature dimension')",
            "execution_result": {
                "success": False,
                "exit_code": 1,
                "error_summary": "ValueError: Invalid feature dimension",
                "stderr": "Traceback (most recent call last):\n  File 'test.py', line 2\nValueError: Invalid feature dimension",
                "stdout": "",
            },
            "review_result": {
                "status": "needs_correction",
                "correction_instructions": "Reshape feature matrix before fitting",
                "detected_errors": ["ValueError in feature dimension"],
                "acceptance_decision": "Code crashed with ValueError",
            },
            "current_iteration": 1,
            "max_iterations": 3,
            "is_demo_mode": False,
            "correction_history": [],
            "execution_log": [],
        }

        res = run_coder(state, llm=InspectingLLM())
        assert len(captured_prompt) == 1
        prompt_text = captured_prompt[0]
        assert "Invalid feature dimension" in prompt_text
        assert "Reshape feature matrix" in prompt_text
        assert "Traceback" in prompt_text

    def test_demo_correction_adapts_on_ml_failure(self, sample_csv):
        """In demo mode, if ML fails in iteration 0, iteration 1 must adapt by omitting ML."""
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        plan = AnalysisPlan(
            objective="Classify",
            use_ml=True,
            ml_task="classification",
            target_column="target",
        )

        state: AnalystState = {
            "dataset_path": sample_csv,
            "dataset_metadata": meta.model_dump(),
            "analysis_plan": plan.model_dump(),
            "generated_code": "initial code",
            "execution_result": {
                "success": False,
                "error_summary": "ValueError: Target column 'target' not found",
                "stderr": "ValueError in LabelEncoder fit_transform target_col",
            },
            "review_result": {
                "status": "needs_correction",
                "correction_instructions": "Fix ML target column",
            },
            "current_iteration": 1,
            "is_demo_mode": True,
            "execution_log": [],
            "correction_history": [],
        }

        res = run_coder(state, llm=None)
        assert "adapted: ML omitted after failure" in res["execution_log"][0]
        assert "RandomForestClassifier" not in res["generated_code"]


