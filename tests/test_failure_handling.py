"""Comprehensive verification of failure handling, self-correction, and edge cases."""

import pytest
import os
import json
from unittest.mock import MagicMock

from app.config import AppConfig, LLMConfig, ExecutionConfig
from app.state import AnalystState, DatasetMetadata, AnalysisPlan, ExecutionResult
from app.agents.planner import run_planner
from app.agents.coder import run_coder
from app.agents.reviewer import run_reviewer
from app.tools.code_executor import SubprocessExecutor, validate_code_safety
from app.tools.dataset_tools import load_csv, profile_dataset, DatasetError
from app.tools.report_generator import generate_report
from app.graph import build_graph, run_analysis


@pytest.fixture
def sample_csv(tmp_path):
    path = str(tmp_path / "sample.csv")
    with open(path, "w") as f:
        f.write("a,b,target\n1,2,x\n3,4,y\n5,6,x\n")
    return path


class TestFailureHandlingScenarios:
    """Explicitly verify the 7 required failure scenarios."""

    # Scenario 1: Model returns malformed output
    def test_planner_malformed_llm_output(self, sample_csv):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Here is some invalid non-json text without braces")
        
        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        state = {
            "dataset_path": sample_csv,
            "dataset_metadata": meta.model_dump(),
            "user_request": "Analyze this",
            "is_demo_mode": False,
            "execution_log": [],
        }
        res = run_planner(state, llm=mock_llm)
        assert res.get("is_fallback") is True
        assert res.get("generation_mode") == "fallback"
        assert "Planner LLM failed" in res.get("fallback_reason", "")
        assert "steps" in res.get("analysis_plan", {})

    def test_reviewer_malformed_llm_output(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="I think the code looks ok but not json")
        
        state = {
            "execution_result": ExecutionResult(success=True, stdout="output").model_dump(),
            "analysis_plan": {"steps": [{"name": "s1", "output": "summary"}]},
            "generated_artifact_paths": [],
            "is_demo_mode": False,
            "current_iteration": 0,
            "max_iterations": 3,
            "execution_log": [],
        }
        res = run_reviewer(state, llm=mock_llm)
        assert res.get("is_fallback") is True
        assert res.get("generation_mode") == "fallback"
        assert res.get("review_result", {}).get("status") in ("accepted", "needs_correction")

    # Scenario 2: Model API fails (e.g. network/auth error)
    def test_api_failure_triggers_honest_fallback(self, sample_csv):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("503 Service Unavailable / Rate Limit Exceeded")

        df = load_csv(sample_csv)
        meta = profile_dataset(df, "sample.csv")
        state = {
            "dataset_path": sample_csv,
            "dataset_metadata": meta.model_dump(),
            "user_request": "Analyze this",
            "is_demo_mode": False,
            "execution_log": [],
        }
        res = run_planner(state, llm=mock_llm)
        assert res["is_fallback"] is True
        assert "503 Service Unavailable" in res["fallback_reason"]
        assert res["generation_mode"] == "fallback"

    # Scenario 3: Generated Python code contains a syntax error
    def test_syntax_error_execution_and_review(self, tmp_path, sample_csv):
        executor = SubprocessExecutor()
        bad_code = "import pandas as pd\nprint('Missing closing parenthesis\n"
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)
        res = executor.execute(bad_code, sample_csv, out_dir, timeout=10)
        assert res.success is False
        assert "SyntaxError" in res.stderr or "SyntaxError" in res.error_summary

    # Scenario 4: Analysis execution times out
    def test_execution_timeout(self, tmp_path, sample_csv):
        executor = SubprocessExecutor()
        hang_code = "import time\ntime.sleep(10)\n"
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)
        res = executor.execute(hang_code, sample_csv, out_dir, timeout=1)
        assert res.success is False
        assert "timed out" in res.error_summary.lower()
        assert res.exit_code == -1

    # Scenario 5: Requested chart cannot be generated
    def test_missing_chart_detected_by_reviewer(self):
        plan = AnalysisPlan(
            objective="Plot",
            steps=[{"name": "plot_step", "output": "histograms"}],
            suggested_visualizations=["histogram"]
        )
        # Execution succeeded, but NO chart files generated
        exec_res = ExecutionResult(success=True, stdout="Done without charts", exit_code=0)
        state = {
            "execution_result": exec_res.model_dump(),
            "analysis_plan": plan.model_dump(),
            "generated_artifact_paths": [],  # Empty!
            "is_demo_mode": True,
            "current_iteration": 0,
            "max_iterations": 3,
            "execution_log": [],
        }
        res = run_reviewer(state, llm=None)
        review = res["review_result"]
        assert "plot_step" in review["missing_steps"]

    # Scenario 6: Dataset is empty or malformed
    def test_empty_dataset_error(self, tmp_path):
        empty_csv = str(tmp_path / "empty.csv")
        with open(empty_csv, "w") as f:
            f.write("")
        with pytest.raises(DatasetError):
            load_csv(empty_csv)

    def test_malformed_dataset_error(self, tmp_path):
        bad_csv = str(tmp_path / "bad.csv")
        with open(bad_csv, "wb") as f:
            f.write(b"\x00\xff\xfe\x00corrupted binary data")
        # Attempting to load binary or non-csv raises error
        with pytest.raises((DatasetError, Exception)):
            load_csv(bad_csv)

    # Scenario 7: Correction limit is reached
    def test_correction_limit_enforced_in_graph(self, sample_csv, tmp_path):
        config = AppConfig(
            llm=LLMConfig(
                provider="nvidia",
                api_key="",
                model="test-model",
                base_url="http://localhost",
                available=False,
            ),
            execution=ExecutionConfig(
                max_correction_attempts=2,
                code_execution_timeout=5,
                max_code_size_kb=100,
                max_output_size_kb=500,
            ),
        )
        
        # Build state where iteration equals max_iterations and status needs correction
        state: AnalystState = {
            "dataset_path": sample_csv,
            "execution_result": ExecutionResult(success=False, error_summary="Repeated error").model_dump(),
            "analysis_plan": {"steps": [{"name": "step1"}]},
            "current_iteration": 2,
            "max_iterations": 2,
            "is_demo_mode": True,
            "execution_log": [],
        }
        res = run_reviewer(state, llm=None)
        assert res["review_result"]["status"] == "terminated"
        assert "Maximum" in res["review_result"]["acceptance_decision"]
