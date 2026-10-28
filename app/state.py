"""LangGraph state definition for the Autonomous Data Analyst workflow."""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class AnalysisPlan(BaseModel):
    """Structured output from the Planner agent."""
    objective: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)
    suggested_visualizations: list[str] = Field(default_factory=list)
    use_ml: bool = False
    ml_task: str = ""  # "classification", "regression", or ""
    target_column: str = ""
    ml_justification: str = ""
    expected_outputs: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Structured output from the Reviewer agent."""
    status: Literal["accepted", "needs_correction", "terminated"] = "needs_correction"
    completed_steps: list[str] = Field(default_factory=list)
    missing_steps: list[str] = Field(default_factory=list)
    detected_errors: list[str] = Field(default_factory=list)
    correction_instructions: str = ""
    acceptance_decision: str = ""


class ExecutionResult(BaseModel):
    """Result of code execution."""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_seconds: float = 0.0
    generated_files: list[str] = Field(default_factory=list)
    error_summary: str = ""


class CorrectionRecord(BaseModel):
    """Record of one correction attempt."""
    iteration: int = 0
    error: str = ""
    reviewer_feedback: str = ""
    code_hash: str = ""
    outcome: str = ""


class DatasetMetadata(BaseModel):
    """Profiling information about the uploaded dataset."""
    filename: str = ""
    rows: int = 0
    columns: int = 0
    column_names: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    missing_values: dict[str, int] = Field(default_factory=dict)
    missing_pct: dict[str, float] = Field(default_factory=dict)
    duplicate_rows: int = 0
    numeric_summary: dict[str, Any] = Field(default_factory=dict)
    categorical_summary: dict[str, Any] = Field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    file_size_bytes: int = 0
    quality_warnings: list[str] = Field(default_factory=list)


from typing import TypedDict, Annotated
import operator


class AnalystState(TypedDict, total=False):
    """Top-level LangGraph state for the autonomous data analyst."""
    # Dataset
    dataset_path: str
    dataset_metadata: dict  # Serialized DatasetMetadata
    user_request: str

    # Plan
    analysis_plan: dict  # Serialized AnalysisPlan

    # Code & Execution
    generated_code: str
    execution_result: dict  # Serialized ExecutionResult
    execution_status: str  # "pending", "success", "failed", "timeout"

    # Review
    review_result: dict  # Serialized ReviewResult

    # Correction loop
    correction_history: Annotated[list[dict], operator.add]  # append-only
    current_iteration: int
    max_iterations: int

    # Artifacts
    output_dir: str
    generated_artifact_paths: list[str]
    final_report: str
    report_path: str

    # Failure
    failure_info: str

    # Workflow control
    workflow_status: str  # "running", "completed", "failed"
    is_demo_mode: bool
    is_fallback: bool  # True if an LLM failure triggered a fallback to deterministic logic
    fallback_reason: str  # Description of LLM failure causing fallback
    generation_mode: str  # "llm", "demo", or "fallback"
    # Per-stage observability; stage modes are "llm", "demo", "fallback", or "unknown".
    planner_generation_mode: str
    coder_generation_mode: str
    reviewer_generation_mode: str
    planner_fallback_reason: str
    coder_fallback_reason: str
    reviewer_fallback_reason: str
    fallback_used: bool
    execution_success: bool
    live_review_success: bool
    execution_log: Annotated[list[str], operator.add]  # append-only event log

