"""LangGraph workflow for the Autonomous Data Analyst.

Defines the state graph with nodes for:
  inspect_dataset → planner → coder → execute_analysis → reviewer → conditional_router
"""

from __future__ import annotations

import logging
import os
import json
from typing import Any, Literal

from langgraph.graph import StateGraph, END

from app.state import (
    AnalystState, DatasetMetadata, AnalysisPlan,
    ExecutionResult, ReviewResult, CorrectionRecord,
)
from app.agents.planner import run_planner
from app.agents.coder import run_coder
from app.agents.reviewer import run_reviewer
from app.tools.dataset_tools import load_csv, profile_dataset
from app.tools.code_executor import get_executor, code_hash
from app.tools.report_generator import generate_report
from app.config import AppConfig, get_llm

logger = logging.getLogger(__name__)


def build_graph(config: AppConfig) -> StateGraph:
    """Build and compile the LangGraph state graph.
    
    Centrally initializes the NVIDIA Muse Glimmer LLM instance and
    injects it into the Planner, Coder, and Reviewer nodes.
    """
    llm = get_llm(config)
    is_demo = config.llm.is_demo_mode

    # --- Node functions ---

    def inspect_dataset(state: AnalystState) -> dict:
        """Load and profile the dataset."""
        dataset_path = state["dataset_path"]
        logger.info(f"Inspecting dataset: {dataset_path}")

        try:
            df = load_csv(dataset_path)
            metadata = profile_dataset(df, filename=os.path.basename(dataset_path))
            return {
                "dataset_metadata": metadata.model_dump(),
                "is_demo_mode": is_demo,
                "is_fallback": False,
                "fallback_reason": "",
                "generation_mode": "demo" if is_demo else "llm",
                "current_iteration": 0,
                "max_iterations": config.execution.max_correction_attempts,
                "workflow_status": "running",
                "execution_log": [f"[Dataset] Loaded {metadata.rows}×{metadata.columns} dataset"],
            }
        except Exception as e:
            return {
                "workflow_status": "failed",
                "failure_info": f"Dataset loading failed: {e}",
                "execution_log": [f"[Dataset] ERROR: {e}"],
            }

    def planner_node(state: AnalystState) -> dict:
        if state.get("workflow_status") == "failed":
            return {}
        return run_planner(state, llm=llm)

    def coder_node(state: AnalystState) -> dict:
        if state.get("workflow_status") == "failed":
            return {}
        return run_coder(state, llm=llm)

    def execute_analysis(state: AnalystState) -> dict:
        """Execute the generated code through the execution service."""
        if state.get("workflow_status") == "failed":
            return {}

        code = state.get("generated_code", "")
        dataset_path = state["dataset_path"]

        if not code:
            return {
                "execution_result": ExecutionResult(
                    success=False, error_summary="No code generated"
                ).model_dump(),
                "execution_status": "failed",
                "execution_log": ["[Executor] ERROR: No code to execute"],
            }

        # Create output directory
        output_dir = state.get("output_dir") or os.path.join(os.path.dirname(dataset_path), "..", "artifacts")
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        executor = get_executor(is_demo=is_demo, backend=config.execution.backend)
        logger.info(f"Executing with backend: {executor.backend_name}")

        result = executor.execute(
            code=code,
            dataset_path=dataset_path,
            output_dir=output_dir,
            timeout=config.execution.code_execution_timeout,
        )

        # Collect artifact paths
        artifact_paths = [
            os.path.join(output_dir, f) for f in result.generated_files
        ]

        status = "success" if result.success else "failed"

        return {
            "execution_result": result.model_dump(),
            "execution_status": status,
            "generated_artifact_paths": artifact_paths,
            "execution_log": [
                f"[Executor] {executor.backend_name}: exit={result.exit_code}, "
                f"duration={result.duration_seconds}s, files={len(result.generated_files)}"
            ],
        }

    def reviewer_node(state: AnalystState) -> dict:
        if state.get("workflow_status") == "failed":
            return {}
        return run_reviewer(state, llm=llm)

    def conditional_router(state: AnalystState) -> Literal["coder", "report_generator", "failure_handler"]:
        """Route based on review result."""
        if state.get("workflow_status") == "failed":
            return "failure_handler"

        review_dict = state.get("review_result", {})
        status = review_dict.get("status", "needs_correction")

        if status == "accepted":
            return "report_generator"
        elif status == "terminated":
            return "failure_handler"
        else:  # needs_correction
            iteration = state.get("current_iteration", 0)
            max_iter = state.get("max_iterations", 3)
            if iteration >= max_iter:
                return "failure_handler"
            return "coder"

    def pre_correction(state: AnalystState) -> dict:
        """Update state before a correction attempt."""
        iteration = state.get("current_iteration", 0)
        exec_dict = state.get("execution_result", {})
        review_dict = state.get("review_result", {})
        code = state.get("generated_code", "")

        record = CorrectionRecord(
            iteration=iteration,
            error=exec_dict.get("error_summary", ""),
            reviewer_feedback=review_dict.get("correction_instructions", ""),
            code_hash=code_hash(code) if code else "",
            outcome="correcting",
        ).model_dump()

        return {
            "current_iteration": iteration + 1,
            "correction_history": [record],
            "execution_log": [f"[Workflow] Correction attempt {iteration + 1}"],
        }

    def report_generator_node(state: AnalystState) -> dict:
        """Generate the final analysis report."""
        metadata = DatasetMetadata(**state.get("dataset_metadata", {}))
        plan = AnalysisPlan(**state.get("analysis_plan", {}))
        exec_result = ExecutionResult(**state.get("execution_result", {}))
        review = ReviewResult(**state.get("review_result", {}))
        correction_history = state.get("correction_history", [])

        # Use configured or parent artifacts dir
        dataset_path = state["dataset_path"]
        output_dir = state.get("output_dir") or os.path.join(os.path.dirname(dataset_path), "..", "artifacts")
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        artifact_paths = state.get("generated_artifact_paths", [])
        file_names = [os.path.basename(p) for p in artifact_paths]

        report = generate_report(
            metadata=metadata,
            plan=plan,
            execution_stdout=exec_result.stdout,
            generated_files=file_names,
            review=review,
            correction_history=correction_history,
            output_dir=output_dir,
            is_demo=is_demo,
            is_fallback=state.get("is_fallback", False),
            workflow_status="completed",
        )

        report_path = os.path.join(output_dir, "analysis_report.md")

        return {
            "final_report": report,
            "report_path": report_path,
            "workflow_status": "completed",
            "execution_log": [f"[Report] Generated ({len(report)} chars) at {report_path}"],
        }

    def failure_handler(state: AnalystState) -> dict:
        """Handle terminal failures."""
        review_dict = state.get("review_result", {})
        exec_dict = state.get("execution_result", {})
        iteration = state.get("current_iteration", 0)

        failure_info = (
            f"Analysis failed after {iteration} attempts. "
            f"Last review: {review_dict.get('acceptance_decision', 'N/A')}. "
            f"Last error: {exec_dict.get('error_summary', 'N/A')}"
        )

        # Still try to generate a partial report
        try:
            metadata = DatasetMetadata(**state.get("dataset_metadata", {}))
            plan = AnalysisPlan(**state.get("analysis_plan", {}))
            exec_result = ExecutionResult(**state.get("execution_result", {}))
            review = ReviewResult(
                status="terminated",
                acceptance_decision=failure_info,
            )

            dataset_path = state["dataset_path"]
            output_dir = state.get("output_dir") or os.path.join(os.path.dirname(dataset_path), "..", "artifacts")
            output_dir = os.path.realpath(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            report = generate_report(
                metadata=metadata, plan=plan,
                execution_stdout=exec_result.stdout,
                generated_files=[],
                review=review,
                correction_history=state.get("correction_history", []),
                output_dir=output_dir,
                is_demo=is_demo,
                is_fallback=state.get("is_fallback", False),
                workflow_status="failed",
                failure_info=failure_info,
            )
            report_path = os.path.join(output_dir, "analysis_report.md")
        except Exception:
            report = ""
            report_path = ""

        return {
            "workflow_status": "failed",
            "failure_info": failure_info,
            "final_report": report,
            "report_path": report_path,
            "execution_log": [f"[Failure] {failure_info}"],
        }

    # --- Build the graph ---
    graph = StateGraph(AnalystState)

    graph.add_node("inspect_dataset", inspect_dataset)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("execute_analysis", execute_analysis)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("pre_correction", pre_correction)
    graph.add_node("report_generator", report_generator_node)
    graph.add_node("failure_handler", failure_handler)

    # Edges
    graph.set_entry_point("inspect_dataset")
    graph.add_edge("inspect_dataset", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "execute_analysis")
    graph.add_edge("execute_analysis", "reviewer")

    # Conditional routing after reviewer
    graph.add_conditional_edges(
        "reviewer",
        conditional_router,
        {
            "coder": "pre_correction",
            "report_generator": "report_generator",
            "failure_handler": "failure_handler",
        },
    )

    graph.add_edge("pre_correction", "coder")
    graph.add_edge("report_generator", END)
    graph.add_edge("failure_handler", END)

    return graph.compile()


def run_analysis(
    dataset_path: str,
    user_request: str,
    config: AppConfig,
    progress_callback=None,
    output_dir: str | None = None,
) -> dict:
    """Run the full analysis workflow. Returns the final state."""
    graph = build_graph(config)

    initial_state: AnalystState = {
        "dataset_path": dataset_path,
        "user_request": user_request or "Perform exploratory data analysis.",
        "current_iteration": 0,
        "max_iterations": config.execution.max_correction_attempts,
        "correction_history": [],
        "execution_log": [],
        "generated_artifact_paths": [],
        "workflow_status": "running",
        "is_demo_mode": config.llm.is_demo_mode,
    }
    if output_dir:
        initial_state["output_dir"] = os.path.realpath(output_dir)

    logger.info(f"Starting analysis workflow (demo={config.llm.is_demo_mode})")

    # Stream through the graph to capture state progression and deliver progress updates
    final_state = dict(initial_state)
    seen_logs = 0
    for state_snapshot in graph.stream(initial_state, stream_mode="values"):
        final_state = state_snapshot
        if progress_callback:
            logs = state_snapshot.get("execution_log", [])
            for entry in logs[seen_logs:]:
                progress_callback("workflow", entry)
            seen_logs = len(logs)

    return final_state

