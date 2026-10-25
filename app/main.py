"""Streamlit UI for the Autonomous Data Analyst Agent."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import load_config
from app.tools.dataset_tools import load_csv, profile_dataset, DatasetError
from app.tools.code_executor import get_executor
from app.graph import run_analysis
from app.state import DatasetMetadata, AnalysisPlan

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    st.set_page_config(
        page_title="Autonomous Data Analyst",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Autonomous Data Analyst Agent")
    st.markdown("Upload a CSV file and let the AI analyze it automatically.")

    # Load config
    config = load_config()

    # --- Sidebar: Status & Config ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Mode indicator
        if config.llm.is_demo_mode:
            st.warning("🔸 **DEMONSTRATION MODE**\nNo LLM API key configured. Using deterministic analysis.")
        else:
            st.success(f"✅ **LLM Mode**\nProvider: {config.llm.provider}\nModel: {config.llm.model}")

        # Execution backend
        executor = get_executor(is_demo=config.llm.is_demo_mode, backend=config.execution.backend)
        st.info(f"🖥️ **Execution Backend**\n{executor.backend_name}")
        if not executor.is_sandboxed:
            st.caption("⚠️ Host subprocess execution. Docker isolation inactive.")
        else:
            st.caption("🛡️ Isolated Docker container sandbox.")

        st.divider()
        st.markdown("**Correction Loop**")
        st.markdown(f"- Max attempts: {config.execution.max_correction_attempts}")
        st.markdown(f"- Timeout: {config.execution.code_execution_timeout}s")

        st.divider()
        st.caption("Phase 1 — Docker isolation deferred to Phase 2.")

    # --- Main content ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Maximum file size: 50 MB",
        )

    with col2:
        st.subheader("🎯 Analysis Objective (Optional)")
        user_request = st.text_area(
            "Describe what you want to analyze",
            placeholder="e.g., 'Analyze survival factors in the Titanic dataset' or leave blank for general EDA",
            height=100,
        )

    if uploaded_file is not None:
        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp(prefix="analyst_data_")
        dataset_path = os.path.join(temp_dir, uploaded_file.name)
        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and preview
        try:
            df = load_csv(dataset_path)
            metadata = profile_dataset(df, filename=uploaded_file.name)

            # Dataset preview
            st.subheader("📋 Dataset Preview")
            preview_col1, preview_col2 = st.columns([2, 1])

            with preview_col1:
                st.dataframe(df.head(10), use_container_width=True)

            with preview_col2:
                st.metric("Rows", f"{metadata.rows:,}")
                st.metric("Columns", metadata.columns)
                st.metric("Duplicate Rows", metadata.duplicate_rows)

                missing_total = sum(metadata.missing_values.values())
                st.metric("Missing Values", f"{missing_total:,}")

            # Data quality
            if metadata.quality_warnings:
                with st.expander("⚠️ Data Quality Warnings", expanded=True):
                    for w in metadata.quality_warnings:
                        st.warning(w)

            # Schema
            with st.expander("📐 Schema & Data Types"):
                schema_df = pd.DataFrame({
                    "Column": metadata.column_names,
                    "Type": [metadata.dtypes.get(c, "unknown") for c in metadata.column_names],
                    "Missing": [metadata.missing_values.get(c, 0) for c in metadata.column_names],
                    "Missing %": [f"{metadata.missing_pct.get(c, 0):.1f}%" for c in metadata.column_names],
                })
                st.dataframe(schema_df, use_container_width=True, hide_index=True)

            # Analysis button
            st.divider()
            run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

            if run_button:
                _run_analysis_workflow(dataset_path, user_request, config, metadata)

        except DatasetError as e:
            st.error(f"❌ Dataset Error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {type(e).__name__}: {e}")
            logger.exception("Error loading dataset")


def _run_analysis_workflow(dataset_path: str, user_request: str, config, metadata):
    """Execute the analysis workflow with progress display."""
    st.subheader("🔬 Analysis in Progress")

    progress_container = st.container()
    status_placeholder = st.empty()
    log_placeholder = st.empty()

    logs = []

    def progress_callback(node: str, message: str):
        logs.append(message)
        with log_placeholder.container():
            for log in logs[-10:]:
                st.text(log)

    with st.spinner("Running analysis workflow..."):
        try:
            final_state = run_analysis(
                dataset_path=dataset_path,
                user_request=user_request or "Perform exploratory data analysis.",
                config=config,
                progress_callback=progress_callback,
            )
        except Exception as e:
            st.error(f"❌ Workflow Error: {type(e).__name__}: {e}")
            logger.exception("Workflow failed")
            return

    # Display results
    workflow_status = final_state.get("workflow_status", "unknown")
    is_fallback = final_state.get("is_fallback", False)

    if is_fallback:
        st.warning(
            f"⚠️ **Deterministic Fallback Active**: {final_state.get('fallback_reason', 'LLM call failed')}. "
            "Results were generated using deterministic routines, not an LLM."
        )

    if workflow_status == "completed":
        status_placeholder.success("✅ Analysis completed successfully!")
    elif workflow_status == "failed":
        status_placeholder.error(f"❌ Analysis failed: {final_state.get('failure_info', 'Unknown')}")
    else:
        status_placeholder.warning(f"⚠️ Workflow ended with status: {workflow_status}")

    # Show execution log
    all_logs = final_state.get("execution_log", [])
    if all_logs:
        with st.expander("📜 Execution Log", expanded=False):
            for log in all_logs:
                st.text(log)

    # Show analysis plan
    plan_dict = final_state.get("analysis_plan", {})
    if plan_dict:
        with st.expander("📋 Analysis Plan", expanded=False):
            plan = AnalysisPlan(**plan_dict)
            st.markdown(f"**Objective:** {plan.objective}")
            for step in plan.steps:
                st.markdown(f"- **{step.get('name', '?')}**: {step.get('description', '')}")
            if plan.use_ml:
                st.markdown(f"**ML Task:** {plan.ml_task} on `{plan.target_column}`")
                st.markdown(f"**Justification:** {plan.ml_justification}")

    # Correction history
    corrections = final_state.get("correction_history", [])
    if corrections:
        with st.expander(f"🔄 Correction History ({len(corrections)} attempts)", expanded=False):
            for rec in corrections:
                st.markdown(f"**Attempt {rec.get('iteration', '?')}**: {rec.get('outcome', 'unknown')}")
                if rec.get("error"):
                    st.code(rec["error"][:500], language="text")

    # Charts
    artifact_paths = final_state.get("generated_artifact_paths", [])
    chart_files = [p for p in artifact_paths if p.lower().endswith((".png", ".jpg", ".svg"))]

    if chart_files:
        st.subheader("📈 Generated Charts")
        # Display in a grid
        cols = st.columns(min(len(chart_files), 2))
        for i, chart_path in enumerate(chart_files):
            if os.path.isfile(chart_path):
                with cols[i % len(cols)]:
                    st.image(chart_path, caption=os.path.basename(chart_path), use_container_width=True)

    # Report
    report = final_state.get("final_report", "")
    report_path = final_state.get("report_path", "")

    if report:
        st.subheader("📄 Analysis Report")
        with st.expander("View Full Report", expanded=True):
            st.markdown(report)

        # Download buttons
        st.subheader("📥 Downloads")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                "📄 Download Report (Markdown)",
                data=report,
                file_name="analysis_report.md",
                mime="text/markdown",
            )

        with dl_col2:
            if chart_files:
                import zipfile
                import io

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for path in chart_files:
                        if os.path.isfile(path):
                            zf.write(path, os.path.basename(path))
                    if report_path and os.path.isfile(report_path):
                        zf.write(report_path, "analysis_report.md")

                st.download_button(
                    "📦 Download All Artifacts (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="analysis_artifacts.zip",
                    mime="application/zip",
                )


if __name__ == "__main__":
    main()

