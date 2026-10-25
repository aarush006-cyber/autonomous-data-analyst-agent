"""Planner agent — inspects dataset metadata and creates an analysis plan."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.state import AnalysisPlan, AnalystState, DatasetMetadata
from app.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from app.tools.dataset_tools import get_safe_metadata_for_llm

logger = logging.getLogger(__name__)


def _build_demo_plan(metadata: DatasetMetadata) -> AnalysisPlan:
    """Build a deterministic analysis plan without an LLM."""
    steps = [
        {"step": 1, "name": "data_overview", "description": "Dataset shape, types, and preview", "output": "summary"},
        {"step": 2, "name": "data_quality", "description": "Missing values, duplicates, warnings", "output": "quality_report"},
        {"step": 3, "name": "descriptive_stats", "description": "Summary statistics for numeric columns", "output": "statistics"},
    ]

    visualizations = []
    numeric_cols = [c for c, t in metadata.dtypes.items()
                    if any(x in t for x in ("int", "float"))]
    cat_cols = [c for c, t in metadata.dtypes.items()
                if t == "object" or t == "category" or t == "bool"]

    if numeric_cols:
        steps.append({"step": 4, "name": "distributions", "description": "Distribution plots for numeric columns", "output": "histograms"})
        visualizations.append("histograms of numeric columns")

        if len(numeric_cols) >= 2:
            steps.append({"step": 5, "name": "correlation_analysis", "description": "Correlation matrix and heatmap", "output": "correlation_heatmap"})
            visualizations.append("correlation heatmap")

            steps.append({"step": 6, "name": "boxplots", "description": "Box plots for numeric columns", "output": "boxplots"})
            visualizations.append("box plots")

    if cat_cols:
        steps.append({"step": len(steps) + 1, "name": "categorical_analysis", "description": "Frequency analysis of categorical columns", "output": "bar_charts"})
        visualizations.append("bar charts for categorical columns")

    # Decide on ML — only for well-known tasks
    use_ml = False
    ml_task = ""
    target_col = ""
    ml_justification = ""

    col_names_lower = [c.lower() for c in metadata.column_names]
    if metadata.rows >= 50:
        # Classification candidates
        for candidate in ["target", "label", "class", "survived", "diagnosis", "species"]:
            if candidate in col_names_lower:
                idx = col_names_lower.index(candidate)
                actual_col = metadata.column_names[idx]
                n_unique = metadata.categorical_summary.get(actual_col, {}).get("unique", 0)
                if n_unique == 0:
                    # Might be numeric with few unique vals
                    n_unique = len(set()) # will be 0, handled below
                if n_unique and 2 <= n_unique <= 20:
                    use_ml = True
                    ml_task = "classification"
                    target_col = actual_col
                    ml_justification = f"Column '{actual_col}' appears to be a categorical target with {n_unique} classes."
                    break
                elif any(x in metadata.dtypes.get(actual_col, "") for x in ("int", "float")):
                    use_ml = True
                    ml_task = "classification"
                    target_col = actual_col
                    ml_justification = f"Column '{actual_col}' appears to be a classification target."
                    break

        # Regression candidates
        if not use_ml:
            for candidate in ["price", "medv", "medhouseval", "quality"]:
                if candidate in col_names_lower:
                    idx = col_names_lower.index(candidate)
                    actual_col = metadata.column_names[idx]
                    if any(x in metadata.dtypes.get(actual_col, "") for x in ("int", "float")):
                        use_ml = True
                        ml_task = "regression"
                        target_col = actual_col
                        ml_justification = f"Column '{actual_col}' appears suitable for regression prediction."
                        break

    if use_ml:
        step_n = len(steps) + 1
        steps.append({"step": step_n, "name": "ml_modeling", "description": f"{ml_task.title()} model for '{target_col}'", "output": "model_metrics"})
        if ml_task == "classification":
            visualizations.extend(["confusion matrix", "feature importance"])
        else:
            visualizations.extend(["actual vs predicted scatter", "residual plot"])

    expected_outputs = ["data_quality_report", "statistical_summary", "charts"]
    if use_ml:
        expected_outputs.append("ml_metrics")

    return AnalysisPlan(
        objective="Comprehensive exploratory data analysis" + (f" with {ml_task}" if use_ml else ""),
        steps=steps,
        suggested_visualizations=visualizations,
        use_ml=use_ml,
        ml_task=ml_task,
        target_column=target_col,
        ml_justification=ml_justification,
        expected_outputs=expected_outputs,
    )


def _parse_plan_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError("No valid JSON found in LLM response")


def run_planner(state: AnalystState, llm=None) -> dict:
    """Planner node for the LangGraph workflow."""
    metadata_dict = state.get("dataset_metadata", {})
    metadata = DatasetMetadata(**metadata_dict)
    is_demo = state.get("is_demo_mode", True)
    user_request = state.get("user_request", "Perform exploratory data analysis.")

    if is_demo or llm is None:
        logger.info("Planner: running in demonstration mode")
        plan = _build_demo_plan(metadata)
        return {
            "analysis_plan": plan.model_dump(),
            "generation_mode": "demo",
            "execution_log": [f"[Planner] Demo plan created with {len(plan.steps)} steps"],
        }

    # LLM-powered planning
    try:
        safe_meta = get_safe_metadata_for_llm(metadata)
        user_prompt = PLANNER_USER_PROMPT.format(
            filename=safe_meta["filename"],
            rows=safe_meta["rows"],
            columns=safe_meta["columns"],
            column_names=safe_meta["column_names"],
            dtypes=json.dumps(safe_meta["dtypes"], indent=2),
            missing_values=json.dumps(safe_meta["missing_values"], indent=2),
            duplicate_rows=safe_meta["duplicate_rows"],
            quality_warnings=safe_meta["quality_warnings"],
            numeric_summary=json.dumps(safe_meta["numeric_summary"], indent=2),
            categorical_summary=json.dumps(safe_meta["categorical_summary"], indent=2),
            sample_rows=json.dumps(safe_meta["sample_rows"], indent=2),
            user_request=user_request[:500],
        )

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        elif not content and hasattr(response, "additional_kwargs"):
            content = response.additional_kwargs.get("reasoning_content", "")
        plan_dict = _parse_plan_json(content)
        plan = AnalysisPlan(**plan_dict)

        return {
            "analysis_plan": plan.model_dump(),
            "generation_mode": "llm",
            "execution_log": [f"[Planner] LLM plan created with {len(plan.steps)} steps"],
        }

    except Exception as e:
        logger.warning(f"Planner LLM call failed: {e}, falling back to demo plan")
        plan = _build_demo_plan(metadata)
        return {
            "analysis_plan": plan.model_dump(),
            "is_fallback": True,
            "fallback_reason": f"Planner LLM failed ({type(e).__name__}: {e})",
            "generation_mode": "fallback",
            "execution_log": [f"[Planner] LLM failed ({type(e).__name__}: {e}); explicitly fell back to deterministic demo plan with {len(plan.steps)} steps"],
        }

