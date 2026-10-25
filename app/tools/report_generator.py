"""Report generator — produces Markdown analysis reports."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from app.state import AnalysisPlan, DatasetMetadata, ReviewResult

logger = logging.getLogger(__name__)


def generate_report(
    metadata: DatasetMetadata,
    plan: AnalysisPlan,
    execution_stdout: str,
    generated_files: list[str],
    review: ReviewResult,
    correction_history: list[dict],
    output_dir: str,
    is_demo: bool = False,
    is_fallback: bool = False,
    workflow_status: str = "completed",
    failure_info: str = "",
) -> str:
    """Generate a Markdown analysis report.
    
    Returns the report content as a string and saves it to output_dir.
    Explicitly distinguishes between successful LLM runs, demo runs,
    fallback runs, and failed/terminated analyses.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Try to parse results from stdout
    results_data = _extract_results(execution_stdout)

    # Build report sections
    sections = []

    # Header
    is_failure = workflow_status == "failed" or review.status == "terminated"
    status_suffix = " [TERMINATED / FAILED]" if is_failure else ""
    sections.append(f"# Data Analysis Report — {metadata.filename}{status_suffix}\n")
    sections.append(f"*Generated: {timestamp}*\n")

    # Mode banners
    if is_failure:
        reason = failure_info or review.acceptance_decision or "Analysis terminated before completion."
        sections.append(
            f"> ❌ **ANALYSIS STATUS: FAILED / TERMINATED**\n"
            f">\n"
            f"> The analysis could not be completed successfully.\n"
            f"> **Reason**: {reason}\n"
        )
    elif is_fallback:
        sections.append(
            "> ⚠️ **EXECUTION MODE: DETERMINISTIC FALLBACK (LLM FAILED/UNAVAILABLE)**\n"
            ">\n"
            "> The LLM call failed or was unavailable during execution. This analysis was generated via deterministic fallback routines and must not be represented as an LLM-powered output.\n"
        )
    elif is_demo:
        sections.append(
            "> ℹ️ **[DEMONSTRATION MODE]** — Results generated using deterministic routines without an LLM.\n"
        )

    # Dataset Overview
    sections.append("## 1. Dataset Overview\n")
    sections.append(f"- **File**: {metadata.filename}")
    sections.append(f"- **Rows**: {metadata.rows:,}")
    sections.append(f"- **Columns**: {metadata.columns}")
    sections.append(f"- **Column Names**: {', '.join(metadata.column_names[:30])}")
    if len(metadata.column_names) > 30:
        sections.append(f"  *(and {len(metadata.column_names) - 30} more)*")
    sections.append("")

    # Analysis Objective
    sections.append("## 2. Analysis Objective\n")
    sections.append(plan.objective or "General exploratory data analysis.")
    sections.append("")

    # Analysis Plan
    sections.append("## 3. Analysis Plan\n")
    for step in plan.steps:
        name = step.get("name", step.get("step", ""))
        desc = step.get("description", "")
        sections.append(f"- **{name}**: {desc}")
    sections.append("")

    # Data Quality
    sections.append("## 4. Data Quality Findings\n")
    if metadata.quality_warnings:
        for w in metadata.quality_warnings:
            sections.append(f"- ⚠️ {w}")
    else:
        sections.append("- ✅ No significant data quality issues detected.")
    sections.append("")

    missing_any = {k: v for k, v in metadata.missing_values.items() if v > 0}
    if missing_any:
        sections.append("### Missing Values\n")
        sections.append("| Column | Missing | % |")
        sections.append("|--------|---------|---|")
        for col, count in sorted(missing_any.items(), key=lambda x: -x[1]):
            pct = metadata.missing_pct.get(col, 0)
            sections.append(f"| {col} | {count} | {pct:.1f}% |")
        sections.append("")

    if metadata.duplicate_rows > 0:
        sections.append(f"- **Duplicate rows**: {metadata.duplicate_rows}")
        sections.append("")

    # Statistical Observations
    sections.append("## 5. Statistical Observations\n")
    if results_data and "results" in results_data:
        r = results_data["results"]
        for key, val in r.items():
            if isinstance(val, dict):
                sections.append(f"### {key.replace('_', ' ').title()}\n")
                for k2, v2 in val.items():
                    sections.append(f"- **{k2}**: {v2}")
            elif isinstance(val, list):
                sections.append(f"### {key.replace('_', ' ').title()}\n")
                for item in val[:20]:
                    sections.append(f"- {item}")
            else:
                sections.append(f"- **{key.replace('_', ' ').title()}**: {val}")
        sections.append("")
    elif metadata.numeric_summary:
        sections.append("### Descriptive Statistics\n")
        sections.append("| Column | Mean | Std | Min | Max |")
        sections.append("|--------|------|-----|-----|-----|")
        for col, stats in list(metadata.numeric_summary.items())[:20]:
            mean = stats.get("mean", "N/A")
            std = stats.get("std", "N/A")
            mn = stats.get("min", "N/A")
            mx = stats.get("max", "N/A")
            sections.append(f"| {col} | {mean} | {std} | {mn} | {mx} |")
        sections.append("")

    # Visualizations
    chart_files = [f for f in generated_files if f.lower().endswith((".png", ".jpg", ".svg"))]
    if chart_files:
        sections.append("## 6. Visualizations\n")
        for chart in chart_files:
            name = os.path.splitext(chart)[0].replace("_", " ").title()
            sections.append(f"### {name}\n")
            sections.append(f"![{name}]({chart})\n")
        sections.append("")

    # ML Results
    if plan.use_ml and results_data and "results" in results_data:
        ml_keys = [k for k in results_data["results"] if any(
            term in k.lower() for term in ["accuracy", "precision", "recall", "f1", "rmse", "mae", "r2", "model", "classification", "regression"]
        )]
        if ml_keys:
            sections.append("## 7. Machine Learning Results\n")
            for k in ml_keys:
                sections.append(f"- **{k}**: {results_data['results'][k]}")
            sections.append("")

    # Limitations
    sections.append("## 8. Limitations\n")
    sections.append("- This analysis is automated and may miss domain-specific nuances.")
    if is_failure:
        sections.append("- ❌ Analysis failed or was terminated before all steps were verified.")
    if is_fallback:
        sections.append("- ⚠️ Analysis executed in deterministic fallback mode due to an LLM service failure.")
    elif is_demo:
        sections.append("- Results were generated in demonstration mode without an LLM.")
        sections.append("- Analysis steps are deterministic and may not be optimally tailored.")
    sections.append("- Correlation does not imply causation.")
    if plan.use_ml:
        sections.append("- ML metrics are computed on a held-out test set but may not generalize to new data.")
    sections.append("")

    # Corrections
    if correction_history:
        sections.append("## 9. Correction History\n")
        for rec in correction_history:
            sections.append(f"- **Attempt {rec.get('iteration', '?')}**: {rec.get('outcome', 'unknown')} — {rec.get('error', 'N/A')[:200]}")
        sections.append("")

    # Conclusion
    sections.append("## 10. Conclusion\n")
    sections.append(f"- **Review Status**: {review.status}")
    sections.append(f"- **Decision**: {review.acceptance_decision}")
    if review.completed_steps:
        sections.append(f"- **Completed**: {', '.join(review.completed_steps)}")
    if review.missing_steps:
        sections.append(f"- **Missing**: {', '.join(review.missing_steps)}")
    sections.append("")

    report_content = "\n".join(sections)

    # Save report
    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Report saved to {report_path}")
    return report_content


def _extract_results(stdout: str) -> dict | None:
    """Try to extract JSON results from execution stdout."""
    if not stdout:
        return None

    # Look for the last JSON object in stdout
    lines = stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Try to find JSON block
    try:
        start = stdout.rfind("{")
        end = stdout.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(stdout[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    return None

