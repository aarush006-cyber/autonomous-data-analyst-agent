"""Reviewer agent — evaluates execution results against the analysis plan."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.state import AnalysisPlan, AnalystState, ExecutionResult, ReviewResult
from app.prompts.reviewer_prompt import REVIEWER_SYSTEM_PROMPT, REVIEWER_USER_PROMPT

logger = logging.getLogger(__name__)


def _demo_review(exec_result: ExecutionResult, plan: AnalysisPlan, generated_files: list[str]) -> ReviewResult:
    """Deterministic review without LLM."""
    if not exec_result.success:
        return ReviewResult(
            status="needs_correction",
            completed_steps=[],
            missing_steps=[s.get("name", f"step_{s.get('step', '?')}") for s in plan.steps],
            detected_errors=[exec_result.error_summary or "Execution failed"],
            correction_instructions="Fix the execution error and retry.",
            acceptance_decision="Execution failed. Code correction required.",
        )

    # Check what was produced
    chart_files = [f for f in generated_files if f.endswith((".png", ".jpg", ".svg"))]
    has_charts = len(chart_files) > 0
    has_output = bool(exec_result.stdout.strip())

    completed = []
    missing = []
    errors = []

    for step in plan.steps:
        name = step.get("name", f"step_{step.get('step', '?')}")
        output_type = step.get("output", "")

        if "chart" in output_type or "plot" in output_type or "heatmap" in output_type or "histogram" in output_type or "boxplot" in output_type:
            if has_charts:
                completed.append(name)
            else:
                missing.append(name)
        elif "metric" in output_type or "model" in output_type:
            if plan.use_ml and any(kw in exec_result.stdout.lower() for kw in ["accuracy", "mae", "rmse", "r2", "f1"]):
                completed.append(name)
            elif plan.use_ml:
                missing.append(name)
            else:
                completed.append(name)
        else:
            if has_output:
                completed.append(name)
            else:
                missing.append(name)

    # Check for error indicators in output
    if exec_result.stderr and "error" in exec_result.stderr.lower():
        errors.append(f"Stderr contains errors: {exec_result.stderr[:200]}")

    if not has_output:
        errors.append("No stdout output produced")

    # Decision
    if not missing and not errors:
        status = "accepted"
        decision = "Analysis completed successfully. All planned steps produced output."
    elif len(missing) <= 1 and not errors:
        status = "accepted"
        decision = f"Analysis substantially complete. Minor missing: {', '.join(missing)}"
    else:
        status = "needs_correction"
        decision = f"Analysis incomplete. Missing: {', '.join(missing)}. Errors: {'; '.join(errors)}"

    return ReviewResult(
        status=status,
        completed_steps=completed,
        missing_steps=missing,
        detected_errors=errors,
        correction_instructions=f"Address: {'; '.join(missing + errors)}" if missing or errors else "",
        acceptance_decision=decision,
    )


def run_reviewer(state: AnalystState, llm=None) -> dict:
    """Reviewer node for the LangGraph workflow."""
    exec_dict = state.get("execution_result", {})
    exec_result = ExecutionResult(**exec_dict)
    plan_dict = state.get("analysis_plan", {})
    plan = AnalysisPlan(**plan_dict)
    is_demo = state.get("is_demo_mode", True)
    iteration = state.get("current_iteration", 0)
    max_iter = state.get("max_iterations", 3)
    generated_files = state.get("generated_artifact_paths", [])

    if is_demo or llm is None:
        logger.info("Reviewer: running in demonstration mode")
        review = _demo_review(exec_result, plan, generated_files)

        # Force termination if at max iterations
        if review.status == "needs_correction" and iteration >= max_iter:
            review.status = "terminated"
            review.acceptance_decision = f"Maximum correction attempts ({max_iter}) reached. Terminating."

        return {
            "review_result": review.model_dump(),
            "generation_mode": "demo",
            "execution_log": [f"[Reviewer] Demo review: {review.status} (iteration {iteration}/{max_iter})"],
        }

    # LLM-powered review
    try:
        user_prompt = REVIEWER_USER_PROMPT.format(
            analysis_plan=json.dumps(plan_dict, indent=2)[:2000],
            execution_success=exec_result.success,
            exit_code=exec_result.exit_code,
            duration=exec_result.duration_seconds,
            stdout=exec_result.stdout[:2000],
            stderr=exec_result.stderr[:1000],
            generated_files=generated_files,
            current_iteration=iteration,
            max_iterations=max_iter,
        )

        system = REVIEWER_SYSTEM_PROMPT.format(max_iterations=max_iter)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        elif not content and hasattr(response, "additional_kwargs"):
            content = response.additional_kwargs.get("reasoning_content", "")

        # Parse review JSON
        review_data = _parse_review_json(content)
        review = ReviewResult(**review_data)

        # Enforce max iterations
        if review.status == "needs_correction" and iteration >= max_iter:
            review.status = "terminated"
            review.acceptance_decision = f"Maximum correction attempts ({max_iter}) reached."

        return {
            "review_result": review.model_dump(),
            "generation_mode": "llm",
            "execution_log": [f"[Reviewer] LLM review: {review.status} (iteration {iteration}/{max_iter})"],
        }

    except Exception as e:
        logger.warning(f"Reviewer LLM call failed: {e}, falling back to demo review")
        review = _demo_review(exec_result, plan, generated_files)

        if review.status == "needs_correction" and iteration >= max_iter:
            review.status = "terminated"
            review.acceptance_decision = f"Maximum corrections ({max_iter}) reached."

        return {
            "review_result": review.model_dump(),
            "is_fallback": True,
            "fallback_reason": f"Reviewer LLM failed ({type(e).__name__}: {e})",
            "generation_mode": "fallback",
            "execution_log": [f"[Reviewer] LLM failed ({type(e).__name__}: {e}); explicitly fell back to deterministic demo review: {review.status}"],
        }


def _parse_review_json(text: str) -> dict:
    """Parse JSON from LLM review response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError("No valid JSON found in reviewer response")

