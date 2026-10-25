"""Five-dataset evaluation script for the Autonomous Data Analyst.

Evaluates the agent on:
1. Iris
2. Titanic
3. Wine Quality
4. Breast Cancer Wisconsin
5. California Housing

Executes the complete LangGraph workflow, capturing:
- Data validation & profiling
- Structured analysis planning
- Subprocess/Docker code execution
- Artifact generation (charts and reports)
- Machine learning metrics (accuracy, F1, RMSE, R²)
- Reviewer acceptance and self-correction iterations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import load_config, AppConfig, LLMConfig
from app.graph import run_analysis
from app.state import AnalysisPlan

logger = logging.getLogger(__name__)

DATASETS = [
    {
        "name": "Iris",
        "source": "sklearn.datasets.load_iris",
        "license": "Public domain (Fisher, 1936)",
        "loader": "_load_iris",
    },
    {
        "name": "Titanic",
        "source": "seaborn built-in dataset",
        "license": "Public domain",
        "loader": "_load_titanic",
    },
    {
        "name": "Wine Quality",
        "source": "sklearn.datasets.load_wine (substitute for UCI Wine Quality)",
        "license": "BSD (sklearn) / CC BY 4.0 (UCI original)",
        "loader": "_load_wine",
    },
    {
        "name": "Breast Cancer Wisconsin",
        "source": "sklearn.datasets.load_breast_cancer",
        "license": "Public domain (UCI ML Repository)",
        "loader": "_load_breast_cancer",
    },
    {
        "name": "California Housing",
        "source": "sklearn.datasets.fetch_california_housing",
        "license": "Public domain (StatLib / Pace & Barry, 1997)",
        "loader": "_load_california_housing",
    },
]


def _load_iris(output_dir: str) -> str:
    path = os.path.join(output_dir, "iris.csv")
    if os.path.isfile(path):
        return path
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = pd.Categorical.from_codes(data.target, data.target_names)
    df.to_csv(path, index=False)
    return path


def _load_titanic(output_dir: str) -> str:
    path = os.path.join(output_dir, "titanic.csv")
    if os.path.isfile(path):
        return path
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
    except Exception:
        np.random.seed(42)
        n = 891
        df = pd.DataFrame({
            "survived": np.random.choice([0, 1], n, p=[0.62, 0.38]),
            "pclass": np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            "sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
            "age": np.where(np.random.random(n) > 0.2, np.random.uniform(1, 80, n).round(1), np.nan),
            "sibsp": np.random.choice(range(6), n, p=[0.68, 0.23, 0.03, 0.02, 0.02, 0.02]),
            "parch": np.random.choice(range(4), n, p=[0.76, 0.13, 0.09, 0.02]),
            "fare": np.random.exponential(30, n).round(2),
            "embarked": np.random.choice(["S", "C", "Q", np.nan], n, p=[0.70, 0.19, 0.09, 0.02]),
        })
    df.to_csv(path, index=False)
    return path


def _load_wine(output_dir: str) -> str:
    path = os.path.join(output_dir, "wine.csv")
    if os.path.isfile(path):
        return path
    from sklearn.datasets import load_wine
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df.to_csv(path, index=False)
    return path


def _load_breast_cancer(output_dir: str) -> str:
    path = os.path.join(output_dir, "breast_cancer.csv")
    if os.path.isfile(path):
        return path
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = pd.Categorical.from_codes(data.target, ["malignant", "benign"])
    df.to_csv(path, index=False)
    return path


def _load_california_housing(output_dir: str) -> str:
    path = os.path.join(output_dir, "california_housing.csv")
    if os.path.isfile(path):
        return path
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target
    df.to_csv(path, index=False)
    return path


LOADERS = {
    "_load_iris": _load_iris,
    "_load_titanic": _load_titanic,
    "_load_wine": _load_wine,
    "_load_breast_cancer": _load_breast_cancer,
    "_load_california_housing": _load_california_housing,
}


def _extract_ml_metrics(stdout: str) -> dict:
    """Extract machine learning metrics from execution stdout."""
    metrics = {}
    if not stdout:
        return metrics
    try:
        start = stdout.rfind("{")
        end = stdout.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(stdout[start:end])
            results = parsed.get("results", {})
            for k, v in results.items():
                if any(m in k.lower() for m in ["acc", "f1", "precision", "recall", "mae", "rmse", "r2", "score"]):
                    metrics[k] = v
    except Exception:
        pass
    return metrics


def evaluate_dataset(dataset_info: dict, config: AppConfig, output_base: str) -> dict:
    """Evaluate a single dataset. Returns a rich evaluation record."""
    name = dataset_info["name"]
    slug = name.lower().replace(" ", "_")

    record = {
        "dataset": name,
        "slug": slug,
        "source": dataset_info["source"],
        "license": dataset_info["license"],
        "ingestion_success": False,
        "shape": None,
        "plan_created": False,
        "plan_objective": "",
        "use_ml": False,
        "ml_task": "",
        "target_column": "",
        "execution_success": False,
        "corrections_required": False,
        "correction_attempts": 0,
        "reviewer_accepted": False,
        "report_generated": False,
        "report_path": "",
        "generated_charts": [],
        "ml_metrics": {},
        "failure": None,
        "failure_cause": None,
        "duration_seconds": 0,
    }

    dataset_dir = os.path.join(output_base, slug)
    os.makedirs(dataset_dir, exist_ok=True)

    # Dedicated reports output directory for this dataset
    report_dir = os.path.join(PROJECT_ROOT, "reports", slug)
    os.makedirs(report_dir, exist_ok=True)

    start = time.time()

    # Load dataset
    try:
        loader = LOADERS[dataset_info["loader"]]
        csv_path = loader(dataset_dir)
        df_preview = pd.read_csv(csv_path)
        record["shape"] = f"{df_preview.shape[0]} rows × {df_preview.shape[1]} cols"
        record["ingestion_success"] = True
        print(f"  ✓ {name}: Dataset loaded ({record['shape']})")
    except Exception as e:
        record["failure"] = "ingestion"
        record["failure_cause"] = str(e)
        print(f"  ✗ {name}: Ingestion failed — {e}")
        record["duration_seconds"] = round(time.time() - start, 2)
        return record

    # Run analysis workflow with dedicated output_dir
    try:
        final_state = run_analysis(
            dataset_path=csv_path,
            user_request=f"Perform comprehensive analysis of the {name} dataset.",
            config=config,
            output_dir=report_dir,
        )

        # Plan evaluation
        plan = final_state.get("analysis_plan", {})
        if plan:
            record["plan_created"] = True
            record["plan_objective"] = plan.get("objective", "")
            record["use_ml"] = plan.get("use_ml", False)
            record["ml_task"] = plan.get("ml_task", "")
            record["target_column"] = plan.get("target_column", "")

        status = final_state.get("workflow_status", "unknown")
        corrections = final_state.get("correction_history", [])
        record["correction_attempts"] = len(corrections)
        record["corrections_required"] = len(corrections) > 0

        exec_result = final_state.get("execution_result", {})
        record["execution_success"] = exec_result.get("success", False)

        review = final_state.get("review_result", {})
        record["reviewer_accepted"] = review.get("status") == "accepted"

        # Charts and artifacts
        artifacts = final_state.get("generated_artifact_paths", [])
        chart_files = [os.path.basename(p) for p in artifacts if p.lower().endswith((".png", ".jpg", ".svg"))]
        record["generated_charts"] = chart_files

        # ML Metrics
        record["ml_metrics"] = _extract_ml_metrics(exec_result.get("stdout", ""))

        # Report
        report = final_state.get("final_report", "")
        report_path = final_state.get("report_path", "")
        record["report_generated"] = bool(report and len(report) > 100)
        record["report_path"] = report_path

        # Also place a copy at reports/{slug}_report.md for direct access
        if report_path and os.path.isfile(report_path):
            flat_report_path = os.path.join(PROJECT_ROOT, "reports", f"{slug}_report.md")
            shutil.copyfile(report_path, flat_report_path)

        if status == "failed":
            record["failure"] = final_state.get("failure_info", "Unknown")
            record["failure_cause"] = "workflow_failed"

        print(f"  {'✓' if status == 'completed' else '✗'} {name}: {status} "
              f"(exec={'✓' if record['execution_success'] else '✗'}, "
              f"review={'✓' if record['reviewer_accepted'] else '✗'}, "
              f"charts={len(chart_files)}, "
              f"report={'✓' if record['report_generated'] else '✗'})")

    except Exception as e:
        record["failure"] = str(e)
        record["failure_cause"] = type(e).__name__
        print(f"  ✗ {name}: Error — {type(e).__name__}: {e}")

    record["duration_seconds"] = round(time.time() - start, 2)
    return record


def run_evaluation(mode: str = "demo", backend: str = "auto", target_dataset: str | None = None):
    """Run the 5-dataset evaluation suite."""
    print("=" * 70)
    print("Autonomous Data Analyst — 5-Dataset Evaluation Suite")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Configure execution mode
    base_config = load_config()
    if mode.lower() == "demo":
        llm_cfg = LLMConfig(
            provider="none",
            api_key="",
            model="deterministic-demo",
            base_url="none",
            available=False,
        )
    else:
        llm_cfg = base_config.llm

    # Override backend if specified
    from app.config import ExecutionConfig
    exec_cfg = ExecutionConfig(
        max_correction_attempts=base_config.execution.max_correction_attempts,
        code_execution_timeout=base_config.execution.code_execution_timeout,
        max_code_size_kb=base_config.execution.max_code_size_kb,
        max_output_size_kb=base_config.execution.max_output_size_kb,
        backend=backend,
    )
    config = AppConfig(llm=llm_cfg, execution=exec_cfg)

    print(f"Mode: {'DEMONSTRATION (Deterministic)' if config.llm.is_demo_mode else 'LIVE LLM (' + config.llm.model + ')'}")
    print(f"Execution Backend: {backend}")
    print(f"Max Corrections: {config.execution.max_correction_attempts}")
    print()

    output_base = os.path.join(PROJECT_ROOT, "datasets")
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)

    datasets_to_run = [d for d in DATASETS if target_dataset is None or d["name"].lower() == target_dataset.lower()]

    results = []
    for ds in datasets_to_run:
        print(f"\n[{ds['name']}]")
        record = evaluate_dataset(ds, config, output_base)
        results.append(record)

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<24} {'Ingest':>7} {'Plan':>6} {'Exec':>6} {'Review':>8} {'Charts':>8} {'Report':>8} {'Time':>6}")
    print("-" * 70)

    for r in results:
        def mark(b): return "✓" if b else "✗"
        print(f"{r['dataset']:<24} {mark(r['ingestion_success']):>7} "
              f"{mark(r['plan_created']):>6} {mark(r['execution_success']):>6} "
              f"{mark(r['reviewer_accepted']):>8} {len(r['generated_charts']):>8} "
              f"{mark(r['report_generated']):>8} "
              f"{r['duration_seconds']:>5.1f}s")

    passed = sum(1 for r in results if r["report_generated"])
    print(f"\nPassed: {passed}/{len(results)}")

    # Save summary results
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": "DEMONSTRATION" if config.llm.is_demo_mode else "LLM",
        "backend": backend,
        "results": results,
    }

    # Save to both datasets/ and reports/
    with open(os.path.join(output_base, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(os.path.join(PROJECT_ROOT, "reports", "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - {os.path.join(output_base, 'evaluation_results.json')}")
    print(f"  - {os.path.join(PROJECT_ROOT, 'reports', 'evaluation_results.json')}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Autonomous Data Analyst on 5 datasets")
    parser.add_argument("--mode", choices=["demo", "llm"], default="demo", help="Execution mode (demo or llm)")
    parser.add_argument("--backend", choices=["auto", "subprocess", "docker"], default="auto", help="Execution backend")
    parser.add_argument("--dataset", type=str, default=None, help="Target specific dataset (e.g. Iris)")
    args = parser.parse_args()

    run_evaluation(mode=args.mode, backend=args.backend, target_dataset=args.dataset)
