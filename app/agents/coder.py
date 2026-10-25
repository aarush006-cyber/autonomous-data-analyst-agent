"""Coder agent — generates and corrects analysis code."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.state import AnalysisPlan, AnalystState, DatasetMetadata
from app.prompts.coder_prompt import (
    CODER_SYSTEM_PROMPT,
    CODER_USER_PROMPT,
    CODER_CORRECTION_PROMPT,
)

logger = logging.getLogger(__name__)


def _sanitize_filename_component(name: Any) -> str:
    """Create a safe filesystem-friendly filename component from a column name."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(name)).strip("_")
    return clean[:40] if clean else "col"


def _generate_demo_code(
    metadata: DatasetMetadata,
    plan: AnalysisPlan,
    skip_ml: bool = False,
) -> str:
    """Generate deterministic analysis code for demonstration mode.
    
    Guarantees safe column name serialization, handles non-finite values,
    supports empty datasets, and prevents execution crashes.
    """
    code_parts = [
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import json",
        "import os",
        "import re",
        "",
        "# Load dataset",
        "df = pd.read_csv(DATASET_PATH)",
        'print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")',
        "",
        "results = {}",
        "generated_files = []",
        "",
        "# --- Data Overview ---",
        'results["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}',
        'results["dtypes"] = {str(col): str(dtype) for col, dtype in df.dtypes.items()}',
        'print("Data overview complete")',
        "",
        "# --- Data Quality ---",
        "n_rows = len(df)",
        "missing = df.isnull().sum()",
        "missing_pct = (missing / n_rows * 100).round(2) if n_rows > 0 else (missing * 0.0)",
        'results["missing_values"] = {str(col): {"count": int(missing[col]), "pct": float(missing_pct[col])} for col in df.columns if missing[col] > 0}',
        'results["duplicate_rows"] = int(df.duplicated().sum()) if n_rows > 0 else 0',
        'print(f"Missing values: {int(missing.sum())} total")',
        'print(f"Duplicate rows: {results[\'duplicate_rows\']}")',
        "",
    ]

    # Handle empty datasets early
    code_parts.extend([
        "if n_rows == 0:",
        '    results["warning"] = "Dataset is empty (0 rows). Skipping visual and modeling analyses."',
        '    print("Dataset contains 0 rows; skipping visualizations.")',
        "else:",
    ])

    # Indented block for non-empty datasets
    ind = "    "

    # Descriptive statistics
    code_parts.extend([
        f"{ind}# --- Descriptive Statistics ---",
        f"{ind}numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()",
        f"{ind}if numeric_cols:",
        f"{ind}    # Replace inf/-inf with NaN for reliable statistics",
        f"{ind}    num_finite = df[numeric_cols].replace([np.inf, -np.inf], np.nan)",
        f"{ind}    desc = num_finite.describe().round(4)",
        f'{ind}    results["descriptive_stats"] = {{str(k): v for k, v in desc.to_dict().items()}}',
        f'{ind}    print("Descriptive statistics computed")',
        "",
    ])

    # Distribution histograms
    code_parts.extend([
        f"{ind}# --- Distribution Plots ---",
        f"{ind}try:",
        f"{ind}    valid_hist_cols = [c for c in numeric_cols if df[c].replace([np.inf, -np.inf], np.nan).dropna().nunique() > 1][:8]",
        f"{ind}    if valid_hist_cols:",
        f"{ind}        n_cols = min(len(valid_hist_cols), 3)",
        f"{ind}        n_rows_plot = (len(valid_hist_cols) + n_cols - 1) // n_cols",
        f"{ind}        fig, axes = plt.subplots(n_rows_plot, n_cols, figsize=(5 * n_cols, 4 * n_rows_plot))",
        f"{ind}        if n_rows_plot * n_cols == 1:",
        f"{ind}            axes = np.array([axes])",
        f"{ind}        axes = axes.flatten()",
        f"{ind}        for i, col in enumerate(valid_hist_cols):",
        f"{ind}            ax = axes[i]",
        f"{ind}            s = df[col].replace([np.inf, -np.inf], np.nan).dropna()",
        f"{ind}            if len(s) > 0:",
        f"{ind}                s.hist(ax=ax, bins=min(30, max(5, s.nunique())), edgecolor='black', alpha=0.7)",
        f"{ind}            title_clean = str(col)[:30]",
        f'{ind}            ax.set_title(f"Distribution: {{title_clean}}", fontsize=10)',
        f"{ind}            ax.set_xlabel(title_clean)",
        f'{ind}            ax.set_ylabel("Frequency")',
        f"{ind}        for j in range(len(valid_hist_cols), len(axes)):",
        f"{ind}            axes[j].set_visible(False)",
        f"{ind}        plt.tight_layout()",
        f"{ind}        p_path = os.path.join(OUTPUT_DIR, 'distributions.png')",
        f"{ind}        plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{ind}        plt.close(fig)",
        f"{ind}        generated_files.append('distributions.png')",
        f'{ind}        print("Distribution plots saved")',
        f"{ind}except Exception as e:",
        f'{ind}    results["warning_distributions"] = str(e)',
        f'{ind}    print(f"Warning in distribution plots: {{e}}")',
        f"{ind}    plt.close('all')",
        "",
    ])

    # Correlation heatmap
    code_parts.extend([
        f"{ind}# --- Correlation Analysis ---",
        f"{ind}try:",
        f"{ind}    corr_cols = [c for c in numeric_cols if df[c].replace([np.inf, -np.inf], np.nan).dropna().nunique() > 1][:15]",
        f"{ind}    if len(corr_cols) >= 2:",
        f"{ind}        corr_df = df[corr_cols].replace([np.inf, -np.inf], np.nan)",
        f"{ind}        corr = corr_df.corr().round(3)",
        f"{ind}        if not corr.empty and not corr.isna().all().all():",
        f'{ind}            results["correlations"] = {{',
        f'{ind}                f"{{c1}}_vs_{{c2}}": float(corr.loc[c1, c2])',
        f'{ind}                for c1 in corr.columns for c2 in corr.columns',
        f'{ind}                if c1 < c2 and pd.notna(corr.loc[c1, c2]) and abs(corr.loc[c1, c2]) > 0.5',
        f'{ind}            }}',
        f"{ind}            fig, ax = plt.subplots(figsize=(max(8, len(corr_cols)), max(6, len(corr_cols) * 0.8)))",
        f"{ind}            sns.heatmap(corr, annot=len(corr_cols) <= 10, cmap='coolwarm', center=0,",
        f"{ind}                        fmt='.2f' if len(corr_cols) <= 10 else '', ax=ax)",
        f'{ind}            ax.set_title("Correlation Heatmap")',
        f"{ind}            p_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')",
        f"{ind}            plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{ind}            plt.close(fig)",
        f"{ind}            generated_files.append('correlation_heatmap.png')",
        f'{ind}            print("Correlation heatmap saved")',
        f"{ind}except Exception as e:",
        f'{ind}    results["warning_correlation"] = str(e)',
        f'{ind}    print(f"Warning in correlation analysis: {{e}}")',
        f"{ind}    plt.close('all')",
        "",
    ])

    # Box plots
    code_parts.extend([
        f"{ind}# --- Box Plots ---",
        f"{ind}try:",
        f"{ind}    box_cols = [c for c in numeric_cols if df[c].replace([np.inf, -np.inf], np.nan).dropna().nunique() > 1][:8]",
        f"{ind}    if box_cols:",
        f"{ind}        fig, axes = plt.subplots(1, len(box_cols), figsize=(max(4, 4 * len(box_cols)), 5))",
        f"{ind}        if len(box_cols) == 1:",
        f"{ind}            axes = [axes]",
        f"{ind}        for i, col in enumerate(box_cols):",
        f"{ind}            s = df[col].replace([np.inf, -np.inf], np.nan).dropna()",
        f"{ind}            if len(s) > 0:",
        f"{ind}                s.plot.box(ax=axes[i])",
        f"{ind}            title_clean = str(col)[:30]",
        f"{ind}            axes[i].set_title(title_clean, fontsize=10)",
        f'{ind}        plt.suptitle("Box Plots", fontsize=14)',
        f"{ind}        plt.tight_layout()",
        f"{ind}        p_path = os.path.join(OUTPUT_DIR, 'boxplots.png')",
        f"{ind}        plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{ind}        plt.close(fig)",
        f"{ind}        generated_files.append('boxplots.png')",
        f'{ind}        print("Box plots saved")',
        f"{ind}except Exception as e:",
        f'{ind}    results["warning_boxplots"] = str(e)',
        f'{ind}    print(f"Warning in box plots: {{e}}")',
        f"{ind}    plt.close('all')",
        "",
    ])

    # Categorical bar charts
    code_parts.extend([
        f"{ind}# --- Categorical Analysis ---",
        f"{ind}try:",
        f'{ind}    cat_candidates = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()',
        f"{ind}    plot_cats = [c for c in cat_candidates if 1 < df[c].nunique(dropna=True) <= 30][:4]",
        f"{ind}    for col in plot_cats:",
        f"{ind}        fig, ax = plt.subplots(figsize=(8, 5))",
        f"{ind}        vc = df[col].astype(str).value_counts().head(15)",
        f"{ind}        vc.plot.bar(ax=ax, edgecolor='black', alpha=0.7)",
        f"{ind}        title_clean = str(col)[:40]",
        f'{ind}        ax.set_title(f"Frequency: {{title_clean}}")',
        f"{ind}        ax.set_xlabel(title_clean)",
        f'{ind}        ax.set_ylabel("Count")',
        f"{ind}        plt.xticks(rotation=45, ha='right')",
        f"{ind}        plt.tight_layout()",
        f"{ind}        safe_col_name = re.sub(r'[^a-zA-Z0-9_\\-]', '_', str(col)).strip('_')[:30] or 'cat'",
        f'{ind}        fname = f"categorical_{{safe_col_name}}.png"',
        f"{ind}        p_path = os.path.join(OUTPUT_DIR, fname)",
        f"{ind}        plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{ind}        plt.close(fig)",
        f"{ind}        generated_files.append(fname)",
        f'{ind}        print(f"Saved categorical plot: {{fname}}")',
        f"{ind}except Exception as e:",
        f'{ind}    results["warning_categorical"] = str(e)',
        f'{ind}    print(f"Warning in categorical analysis: {{e}}")',
        f"{ind}    plt.close('all')",
        "",
    ])

    # Machine Learning section (if not skipped and planned)
    if plan.use_ml and plan.target_column and not skip_ml:
        target = plan.target_column
        if plan.ml_task == "classification":
            code_parts.extend(_generate_classification_code(target, metadata, indent=ind))
        elif plan.ml_task == "regression":
            code_parts.extend(_generate_regression_code(target, metadata, indent=ind))

    # Final summary output
    code_parts.extend([
        "",
        "# --- Final Summary ---",
        'print(json.dumps({"results": results, "generated_files": generated_files}, indent=2, default=str))',
        'print("Analysis complete.")',
    ])

    return "\n".join(code_parts)


def _generate_classification_code(
    target: str,
    metadata: DatasetMetadata,
    indent: str = "    ",
) -> list[str]:
    """Generate robust classification ML code with safe target serialization."""
    target_lit = repr(str(target))

    return [
        f"{indent}# --- Machine Learning: Classification ---",
        f"{indent}try:",
        f"{indent}    from sklearn.model_selection import train_test_split",
        f"{indent}    from sklearn.ensemble import RandomForestClassifier",
        f"{indent}    from sklearn.preprocessing import LabelEncoder",
        f"{indent}    from sklearn.metrics import (",
        f"{indent}        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix",
        f"{indent}    )",
        f"",
        f"{indent}    target_col = {target_lit}",
        f"{indent}    if target_col in df.columns:",
        f"{indent}        # Clean target: remove NaN and infinite values",
        f"{indent}        ml_df = df.dropna(subset=[target_col]).copy()",
        f"{indent}        if pd.api.types.is_numeric_dtype(ml_df[target_col]):",
        f"{indent}            ml_df = ml_df[~ml_df[target_col].isin([np.inf, -np.inf])]",
        f"",
        f"{indent}        target_series = ml_df[target_col].astype(str)",
        f"{indent}        unique_classes = target_series.nunique()",
        f"{indent}        class_counts = target_series.value_counts()",
        f"",
        f"{indent}        # Validate target suitability",
        f"{indent}        if len(ml_df) < 10:",
        f'{indent}            results["ml_note"] = "Classification skipped: insufficient samples (< 10 rows)"',
        f'{indent}            print("Skipping classification: insufficient samples (< 10 rows)")',
        f"{indent}        elif unique_classes < 2:",
        f'{indent}            results["ml_note"] = f"Classification skipped: target \'{{target_col}}\' has only {{unique_classes}} class"',
        f'{indent}            print(f"Skipping classification: target has only {{unique_classes}} class")',
        f"{indent}        elif unique_classes > 50 and pd.api.types.is_numeric_dtype(df[target_col]):",
        f'{indent}            results["ml_note"] = f"Classification skipped: target \'{{target_col}}\' appears continuous ({{unique_classes}} unique values)"',
        f'{indent}            print("Skipping classification: continuous target in classification task")',
        f"{indent}        else:",
        f"{indent}            # Filter out rare classes with fewer than 2 samples to permit robust splitting",
        f"{indent}            valid_classes = class_counts[class_counts >= 2].index",
        f"{indent}            if len(valid_classes) >= 2:",
        f"{indent}                ml_df = ml_df[target_series.isin(valid_classes)].copy()",
        f"{indent}                target_series = ml_df[target_col].astype(str)",
        f"{indent}            ",
        f"{indent}            if len(ml_df) >= 10 and target_series.nunique() >= 2:",
        f"{indent}                le = LabelEncoder()",
        f"{indent}                y = le.fit_transform(target_series)",
        f"",
        f"{indent}                # Prepare features",
        f"{indent}                feature_dfs = []",
        f"{indent}                feature_names = []",
        f"",
        f"{indent}                # Numeric features: sanitize non-finite values and impute with median",
        f"{indent}                num_cols = [c for c in ml_df.select_dtypes(include=[np.number]).columns if c != target_col]",
        f"{indent}                if num_cols:",
        f"{indent}                    num_df = ml_df[num_cols].replace([np.inf, -np.inf], np.nan)",
        f"{indent}                    num_df = num_df.fillna(num_df.median().fillna(0))",
        f"{indent}                    var_cols = [c for c in num_df.columns if num_df[c].nunique() > 1]",
        f"{indent}                    if var_cols:",
        f"{indent}                        feature_dfs.append(num_df[var_cols])",
        f"{indent}                        feature_names.extend([str(c) for c in var_cols])",
        f"",
        f"{indent}                # Categorical features: encode features with reasonable cardinality",
        f'{indent}                cat_cols = [c for c in ml_df.select_dtypes(include=["object", "category", "bool", "string"]).columns if c != target_col]',
        f"{indent}                for c in cat_cols:",
        f"{indent}                    if 1 < ml_df[c].nunique(dropna=True) <= 20:",
        f"{indent}                        c_clean = ml_df[c].fillna('Missing').astype(str)",
        f"{indent}                        enc = LabelEncoder().fit_transform(c_clean)",
        f"{indent}                        enc_name = f'{{c}}_enc'",
        f"{indent}                        feature_dfs.append(pd.Series(enc, index=ml_df.index, name=enc_name).to_frame())",
        f"{indent}                        feature_names.append(enc_name)",
        f"",
        f"{indent}                if feature_dfs and len(feature_names) >= 1:",
        f"{indent}                    X = pd.concat(feature_dfs, axis=1)",
        f"{indent}                    y_counts = pd.Series(y).value_counts()",
        f"{indent}                    can_stratify = (y_counts.min() >= 2) and (len(y_counts) > 1)",
        f"",
        f"{indent}                    X_train, X_test, y_train, y_test = train_test_split(",
        f"{indent}                        X, y, test_size=0.2, random_state=42,",
        f"{indent}                        stratify=y if can_stratify else None,",
        f"{indent}                    )",
        f"",
        f"{indent}                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)",
        f"{indent}                    model.fit(X_train, y_train)",
        f"{indent}                    y_pred = model.predict(X_test)",
        f"",
        f"{indent}                    avg = 'binary' if len(np.unique(y)) == 2 else 'weighted'",
        f'{indent}                    results["ml_accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)',
        f'{indent}                    results["ml_precision"] = round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4)',
        f'{indent}                    results["ml_recall"] = round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4)',
        f'{indent}                    results["ml_f1"] = round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)',
        f"",
        f'{indent}                    print(f"Classification Results:")',
        f'{indent}                    print(f"  Accuracy:  {{results[\'ml_accuracy\']}}")',
        f'{indent}                    print(f"  Precision: {{results[\'ml_precision\']}}")',
        f'{indent}                    print(f"  Recall:    {{results[\'ml_recall\']}}")',
        f'{indent}                    print(f"  F1 Score:  {{results[\'ml_f1\']}}")',
        f"",
        f"{indent}                    # Confusion matrix (guard against excessive classes)",
        f"{indent}                    if len(le.classes_) <= 20:",
        f"{indent}                        cm = confusion_matrix(y_test, y_pred)",
        f"{indent}                        fig, ax = plt.subplots(figsize=(max(6, len(le.classes_) * 0.8), max(5, len(le.classes_) * 0.6)))",
        f"{indent}                        labels = [str(cl)[:15] for cl in le.classes_]",
        f"{indent}                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,",
        f"{indent}                                    xticklabels=labels, yticklabels=labels)",
        f'{indent}                        ax.set_title("Confusion Matrix")',
        f'{indent}                        ax.set_xlabel("Predicted")',
        f'{indent}                        ax.set_ylabel("Actual")',
        f"{indent}                        p_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')",
        f"{indent}                        plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{indent}                        plt.close(fig)",
        f"{indent}                        generated_files.append('confusion_matrix.png')",
        f"",
        f"{indent}                    # Feature importance",
        f"{indent}                    importances = model.feature_importances_",
        f"{indent}                    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)",
        f"{indent}                    fig, ax = plt.subplots(figsize=(8, max(4, len(feat_imp) * 0.4)))",
        f"{indent}                    feat_imp.plot.barh(ax=ax, edgecolor='black', alpha=0.7)",
        f'{indent}                    ax.set_title("Feature Importance")',
        f'{indent}                    ax.set_xlabel("Importance")',
        f"{indent}                    p_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')",
        f"{indent}                    plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{indent}                    plt.close(fig)",
        f"{indent}                    generated_files.append('feature_importance.png')",
        f'{indent}                    print("ML classification complete")',
        f"{indent}                else:",
        f'{indent}                    results["ml_note"] = "Insufficient features with variance for classification"',
        f'{indent}                    print("Skipping ML: insufficient valid features")',
        f"{indent}            else:",
        f'{indent}                results["ml_note"] = "Insufficient samples after filtering rare classes"',
        f'{indent}                print("Skipping ML: insufficient samples after rare class filtering")',
        f"{indent}    else:",
        f'{indent}        results["ml_note"] = f"Target column \'{{target_col}}\' not found in dataset"',
        f'{indent}        print(f"Target column \'{{target_col}}\' not found")',
        f"{indent}except Exception as e:",
        f'{indent}    results["ml_error"] = str(e)',
        f'{indent}    print(f"ML classification failed: {{e}}")',
        f"{indent}    plt.close('all')",
        "",
    ]


def _generate_regression_code(
    target: str,
    metadata: DatasetMetadata,
    indent: str = "    ",
) -> list[str]:
    """Generate robust regression ML code with safe target serialization."""
    target_lit = repr(str(target))

    return [
        f"{indent}# --- Machine Learning: Regression ---",
        f"{indent}try:",
        f"{indent}    from sklearn.model_selection import train_test_split",
        f"{indent}    from sklearn.ensemble import RandomForestRegressor",
        f"{indent}    from sklearn.preprocessing import LabelEncoder",
        f"{indent}    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        f"",
        f"{indent}    target_col = {target_lit}",
        f"{indent}    if target_col in df.columns:",
        f"{indent}        # Validate and coerce target to numeric",
        f"{indent}        ml_df = df.copy()",
        f"{indent}        y_raw = pd.to_numeric(ml_df[target_col], errors='coerce')",
        f"{indent}        valid_mask = y_raw.notna() & ~y_raw.isin([np.inf, -np.inf])",
        f"{indent}        ml_df = ml_df[valid_mask].copy()",
        f"{indent}        y = y_raw[valid_mask]",
        f"",
        f"{indent}        if len(y) < 10:",
        f'{indent}            results["ml_note"] = "Regression skipped: insufficient numeric target samples (< 10 rows)"',
        f'{indent}            print("Skipping regression: insufficient samples (< 10 rows)")',
        f"{indent}        elif y.nunique() <= 1 or (y.std() == 0 or np.isnan(y.std())):",
        f'{indent}            results["ml_note"] = f"Regression skipped: target \'{{target_col}}\' has zero variance (constant values)"',
        f'{indent}            print("Skipping regression: constant target values")',
        f"{indent}        else:",
        f"{indent}            feature_dfs = []",
        f"{indent}            feature_names = []",
        f"",
        f"{indent}            # Numeric features: sanitize non-finite values and impute with median",
        f"{indent}            num_cols = [c for c in ml_df.select_dtypes(include=[np.number]).columns if c != target_col]",
        f"{indent}            if num_cols:",
        f"{indent}                num_df = ml_df[num_cols].replace([np.inf, -np.inf], np.nan)",
        f"{indent}                num_df = num_df.fillna(num_df.median().fillna(0))",
        f"{indent}                var_cols = [c for c in num_df.columns if num_df[c].nunique() > 1]",
        f"{indent}                if var_cols:",
        f"{indent}                    feature_dfs.append(num_df[var_cols])",
        f"{indent}                    feature_names.extend([str(c) for c in var_cols])",
        f"",
        f"{indent}            # Categorical features: encode features with reasonable cardinality",
        f'{indent}            cat_cols = [c for c in ml_df.select_dtypes(include=["object", "category", "bool", "string"]).columns if c != target_col]',
        f"{indent}            for c in cat_cols:",
        f"{indent}                if 1 < ml_df[c].nunique(dropna=True) <= 20:",
        f"{indent}                    c_clean = ml_df[c].fillna('Missing').astype(str)",
        f"{indent}                    enc = LabelEncoder().fit_transform(c_clean)",
        f"{indent}                    enc_name = f'{{c}}_enc'",
        f"{indent}                    feature_dfs.append(pd.Series(enc, index=ml_df.index, name=enc_name).to_frame())",
        f"{indent}                    feature_names.append(enc_name)",
        f"",
        f"{indent}            if feature_dfs and len(feature_names) >= 1:",
        f"{indent}                X = pd.concat(feature_dfs, axis=1)",
        f"{indent}                X_train, X_test, y_train, y_test = train_test_split(",
        f"{indent}                    X, y, test_size=0.2, random_state=42",
        f"{indent}                )",
        f"",
        f"{indent}                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)",
        f"{indent}                model.fit(X_train, y_train)",
        f"{indent}                y_pred = model.predict(X_test)",
        f"",
        f"{indent}                mae = mean_absolute_error(y_test, y_pred)",
        f"{indent}                rmse = np.sqrt(mean_squared_error(y_test, y_pred))",
        f"{indent}                r2 = r2_score(y_test, y_pred) if y_test.nunique() > 1 else 0.0",
        f"",
        f'{indent}                results["ml_mae"] = round(float(mae), 4)',
        f'{indent}                results["ml_rmse"] = round(float(rmse), 4)',
        f'{indent}                results["ml_r2"] = round(float(r2), 4)',
        f"",
        f'{indent}                print(f"Regression Results:")',
        f'{indent}                print(f"  MAE:  {{results[\'ml_mae\']}}")',
        f'{indent}                print(f"  RMSE: {{results[\'ml_rmse\']}}")',
        f'{indent}                print(f"  R²:   {{results[\'ml_r2\']}}")',
        f"",
        f"{indent}                # Actual vs Predicted scatter plot",
        f"{indent}                fig, ax = plt.subplots(figsize=(8, 6))",
        f"{indent}                ax.scatter(y_test, y_pred, alpha=0.5, s=15)",
        f"{indent}                mn = min(float(y_test.min()), float(y_pred.min()))",
        f"{indent}                mx = max(float(y_test.max()), float(y_pred.max()))",
        f"{indent}                ax.plot([mn, mx], [mn, mx], 'r--', label='Perfect prediction')",
        f'{indent}                ax.set_xlabel("Actual")',
        f'{indent}                ax.set_ylabel("Predicted")',
        f'{indent}                ax.set_title("Actual vs Predicted")',
        f"{indent}                ax.legend()",
        f"{indent}                p_path = os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png')",
        f"{indent}                plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{indent}                plt.close(fig)",
        f"{indent}                generated_files.append('actual_vs_predicted.png')",
        f"",
        f"{indent}                # Residual plot",
        f"{indent}                residuals = y_test - y_pred",
        f"{indent}                fig, ax = plt.subplots(figsize=(8, 6))",
        f"{indent}                ax.scatter(y_pred, residuals, alpha=0.5, s=15)",
        f"{indent}                ax.axhline(y=0, color='r', linestyle='--')",
        f'{indent}                ax.set_xlabel("Predicted")',
        f'{indent}                ax.set_ylabel("Residuals")',
        f'{indent}                ax.set_title("Residual Plot")',
        f"{indent}                p_path = os.path.join(OUTPUT_DIR, 'residual_plot.png')",
        f"{indent}                plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{indent}                plt.close(fig)",
        f"{indent}                generated_files.append('residual_plot.png')",
        f"",
        f"{indent}                # Feature importance",
        f"{indent}                importances = model.feature_importances_",
        f"{indent}                feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)",
        f"{indent}                fig, ax = plt.subplots(figsize=(8, max(4, len(feat_imp) * 0.4)))",
        f"{indent}                feat_imp.plot.barh(ax=ax, edgecolor='black', alpha=0.7)",
        f'{indent}                ax.set_title("Feature Importance")',
        f'{indent}                ax.set_xlabel("Importance")',
        f"{indent}                p_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')",
        f"{indent}                plt.savefig(p_path, bbox_inches='tight', dpi=100)",
        f"{indent}                plt.close(fig)",
        f"{indent}                generated_files.append('feature_importance.png')",
        f'{indent}                print("ML regression complete")',
        f"{indent}            else:",
        f'{indent}                results["ml_note"] = "Insufficient features with variance for regression"',
        f'{indent}                print("Skipping ML: insufficient valid features")',
        f"{indent}    else:",
        f'{indent}        results["ml_note"] = f"Target column \'{{target_col}}\' not found in dataset"',
        f'{indent}        print(f"Target column \'{{target_col}}\' not found")',
        f"{indent}except Exception as e:",
        f'{indent}    results["ml_error"] = str(e)',
        f'{indent}    print(f"ML regression failed: {{e}}")',
        f"{indent}    plt.close('all')",
        "",
    ]


def _clean_code_response(text: str) -> str:
    """Extract clean Python code from LLM response."""
    text = text.strip()
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def run_coder(state: AnalystState, llm=None) -> dict:
    """Coder node for the LangGraph workflow.
    
    Generates Python code based on the analysis plan, or updates existing
    code using execution errors and reviewer feedback during correction iterations.
    Explicitly tracks and records LLM failures and demo fallbacks.
    """
    metadata_dict = state.get("dataset_metadata", {})
    metadata = DatasetMetadata(**metadata_dict)
    plan_dict = state.get("analysis_plan", {})
    plan = AnalysisPlan(**plan_dict)
    is_demo = state.get("is_demo_mode", True)
    iteration = state.get("current_iteration", 0)

    # Check if this is a correction attempt
    correction_history = state.get("correction_history", [])
    previous_code = state.get("generated_code", "")
    review_dict = state.get("review_result", {})
    exec_dict = state.get("execution_result", {})

    is_correction = iteration > 0 and bool(previous_code)

    # Build rich execution error context
    error_summary = exec_dict.get("error_summary", "")
    stderr_text = exec_dict.get("stderr", "").strip()
    stdout_text = exec_dict.get("stdout", "").strip()

    error_parts = []
    if error_summary:
        error_parts.append(f"Summary: {error_summary}")
    if stderr_text:
        error_parts.append(f"Stderr Traceback:\n{stderr_text[-1500:]}")
    elif not exec_dict.get("success", False) and stdout_text:
        error_parts.append(f"Stdout (last 1000 chars):\n{stdout_text[-1000:]}")
    combined_error = "\n".join(error_parts) if error_parts else "Execution failed with non-zero exit code"

    # Build rich reviewer feedback context
    feedback_parts = []
    if review_dict.get("acceptance_decision"):
        feedback_parts.append(f"Reviewer Decision: {review_dict['acceptance_decision']}")
    if review_dict.get("correction_instructions"):
        feedback_parts.append(f"Correction Instructions: {review_dict['correction_instructions']}")
    if review_dict.get("detected_errors"):
        feedback_parts.append(f"Detected Errors: {', '.join(review_dict['detected_errors'])}")
    if review_dict.get("missing_steps"):
        feedback_parts.append(f"Missing Steps: {', '.join(review_dict['missing_steps'])}")
    combined_feedback = "\n".join(feedback_parts) if feedback_parts else "Please correct execution errors and complete missing planned steps."

    # Check if previous failure in demo mode was ML-related so we can adapt
    combined_err_lower = (combined_error + " " + combined_feedback).lower()
    skip_ml_in_demo = is_correction and any(
        kw in combined_err_lower
        for kw in ["ml", "target", "classifier", "regressor", "labelencoder", "stratify", "syntaxerror", "indexerror"]
    )

    if is_demo or llm is None:
        logger.info(f"Coder: generating demo code (iteration {iteration}, skip_ml={skip_ml_in_demo})")
        code = _generate_demo_code(metadata, plan, skip_ml=skip_ml_in_demo)
        return {
            "generated_code": code,
            "generation_mode": "demo",
            "execution_log": [
                f"[Coder] Deterministic demo code generated ({len(code)} chars, iteration {iteration})"
                + (" [adapted: ML omitted after failure]" if skip_ml_in_demo else "")
            ],
        }

    # LLM-powered code generation
    try:
        if is_correction:
            prompt = CODER_CORRECTION_PROMPT.format(
                previous_code=previous_code[:3000],
                error=combined_error[:1500],
                reviewer_feedback=combined_feedback[:1000],
                correction_history=json.dumps(correction_history[-3:], indent=2)[:1000],
                column_names=metadata.column_names,
                dtypes=json.dumps(dict(list(metadata.dtypes.items())[:50])),
            )
        else:
            prompt = CODER_USER_PROMPT.format(
                filename=metadata.filename,
                rows=metadata.rows,
                columns=metadata.columns,
                column_names=metadata.column_names,
                dtypes=json.dumps(dict(list(metadata.dtypes.items())[:50])),
                analysis_plan=json.dumps(plan_dict, indent=2)[:2000],
            )

        messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        elif not content and hasattr(response, "additional_kwargs"):
            content = response.additional_kwargs.get("reasoning_content", "")
        code = _clean_code_response(content)

        return {
            "generated_code": code,
            "generation_mode": "llm",
            "execution_log": [f"[Coder] LLM code generated ({len(code)} chars, iteration {iteration})"],
        }

    except Exception as e:
        logger.warning(f"Coder LLM call failed: {e}, explicitly falling back to deterministic demo code")
        code = _generate_demo_code(metadata, plan, skip_ml=skip_ml_in_demo)
        fallback_msg = f"Coder LLM call failed ({type(e).__name__}: {e})"
        return {
            "generated_code": code,
            "is_fallback": True,
            "fallback_reason": fallback_msg,
            "generation_mode": "fallback",
            "execution_log": [
                f"[Coder] LLM call failed ({type(e).__name__}: {e}); explicitly fell back to deterministic demo code ({len(code)} chars, iteration {iteration})"
            ],
        }
