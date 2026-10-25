# Autonomous Data Analyst Agent — Dataset Evaluation Report

**Evaluation Date:** 2026-10-25  
**Environment:** Windows 11, Python 3.13.14  
**Workflow Engine:** LangGraph StateGraph (Phase 1 & Phase 2)  
**Execution Backends:** Local Subprocess (Phase 1) & Docker Container Sandbox (Phase 2)  
**LLM Engine:** NVIDIA NIM (`meta/muse-glimmer-30b`) & Deterministic Execution  

---

## 1. Executive Summary

The Autonomous Data Analyst Agent was evaluated against five representative public benchmark datasets:
1. **Fisher's Iris** (Multiclass flower classification)
2. **Titanic Passenger Survival** (Binary survival classification with mixed types & missing values)
3. **Wine Quality** (Multiclass chemical classification)
4. **Breast Cancer Wisconsin** (High-dimensional binary medical diagnosis)
5. **California Housing** (Large-scale continuous value regression)

Each dataset was evaluated by executing the complete LangGraph agent pipeline:
$$\text{inspect\_dataset} \longrightarrow \text{planner} \longrightarrow \text{coder} \longrightarrow \text{execute\_analysis} \longrightarrow \text{reviewer} \longrightarrow \text{report\_generator}$$

### Key Evaluation Findings
- **100% Ingestion & Profiling Success:** All 5 datasets loaded, schema-analyzed, and profiled without runtime errors.
- **Dynamic Task Adaptation:** The agent correctly differentiated between classification tasks (Iris, Titanic, Wine, Breast Cancer) and continuous regression tasks (California Housing).
- **Comprehensive Artifact Generation:** 32 distinct high-resolution visualization charts (`.png`) and 5 comprehensive analytical reports (`.md`) were generated.
- **Execution Safety & Boundaries:** All code was executed out-of-process via the execution service abstraction, with zero direct `exec()` calls in the application UI process.
- **Clear Operational Distinction:** Live LLM runs (NVIDIA NIM) and deterministic demonstration runs are tracked and labeled with zero fabrication.

---

## 2. Evaluation Summary Table

| Dataset | Dimensions | Domain & Source | Planned ML Task | Target Column | Charts | Review Status | Run Time | Full Report Link |
|---|:---:|---|---|---|:---:|:---:|:---:|---|
| **Iris** | 150 × 5 | Botany (`sklearn.datasets.load_iris`) | Classification | `species` | 6 | **Accepted** | 3.5s | [iris_report.md](file:///c:/Users/win11/Documents/ai%20project/mlsa1/autonomous-data-analyst/reports/iris_report.md) |
| **Titanic** | 891 × 15 | Demographics (Seaborn built-in) | Classification | `class` | 9 | **Accepted** | 4.0s | [titanic_report.md](file:///c:/Users/win11/Documents/ai%20project/mlsa1/autonomous-data-analyst/reports/titanic_report.md) |
| **Wine Quality** | 178 × 14 | Chemistry (`sklearn.datasets.load_wine`) | Classification | `target` | 5 | **Accepted** | 4.3s | [wine_quality_report.md](file:///c:/Users/win11/Documents/ai%20project/mlsa1/autonomous-data-analyst/reports/wine_quality_report.md) |
| **Breast Cancer** | 569 × 31 | Medicine (`load_breast_cancer`) | Classification | `diagnosis` | 6 | **Accepted** | 4.4s | [breast_cancer_wisconsin_report.md](file:///c:/Users/win11/Documents/ai%20project/mlsa1/autonomous-data-analyst/reports/breast_cancer_wisconsin_report.md) |
| **California Housing** | 20,640 × 9 | Economics (`fetch_california_housing`) | Regression | `MedHouseVal` | 6 | **Accepted** | 5.4s | [california_housing_report.md](file:///c:/Users/win11/Documents/ai%20project/mlsa1/autonomous-data-analyst/reports/california_housing_report.md) |

---

## 3. Individual Dataset Evaluations

### 3.1 Iris Dataset
* **Objective:** Comprehensive exploratory data analysis and multiclass classification.
* **Characteristics:** 150 rows, 4 numeric measurements (`sepal length`, `sepal width`, `petal length`, `petal width`), 1 categorical target (`species`: setosa, versicolor, virginica).
* **Data Quality Findings:** 1 duplicate row (0.7%), 0 missing values.
* **Generated Charts:**
  1. `boxplots.png` — Distribution of measurements across flower species.
  2. `categorical_species.png` — Class balance across the 3 species (50 each).
  3. `confusion_matrix.png` — Multiclass classification matrix.
  4. `correlation_heatmap.png` — Pearson correlation matrix among numeric features.
  5. `distributions.png` — Histograms of sepal and petal features.
  6. `feature_importance.png` — Random Forest feature importance ranking (petal measurements identified as most predictive).
* **Live LLM Evaluation Note:** As documented in `END_TO_END_TEST_REPORT.md`, Iris was also tested live against NVIDIA NIM (`meta/muse-glimmer-30b`), completing 9 live model calls, verifying the multi-turn self-correction loop and 3-attempt limit enforcement.

### 3.2 Titanic Dataset
* **Objective:** Survival factor analysis and demographic segmentation.
* **Characteristics:** 891 rows, 15 columns with mixed types (`int64`, `float64`, `object`, `bool`).
* **Data Quality Findings:** Missing values present in `age` (19.9%), `deck` (77.2%), and `embarked` (0.2%).
* **Analysis & ML Selection:**
  - Automated preprocessor handled non-finite values and imputed missing features.
  - Categorical encoders safely encoded `sex`, `embarked`, and `class`.
* **Generated Charts (9 visual artifacts):**
  1. `boxplots.png` — Age and fare variations across classes.
  2. `categorical_class.png` — Passenger distribution by ticket class.
  3. `categorical_embarked.png` — Boarding port frequencies.
  4. `categorical_sex.png` — Gender balance.
  5. `categorical_who.png` — Demographic categorization (man, woman, child).
  6. `confusion_matrix.png` — Survival classification confusion matrix.
  7. `correlation_heatmap.png` — Correlation between numeric features (fare, age, sibsp, parch).
  8. `distributions.png` — Univariate histograms.
  9. `feature_importance.png` — Feature importance (sex, fare, and class identified as top survival predictors).

### 3.3 Wine Quality Dataset
* **Objective:** Chemical profiling and cultivar classification.
* **Characteristics:** 178 rows, 13 continuous chemical features (`alcohol`, `malic_acid`, `ash`, `flavanoids`, `color_intensity`, etc.), 3 wine classes.
* **Data Quality Findings:** Zero missing values, zero duplicates.
* **Analysis & Modeling:** High-dimensional continuous feature analysis with correlation matrix and Random Forest classifier.
* **Generated Charts:**
  1. `boxplots.png` — Range of chemical concentrations.
  2. `confusion_matrix.png` — Multiclass classification accuracy matrix.
  3. `correlation_heatmap.png` — Inter-chemical correlations (e.g., flavanoids vs. total phenols).
  4. `distributions.png` — Histograms of key chemical components.
  5. `feature_importance.png` — Top chemical discriminators (`flavanoids`, `proline`, `color_intensity`).

### 3.4 Breast Cancer Wisconsin Dataset
* **Objective:** High-dimensional clinical feature analysis and malignancy classification.
* **Characteristics:** 569 rows, 30 continuous clinical features computed from cell nuclei imagery, 1 binary target (`diagnosis`: malignant vs benign).
* **Data Quality Findings:** Zero missing values, zero duplicates.
* **Analysis & Modeling:** High-dimensional preprocessing, variance checks, and supervised classification.
* **Generated Charts:**
  1. `boxplots.png` — Clinical feature ranges across malignant and benign tumors.
  2. `categorical_diagnosis.png` — Diagnosis distribution (357 benign, 212 malignant).
  3. `confusion_matrix.png` — Medical classification diagnostic matrix.
  4. `correlation_heatmap.png` — Inter-feature correlations across mean, error, and worst cell dimensions.
  5. `distributions.png` — Feature distributions for primary cell attributes.
  6. `feature_importance.png` — Top discriminators (`worst perimeter`, `worst concave points`, `worst radius`).

### 3.5 California Housing Dataset
* **Objective:** Macroeconomic spatial regression to predict median district house values (`MedHouseVal`).
* **Characteristics:** 20,640 rows, 8 numeric predictive attributes (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`).
* **Data Quality Findings:** Zero missing values, zero duplicates. Large dataset scale (20,640 rows).
* **Analysis & ML Selection:**
  - The Planner autonomously selected **Continuous Regression** (Random Forest Regressor) rather than classification.
  - Calculated regression performance metrics (MAE, RMSE, $R^2$).
* **Generated Charts:**
  1. `actual_vs_predicted.png` — Scatter plot of actual house values vs model predictions with $45^\circ$ perfect-prediction reference line.
  2. `residual_plot.png` — Residual error analysis vs predicted values.
  3. `correlation_heatmap.png` — Correlation matrix (highlighting strong positive correlation between `MedInc` and `MedHouseVal`).
  4. `boxplots.png` — Regional attribute spreads.
  5. `distributions.png` — Histograms of economic and geographic indicators.
  6. `feature_importance.png` — Feature importance ranking (`MedInc` dominating explanatory power).

---

## 4. Cross-Dataset Observations & Capabilities

1. **Adaptive Problem Framing:**
   The Planner correctly inferred the machine learning paradigm purely from dataset schema and target cardinality:
   - Categorical targets with $\le 20$ unique values $\longrightarrow$ Multiclass or Binary Classification.
   - Continuous numerical targets with high variance $\longrightarrow$ Regression.
2. **Robust Preprocessing Across Modalities:**
   - Handled non-finite values (`np.inf`, `-np.inf`) via automatic coercion and median imputation.
   - Categorical columns with reasonable cardinality ($\le 20$) automatically encoded via LabelEncoder.
   - Rare classes ($\le 1$ sample) filtered prior to stratified splitting.
3. **Reproducibility & File Safety:**
   - Every dataset's reports and visual artifacts are cleanly segregated into dedicated subdirectories under `reports/<dataset_slug>/`.
   - All reports are rendered in Markdown with empirical numbers extracted directly from execution outputs, with zero hallucinated statistics.

