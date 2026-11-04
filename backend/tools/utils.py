import os
import pandas as pd
import io
import json
import datetime
from backend.tools.preprocessing import (
    run_base_preprocessing,    run_categorical_preprocessing,
    run_numerical_preprocessing,     convert_numeric_strings,
)


def create_working_dir(filename: str) -> str:
    """Generates a unique project working directory under data/ and creates it."""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    i = 1
    while os.path.exists(f"/tmp/data/{base_name}_project{i}"):
        i += 1
    working_dir = f"/tmp/data/{base_name}_project{i}"
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def save_step_output(df: pd.DataFrame, working_dir: str, filename: str = "working_dataset.csv") -> str:
    """Overwrites a single working copy of the dataset inside working_dir 
    to keep intermediate disk usage clean and simple."""
    try:
        os.makedirs(working_dir, exist_ok=True)
        output_path = os.path.join(working_dir, filename)
        df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        print(f"Error saving dataset to {working_dir}/{filename}: {e}")
        raise

def process_and_save_upload(body_bytes: bytes, filename: str) -> tuple[str, list[str]]:
    """Validates extension, converts Excel if needed, creates project dir, saves CSV, returns working_dir & columns."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.csv', '.xlsx', '.xls']:
        raise ValueError(f"Unsupported format '{ext}'. Only CSV and Excel (.xlsx, .xls) are allowed.")
    if ext == '.csv':
        df = pd.read_csv(io.BytesIO(body_bytes), sep=None, engine="python")
    else:
        df = pd.read_excel(io.BytesIO(body_bytes))
    working_dir = create_working_dir(filename)
    save_step_output(df, working_dir, "working_dataset.csv")
    return working_dir, list(df.columns)

def log_step(working_dir: str, step: str, details: dict) -> dict:
    """Appends one step (what we did + facts) to report.json inside working_dir."""
    path = os.path.join(working_dir, "report.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            report = json.load(f)
    else:
        report = {"steps": []}
    report["steps"].append({
        "step": step,
        "time": datetime.datetime.now().isoformat(),
        **details
    })
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def build_summary(report: dict) -> str:
    """Builds a plain-text summary directly from report.json (no LLM)."""
    steps = {s["step"]: s for s in report.get("steps", [])}
    lines = ["PRELIMINARY QUICK REPORT (automated baseline: default settings, no tuning).", ""]

    ins = steps.get("inspect_dataset_summary")
    if ins:
        lines.append(f"- Dataset: {ins['shape']['rows']} rows x {ins['shape']['columns']} columns, {ins['total_null_cells']} missing cells.")

    base = steps.get("tool_run_base_cleanup")
    if base:
        lines.append(f"- Base cleanup: {base['initial_shape']['rows']} -> {base['cleaned_shape']['rows']} rows.")
        if base.get("columns_converted_to_numeric"):
            lines.append(f"- Converted to numeric: {', '.join(base['columns_converted_to_numeric'])}.")
        lines.append(f"- Rows dropped: {base.get('rows_dropped_because_target_missing', 0)} (target missing), {base.get('rows_dropped_by_missing_value_handling', 0)} (missing-value rules).")

    dt = steps.get("tool_drop_datetime_columns")
    if dt:
        lines.append(f"- Date/time columns dropped: {', '.join(dt['dropped_columns']) or 'none found'}.")

    irr = steps.get("tool_drop_irrelevant_columns")
    if irr:
        lines.append(f"- Irrelevant columns dropped: {', '.join(irr['dropped_columns']) or 'none'}.")

    sent = steps.get("tool_extract_text_sentiment")
    if sent:
        lines.append(f"- Text column '{sent['text_column']}' converted to a sentiment score.")

    cat = steps.get("tool_preprocess_categorical_features")
    if cat:
        lines.append(f"- Encoded text columns: {', '.join(cat['text_columns_encoded']) or 'none'} (now {cat['shape_after']['columns']} columns).")

    num = steps.get("tool_process_and_reduce_numeric_features")
    if num:
        capped = num.get("outlier_values_capped_total", 0)
        lines.append(f"- Numeric step: {capped} outlier values capped (rows kept), {len(num['columns_removed'])} columns removed, final {num['shape_after']['columns']} columns.")
        for reason, cols in num.get("columns_removed_by_reason", {}).items():
            if cols:
                lines.append(f"- Removed because {reason}: {', '.join(cols)}.")
    train = steps.get("train_and_evaluate_models")
    if train:
        metric_names = {"r2_score": "R2", "balanced_accuracy": "balanced accuracy", "accuracy": "accuracy"}
        metric = metric_names.get(train.get("metric"), "accuracy")
        lines.append(f"- Task: {train['task_type']}, target: {train['target']}, metric: {metric}.")
        if train.get("target_log_transformed"):
            lines.append(f"- Target was highly skewed (skew {train['target_skew']:.1f}), so models were trained on its log and predictions converted back.")
        if train.get("label_mapping"):
            mapping_text = ", ".join(f"{k} = {v}" for k, v in train["label_mapping"].items())
            lines.append(f"- Class labels used by the models: {mapping_text}.")

        if train["task_type"] == "classification":
            if train.get("class_counts_train"):
                tr = ", ".join(f"{k}: {v}" for k, v in train["class_counts_train"].items())
                te = ", ".join(f"{k}: {v}" for k, v in train["class_counts_test"].items())
                lines.append(f"- Class counts in training set: {tr}. In test set: {te}.")
            if train.get("imbalanced"):
                lines.append(f"- Data is imbalanced: the smallest class is {train['minority_share'] * 100:.1f}% of the rows. Class weighting was used so rare classes count more.")
                lines.append(f"- Because of this, scores below are balanced accuracy (average recall over the classes). Random guessing scores {train['metric_baseline']:.3f} on this metric.")
                lines.append(f"- Plain accuracy of always guessing the most common class is {train['majority_class_baseline_accuracy']:.3f}, so plain accuracy alone is misleading here.")
                if train.get("minority_count") is not None and train["minority_count"] < 50:
                    lines.append(f"- Warning: the smallest class has only {train['minority_count']} rows, so all scores are noisy.")
            elif train.get("majority_class_baseline_accuracy") is not None:
                lines.append(f"- Baseline: always guessing the most common class scores {train['majority_class_baseline_accuracy']:.3f} accuracy, so models should beat this.")

        for m in train["model_results"]:
            line = f"- {m['model_name']}: train {m['train_acc']:.3f}, test {m['test_acc']:.3f}, CV {m['cv_score']:.3f} (CV std {m['cv_std']:.3f}, lower std = more stable)"
            if train.get("imbalanced") and m.get("test_accuracy_plain") is not None:
                line += f", plain test accuracy {m['test_accuracy_plain']:.3f}"
            lines.append(line)
        lines.append(f"- Best model: {train['best_model']} (highest cross-validation {metric}).")

        best = next((m for m in train["model_results"] if m["model_name"] == train["best_model"]), None)
        if best and best.get("per_class"):
            lines.append(f"- Per-class test results for {train['best_model']}:")
            for cls, v in best["per_class"].items():
                lines.append(f"  - class '{cls}' ({v['test_rows']} test rows): precision {v['precision']:.3f}, recall {v['recall']:.3f}, F1 {v['f1']:.3f}")

        lines.append("- Note: preprocessing was fitted on the full dataset before the split, so test scores may be slightly optimistic.")

    lines.append("Ask me anything about these results below.")
    return "\n".join(lines)


def strip_key(obj, key="evaluation_summary"):
    """Remove a key everywhere inside a nested dict/list (returns a new copy)."""
    if isinstance(obj, dict):
        return {k: strip_key(v, key) for k, v in obj.items() if k != key}
    if isinstance(obj, list):
        return [strip_key(item, key) for item in obj]
    return obj


# ------------------- COMPULSORY PIPELINE STEPS (always run, in fixed order) ------------------- #

def _read_working(working_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(working_dir, "working_dataset.csv"))


def inspect_dataset(working_dir: str) -> dict:
    """Inspects the dataset (code runs it, result is given to the LLM in its prompt)."""
    df = _read_working(working_dir)
    result = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "sample_data": df.head(5).to_dict(orient="records"),
    }
    log_step(working_dir, "inspect_dataset_summary", {
        "shape": result["shape"],
        "dtypes": result["dtypes"],
        "missing_pct": result["missing_pct"],
        "total_null_cells": int(df.isnull().sum().sum()),
    })
    return result


def run_base_cleanup(working_dir: str, target: str) -> None:
    df = _read_working(working_dir)
    initial_shape = df.shape
    nulls_before = int(df.isnull().sum().sum())
    cols_before = set(df.columns)

    df, converted_cols = convert_numeric_strings(df)

    rows_dropped_missing_target = 0
    if target in df.columns:
        before = len(df)
        df = df.dropna(subset=[target])
        rows_dropped_missing_target = before - len(df)

    rows_before_missing = len(df)
    nulls_by_column = {c: int(n) for c, n in df.isnull().sum().items() if n > 0}
    df = run_base_preprocessing(df)
    rows_dropped_missing_handling = rows_before_missing - len(df)

    save_step_output(df, working_dir)
    log_step(working_dir, "tool_run_base_cleanup", {
        "initial_shape": {"rows": initial_shape[0], "columns": initial_shape[1]},
        "cleaned_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "nulls_before": nulls_before,
        "nulls_after": int(df.isnull().sum().sum()),
        "dropped_columns": list(cols_before - set(df.columns)),
        "columns_converted_to_numeric": converted_cols,
        "rows_dropped_because_target_missing": rows_dropped_missing_target,
        "missing_values_by_column_before_imputation": nulls_by_column,
        "rows_dropped_by_missing_value_handling": rows_dropped_missing_handling,
        "missing_value_rules": "numeric columns with more than 30% missing are dropped; text columns with under 5% missing lose those rows; other missing values are filled with median, mean or most frequent value",
        "actions": "converted number-like text to numeric, dropped rows with missing target, trimmed whitespace, converted inf to NaN, dropped empty columns, imputed missing values",
    })


def run_categorical(working_dir: str, target: str) -> None:
    df = _read_working(working_dir)
    shape_before = df.shape
    cols_before = list(df.columns)
    text_cols = [c for c in df.select_dtypes(include="object").columns if c != target]

    target_series = None
    if target in df.columns:
        target_series = df[target]
        df = df.drop(columns=[target])
    df = run_categorical_preprocessing(df)
    if target_series is not None:
        df[target] = target_series.values

    save_step_output(df, working_dir)
    new_cols = [c for c in df.columns if c not in cols_before]
    log_step(working_dir, "tool_preprocess_categorical_features", {
        "shape_before": {"rows": shape_before[0], "columns": shape_before[1]},
        "shape_after": {"rows": df.shape[0], "columns": df.shape[1]},
        "text_columns_encoded": text_cols,
        "new_columns_created_count": len(new_cols),
        "target_protected": target,
        "actions": "lowercased text, grouped rare categories into 'other', converted numeric strings, one-hot encoded text columns (target left unchanged)",
    })


def run_numeric(working_dir: str, target: str, task_type: str) -> None:
    df = _read_working(working_dir)
    shape_before = df.shape
    cols_before = set(df.columns)
    outlier_log = {}
    df = run_numerical_preprocessing(df, target_col=target, task_type=task_type, log=outlier_log)

    save_step_output(df, working_dir)
    capped = outlier_log.get("values_capped_by_column", {})
    log_step(working_dir, "tool_process_and_reduce_numeric_features", {
        "shape_before": {"rows": shape_before[0], "columns": shape_before[1]},
        "shape_after": {"rows": df.shape[0], "columns": df.shape[1]},
        "rows_removed_as_outliers": shape_before[0] - df.shape[0],
        "outlier_values_capped_by_column": capped,
        "outlier_rows_dropped_by_column": outlier_log.get("rows_dropped_by_column", {}),
        "outlier_values_capped_total": sum(capped.values()),
        "columns_removed": sorted(cols_before - set(df.columns)),
        "columns_removed_by_reason": outlier_log.get("removed_columns_by_reason", {}),
        "kept_columns": list(df.columns),
        "k_features_limit": 10,
        "pca_used": False,
        "actions": "dropped near-constant columns, removed or capped outliers (IQR), scaled with Yeo-Johnson, removed highly correlated columns (above 0.9), selected top features",
    })


def run_training(working_dir: str, target: str, task_type: str) -> None:
    from backend.graph.workflow import app   # imported here to avoid a circular import
    df = _read_working(working_dir)
    result = app.invoke({
        "working_dir": working_dir,
        "df": df,
        "task_type": task_type,
        "target": target,
        "model_name": None,
        "model_history": [],
        "evaluation_summary": None,
        "export_path": None,
    })
    history = result.get("model_history", [])
    first = history[0] if history else {}
    log_step(working_dir, "train_and_evaluate_models", {
        "task_type": task_type,
        "target": target,
        "rows_used": int(df.shape[0]),
        "features_used": [c for c in df.columns if c != target],
        "models_evaluated": len(history),
        "model_results": history,
        "best_model": result.get("model_name", "Unknown"),
        "best_model_selected_by": f"highest cross-validation score ({first.get('metric', 'accuracy')})",
        "known_limitations": [
            "Preprocessing (scaling, outlier handling, feature selection, missing-value filling) was fitted on the full dataset before the train/test split, so test scores may be slightly optimistic (data leakage).",
            "Models use default settings with no hyperparameter tuning.",
            "Scores come from a single train/test split plus 5-fold cross-validation.",
        ],
        "evaluation_summary": result.get("evaluation_summary", ""),
        "target_skew": first.get("target_skew"),
        "target_log_transformed": bool(first.get("target_log_transformed")),
        "label_mapping": first.get("label_mapping"),
        "majority_class_baseline_accuracy": first.get("majority_baseline"),
        "metric": first.get("metric"),
        "imbalanced": bool(first.get("imbalanced")),
        "minority_share": first.get("minority_share"),
        "minority_count": first.get("minority_count"),
        "class_weighting_used": bool(first.get("class_weighting_used")),
        "metric_baseline": first.get("metric_baseline"),
        "class_counts_train": first.get("class_counts_train"),
        "class_counts_test": first.get("class_counts_test"),
    })

def run_compulsory_steps(working_dir: str, target: str, task_type: str) -> None:
    """All compulsory steps in one function, fixed order. The LLM cannot skip any."""
    run_base_cleanup(working_dir, target)
    run_categorical(working_dir, target)
    run_numeric(working_dir, target, task_type)
    run_training(working_dir, target, task_type)