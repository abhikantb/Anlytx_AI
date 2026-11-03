import os
import pandas as pd
import io
import json
import datetime

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
        capped = num.get("outlier_values_capped", 0)
        lines.append(f"- Numeric step: {capped} outlier values capped (rows kept), {len(num['columns_removed'])} columns removed, final {num['shape_after']['columns']} columns.")
    train = steps.get("train_and_evaluate_models")
    if train:
        metric = "R2" if train["task_type"] == "regression" else "accuracy"
        lines.append(f"- Task: {train['task_type']}, target: {train['target']}, metric: {metric}.")
        if train.get("target_log_transformed"):
            lines.append(f"- Target was highly skewed (skew {train['target_skew']:.1f}), so models were trained on its log and predictions converted back.")
        if train.get("label_mapping"):
            mapping_text = ", ".join(f"{k} = {v}" for k, v in train["label_mapping"].items())
            lines.append(f"- Class labels used by the models: {mapping_text}.")
        if train.get("majority_class_baseline_accuracy") is not None:
            lines.append(f"- Baseline: always guessing the most common class scores {train['majority_class_baseline_accuracy']:.3f} accuracy, so models should beat this.")
        for m in train["model_results"]:
            lines.append(f"- {m['model_name']}: train {m['train_acc']:.3f}, test {m['test_acc']:.3f}, CV {m['cv_score']:.3f}")
        lines.append(f"- Best model: {train['best_model']} (highest cross-validation score).")
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