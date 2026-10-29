import os
import json
import io
from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional,Any, Dict
import pandas as pd

from fastapi import FastAPI
from backend.api_routes import router as api_router

from fastmcp import FastMCP
mcp = FastMCP("Anlytx_AI")
mcp.app.include_router(api_router)

# Import required packages from other modules
from backend.graph.workflow import app
from backend.tools.utils import create_working_dir,save_step_output,ensure_csv_format

from backend.tools.preprocessing import (
    run_base_preprocessing,     run_categorical_preprocessing,     run_numerical_preprocessing,
    drop_datetime_columns,     extract_text_sentiment,     drop_irrelevant_columns )





# ------------------- 1. DISCOVERY & TASK SETUP TOOLS ------------------- #
@mcp.tool()
def inspect_dataset_summary(file_path: str) -> Dict[str, Any]:
    """CRITICAL: You MUST ALWAYS use this tool FIRST for any dataset or CSV uploaded in the chat. 
    DO NOT write your own Python code or use your local sandbox to read datasets. 
    You are strictly required to call this tool to inspect the data shape, columns, and summary stats.
    """
    processed_file_path = ensure_csv_format(file_path)
    if not os.path.exists(processed_file_path):
        return {"error": f"File not found at {processed_file_path}"}
    try:
        working_dir = create_working_dir(processed_file_path)
        df = pd.read_csv(processed_file_path)
        save_step_output(df, working_dir)
        return {
            "file_path": processed_file_path,
            "working_dir": working_dir,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "sample_data": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        return {"error": f"Failed to inspect file at {file_path}: {str(e)}"}


@mcp.tool()
def detect_task_and_target(file_path: str, user_task: Optional[str] = None,
                            user_target: Optional[str] = None) -> Dict[str, Any]:
    """Inspects a CSV file and validates or auto-detects task type and target column."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Failed to read file at {file_path}: {str(e)}"}
    columns = list(df.columns)

    if not user_task or not user_target:
        return {
            "message": "Please specify task ('classification' or 'regression') and target column.",
            "available_columns": columns,
            "sample_data": df.head(2).to_dict()
        }

    task = user_task.lower().strip()
    valid_tasks = ['classification', 'regression']
    if task not in valid_tasks:
        return {"error": f"Invalid task. Choose from {valid_tasks}"}

    column_list = {c.lower().strip(): c for c in columns}
    target_clean = user_target.lower().strip()
    if target_clean not in column_list:
        return {"error": f"Column '{user_target}' not found. Available columns: {columns}"}
    target_col = column_list[target_clean]
    return {"status": "success","task_type": task,"target": target_col,"message": f"Task set to '{task}' and target set to '{target_col}'."}

# ------------------- 2. COMPULSORY BASE PREPROCESSING TOOL ------------------- #

@mcp.tool()
def tool_run_base_cleanup(working_dir: str) -> Dict[str, Any]:
    """Runs compulsory baseline dataset cleaning in optimal sequence:
    1. Removes extra spaces / trims whitespaces
    2. Handles infinite values (converts inf to NaN)
    3. Drops completely empty or constant single-value columns
    4. Imputes remaining missing values
    """
    try:
        file_path = os.path.join(working_dir, "working_dataset.csv")
        df = pd.read_csv(file_path)
        initial_shape = df.shape
        # Executes the unified compulsory function
        df = run_base_preprocessing(df)
        out_path = save_step_output(df, working_dir)
        return {
            "status": "success",
            "output_path": out_path,
            "initial_shape": {"rows": initial_shape[0], "columns": initial_shape[1]},
            "cleaned_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "remaining_nulls": int(df.isnull().sum().sum())
        }
    except Exception as e:
        return {"error": f"Base cleanup failed: {str(e)}"}

@mcp.tool()
def tool_preprocess_categorical_features(working_dir: str) -> Dict[str, Any]:
    """MACRO TOOL 2: Cleans, groups rare categories, converts numeric strings, and encodes object columns."""
    try:
        file_path = os.path.join(working_dir, "working_dataset.csv")
        df = pd.read_csv(file_path)
        df = run_categorical_preprocessing(df)
        out_path = save_step_output(df, working_dir)
        return {
            "status": "success",
            "output_path": out_path,
            "resulting_shape": {"rows": df.shape[0], "columns": df.shape[1]}
        }
    except Exception as e:
        return {"error": f"Categorical preprocessing failed: {str(e)}"}


@mcp.tool()
def tool_process_and_reduce_numeric_features(working_dir: str, target_col: Optional[str] = None,
                                            task_type: Optional[str] = None, k: int = 10, use_pca: bool = False) -> Dict[str, Any]:
    """MACRO TOOL 3: Drops low variance, handles outliers, scales features, removes collinearity, and performs feature selection/PCA."""
    try:
        file_path = os.path.join(working_dir, "working_dataset.csv")
        df = pd.read_csv(file_path)
        df = run_numerical_preprocessing(df, target_col=target_col, task_type=task_type, k=k, use_pca=use_pca)
        out_path = save_step_output(df, working_dir)
        return {
            "status": "success",
            "output_path": out_path,
            "final_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "remaining_columns": list(df.columns)
        }
    except Exception as e:
        return {"error": f"Numerical processing failed: {str(e)}"}   



# ------------------- # 3. CONDITIONAL / DATA-DEPENDENT MCP TOOLS (Called selectively by Claude after data inspection) ------------------- #
@mcp.tool()
def tool_drop_datetime_columns(working_dir: str) -> Dict[str, Any]:
    """Detects and drops date/time columns if identified as non-predictive timestamps."""
    try:
        file_path = os.path.join(working_dir, "working_dataset.csv")
        df = pd.read_csv(file_path)
        initial_cols = df.columns.tolist()
        df = drop_datetime_columns(df)
        dropped = list(set(initial_cols) - set(df.columns))
        out_path = save_step_output(df, working_dir)
        return {"status": "success", "output_path": out_path, "dropped_columns": dropped}
    except Exception as e:
        return {"error": f"Failed to drop datetime columns: {str(e)}"}


@mcp.tool()
def tool_extract_text_sentiment(text_col: str, working_dir: str, model_type: str = "clef:27b") -> Dict[str, Any]:
    """Extracts sentiment score from a specific review or text feedback column."""
    try:
        file_path = os.path.join(working_dir, "working_dataset.csv")
        df = pd.read_csv(file_path)
        if text_col not in df.columns:
            return {"error": f"Column '{text_col}' not found."}
        df = extract_text_sentiment(df, text_col=text_col, model_type=model_type)
        out_path = save_step_output(df, working_dir)
        return {"status": "success", "output_path": out_path, "added_column": f"{text_col}_sentiment"}
    except Exception as e:
        return {"error": f"Failed to extract sentiment: {str(e)}"}


@mcp.tool()
def tool_drop_irrelevant_columns(cols_to_drop: List[str], working_dir: str) -> Dict[str, Any]:
    """Drops high-cardinality IDs, hashes, or unneeded text columns explicitly."""
    try:
        file_path = os.path.join(working_dir, "working_dataset.csv")
        df = pd.read_csv(file_path)
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols)
        out_path = save_step_output(df, working_dir)
        return {"status": "success", "output_path": out_path, "dropped": existing_cols}
    except Exception as e:
        return {"error": f"Failed to drop irrelevant columns: {str(e)}"}

# ------------------- 4. MODEL TRAINING & EVALUATION TOOL ------------------- #

@mcp.tool()
def train_and_evaluate_models(working_dir: str, task_type: str, target: str) -> str:
    """Runs the complete AutoML pipeline on the attached dataset using the established working_dir once target and task are confirmed."""
    try:
        file_path = os.path.join(working_dir, "working_dataset.csv")
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error reading CSV file at {file_path}: {e}"

    os.makedirs(working_dir, exist_ok=True)

    initial_state = {
        "working_dir": working_dir,
        "df": df,
        "task_type": task_type,
        "target": target,
        "model_name": None,
        "model_history": [],
        "evaluation_summary": None,
        "export_path": None,
    }

    try:
        result = app.invoke(initial_state)
        history = result.get("model_history", [])
        best_model = result.get("model_name", "Unknown")
        summary = result.get("evaluation_summary", "")
        return (
            f"Pipeline completed successfully!\n"
            f"Working Directory: {working_dir}\n"
            f"Models evaluated: {len(history)}\n"
            f"Recommended Best Model: {best_model}\n"
            f"Evaluation Summary: {summary}\n"
            f"Outputs are saved in '{working_dir}' folder."
        )
    except Exception as e:
        return f"Pipeline Execution Error: {e}"


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", 8000))
    mcp.run(transport="sse", host=host, port=port)