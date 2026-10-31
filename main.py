import os
import json
from starlette.requests import Request
from starlette.responses import JSONResponse
import io
from typing import List, Optional,Any, Dict
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from fastmcp import FastMCP
mcp = FastMCP("Anlytx_AI")

# Import required packages from other modules
from backend.graph.workflow import app
from backend.tools.utils import create_working_dir,save_step_output,process_and_save_upload,log_step
from backend.tools.preprocessing import (
    run_base_preprocessing,     run_categorical_preprocessing,     run_numerical_preprocessing,
    drop_datetime_columns,     extract_text_sentiment,     drop_irrelevant_columns ,    convert_numeric_strings)


# ------------------- 1. DISCOVERY & TASK SETUP TOOLS ------------------- #
@mcp.custom_route("/api/processing_data", methods=["POST"])
async def processing_data(request: Request):
    try:
        body = await request.body()
        filename = request.query_params["filename"]
        task_type = request.query_params["task_type"]
        working_dir, columns = process_and_save_upload(body, filename)
        return JSONResponse({
            "status": "success",
            "working_dir": working_dir,
            "columns": columns,
            "task_type": task_type
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@mcp.custom_route("/api/run-pipeline", methods=["POST"])
async def run_pipeline(request: Request):
    try:
        body2 = await request.json()
        working_dir = body2["working_dir"]
        task_type = body2["task_type"]
        target = body2["target"]
        tools = [inspect_dataset_summary,tool_run_base_cleanup,tool_preprocess_categorical_features,train_and_evaluate_models,
            tool_process_and_reduce_numeric_features,tool_drop_datetime_columns,tool_extract_text_sentiment,tool_drop_irrelevant_columns,]
        # Use agent_llm to avoid shadowing the global llm instance
        primary = ChatGroq(model="openai/gpt-oss-120b", temperature=0).bind_tools(tools).with_retry(stop_after_attempt=3)
        backup1 = ChatGroq(model="openai/gpt-oss-20b", temperature=0).bind_tools(tools).with_retry(stop_after_attempt=3)
        backup2 = ChatGroq(model="qwen/qwen3.8-27b", temperature=0).bind_tools(tools).with_retry(stop_after_attempt=3)
        agent_llm = primary.with_fallbacks([backup1, backup2])
        
        system_instruction = (
            "You are an expert AutoML data scientist. You must:\n"
            "1. ALWAYS call inspect_dataset_summary first using the provided working_dir.\n"
            "2. Carefully read the sample data rows, column types, shape, and null percentages.\n"
            "3. Decide which cleaning and preprocessing tools to call based on the observed data.\n"
            "4. Finally, call train_and_evaluate_models to finish the training process.\n"
            "5. Your final answer must be plain text only: no markdown, no tables, no asterisks. Use short lines starting with a dash, and keep it under 200 words."
        )
        user_prompt = (
            f"Dataset working_dir: '{working_dir}'\n"
            f"Task Type: '{task_type}'\n"
            f"Target Column: '{target}'\n"
            "Inspect the dataset summary and sample rows, execute necessary preprocessing tools, and train the models."
        )
        messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_prompt)]
        tool_map = {
            "inspect_dataset_summary": inspect_dataset_summary,
            "tool_run_base_cleanup": tool_run_base_cleanup,
            "tool_preprocess_categorical_features": tool_preprocess_categorical_features,
            "tool_process_and_reduce_numeric_features": tool_process_and_reduce_numeric_features,
            "tool_drop_datetime_columns": tool_drop_datetime_columns,
            "tool_extract_text_sentiment": tool_extract_text_sentiment,
            "tool_drop_irrelevant_columns": tool_drop_irrelevant_columns,
            "train_and_evaluate_models": train_and_evaluate_models
        }
        while True:
            response = agent_llm.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tool_call in response.tool_calls:
                tool_fn = tool_map.get(tool_call["name"])
                # Execute tool directly
                tool_output = tool_fn(**tool_call["args"])
                raw = json.dumps(tool_output, default=str) if isinstance(tool_output, dict) else str(tool_output)
                content = raw if len(raw) <= 1500 else raw[:1500] + " ...[truncated]"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": content
                })

        report_path = os.path.join(working_dir, "report.json")
        report = {}
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
        return JSONResponse({"status": "success","working_dir": working_dir,"final_summary": messages[-1].content,"report": report})
    except Exception as e:
        return JSONResponse({"error": f"Pipeline execution failed: {str(e)}"}, status_code=500)

@mcp.custom_route("/api/run-pipeline-stream", methods=["POST"])
async def run_pipeline_stream(request: Request):
    # Instead of waiting for the end, yield updates as they happen:
    async def event_generator():
        yield "data: 🔍 Inspecting dataset...\n\n"
        # run inspect_dataset_summary
        yield "data: 🧹 Running base cleanup...\n\n"
        # run tool_run_base_cleanup
        yield "data: 🤖 Training models...\n\n"
        # stream final LLM summary word-by-word
        for chunk in agent_llm.stream(messages):
            yield f"data: {chunk.content}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")



@mcp.tool()
def inspect_dataset_summary(working_dir: str) -> Dict[str, Any]:
    """CRITICAL: You MUST ALWAYS use this tool FIRST on the provided working_dir.
    Inspects the dataset shape, dtypes, null percentages, and first sample rows 
    to decide which preprocessing tools to execute.
    """
    file_path = os.path.join(working_dir, "working_dataset.csv")
    if not os.path.exists(file_path):
        return {"error": f"Working file not found at {file_path}"}
    try:
        df = pd.read_csv(file_path)
        result = {
            "working_dir": working_dir,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "sample_data": df.head(5).to_dict(orient="records")
        }
        # Save the facts (without sample rows) to report.json
        log_step(working_dir, "inspect_dataset_summary", {
            "shape": result["shape"],
            "dtypes": result["dtypes"],
            "missing_pct": result["missing_pct"],
            "total_null_cells": int(df.isnull().sum().sum()),
        })
        return result
    except Exception as e:
        return {"error": f"Failed to inspect working dataset: {str(e)}"}


# ------------------- 2. COMPULSORY BASE PREPROCESSING TOOL ------------------- #

@mcp.tool()
def tool_run_base_cleanup(working_dir: str, target: Optional[str] = None) -> Dict[str, Any]:
    """Runs compulsory baseline dataset cleaning in optimal sequence:
    1. Converts number-like text columns (e.g. '4,25,000', '45,000 kms') to numbers
    2. Drops rows where the target column is empty (never imputes the target)
    3. Trims whitespace, handles infinite values, drops empty columns, imputes missing values
    Always pass the target column name in `target`.
    """
    try:
        file_path = os.path.join(working_dir, "working_dataset.csv")
        df = pd.read_csv(file_path)
        initial_shape = df.shape
        nulls_before = int(df.isnull().sum().sum())
        cols_before = set(df.columns)

        df, converted_cols = convert_numeric_strings(df)

        rows_dropped_missing_target = 0
        if target and target in df.columns:
            before = len(df)
            df = df.dropna(subset=[target])
            rows_dropped_missing_target = before - len(df)

        df = run_base_preprocessing(df)

        out_path = save_step_output(df, working_dir)
        dropped_cols = list(cols_before - set(df.columns))
        remaining_nulls = int(df.isnull().sum().sum())

        log_step(working_dir, "tool_run_base_cleanup", {
            "initial_shape": {"rows": initial_shape[0], "columns": initial_shape[1]},
            "cleaned_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "nulls_before": nulls_before,
            "nulls_after": remaining_nulls,
            "dropped_columns": dropped_cols,
            "columns_converted_to_numeric": converted_cols,
            "rows_dropped_because_target_missing": rows_dropped_missing_target,
            "actions": "converted number-like text to numeric, dropped rows with missing target, trimmed whitespace, converted inf to NaN, dropped empty columns, imputed missing values",
        })

        return {
            "status": "success",
            "output_path": out_path,
            "initial_shape": {"rows": initial_shape[0], "columns": initial_shape[1]},
            "cleaned_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "remaining_nulls": remaining_nulls,
            "columns_converted_to_numeric": converted_cols
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

        log_step(working_dir, "train_and_evaluate_models", {
            "task_type": task_type,
            "target": target,
            "rows_used": int(df.shape[0]),
            "features_used": [c for c in df.columns if c != target],
            "models_evaluated": len(history),
            "model_results": history,
            "best_model": best_model,
            "best_model_selected_by": "highest cross-validation score",
            "evaluation_summary": summary,
        })

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