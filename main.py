import os
import json
from typing import List, Optional, Any, Dict
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from starlette.requests import Request
from starlette.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from fastmcp import FastMCP

mcp = FastMCP("Anlytx_AI")

from backend.tools.utils import (
    process_and_save_upload, save_step_output, log_step, build_summary, strip_key,
    inspect_dataset, run_compulsory_steps,
)
from backend.tools.preprocessing import drop_datetime_columns, extract_text_sentiment


# ------------------- 1. UPLOAD ROUTE ------------------- #
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


# ------------------- 2. OPTIONAL TOOLS (the LLM decides if they are needed) ------------------- #
@mcp.tool()
def tool_drop_datetime_columns(working_dir: str, target: Optional[str] = None) -> Dict[str, Any]:
    """Detects and drops date/time columns if identified as non-predictive timestamps. Never drops the target."""
    try:
        df = pd.read_csv(os.path.join(working_dir, "working_dataset.csv"))
        initial_cols = df.columns.tolist()
        target_series = None
        if target and target in df.columns:
            target_series = df[target]
            df = df.drop(columns=[target])
        df = drop_datetime_columns(df)
        if target_series is not None:
            df[target] = target_series.values
        dropped = [c for c in initial_cols if c not in df.columns]
        save_step_output(df, working_dir)
        log_step(working_dir, "tool_drop_datetime_columns", {
            "dropped_columns": dropped,
            "shape_after": {"rows": df.shape[0], "columns": df.shape[1]},
            "actions": "dropped columns that look like dates or timestamps (they rarely help prediction)",
        })
        return {"status": "success", "dropped_columns": dropped}
    except Exception as e:
        return {"error": f"Failed to drop datetime columns: {str(e)}"}


@mcp.tool()
def tool_extract_text_sentiment(text_col: str, working_dir: str) -> Dict[str, Any]:
    """Replaces a long free-text column (reviews, comments, sentences) with a numeric sentiment score (-1 to +1). English text only."""
    try:
        df = pd.read_csv(os.path.join(working_dir, "working_dataset.csv"))
        if text_col not in df.columns:
            return {"error": f"Column '{text_col}' not found."}
        df = extract_text_sentiment(df, text_col=text_col)
        save_step_output(df, working_dir)
        log_step(working_dir, "tool_extract_text_sentiment", {
            "text_column": text_col,
            "new_column": f"{text_col}_sentiment",
            "actions": f"replaced text column '{text_col}' with a sentiment score from -1 (negative) to +1 (positive) using VADER",
        })
        return {"status": "success", "added_column": f"{text_col}_sentiment"}
    except Exception as e:
        return {"error": f"Failed to extract sentiment: {str(e)}"}


@mcp.tool()
def tool_drop_irrelevant_columns(cols_to_drop: List[str], working_dir: str, target: Optional[str] = None) -> Dict[str, Any]:
    """Drops useless columns such as IDs, names, hashes, links or unneeded text. Never drops the target."""
    try:
        df = pd.read_csv(os.path.join(working_dir, "working_dataset.csv"))
        existing_cols = [c for c in cols_to_drop if c in df.columns and c != target]
        df = df.drop(columns=existing_cols)
        save_step_output(df, working_dir)
        log_step(working_dir, "tool_drop_irrelevant_columns", {
            "dropped_columns": existing_cols,
            "shape_after": {"rows": df.shape[0], "columns": df.shape[1]},
            "actions": "dropped ID-like or free-text columns chosen by the agent after looking at the data",
        })
        return {"status": "success", "dropped": existing_cols}
    except Exception as e:
        return {"error": f"Failed to drop irrelevant columns: {str(e)}"}


# ------------------- 3. MAIN PIPELINE ROUTE ------------------- #
@mcp.custom_route("/api/run-pipeline", methods=["POST"])
async def run_pipeline(request: Request):
    try:
        body2 = await request.json()
        working_dir = body2["working_dir"]
        task_type = body2["task_type"]
        target = body2["target"]

        # Step A: code inspects the dataset (always)
        info = inspect_dataset(working_dir)
        short_info = {
            "shape": info["shape"],
            "dtypes": info["dtypes"],
            "columns_with_missing_pct": {c: v for c, v in info["missing_pct"].items() if v > 0},
            "sample_rows": [{k: str(v)[:80] for k, v in row.items()} for row in info["sample_data"][:3]],
        }

        # Step B: the LLM only decides the OPTIONAL steps
        optional_tools = [tool_drop_datetime_columns, tool_extract_text_sentiment, tool_drop_irrelevant_columns]
        tool_map = {
            "tool_drop_datetime_columns": tool_drop_datetime_columns,
            "tool_extract_text_sentiment": tool_extract_text_sentiment,
            "tool_drop_irrelevant_columns": tool_drop_irrelevant_columns,
        }
        primary = ChatGroq(model="openai/gpt-oss-120b", temperature=0).bind_tools(optional_tools).with_retry(stop_after_attempt=3)
        backup1 = ChatGroq(model="qwen/qwen3.8-27b", temperature=0).bind_tools(optional_tools).with_retry(stop_after_attempt=3)
        backup2 = ChatGroq(model="openai/gpt-oss-20b", temperature=0).bind_tools(optional_tools).with_retry(stop_after_attempt=3)
        agent_llm = primary.with_fallbacks([backup1, backup2])

        system_instruction = (
            "You are an expert AutoML data scientist. You get a summary of a dataset.\n"
            "Call a tool ONLY if the data needs it:\n"
            "- tool_extract_text_sentiment: for a long free-text column (reviews, comments, sentences).\n"
            "- tool_drop_irrelevant_columns: for useless columns (IDs, names, links, hashes).\n"
            "- tool_drop_datetime_columns: if date or time columns exist.\n"
            "Never touch the target column. Cleaning, encoding, scaling and training run automatically "
            "after you, so do not do them. When you are finished, or if nothing is needed, reply with the single word DONE."
        )
        user_prompt = (
            f"working_dir: '{working_dir}'\n"
            f"Task type: '{task_type}'\n"
            f"Target column: '{target}'\n"
            f"Dataset summary: {json.dumps(short_info, default=str)}"
        )
        messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_prompt)]

        for _ in range(4):
            response = agent_llm.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                fn = tool_map.get(tc["name"])
                if fn is None:
                    content = f"Unknown tool: {tc['name']}"
                elif tc["args"].get("text_col") == target:
                    content = "Error: the target column cannot be used here."
                else:
                    args = dict(tc["args"])
                    args["working_dir"] = working_dir
                    if tc["name"] != "tool_extract_text_sentiment":
                        args["target"] = target
                    content = json.dumps(fn(**args), default=str)[:1500]
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})

        # Step C: code runs all compulsory steps in a fixed order (cannot be skipped)
        run_compulsory_steps(working_dir, target, task_type)

        # Step D: build the response
        report_path = os.path.join(working_dir, "report.json")
        report = {}
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
        return JSONResponse({
            "status": "success",
            "working_dir": working_dir,
            "final_summary": build_summary(report),
            "report": strip_key(report),
        })
    except Exception as e:
        return JSONResponse({"error": f"Pipeline execution failed: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", 8000))
    mcp.run(transport="sse", host=host, port=port)