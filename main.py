import os
import json
from typing import List, Optional, Any, Dict
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from starlette.requests import Request
from starlette.responses import JSONResponse,Response
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from fastmcp import FastMCP

mcp = FastMCP("Anlytx_AI")

def allow_browser(handler):
    async def wrapper(request: Request):
        if request.method == "OPTIONS":
            response = Response(status_code=204)   # the browser's question gets an empty "ok"
        else:
            response = await handler(request)       # a normal request runs the real route
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    return wrapper



from backend.tools.utils import (
    process_and_save_upload, save_step_output, log_step, build_summary, strip_key,
    inspect_dataset, run_compulsory_steps,reset_working_data
)
from backend.tools.preprocessing import drop_datetime_columns, extract_text_sentiment


# ------------------- 1. UPLOAD ROUTE ------------------- #
@mcp.custom_route("/api/processing_data", methods=["POST","OPTIONS"])
@allow_browser
async def processing_data(request: Request):
    try:
        body = await request.body()
        filename = request.query_params["filename"]
        task_type = request.query_params["task_type"]
        working_dir, columns = process_and_save_upload(body, filename)
        # first 5 rows as text, for the sample-rows box on the page (empty cells stay empty)
        head = pd.read_csv(os.path.join(working_dir, "working_dataset.csv"), nrows=5, dtype=str, keep_default_na=False)
        return JSONResponse({
            "status": "success",
            "working_dir": working_dir,
            "columns": columns,
            "preview": {"columns": list(head.columns), "rows": head.values.tolist()},
            "task_type": task_type
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ------------------- 2. OPTIONAL TOOLS (the LLM decides if they are needed) ------------------- #

@mcp.tool()
def tool_drop_id_columns(id_cols: List[str], working_dir: str, target: Optional[str] = None) -> Dict[str, Any]:
    """Drops ID columns (row numbers, customer/account codes, any column that only identifies a row). Judge by the sample values and unique_ratio, not only the column name. Never drops the target."""
    try:
        df = pd.read_csv(os.path.join(working_dir, "working_dataset.csv"))
        existing_cols = [c for c in id_cols if c in df.columns and c != target]
        df = df.drop(columns=existing_cols)
        save_step_output(df, working_dir)
        log_step(working_dir, "tool_drop_id_columns", {
            "dropped_columns": existing_cols,
            "shape_after": {"rows": df.shape[0], "columns": df.shape[1]},
            "actions": "dropped ID columns chosen by the agent after looking at the sample values",
        })
        return {"status": "success", "dropped": existing_cols}
    except Exception as e:
        return {"error": f"Failed to drop ID columns: {str(e)}"}

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
    """Drops useless text columns such as names, hashes, links or unneeded long text. Do not use for ID columns. Never drops the target."""
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

@mcp.tool()
def tool_drop_auto_date_columns(date_cols: List[str], working_dir: str, target: Optional[str] = None) -> Dict[str, Any]:
    """Drops date/time columns (dates, timestamps, times of day) that are still in the data. Judge by the sample values, not only the column name. Plain numbers like age, price, visits, or day-of-week counts are NOT dates, keep them. Keep the column if unsure. Never drops the target."""
    try:
        df = pd.read_csv(os.path.join(working_dir, "working_dataset.csv"))
        existing_cols = [c for c in date_cols if c in df.columns and c != target]
        df = df.drop(columns=existing_cols)
        save_step_output(df, working_dir)
        log_step(working_dir, "tool_drop_auto_date_columns", {
            "dropped_columns": existing_cols,
            "shape_after": {"rows": df.shape[0], "columns": df.shape[1]},
            "actions": "dropped leftover date/time columns chosen by the agent after looking at the sample values",
        })
        return {"status": "success", "dropped": existing_cols}
    except Exception as e:
        return {"error": f"Failed to drop date columns: {str(e)}"}

# ------------------- 3. MAIN PIPELINE ROUTE ------------------- #
@mcp.custom_route("/api/run-pipeline", methods=["POST", "OPTIONS"])
@allow_browser
async def run_pipeline(request: Request):
    try:
        body2 = await request.json()
        working_dir = body2["working_dir"]
        task_type = body2["task_type"]
        target = body2["target"]
        date_column = body2.get("date_column")

        reset_working_data(working_dir)

        # Step 0: drop the date/time column chosen by the user (preprocessing function does the work)
        df = pd.read_csv(os.path.join(working_dir, "working_dataset.csv"))
        df_after = drop_datetime_columns(df, date_column, target)
        dropped_dt = [c for c in df.columns if c not in df_after.columns]
        if dropped_dt:
            save_step_output(df_after, working_dir)
            log_step(working_dir, "tool_drop_datetime_columns", {
                "dropped_columns": dropped_dt,
                "shape_after": {"rows": df_after.shape[0], "columns": df_after.shape[1]},
                "actions": "dropped the date/time column chosen by the user",
            })
        df = df_after

        # Step A: code inspects the dataset (always)
        info = inspect_dataset(working_dir)
        short_info = {
            "shape": info["shape"],
            "dtypes": info["dtypes"],
            "unique_ratio": {c: round(df[c].nunique() / len(df), 2) for c in df.columns},
            "columns_with_missing_pct": {c: v for c, v in info["missing_pct"].items() if v > 0},
            "sample_rows": [{k: str(v)[:80] for k, v in row.items()} for row in info["sample_data"][:5]],
        }

        # Step B: the LLM only decides the OPTIONAL steps
        optional_tools = [tool_drop_id_columns, tool_drop_auto_date_columns, tool_extract_text_sentiment, tool_drop_irrelevant_columns]
        tool_map = {
            "tool_drop_id_columns": tool_drop_id_columns,
            "tool_drop_auto_date_columns": tool_drop_auto_date_columns,
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
            "- tool_drop_id_columns: for ID columns (row numbers, customer or account codes). Judge by the sample "
            "values and unique_ratio, not only the name. Do NOT drop decimal numbers like price or amounts, "
            "even if all their values are unique. If unsure, keep the column.\n"
            "- tool_drop_auto_date_columns: ONLY for columns that still hold real dates or times in the sample values "
            "(like calendar dates or timestamps). Plain numbers like age, price, visits or day counts are NOT dates. "
            "If unsure, keep the column.\n"
            "- tool_extract_text_sentiment: for a long free-text column (reviews, comments, sentences).\n"
            "- tool_drop_irrelevant_columns: for other useless text columns (names, links, hashes).\n"
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

# ------------------- 4. CHAT ROUTE ------------------- #
# chat with Jarvis using free OpenRouter models to save Groq limits
@mcp.custom_route("/api/chat", methods=["POST", "OPTIONS"])
@allow_browser
async def chat_handler(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        working_dir = data.get("working_dir")
        history = data.get("history", [])

        if not question:
            return JSONResponse({"answer": "Please ask a question about your dataset or analysis."})

        # read summary report if available to ground the AI answers
        summary_text = ""
        report_path = os.path.join(working_dir, "report.json") if working_dir else ""
        if report_path and os.path.exists(report_path):
            with open(report_path, "r") as f:
                report_data = json.load(f)
            summary_text = build_summary(report_data)

        # system instructions: plain explanation, ground on report, keep it brief
        system_rules = (
            "You are Jarvis, an expert and friendly AutoML assistant for Anlytx AI.\n"
            "Your role is to answer questions strictly about the dataset, models trained, cross-validation metrics, "
            "feature importance, and known limitations from the report provided.\n"
            "Rules:\n"
            "1. Explain in simple, clear terms without hardcoding.\n"
            "2. Keep answers concise and direct.\n"
            "3. If the question asks why a metric or model was picked, cite facts from the summary.\n"
            "4. If something is unknown or not in the report, admit it plainly."
        )

        messages = [SystemMessage(content=system_rules)]

        if summary_text:
            messages.append(SystemMessage(content=f"Current Pipeline Run Summary:\n{summary_text}"))

        # append conversation memory from the browser
        for msg in history:
            role = msg.get("role")
            text = msg.get("text", "")
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "bot":
                messages.append(SystemMessage(content=f"Assistant: {text}"))

        messages.append(HumanMessage(content=question))

        # Free OpenRouter models for chat (zero Groq token usage)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"

        primary = ChatOpenAI(
            model="nvidia/nemotron-3-ultra:free",
            openai_api_base=base_url,
            openai_api_key=openrouter_key,
            temperature=0.2,
        ).with_retry(stop_after_attempt=2)

        backup1 = ChatOpenAI(
            model="nvidia/nemotron-3.5-lightning:free",
            openai_api_base=base_url,
            openai_api_key=openrouter_key,
            temperature=0.2,
        ).with_retry(stop_after_attempt=2)

        backup2 = ChatOpenAI(
            model="cohere/north-mini-code:free",
            openai_api_base=base_url,
            openai_api_key=openrouter_key,
            temperature=0.2,
        ).with_retry(stop_after_attempt=2)

        llm = primary.with_fallbacks([backup1, backup2])

        ai_response = llm.invoke(messages)
        return JSONResponse({"status": "success", "answer": ai_response.content})

    except Exception as e:
        return JSONResponse({"answer": f"Sorry, I ran into an error while processing: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", 8000))
    mcp.run(transport="sse", host=host, port=port)