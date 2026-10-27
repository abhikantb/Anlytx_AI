import os
import json
from typing import Literal, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# LLM model,observability imports
from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

from backend.graph.state import AgentState
from backend.tools.ml_models import train_and_evaluate_models, MODELS


def model_training_node(state: AgentState) -> Dict[str, Any]:
    # Executes training
    model_name = state.get('model_name') or 'RandomForest'
    result = train_and_evaluate_models(state['df'], state['task_type'],state['target'],model_name )
    model_result = {
        "model_name": model_name,
        "train_acc": result['training_accuracy'],
        "test_acc": result['testing_accuracy'],
        "cv_score": result['cross_val_score']
    }
    return {"model_history": state.get('model_history', []) + [model_result]}


def planner_node(state: AgentState) -> Dict[str, Any]:
    history = state.get('model_history', [])
    task = state['task_type']
    all_models = list(MODELS[task].keys()) # Get all models for the task from model_tools.py
    tried_models = {item['model_name'] for item in history}
    untried_models = [m for m in all_models if m not in tried_models]
    if untried_models:
        next_model = untried_models[0]
    else:
        next_model = all_models[0]  # Fallback if all are completed
    return {"model_name": next_model}


def check_status(state: AgentState) -> Literal["planner_node", "evaluation_node"]:
    task = state['task_type']
    if len(state.get('model_history', [])) >= len(MODELS[task]): # stop if we have tried all models
        return "evaluation_node"
    return "planner_node"


def evaluation_node(state: AgentState) -> Dict[str, Any]:
    history = state.get('model_history', [])
    if not history:
        raise ValueError("No model history available to evaluate.")

    best_entry = max(history, key=lambda x: x.get('cv_score', 0))
    best_model_name = best_entry['model_name']
    
    prompt = f"""Analyze these model performance metrics: {history}.
            1. Identify if any model shows signs of overfitting or underfitting.
            2. Explain why {best_model_name} is the best model based on the balance of testing accuracy and cross-validation score.
            """
    try:
        summary = llm.invoke(prompt).content
    except Exception as e:
        summary = f"LLM evaluation failed: {e}"
    return {"model_name": best_model_name, "evaluation_summary": summary}


def export_node(state: AgentState) -> Dict[str, Any]:
    working_dir = state["working_dir"]
    # Save cleaned dataset
    cleaned_data_path = os.path.join(working_dir, "cleaned_dataset.csv")
    if "df" in state and state["df"] is not None:
        state["df"].to_csv(cleaned_data_path, index=False)
    # Save pipeline metrics / summary json
    summary_path = os.path.join(working_dir, "pipeline_summary.json")
    summary_data = {
        "task_type": state.get("task_type"),
        "target": state.get("target"),
        "model_history": state.get("model_history", []),
        "model_name": state.get("model_name")
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    return {}