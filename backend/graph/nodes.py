import os
import json
from typing import Literal, Dict, Any

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
        "cv_score": result['cross_val_score'],
        "cv_std": result['cv_std'],
        "metric": result['metric'],
        "test_accuracy_plain": result['test_accuracy_plain'],
        "per_class": result['per_class'],
        "target_skew": result['target_skew'],
        "target_log_transformed": result['target_log_transformed'],
        "label_mapping": result['label_mapping'],
        "majority_baseline": result['majority_baseline'],
        "imbalanced": result['imbalanced'],
        "minority_share": result['minority_share'],
        "minority_count": result['minority_count'],
        "class_weighting_used": result['class_weighting_used'],
        "metric_baseline": result['metric_baseline'],
        "class_counts_train": result['class_counts_train'],
        "class_counts_test": result['class_counts_test'],
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
    return {"model_name": best_entry['model_name'], "evaluation_summary": ""}


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