#  schema imports
from typing import Any, TypedDict, List, Dict, Optional
import pandas as pd

class AgentState(TypedDict):
    working_dir: Optional[str]
    df: pd.DataFrame  
    task_type: str
    target: str
    model_name: Optional[str]
    model_history: List[Dict[str, Any]]
    evaluation_summary: Optional[str]
    export_path: Optional[str]