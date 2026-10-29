# backend/api_routes.py
import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from typing import Dict, Any

router = APIRouter()

@router.post("/api/get-columns")
async def get_columns(dataset: UploadFile = File(...), task_type: str = Form(...)):
    try:
        # Save uploaded file temporarily
        os.makedirs("/tmp/anlytx_uploads", exist_ok=True)
        file_path = os.path.join("/tmp/anlytx_uploads", dataset.filename)
        
        contents = await dataset.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            
        # Read columns based on extension
        if dataset.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif dataset.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return {"error": "Invalid file format. Upload CSV or Excel."}
            
        return {
            "columns": list(df.columns),
            "file_path": file_path,
            "task_type": task_type
        }
    except Exception as e:
        return {"error": str(e)}