import os
import pandas as pd
import io

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
        df = pd.read_csv(io.BytesIO(body_bytes))
    else:
        df = pd.read_excel(io.BytesIO(body_bytes))
    working_dir = create_working_dir(filename)
    save_step_output(df, working_dir, "working_dataset.csv")
    return working_dir, list(df.columns)