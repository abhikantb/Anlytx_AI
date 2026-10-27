import os
import pandas as pd

def create_working_dir(file_path: str) -> str:
    """Generates a unique project working directory under data/ and creates it."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    i = 1
    while os.path.exists(f"data/{base_name}_project{i}"):
        i += 1
    working_dir = f"data/{base_name}_project{i}"
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

def ensure_csv_format(file_path: str) -> str:
    """
    Checks if the input file is an Excel file. If so, reads it and converts it 
    into a CSV file within the same directory, returning the new CSV file path.
    If it's already a CSV, returns the path as-is.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        # Read Excel and save as CSV in the same location
        df = pd.read_excel(file_path)
        csv_path = os.path.splitext(file_path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    return file_path