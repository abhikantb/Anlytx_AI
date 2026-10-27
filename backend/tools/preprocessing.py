import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import requests
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# ------------------- 1. General Features Preprocessing ------------------- #

# 1.1 Drop empty and null columns
def drop_empty_and_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops completely empty or 100% null columns before processing."""
    df = df.copy()
    null_cols = [col for col in df.columns if df[col].isnull().all() or len(df[col].dropna()) == 0]
    if null_cols:
        df = df.drop(columns=null_cols)
    return df


# 1.2 Date/time Features Preprocessing
def drop_datetime_columns(df: pd.DataFrame, sample_size: int = 5, threshold: float = 0.8) -> pd.DataFrame:
    """
    Intelligently detects and drops date, time, timestamp, or duration columns 
    by checking if sample values parse successfully into datetimes using pd.to_datetime.
    """
    df = df.copy()
    drop_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        # 1. Quick name check for explicit timestamp/date keywords
        if any(kw in col_lower for kw in ['timestamp', 'datetime', 'dob']):
            drop_cols.append(col)
            continue
        # 2. Sample-based parsing check for object/string or numeric date columns
        sample = df[col].dropna().head(sample_size)
        if len(sample) == 0:
            continue
        try:
            # Attempt parsing sample values with errors='coerce' (invalid dates become NaT)
            parsed = pd.to_datetime(sample, errors='coerce')
            success_ratio = parsed.notna().mean()
            # If a high percentage of the sample parses successfully as a date, drop it
            if success_ratio >= threshold:
                drop_cols.append(col)
        except Exception:
            # If pd.to_datetime fails entirely on the sample type, it's safe (not a date)
            continue
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)))
    return df

#1.3 handle infinite values
def handle_infinite_values(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
 
# 1.4 remove extra spaces
def remove_extra_spaces(df:pd.DataFrame):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("", pd.NA)
    return df

# 1.5 Handles all missing values
def handle_missing_values(df: pd.DataFrame, num_drop_thresh: float = 0.3, obj_drop_row_thresh: float = 0.05) -> pd.DataFrame:
    """
    Unified missing value handler for both numeric and categorical columns.
    Assumes 100% empty columns are already dropped by drop_empty_and_null_columns.
    """
    df = df.copy()
    
    # 1. Handle Numerical Missing Values
    for col in df.select_dtypes('number').columns:
        ratio = df[col].isnull().mean()
        if ratio == 0:
            continue
        if ratio > num_drop_thresh:
            df = df.drop(columns=[col])
            continue
        # Fill missing values based on skewness
        if abs(df[col].skew()) > 1:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())

    # 2. Handle Object/Categorical Missing Values
    for col in df.select_dtypes(include='object').columns:
        missing_pct = df[col].isnull().mean()
        if missing_pct == 0:
            continue
        if missing_pct < obj_drop_row_thresh:
            df = df.dropna(subset=[col])
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("missing")
                
    return df

# ------------------- 2. Categorical Features Preprocessing ------------------- #

# 2.1 lowercase all categories
def standardize_categories(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower()
    return df

# 2.2 handle cardinality and rare categories
def handle_categories(df, threshold=0.1, max_categories=5):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        freq = df[col].value_counts(normalize=True)
        # rare logic
        if freq.max() > threshold:
            rare = freq[freq <= threshold].index
            df[col] = df[col].replace(rare, "other")
        # limit categories
        if df[col].nunique() > max_categories:
            top = df[col].value_counts().nlargest(max_categories).index
            df[col] = df[col].where(df[col].isin(top), "other")
    return df


# 2.3 text sentiment extraction if any feedback/comment type columns exist
def extract_text_sentiment(df: pd.DataFrame, text_col: str, model_type: str = "clef:27b") -> pd.DataFrame:
    """
    Extracts numeric sentiment probability (0.0 to 1.0) from a text column 
    using Ollama's /v1/systemone endpoint with the clef:27b model.
    """
    df = df.copy()
    if text_col not in df.columns:
        return df
    api_key = os.getenv("OLLAMA_API_KEY")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{ollama_host}/v1/systemone"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    def get_sentiment_score(text: str) -> float:
        if pd.isna(text) or str(text).strip() == "":
            raise ValueError(f"Empty row found in text column '{text_col}'.")
        payload = {
            "model": model_type,
            "state": {"text": text},
            "questions": {
                "sentiment": {
                    "type": "noul",
                    "instructions": "Is the sentiment of this text positive?"
                }
            }
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            res_data = response.json()
            return res_data.get("answers", {}).get("sentiment", {}).get("noul", 0.5)
        except Exception as e:
            raise RuntimeError(f"Ollama API request failed: {e}")
    df[text_col] = df[text_col].apply(get_sentiment_score)
    return df


# 2.4 drop irrelevant categorical columns
def drop_irrelevant_columns(df, unique_thresh=0.95, text_len_thresh=50, space_thresh=2, ratio_thresh=0.8):
    df = df.copy()
    drop_cols = []
    for col in df.select_dtypes(include='object').columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > unique_thresh:
            drop_cols.append(col)
            continue
        avg_len = df[col].astype(str).str.len().mean()
        if avg_len > text_len_thresh:
            drop_cols.append(col)
            continue
        space_ratio = df[col].astype(str).str.count(' ').gt(space_thresh).mean()
        if space_ratio >= ratio_thresh:
            drop_cols.append(col)
            continue
        link_ratio = df[col].astype(str).str.contains(r'http|www', case=False, regex=True).mean()
        if link_ratio >= ratio_thresh:
            drop_cols.append(col)
    df = df.drop(columns=list(set(drop_cols)))
    return df

# 2.5 convert numeric columns
def convert_numeric(df):
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            print(f"Convert to numeric failed for column '{col}': {e}")
    return df

# 2.6 encoding categorical varibales
import pandas as pd
def encode_object_columns(df):
    df = df.copy()
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True, dtype=int)
    return df


# ------------------- 3. Numerical Features Preprocessing ------------------- #

# 3.1 drop constant or low variance columns
def drop_low_variance(df, threshold=0.99):
    df = df.copy()
    for col in df.columns:
        if df[col].value_counts(normalize=True).iloc[0] >= threshold:
            df = df.drop(columns=col)
    return df

# 3.2 remove outliers
def handle_outliers(df: pd.DataFrame, limit: float = 0.01) -> pd.DataFrame:
    """Detects outliers on raw features using IQR. 
    Drops rows if outlier count is tiny (<= limit), otherwise clips/caps values."""
    df = df.copy()
    for col in df.columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (df[col] < low) | (df[col] > high)
        ratio = outlier_mask.mean()
        # Drop only if outliers represent a tiny fraction (e.g. <= 1%)
        if 0 < ratio <= limit:
            df = df[~outlier_mask]
        elif ratio > limit:
            # Cap extreme values to preserve all rows
            df[col] = df[col].clip(lower=low, upper=high)
    return df

# 3.3 Scaling
from sklearn.preprocessing import PowerTransformer
def scale_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Applies PowerTransformer (Yeo-Johnson) to stabilize variance and standardize all numeric features."""
    X = X.copy()
    if len(X.columns) > 0:
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        pt.fit(X_train)
        X[X.columns] = pt.transform(X[X.columns])
    return X

# -------------------4. Feature selection ------------------- #

#4.1 remove highly correlated numerical features
def remove_correlated_features(df, num_cols, threshold=0.9):
    df = df.copy()
    corr_matrix = df[num_cols].corr().abs()
    drop_cols = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[i]
                drop_cols.add(colname)
    df = df.drop(columns=list(drop_cols))
    return df

# 4.2 feature selection
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
def select_best_features(X, y, task_type, k: int = 10):
    X = X.copy()
    k_actual = min(k, X.shape[1])
    if task_type == "classification":
        selector = SelectKBest(score_func=f_classif, k=k_actual) # annova 
    else:
        selector = SelectKBest(score_func=f_regression, k=k_actual) # correlation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selector.fit(X_train, y_train)
    X_new = selector.transform(X)
    selected_cols = X.columns[selector.get_support()]
    X_new = pd.DataFrame(X_new, columns=selected_cols, index=X.index)
    return X_new

# 4.3 apply PCA
from sklearn.decomposition import PCA
def apply_pca(X, n_components=0.95):
    X = X.copy()
    pca = PCA(n_components=n_components)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    pca.fit(X_train)
    X_pca = pca.transform(X)
    cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=cols, index=X.index)
    return X_pca


# ------------------- A. BASE Preprocessing FUNCTION ------------------- #
def run_base_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Runs all non-negotiable compulsory base cleaning functions sequentially."""
    df = remove_extra_spaces(df)
    df = handle_infinite_values(df)
    df = drop_empty_and_null_columns(df)
    df = handle_missing_values(df)
    return df


# ------------------- B. Grouped Categorical Processing ------------------- #
def run_categorical_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Sequential categorical pipeline: Standardize -> Group Rare -> Convert Numeric Strings -> Encode."""
    df = standardize_categories(df)
    df = handle_categories(df)
    df = convert_numeric(df)
    df = encode_object_columns(df)
    return df

# ------------------- C. Numerical Feature Engineering ------------------- #
def run_numerical_preprocessing(df: pd.DataFrame, target_col: Optional[str] = None, task_type: Optional[str] = None, k: int = 10, use_pca: bool = False, n_components: int = 5) -> pd.DataFrame:
    """Sequential numerical pipeline: Drop Low Variance -> Handle Outliers -> Scale -> Remove Collinear -> Feature Selection / PCA."""
    df = df.copy()
    # C1. Separate features (X) and target (y) at the start
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
    # C2. Process features only
    X = drop_low_variance(X)
    X = handle_outliers(X)
    X = scale_numeric_features(X)
    if len(X.columns) > 1:
        X = remove_correlated_features(X, X.columns.tolist())
    # C3. Apply Feature Selection or PCA on X using y
    if y is not None and task_type:
        if use_pca and X.shape[1] > n_components:
            X = apply_pca(X, n_components=n_components)
        elif X.shape[1] > k:
            X = select_best_features(X, y, task_type=task_type, k=k)
    # C4. Re-attach target column back to dataframe
    if y is not None:
        X[target_col] = y
    return X