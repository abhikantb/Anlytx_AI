import pandas as pd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
                             RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

# Define the model library
MODELS = {
    'classification': {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000)
    },
    'regression': {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge()
    }
}

def train_and_evaluate_models(df, task_type, target, model_name='RandomForest'):
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]

    target_skew = None
    use_log = False
    label_mapping = None

    # Validate target compatibility with task_type
    if task_type == 'classification':
        if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
            raise ValueError(f"Task is 'classification' but target '{target}' is continuous float data.")
        y_cat = y.astype('category')
        if len(y_cat.cat.categories) <= 20:
            label_mapping = {int(i): str(c) for i, c in enumerate(y_cat.cat.categories)}
        y = y_cat.cat.codes
    elif task_type == 'regression':
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Task is 'regression' but target '{target}' is non-numeric/string data.")
        target_skew = float(y.skew())
        use_log = bool(target_skew > 1 and y.min() >= 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    model = MODELS[task_type][model_name]
    if use_log:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    cv_model = model
    if task_type == 'classification':
        training_accuracy = accuracy_score(y_train, train_preds)
        testing_accuracy = accuracy_score(y_test, test_preds)
        metric = "accuracy"
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='accuracy')
    else:
        training_accuracy = r2_score(y_train, train_preds)
        testing_accuracy = r2_score(y_test, test_preds)
        metric = "r2_score"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='r2')

    return {
        "model": model,
        "score": testing_accuracy,
        "metric": metric,
        "model_name": model_name,
        "training_accuracy": training_accuracy,
        "testing_accuracy": testing_accuracy,
        "cross_val_score": cv_scores.mean(),
        "cross_val_scores": cv_scores.tolist(),
        "target_skew": target_skew,
        "target_log_transformed": use_log,
        "label_mapping": label_mapping,
    }