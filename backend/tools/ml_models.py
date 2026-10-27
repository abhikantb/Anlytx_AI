import pandas as pd
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

    # Validate target compatibility with task_type
    if task_type == 'classification':
        if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
            raise ValueError(f"Task is 'classification' but target '{target}' is continuous float data.")
        # Ensure target is discrete/categorical for classification metrics and StratifiedKFold
        y = y.astype('category').cat.codes
    elif task_type == 'regression':
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Task is 'regression' but target '{target}' is non-numeric/string data.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    model = MODELS[task_type][model_name]
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    cv_model = MODELS[task_type][model_name]
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
        "cross_val_scores": cv_scores.tolist()
    }