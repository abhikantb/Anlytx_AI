import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, balanced_accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from backend.tools.preprocessing import get_scaler, get_feature_selector

# If the smallest class is below this share of the data, we oversample the training rows
IMBALANCE_LIMIT = 60
# Number of features kept by feature selection
K_FEATURES = 10
# Algorithms that should NOT be oversampled (add a model name here if it doesn't need it)
NO_BALANCING = set()

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


def build_pipeline(model, task_type, n_features, balance):
    """Scale -> select features -> (oversample) -> model. All steps are fitted on training rows only."""
    steps = [
        ("scale", get_scaler()),
        ("select", get_feature_selector(task_type, k=min(K_FEATURES, n_features))),
    ]
    if balance:
        steps.append(("balance", RandomOverSampler(random_state=42)))
    steps.append(("model", model))
    return Pipeline(steps)

def train_and_evaluate_models(df, task_type, target, model_name='RandomForest'):
    # Split the data into features (X) and the column we want to predict (y)
    X = df.drop(columns=[target])
    y = df[target]

    target_skew = None
    use_log = False
    label_mapping = None
    max_class_percent = None
    min_class_percent = None
    min_class_count = None
    imbalanced = False

    if task_type == 'classification':
        # A target with many different values is a number to predict, not a class
        if y.nunique() > 20:
            raise ValueError(f"Task is 'classification' but target '{target}' has {y.nunique()} unique values, looks like a regression target.")
        # label encoding: turn class names into numbers (no -> 0, yes -> 1)
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y), index=y.index)
        # keep number -> name, so the report can show real class names
        label_mapping = {i: str(name) for i, name in enumerate(encoder.classes_)}

        class_percent = y.value_counts(normalize=True) * 100   # percent of rows in each class
        max_class_percent = float(class_percent.max())          # percent of the biggest class
        min_class_percent = float(class_percent.min())          # percent of the smallest class
        min_class_count = int(y.value_counts().min())           # rows in the smallest class
        imbalanced = min_class_percent < IMBALANCE_LIMIT / 100 * (100 / y.nunique())   # smallest class is under 60% of its fair share -> imbalanced
    elif task_type == 'regression':
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Task is 'regression' but target '{target}' is non-numeric/string data.")
        # a very skewed target is easier to learn on its log
        target_skew = float(y.skew())
        use_log = bool(target_skew > 1 and y.min() >= 0)

    # Keep the same class ratio in train and test (only if every class has at least 2 rows)
    can_stratify = task_type == 'classification' and min_class_count >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )

    # Oversample only when classes are imbalanced (and this model needs it)
    oversampling_used = task_type == 'classification' and imbalanced and model_name not in NO_BALANCING

    # Pipeline = scale -> select features -> (oversample) -> model.
    # clone() gives a fresh copy so the shared model in MODELS is never changed
    model = build_pipeline(clone(MODELS[task_type][model_name]), task_type, X.shape[1], oversampling_used)
    if use_log:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    model.fit(X_train, y_train)   # every step learns from the training rows only

    # Which features did feature selection keep
    fitted_pipeline = model.regressor_ if use_log else model
    selected_features = list(X.columns[fitted_pipeline.named_steps["select"].get_support()])

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    class_counts_train = None
    class_counts_test = None
    per_class = None
    test_accuracy_plain = None
    metric_baseline = None

    if task_type == 'classification':
        if imbalanced:
            # imbalanced -> balanced accuracy (average recall over classes)
            score_fn, scoring, metric = balanced_accuracy_score, 'balanced_accuracy', 'balanced_accuracy'
            metric_baseline = 1.0 / y.nunique()
        else:
            score_fn, scoring, metric = accuracy_score, 'accuracy', 'accuracy'
            metric_baseline = max_class_percent / 100
        training_accuracy = score_fn(y_train, train_preds)
        testing_accuracy = score_fn(y_test, test_preds)
        test_accuracy_plain = accuracy_score(y_test, test_preds)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)   # CV on train rows only, test rows stay unseen

        # Class counts and per-class precision / recall / F1 on the test set
        labels = sorted(int(c) for c in y.unique())
        names = {c: label_mapping.get(c, str(c)) for c in labels}
        class_counts_train = {names[c]: int((y_train == c).sum()) for c in labels}
        class_counts_test = {names[c]: int((y_test == c).sum()) for c in labels}
        p, r, f, s = precision_recall_fscore_support(y_test, test_preds, labels=labels, zero_division=0)
        per_class = {
            names[c]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "test_rows": int(s[i])}
            for i, c in enumerate(labels)
        }
    else:
        training_accuracy = r2_score(y_train, train_preds)
        testing_accuracy = r2_score(y_test, test_preds)
        metric = "r2_score"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')   # CV on train rows only, test rows stay unseen

    return {
        "model": model,
        "score": testing_accuracy,
        "metric": metric,
        "model_name": model_name,
        "training_accuracy": training_accuracy,
        "testing_accuracy": testing_accuracy,
        "cross_val_score": cv_scores.mean(),
        "cv_std": float(cv_scores.std()),
        "cross_val_scores": cv_scores.tolist(),
        "target_skew": target_skew,
        "target_log_transformed": use_log,
        "label_mapping": label_mapping,
        "majority_baseline": None if max_class_percent is None else max_class_percent / 100,
        "imbalanced": imbalanced,
        "minority_share": None if min_class_percent is None else min_class_percent / 100,
        "minority_count": min_class_count,
        "oversampling_used": oversampling_used,
        "selected_features": selected_features,
        "metric_baseline": metric_baseline,
        "test_accuracy_plain": test_accuracy_plain,
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
        "per_class": per_class,
        "X_test": X_test,   # test rows, needed for feature importance
        "y_test": y_test,
    }

def run_all_models(df, task_type, target):
    """Trains every model in MODELS[task_type], returns (history, best_model_name)."""
    history = []
    for model_name in MODELS[task_type]:
        # train one model and get its results
        result = train_and_evaluate_models(df, task_type, target, model_name)
        result.pop("model")   # the fitted model object can't be saved in report.json
        result.pop("score")
        result.pop("X_test")  # test rows can't be saved in report.json
        result.pop("y_test")
        # rename 3 keys to the names that utils.py and the report use
        result["train_acc"] = result.pop("training_accuracy")
        result["test_acc"] = result.pop("testing_accuracy")
        result["cv_score"] = result.pop("cross_val_score")
        history.append(result)
    # best model = highest cross-validation score
    best = max(history, key=lambda r: r["cv_score"])
    return history, best["model_name"]


def get_feature_importance(df, task_type, target, model_name, top_count=5):
    """Permutation importance: which columns the best model depends on, measured on the test rows."""
    # train the best model again (same split, so the same test rows)
    result = train_and_evaluate_models(df, task_type, target, model_name)
    # use the same score as the report (r2_score is called "r2" in sklearn)
    scoring = "r2" if result["metric"] == "r2_score" else result["metric"]
    # shuffle one column at a time in the test rows and see how much the score drops (5 times, then average)
    importance = permutation_importance(
        result["model"], result["X_test"], result["y_test"],
        scoring=scoring, n_repeats=5, random_state=42
    )
    # sort columns from biggest score drop to smallest, keep the top ones
    order = importance.importances_mean.argsort()[::-1][:top_count]
    return [
        {"feature": result["X_test"].columns[i], "score_drop": round(float(importance.importances_mean[i]), 4)}
        for i in order
    ]