import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
                             RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

# If the smallest class is below this share of the data, we treat the data as imbalanced
IMBALANCE_LIMIT = 0.30


class BalancedGradientBoosting(GradientBoostingClassifier):
    """GradientBoosting has no class_weight option, so this gives rare classes more weight while fitting."""
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight)


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
    majority_baseline = None
    imbalanced = False
    minority_share = None
    minority_count = None

    # Validate target compatibility with task_type
    if task_type == 'classification':
        if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
            raise ValueError(f"Task is 'classification' but target '{target}' is continuous float data.")
        y_cat = y.astype('category')
        if len(y_cat.cat.categories) <= 20:
            label_mapping = {int(i): str(c) for i, c in enumerate(y_cat.cat.categories)}
        y = y_cat.cat.codes
        shares = y.value_counts(normalize=True)
        majority_baseline = float(shares.max())
        minority_share = float(shares.min())
        minority_count = int(y.value_counts().min())
        imbalanced = minority_share < IMBALANCE_LIMIT
    elif task_type == 'regression':
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Task is 'regression' but target '{target}' is non-numeric/string data.")
        target_skew = float(y.skew())
        use_log = bool(target_skew > 1 and y.min() >= 0)

    # Keep the same class ratio in train and test (only if every class has at least 2 rows)
    can_stratify = task_type == 'classification' and minority_count >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )

    # Select model (clone so we never change the shared model in MODELS)
    model = clone(MODELS[task_type][model_name])
    class_weighting_used = False
    if task_type == 'classification' and imbalanced:
        if "class_weight" in model.get_params():
            model.set_params(class_weight="balanced")
            class_weighting_used = True
        elif isinstance(model, GradientBoostingClassifier):
            model = BalancedGradientBoosting(random_state=42)
            class_weighting_used = True
    if use_log:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    cv_model = model
    class_counts_train = None
    class_counts_test = None
    per_class = None
    test_accuracy_plain = None
    metric_baseline = None

    if task_type == 'classification':
        if imbalanced:
            score_fn, scoring, metric = balanced_accuracy_score, 'balanced_accuracy', 'balanced_accuracy'
            metric_baseline = 1.0 / y.nunique()
        else:
            score_fn, scoring, metric = accuracy_score, 'accuracy', 'accuracy'
            metric_baseline = majority_baseline
        training_accuracy = score_fn(y_train, train_preds)
        testing_accuracy = score_fn(y_test, test_preds)
        test_accuracy_plain = accuracy_score(y_test, test_preds)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring=scoring)

        # Class counts and per-class precision / recall / F1 on the test set
        labels = sorted(int(c) for c in y.unique())
        names = {c: (label_mapping or {}).get(c, str(c)) for c in labels}
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
        cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='r2')

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
        "majority_baseline": majority_baseline,
        "imbalanced": imbalanced,
        "minority_share": minority_share,
        "minority_count": minority_count,
        "class_weighting_used": class_weighting_used,
        "metric_baseline": metric_baseline,
        "test_accuracy_plain": test_accuracy_plain,
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
        "per_class": per_class,
    }