# local check: runs run_all_models on built-in sklearn datasets (no GitHub, no Horizon, no n8n)
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from backend.tools.ml_models import run_all_models


def show(title, df, task_type, target):
    # run all models and print one short line per model
    print("=====", title, "| rows:", len(df))
    history, best = run_all_models(df, task_type, target)
    for r in history:
        print(r["model_name"], "| metric:", r["metric"],
              "| test:", round(r["test_acc"], 3),
              "| cv:", round(r["cv_score"], 3),
              "| oversampled:", r["oversampling_used"])
        print("    kept features:", r["selected_features"])
    print("best model:", best)


# test 1: balanced classification (3 classes, equal rows in each)
iris = load_iris(as_frame=True).frame
show("balanced classification", iris, "classification", "target")

# test 2: imbalanced classification (keep all class 1 rows, only 40 rows of class 0)
cancer = load_breast_cancer(as_frame=True).frame
cancer = pd.concat([cancer[cancer["target"] == 1], cancer[cancer["target"] == 0].head(40)])
show("imbalanced classification", cancer, "classification", "target")

# test 3: regression
diabetes = load_diabetes(as_frame=True).frame
show("regression", diabetes, "regression", "target")

from backend.tools.ml_models import get_feature_importance
print(get_feature_importance(iris, "classification", "target", "RandomForest"))
print(get_feature_importance(cancer, "classification", "target", "LogisticRegression"))
print(get_feature_importance(diabetes, "regression", "target", "Ridge"))