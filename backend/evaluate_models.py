import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

DATA_PATH = os.getenv("DATA_PATH", "data/telco_customer_churn.csv")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
COST_FP = float(os.getenv("COST_FP", "1.0"))
COST_FN = float(os.getenv("COST_FN", "5.0"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    columns_to_drop = ["customerID"]
    df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"]
    return X, y


def build_preprocessor():
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    numeric_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ]
    )


def build_models():
    base_estimators = [
        ("logistic", LogisticRegression(max_iter=1000)),
        ("tree", DecisionTreeClassifier(random_state=42)),
        ("lgbm", lgb.LGBMClassifier(random_state=42)),
        ("catboost", CatBoostClassifier(iterations=150, random_state=42, verbose=0)),
    ]

    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=5,
    )

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(iterations=150, random_state=42, verbose=0),
        "Stacking Ensemble": ensemble,
    }


def select_threshold(y_true, proba, cost_fp, cost_fn):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_cost = float("inf")
    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        tn = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))
        cost = (fp * cost_fp) + (fn * cost_fn)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold


def score_predictions(y_true, proba, threshold):
    preds = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, pos_label=1, average="binary", zero_division=0
    )
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]

    threshold = select_threshold(y_train.values, proba, COST_FP, COST_FN)
    cv_metrics = score_predictions(y_train.values, proba, threshold)

    pipeline.fit(X_train, y_train)
    holdout_proba = pipeline.predict_proba(X_test)[:, 1]
    holdout_metrics = score_predictions(y_test.values, holdout_proba, threshold)

    return {
        "model": name,
        "cv": cv_metrics,
        "holdout": holdout_metrics,
    }


def main():
    X, y = load_data()
    preprocessor = build_preprocessor()
    models = build_models()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    metrics = [
        evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor)
        for name, model in models.items()
    ]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    json_path = os.path.join(REPORTS_DIR, "metrics.json")
    csv_path = os.path.join(REPORTS_DIR, "metrics.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    flattened = []
    for row in metrics:
        for split in ["cv", "holdout"]:
            entry = {"model": row["model"], "split": split}
            entry.update(row[split])
            flattened.append(entry)
    pd.DataFrame(flattened).to_csv(csv_path, index=False)

    print(f"Saved metrics to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
