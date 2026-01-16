import os

import joblib
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
    classification_report,
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

DATA_PATH = os.getenv("DATA_PATH", "data/telco_customer_churn.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "backend/models/churn_model.joblib")
COST_FP = float(os.getenv("COST_FP", "1.0"))
COST_FN = float(os.getenv("COST_FN", "5.0"))

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

columns_to_drop = ["customerID"]
df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

target_column = "Churn"
X = df_clean.drop(columns=[target_column])
y = df_clean[target_column]

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

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numeric", numeric_transformer, numeric_features),
    ]
)

estimators = [
    ("logistic", LogisticRegression(max_iter=1000)),
    ("tree", DecisionTreeClassifier(random_state=42)),
    ("lgbm", lgb.LGBMClassifier(random_state=42)),
    ("catboost", CatBoostClassifier(iterations=150, random_state=42, verbose=0)),
]

model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
    cv=5,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.05, 0.95, 91)
best_threshold = 0.5
best_cost = float("inf")
for threshold in thresholds:
    preds = (proba >= threshold).astype(int)
    tn = np.sum((y_test == 0) & (preds == 0))
    fp = np.sum((y_test == 0) & (preds == 1))
    fn = np.sum((y_test == 1) & (preds == 0))
    cost = (fp * COST_FP) + (fn * COST_FN)
    if cost < best_cost:
        best_cost = cost
        best_threshold = float(threshold)

y_pred = (proba >= best_threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, pos_label=1, average="binary", zero_division=0
)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Best threshold (cost-based): {best_threshold:.4f}")
print("Classification Report:")
print(report)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
metrics = {
    "accuracy": accuracy,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "threshold": best_threshold,
    "cost_fp": COST_FP,
    "cost_fn": COST_FN,
}

joblib.dump({"model": pipeline, "threshold": best_threshold, "metrics": metrics}, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
