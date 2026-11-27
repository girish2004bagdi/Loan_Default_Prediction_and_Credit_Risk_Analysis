"""Train and save the loan-default model used by the Streamlit app."""
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "credit_risk_dataset.csv"
MODEL_PATH = ROOT / "model.pkl"

TARGET = "loan_status"
FEATURES = [
    "person_age", "person_income", "person_home_ownership",
    "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
    "loan_int_rate", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
    "cred_hist_to_age",
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "credit_risk_dataset.csv was not found. Put it beside train_model.py."
        )

    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    # Apply the same basic quality rules documented in the notebook.
    df = df.drop_duplicates()
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)].copy()
    df["cred_hist_to_age"] = df["cb_person_cred_hist_length"] / df["person_age"]
    df = df.dropna(subset=[TARGET])

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)

    # Print held-out metrics for transparency; do not claim results before this runs.
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
