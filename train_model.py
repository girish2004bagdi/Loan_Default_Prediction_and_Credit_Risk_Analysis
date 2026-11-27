
# ============================================================
# LOAN DEFAULT PREDICTION & CREDIT RISK ANALYSIS
# Complete ML Training, Evaluation and Visualization Pipeline
# ============================================================

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
IMAGE_DIR = BASE_DIR / "images"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("Original dataset shape:", df.shape)

    required_columns = {
        "loan_status",
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    return df


# ============================================================
# 3. DATA CLEANING AND FEATURE ENGINEERING
# ============================================================

def preprocess_dataset(df):

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove invalid age and employment length records
    df = df[
        (df["person_age"] > 0) &
        (df["person_age"] <= 100) &
        (
            df["person_emp_length"].isna() |
            (df["person_emp_length"] <= 60)
        )
    ].copy()

    # Feature Engineering
    df["cred_hist_to_age"] = (
        df["cb_person_cred_hist_length"] /
        df["person_age"].replace(0, np.nan)
    )

    print("\nData cleaning completed!")
    print("Cleaned dataset shape:", df.shape)

    print("\nTarget distribution:")
    print(df["loan_status"].value_counts())

    return df


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    if y.nunique() != 2 or set(y.unique()) != {0, 1}:
        raise ValueError(
            "This training script expects loan_status labels 0 and 1."
        )

    # Identify numerical and categorical features
    numeric_features = [
        col for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col])
    ]

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    print("\nNumerical features:", numeric_features)
    print("\nCategorical features:", categorical_features)

    return X, y, numeric_features, categorical_features


# ============================================================
# 5. BUILD PREPROCESSING PIPELINE
# ============================================================

def build_pipeline(numeric_features, categorical_features):

    numeric_transformer = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_transformer = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ])

    return pipeline


# ============================================================
# 6. TRAIN MODEL
# ============================================================

def train_model(pipeline, X_train, y_train):

    print("\nTraining Random Forest model...")

    pipeline.fit(X_train, y_train)

    print("Model training completed!")

    return pipeline


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

def evaluate_model(pipeline, X_test, y_test):

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, y_probability
        )
    }

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)

    for name, score in metrics.items():
        print(f"{name}: {score:.4f} ({score:.2%})")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Default", "Default"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    return y_pred, y_probability, metrics


# ============================================================
# 8. VISUALIZATION: CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_test, y_pred):

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["No Default", "Default"],
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        "Loan Default Prediction - Confusion Matrix",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix saved.")


# ============================================================
# 9. VISUALIZATION: ROC CURVE
# ============================================================

def plot_roc_curve(y_test, y_probability):

    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_test,
        y_probability,
        ax=ax,
        name="Random Forest"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_title(
        "ROC Curve - Loan Default Prediction",
        fontsize=14
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("ROC curve saved.")


# ============================================================
# 10. VISUALIZATION: PRECISION-RECALL CURVE
# ============================================================

def plot_precision_recall_curve(y_test, y_probability):

    fig, ax = plt.subplots(figsize=(8, 6))

    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_probability,
        ax=ax,
        name="Random Forest"
    )

    ax.set_title(
        "Precision-Recall Curve",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Precision-Recall curve saved.")


# ============================================================
# 11. VISUALIZATION: MODEL PERFORMANCE
# ============================================================

def plot_model_performance(metrics):

    fig, ax = plt.subplots(figsize=(9, 6))

    names = list(metrics.keys())
    scores = list(metrics.values())

    bars = ax.bar(
        names,
        scores
    )

    ax.set_ylim(0, 1.10)

    ax.set_ylabel("Score")

    ax.set_title(
        "Random Forest - Model Performance Metrics",
        fontsize=14
    )

    ax.bar_label(
        bars,
        labels=[f"{score:.2%}" for score in scores],
        padding=4
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "model_performance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Model performance chart saved.")


# ============================================================
# 12. VISUALIZATION: FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(pipeline):

    preprocessor = pipeline.named_steps["preprocessor"]

    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    importance_values = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    top_features = importance_df.head(15).sort_values(
        by="Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    ax.set_title(
        "Top 15 Feature Importances - Random Forest",
        fontsize=14
    )

    ax.set_xlabel("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Save feature importance values
    importance_df.to_csv(
        BASE_DIR / "feature_importance.csv",
        index=False
    )

    print("Feature importance chart saved.")


# ============================================================
# 13. SAVE EVALUATION METRICS
# ============================================================

def save_metrics(metrics):

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Score"]
    )

    metrics_df.to_csv(
        BASE_DIR / "model_metrics.csv",
        index=False
    )

    print("Evaluation metrics saved to model_metrics.csv")


# ============================================================
# 14. SAVE TRAINED MODEL
# ============================================================

def save_model(pipeline):

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print("\nModel saved successfully!")
    print("Model path:", MODEL_PATH)


# ============================================================
# 15. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("LOAN DEFAULT PREDICTION & CREDIT RISK ANALYSIS")
    print("=" * 60)

    # Create images folder
    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load dataset
    df = load_dataset()

    # Clean and engineer features
    df = preprocess_dataset(df)

    # Prepare features
    X, y, numeric_features, categorical_features = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Build pipeline
    pipeline = build_pipeline(
        numeric_features,
        categorical_features
    )

    # Train
    pipeline = train_model(
        pipeline,
        X_train,
        y_train
    )

    # Evaluate
    y_pred, y_probability, metrics = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    # Generate visualizations
    print("\nGenerating graphs...")

    plot_confusion_matrix(y_test, y_pred)

    plot_roc_curve(y_test, y_probability)

    plot_precision_recall_curve(
        y_test,
        y_probability
    )

    plot_model_performance(metrics)

    plot_feature_importance(pipeline)

    # Save evaluation results
    save_metrics(metrics)

    # Save model
    save_model(pipeline)

    print("\n" + "=" * 60)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nGenerated files:")
    print("- model.pkl")
    print("- model_metrics.csv")
    print("- feature_importance.csv")
    print("- images/confusion_matrix.png")
    print("- images/roc_curve.png")
    print("- images/precision_recall_curve.png")
    print("- images/model_performance.png")
    print("- images/feature_importance.png")


if __name__ == "__main__":
    main()
