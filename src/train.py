import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. MLflow Experiment
# ==========================================

mlflow.set_experiment("IBM-Customer-Churn")


# ==========================================
# 2. Load Dataset
# ==========================================

DATA_PATH = "data/Telco_customer_churn.xlsx"

print("Loading dataset...")

df = pd.read_excel(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 3. Select Features and Target
# ==========================================

features = [
    "Monthly Charges",
    "Contract",
    "Paperless Billing"
]

target = "Churn Value"

X = df[features].copy()
y = df[target].copy()

print("\nSelected features:")
print(features)

print("\nTarget:")
print(target)


# ==========================================
# 4. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 5. Define Preprocessing
# ==========================================

numeric_features = [
    "Monthly Charges"
]

categorical_features = [
    "Contract",
    "Paperless Billing"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==========================================
# 6. Create ML Pipeline
# ==========================================

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)


# ==========================================
# 7. Train Model with MLflow
# ==========================================

print("\nStarting model training...")

with mlflow.start_run():

    # Train
    model_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = model_pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # --------------------------------------
    # Log parameters
    # --------------------------------------

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "features",
        ", ".join(features)
    )

    # --------------------------------------
    # Log metrics
    # --------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # --------------------------------------
    # Log model to MLflow
    # --------------------------------------

    mlflow.sklearn.log_model(
        model_pipeline,
        "model"
    )

    # --------------------------------------
    # Display results
    # --------------------------------------

    print("\n========================================")
    print("MODEL TRAINING COMPLETED")
    print("========================================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ==========================================
# 8. Save Model Locally
# ==========================================

joblib.dump(
    model_pipeline,
    "model/model.pkl"
)

print("\n========================================")
print("Model saved to model/model.pkl")
print("========================================")
