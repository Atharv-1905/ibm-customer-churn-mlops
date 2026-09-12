import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================================
# 1. Load Dataset
# ==========================================

DATA_PATH = "data/Telco_customer_churn.xlsx"

df = pd.read_excel(DATA_PATH)


# ==========================================
# 2. Select Features and Target
# ==========================================

features = [
    "Monthly Charges",
    "Contract",
    "Paperless Billing"
]

target = "Churn Value"

X = df[features].copy()
y = df[target].copy()


# ==========================================
# 3. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Preprocessing
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
# 5. ML Pipeline
# ==========================================

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# ==========================================
# 6. Train Model
# ==========================================

print("Training model...")

model_pipeline.fit(X_train, y_train)


# ==========================================
# 7. Evaluate Model
# ==========================================

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed")
print(f"Accuracy: {accuracy:.4f}")


# ==========================================
# 8. Save Model
# ==========================================

joblib.dump(
    model_pipeline,
    "model/model.pkl"
)

print("Model saved to model/model.pkl")
