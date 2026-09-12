from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib


# Load trained model
model = joblib.load("model/model.pkl")

app = FastAPI(
    title="IBM Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0"
)


# Input schema
class CustomerData(BaseModel):
    monthly_charges: float = Field(alias="Monthly Charges")
    contract: str = Field(alias="Contract")
    paperless_billing: str = Field(alias="Paperless Billing")

    model_config = {
        "populate_by_name": True
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([{
        "Monthly Charges": data.monthly_charges,
        "Contract": data.contract,
        "Paperless Billing": data.paperless_billing
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        churn = "Yes"
    else:
        churn = "No"

    return {
        "churn_prediction": churn,
        "churn_value": int(prediction)
    }
