import pandas as pd
import joblib


# Load trained pipeline
model = joblib.load("model/model.pkl")


# New customer data
customer = pd.DataFrame({
    "Monthly Charges": [85.5],
    "Contract": ["Month-to-month"],
    "Paperless Billing": ["Yes"]
})


# Make prediction
prediction = model.predict(customer)[0]


# Display result
if prediction == 1:
    print("Churn Prediction: Yes")
else:
    print("Churn Prediction: No")
