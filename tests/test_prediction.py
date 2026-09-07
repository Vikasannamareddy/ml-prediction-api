import os
import json
import joblib
import pandas as pd
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "churn_model.joblib"
)
THRESHOLD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "threshold.json"
)
def load_model():
    return joblib.load(MODEL_PATH)
def load_threshold():
    with open(THRESHOLD_PATH, "r") as f:
        return json.load(f)["threshold"]
def create_sample_customer():
    return pd.DataFrame({
        "Tenure Months": [12],
        "Monthly Charges": [70.0],
        "Total Charges": [1000.0],
        "CLTV": [5000],
        "Gender": ["Male"],
        "Senior Citizen": ["No"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "Phone Service": ["Yes"],
        "Multiple Lines": ["No"],
        "Internet Service": ["Fiber optic"],
        "Online Security": ["No"],
        "Online Backup": ["Yes"],
        "Device Protection": ["No"],
        "Tech Support": ["No"],
        "Streaming TV": ["Yes"],
        "Streaming Movies": ["Yes"],
        "Contract": ["Month-to-month"],
        "Paperless Billing": ["Yes"],
        "Payment Method": ["Electronic check"]
    })
def test_prediction_probability():
    model = load_model()
    customer = create_sample_customer()
    probability = model.predict_proba(customer)[0][1]
    assert 0 <= probability <= 1
def test_prediction_output():
    model = load_model()
    threshold = load_threshold()
    customer = create_sample_customer()
    probability = model.predict_proba(customer)[0][1]
    prediction = bool(probability >= threshold)
    assert isinstance(prediction, bool)